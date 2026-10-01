"""Cross-language native CLI tests; Python only produces independent fixtures."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from experiments.full22 import codec
from experiments.full22.data import ROOT
from experiments.full22.records import Record,raw_record

BINARY=ROOT/"target/native-a22-current/release/atom-quantizer"


def fixture(path, records):
    prefix=bytearray(b"GGUF"+struct.pack("<IQQ",3,len(records),0)); body=bytearray(); entries=[]
    expected={}
    for index,(name,record) in enumerate(records):
        values=codec.decode(record)
        body.extend(bytes((-len(body))%32)); offset=len(body)
        body.extend(values.astype('<f4').tobytes()); expected[name]=values
        prefix.extend(struct.pack('<Q',len(name.encode()))+name.encode()+struct.pack('<I',values.ndim))
        for dim in reversed(values.shape): prefix.extend(struct.pack('<Q',dim))
        prefix.extend(struct.pack('<IQ',0,offset))
        data=record.dumps(index%2==0)
        entries.append({'name':name,'shape':list(values.shape),'offset':offset,'entry':f'tensors/{index:04}.ar',
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'decoded_sha256':hashlib.sha256(values.astype('<f4').tobytes()).hexdigest(),'data':data})
    prefix.extend(bytes((-len(prefix))%32))
    manifest={'format':'A22-2','source_sha256':hashlib.sha256(prefix+body).hexdigest(),
        'source_bytes':len(prefix)+len(body),'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'tensors':[]}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('gguf-prefix.bin',prefix)
        for entry in entries:
            data=entry.pop('data');entry['gguf_offset']=len(prefix)+entry.pop('offset')
            manifest['tensors'].append(entry);archive.writestr(entry['entry'],data)
        archive.writestr('manifest.json',json.dumps(manifest,separators=(',',':')))
    return bytes(prefix+body)


def change_manifest(source,destination,mutate):
    with zipfile.ZipFile(source) as archive,zipfile.ZipFile(destination,'x') as target:
        for name in archive.namelist():
            data=archive.read(name)
            if name=='manifest.json':
                value=json.loads(data);mutate(value);data=json.dumps(value).encode()
            target.writestr(name,data)


def cli(*args,cwd=None):
    env=dict(os.environ,PATH='/no-python-or-tools-here')
    return subprocess.run([str(BINARY),*map(str,args)],cwd=cwd,env=env,capture_output=True,text=True)


class NativeArchiveTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BINARY.is_file(),'build the native A22 binary before integration tests')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def rejected(self,archive,phrase=None):
        verify=cli('verify',archive); self.assertNotEqual(verify.returncode,0)
        destination=self.root/'rejected.gguf'
        exported=cli('decode',archive,'--out',destination)
        self.assertNotEqual(exported.returncode,0)
        self.assertFalse(destination.exists());self.assertFalse(list(self.root.glob('.*.a22.tmp')))
        if phrase:self.assertIn(phrase,verify.stderr)

    def test_all_scalar_widths_padding_and_stored_correction_ranks_match_python(self):
        rng=np.random.default_rng(8307);records=[]
        for bits in (2,3,4,6,8):
            for rank in (0,1,2):
                weight=rng.normal(size=(5,35)).astype(np.float32)
                inputs=rng.normal(size=(8,35)).astype(np.float32)
                config={'bits':bits}
                if rank:config.update(rank=rank,rank_geometry='activation')
                records.append((f'b{bits}-r{rank}.weight',codec.encode(weight,inputs,config)))
        record=codec.encode(rng.normal(size=(3,35)).astype(np.float32),config={'bits':6,'scale_dtype':'f32'})
        records.append(('scales-f32.weight',record))
        record=codec.encode(rng.normal(size=(5,35)).astype(np.float32),rng.normal(size=(8,35)).astype(np.float32),{'bits':6,'rank':2,'rank_geometry':'diagonal'})
        records.append(('diagonal.weight',record))
        zero=codec.encode(np.arange(70,dtype=np.float32).reshape(2,35),config={'bits':4})
        zero.arrays.update(lowrank_u=np.empty((2,0),np.float16),lowrank_v=np.empty((0,35),np.float16));zero.meta['lowrank_effective_rank']=0
        records.append(('zero-rank.weight',zero))
        archive=self.root/'fixture.a22'; expected=fixture(archive,records)
        verify=cli('verify',archive,cwd=self.root);self.assertEqual(verify.returncode,0,verify.stderr)
        output=self.root/'model.gguf';result=cli('decode',archive,'--out',output,cwd=self.root)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(output.read_bytes(),expected)

    def test_raw_bf16_f32_constant_and_half_scale_extremes(self):
        raw=np.array([0,0x80000000,1,0x80000001,0x7f7fffff],np.uint32).view(np.float32)
        bf=np.array([0,0x80000000,0x00010000,0x80010000,0x7f7f0000],np.uint32).view(np.float32)
        constant=Record({'kind':'constant','shape':[2,7]},{'value':np.array([-0.0],np.float32)})
        tiny=codec.encode(np.linspace(-1e-9,1e-9,64,dtype=np.float32).reshape(2,32),config={'bits':6})
        archive=self.root/'raw.a22';expected=fixture(archive,[('f32',raw_record(raw)),('bf16',raw_record(bf)),('constant',constant),('tiny',tiny)])
        output=self.root/'raw.gguf';result=cli('decode',archive,'--out',output)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(output.read_bytes(),expected)

    def test_malformed_commitments_shapes_offsets_and_extra_members_are_rejected(self):
        source=self.root/'good.a22';fixture(source,[('weight',raw_record(np.arange(32,dtype=np.float32)))])
        mutations=[lambda m:m['tensors'][0].pop('decoded_sha256'),lambda m:m['tensors'][0].update(decoded_sha256='0'*64),
            lambda m:m['tensors'][0].update(shape=[1,32]),lambda m:m['tensors'][0].update(gguf_offset=0),
            lambda m:m.update(source_bytes=m['source_bytes']+1024),lambda m:m.update(format='A22-1')]
        for i,mutate in enumerate(mutations):
            path=self.root/f'bad{i}.a22';change_manifest(source,path,mutate);self.rejected(path)
        extra=self.root/'extra.a22';change_manifest(source,extra,lambda m:None)
        with zipfile.ZipFile(extra,'a') as archive:archive.writestr('extra.bin',b'ignored')
        self.rejected(extra,'member set')
        missing=self.root/'missing.a22'
        with zipfile.ZipFile(source) as a,zipfile.ZipFile(missing,'x') as b:
            for name in a.namelist():
                if not name.endswith('.ar'):b.writestr(name,a.read(name))
        self.rejected(missing,'member set')

    def test_checksum_valid_bad_gguf_prefixes_are_rejected(self):
        source=self.root/'prefix.a22';fixture(source,[('first',raw_record(np.arange(32,dtype=np.float32))),('other',raw_record(np.arange(32,dtype=np.float32)))])
        with zipfile.ZipFile(source) as z:
            prefix=z.read('gguf-prefix.bin');manifest=json.loads(z.read('manifest.json'));members={n:z.read(n) for n in z.namelist() if n not in ('gguf-prefix.bin','manifest.json')}
        positions=[]; offset=24
        for _ in range(2):
            length=struct.unpack_from('<Q',prefix,offset)[0];name=offset+8; offset=name+length
            dims=struct.unpack_from('<I',prefix,offset)[0];dtype=offset+4+dims*8;positions.append((name,dtype,dtype+4));offset=dtype+12
        variants=[]
        duplicate=bytearray(prefix);duplicate[positions[1][0]:positions[1][0]+5]=b'first';variants.append(('duplicate',duplicate))
        overlap=bytearray(prefix);struct.pack_into('<Q',overlap,positions[1][2],0);variants.append(('overlap',overlap))
        dtype=bytearray(prefix);struct.pack_into('<I',dtype,positions[0][1],1);variants.append(('dtype',dtype))
        for name,changed in variants:
            metadata=json.loads(json.dumps(manifest));metadata['prefix_sha256']=hashlib.sha256(changed).hexdigest()
            path=self.root/(name+'-prefix.a22')
            with zipfile.ZipFile(path,'x') as z:
                z.writestr('gguf-prefix.bin',changed);z.writestr('manifest.json',json.dumps(metadata))
                for entry,data in members.items():z.writestr(entry,data)
            self.rejected(path)

    def test_duplicate_members_and_record_checksum_corruption_are_rejected(self):
        source=self.root/'good.a22';fixture(source,[('weight',raw_record(np.arange(32,dtype=np.float32)))])
        duplicate=self.root/'duplicate.a22';duplicate.write_bytes(source.read_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(duplicate,'a') as archive:archive.writestr('manifest.json',b'{}')
        self.rejected(duplicate,'duplicate ZIP member')
        corrupt=self.root/'corrupt.a22'
        with zipfile.ZipFile(source) as a,zipfile.ZipFile(corrupt,'x') as b:
            for name in a.namelist():
                data=a.read(name)
                if name.endswith('.ar'):data=data[:-1]+bytes([data[-1]^1])
                b.writestr(name,data)
        self.rejected(corrupt,'checksum')

    def test_checksum_valid_malformed_array_descriptors_are_rejected(self):
        source=self.root/'good.a22';fixture(source,[('weight',raw_record(np.arange(32,dtype=np.float32)))])
        with zipfile.ZipFile(source) as z:
            base=json.loads(z.read('manifest.json'));prefix=z.read('gguf-prefix.bin')
            raw=Record.loads(z.read(base['tensors'][0]['entry'])).dumps()
        size=struct.unpack_from('<I',raw,4)[0];original=json.loads(raw[8:8+size]);body=raw[8+size:-32]
        mutations=[lambda h:h['arrays'][0].update(offset=1),lambda h:h['arrays'][0].update(bytes=65),
            lambda h:h['arrays'][0].update(shape=[2,16]),lambda h:h['arrays'][0].update(dtype='|O'),
            lambda h:h['arrays'].append(dict(h['arrays'][0]))]
        for index,mutate in enumerate(mutations):
            header=json.loads(json.dumps(original));mutate(header)
            encoded=json.dumps(header,separators=(',',':')).encode();frame=b'AR01'+struct.pack('<I',len(encoded))+encoded+body
            frame+=hashlib.sha256(frame).digest();manifest=json.loads(json.dumps(base));entry=manifest['tensors'][0]
            entry['bytes']=len(frame);entry['sha256']=hashlib.sha256(frame).hexdigest()
            path=self.root/f'array-{index}.a22'
            with zipfile.ZipFile(path,'x') as z:z.writestr('gguf-prefix.bin',prefix);z.writestr(entry['entry'],frame);z.writestr('manifest.json',json.dumps(manifest))
            self.rejected(path)

    def test_zip_crc_and_truncation_are_checked_independently_of_record_hashes(self):
        source=self.root/'good.a22';fixture(source,[('weight',raw_record(np.arange(32,dtype=np.float32)))])
        original=source.read_bytes();damaged=bytearray(original)
        central=damaged.index(b'PK\x01\x02');damaged[central+16]^=1;damaged[14]^=1
        corrupt=self.root/'zip-crc.a22';corrupt.write_bytes(damaged);self.rejected(corrupt)
        for amount in (1,20,len(original)//2):
            path=self.root/f'truncated-{amount}.a22';path.write_bytes(original[:-amount]);self.rejected(path)

    def test_truncated_or_suffixed_inner_zlib_is_rejected_even_with_updated_member_hash(self):
        source=self.root/'good.a22';fixture(source,[('weight',raw_record(np.arange(64,dtype=np.float32)))])
        for name,change in [('truncated',lambda b:b[:-1]),('suffixed',lambda b:b+b'extra')]:
            target=self.root/(name+'.a22')
            with zipfile.ZipFile(source) as a,zipfile.ZipFile(target,'x') as b:
                manifest=json.loads(a.read('manifest.json'));entry=manifest['tensors'][0]
                altered=change(a.read(entry['entry']));entry['bytes']=len(altered);entry['sha256']=hashlib.sha256(altered).hexdigest()
                b.writestr('gguf-prefix.bin',a.read('gguf-prefix.bin'));b.writestr(entry['entry'],altered);b.writestr('manifest.json',json.dumps(manifest))
            self.rejected(target)

    def test_unsupported_decoder_stages_and_nonfinite_values_fail_closed(self):
        rng=np.random.default_rng(10);weights=rng.normal(size=(8,32)).astype(np.float32);inputs=rng.normal(size=(8,32)).astype(np.float32)
        for name,config in [('rotation',{'bits':4,'rotation':8}),('gain',{'bits':4,'post':'norm'}),('rank4',{'bits':4,'rank':4,'rank_geometry':'activation'})]:
            path=self.root/(name+'.a22');fixture(path,[('weight',codec.encode(weights,inputs,config))]);self.rejected(path)
        record=raw_record(np.arange(32,dtype=np.float32));path=self.root/'nonfinite.a22';fixture(path,[('weight',record)])
        with zipfile.ZipFile(path) as a:
            manifest=json.loads(a.read('manifest.json'));prefix=a.read('gguf-prefix.bin')
        bad=Record({'kind':'raw_f32','shape':[32]},{'values':np.full(32,np.nan,np.float32)}).dumps(True)
        entry=manifest['tensors'][0];entry['bytes']=len(bad);entry['sha256']=hashlib.sha256(bad).hexdigest()
        new=self.root/'nan.a22'
        with zipfile.ZipFile(new,'x') as a:a.writestr('gguf-prefix.bin',prefix);a.writestr(entry['entry'],bad);a.writestr('manifest.json',json.dumps(manifest))
        self.rejected(new,'nonfinite')

    def test_existing_outputs_and_dangling_links_are_preserved(self):
        path=self.root/'good.a22';fixture(path,[('weight',raw_record(np.arange(32,dtype=np.float32)))])
        output=self.root/'keep.gguf';output.write_bytes(b'preserve')
        self.assertNotEqual(cli('decode',path,'--out',output).returncode,0);self.assertEqual(output.read_bytes(),b'preserve')
        link=self.root/'link.gguf';link.symlink_to(self.root/'absent')
        self.assertNotEqual(cli('decode',path,'--out',link).returncode,0);self.assertTrue(link.is_symlink())
        self.assertFalse(list(self.root.glob('.*.a22.tmp')))


if __name__=='__main__':unittest.main()
