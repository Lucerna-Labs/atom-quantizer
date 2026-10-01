"""Actual main-app fitting plus explicit false-backend fault fixtures."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

import numpy as np

from experiments.full22 import codec,harness
from experiments.full22.data import ROOT,sha256
from experiments.full22.records import Record
from test_full22_integrity import fixture

BINARY=ROOT/'target/functional-build-current/release/atom-quantizer'
PYTHON=ROOT/'.venv/bin/python'


class FunctionalBuildTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(BINARY.is_file(),'build the I2 native binary first')
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.source,self.names,_=fixture(self.root)
        self.output=self.root/'published.a22';self.work=self.root/'work'

    def command(self,extra=(),engine=ROOT,python=PYTHON):
        return [str(BINARY),'build-functional',str(self.source),'--base-calibration',str(self.root/'fit.acal'),
            '--residual-calibration',str(self.root/'selection.acal'),'--out',str(self.output),'--work-dir',str(self.work),
            '--engine-dir',str(engine),'--python',str(python),*map(str,extra)]

    def run_command(self,**kwargs):
        return subprocess.run(self.command(**kwargs),capture_output=True,text=True)

    def archive(self):
        path=self.root/'backend.a22'
        with contextlib.redirect_stdout(io.StringIO()):
            harness.build_model({'bits':6,'rank':2,'rank_geometry':'activation'},path,self.source,self.root/'fit.acal',residual_calibration=self.root/'selection.acal')
        return path

    def fake_engine(self,archive=None):
        engine=self.root/'fake-engine';package=engine/'experiments/full22';package.mkdir(parents=True)
        (engine/'experiments/__init__.py').write_text('');(package/'__init__.py').write_text('')
        script="import sys,shutil\nfrom pathlib import Path\nout=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True)\n"
        if archive is not None:script+=f"shutil.copyfile({str(archive)!r},out/'model.a22')\n"
        script+="print('FIT_ADMITTED')\n"
        (package/'harness.py').write_text(script)
        return engine

    def rewrite(self,source,destination,manifest_change=None,record_change=None):
        with zipfile.ZipFile(source) as a,zipfile.ZipFile(destination,'x') as b:
            manifest=json.loads(a.read('manifest.json'));changed={}
            if record_change:
                entry=manifest['tensors'][0];record=Record.loads(a.read(entry['entry']));record_change(record)
                data=record.dumps(True);entry['bytes']=len(data);entry['sha256']=hashlib.sha256(data).hexdigest()
                entry['decoded_sha256']=hashlib.sha256(codec.decode(record).tobytes()).hexdigest();changed[entry['entry']]=data
            if manifest_change:manifest_change(manifest)
            for name in a.namelist():
                data=json.dumps(manifest).encode() if name=='manifest.json' else changed.get(name,a.read(name))
                b.writestr(name,data)

    def test_real_command_fits_then_independently_checks_and_publishes(self):
        result=self.run_command();self.assertEqual(result.returncode,0,result.stderr)
        receipt=json.loads(result.stdout)
        self.assertEqual(receipt['status'],'PUBLISHED');self.assertEqual(receipt['fitting_backend'],'python')
        self.assertTrue(receipt['native_consumer']);self.assertEqual(receipt['verified_tensors'],2)
        self.assertEqual(sha256(self.output),receipt['archive_sha256'])
        self.assertEqual(len(receipt['native_reports']),2)
        for item in receipt['native_reports']:
            report=json.loads(Path(item['path']).read_text());self.assertTrue(report['all_passed'])
            self.assertEqual(report['candidate'],str(self.work/'native.gguf'))
        self.source.unlink();(self.root/'fit.acal').unlink();(self.root/'selection.acal').unlink()
        decoded=self.root/'standalone.gguf'
        run=subprocess.run([str(BINARY),'decode',str(self.output),'--out',str(decoded)],env=dict(os.environ,PATH='/no-python'),capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(sha256(decoded),receipt['decoded_sha256'])

    def test_existing_destination_and_dangling_link_are_preserved_before_fitting(self):
        self.output.write_bytes(b'preserve');result=self.run_command();self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.output.read_bytes(),b'preserve');self.assertFalse(self.work.exists())
        self.output.unlink();self.output.symlink_to(self.root/'absent')
        result=self.run_command();self.assertNotEqual(result.returncode,0);self.assertTrue(self.output.is_symlink())
        self.assertFalse(self.work.exists())

    def test_invalid_options_and_missing_interpreter_do_not_start_work(self):
        for extra in (('--start-bits','4'),('--out',str(self.root/'other.a22'))):
            result=self.run_command(extra=extra);self.assertNotEqual(result.returncode,0);self.assertFalse(self.work.exists())
        result=self.run_command(python=self.root/'missing-python');self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.work.exists());self.assertFalse(self.output.exists())

    def test_wrong_source_calibration_stops_before_backend_and_preserves_failure(self):
        path=self.root/'fit.acal';data=bytearray(path.read_bytes());data[4]^=1;data[-32:]=hashlib.sha256(data[:-32]).digest();path.write_bytes(data)
        result=self.run_command();self.assertNotEqual(result.returncode,0);self.assertIn('SHA-256 mismatch',result.stderr)
        self.assertFalse(self.output.exists());self.assertTrue((self.work/'failure.json').exists())
        self.assertFalse((self.work/'backend.stdout').exists())

    def test_false_backend_success_cannot_publish_an_absent_model(self):
        result=self.run_command(engine=self.fake_engine())
        self.assertNotEqual(result.returncode,0);self.assertFalse(self.output.exists())
        self.assertIn('FIT_ADMITTED',(self.work/'backend.stdout').read_text())
        self.assertTrue((self.work/'failure.json').exists())

    def test_wrong_provenance_or_profile_is_rejected_natively(self):
        original=self.archive();changed=self.root/'wrong.a22'
        self.rewrite(original,changed,lambda m:m.update(calibration_sha256='0'*64))
        result=self.run_command(engine=self.fake_engine(changed));self.assertNotEqual(result.returncode,0)
        self.assertIn('provenance',result.stderr);self.assertFalse(self.output.exists())

    def test_wrong_correction_profile_is_not_silently_accepted(self):
        original=self.archive();changed=self.root/'diagonal-profile.a22'
        self.rewrite(original,changed,lambda m:m['config'].update(rank_geometry='diagonal'))
        result=self.run_command(engine=self.fake_engine(changed));self.assertNotEqual(result.returncode,0)
        self.assertIn('functional-six profile',result.stderr);self.assertFalse(self.output.exists())

    def test_duplicate_observation_contents_are_rejected_before_fitting(self):
        (self.root/'selection.acal').write_bytes((self.root/'fit.acal').read_bytes())
        result=self.run_command();self.assertNotEqual(result.returncode,0)
        self.assertIn('distinct contents',result.stderr);self.assertFalse(self.output.exists())
        self.assertFalse((self.work/'backend.stdout').exists())

    def test_backend_source_changes_invalidate_an_otherwise_valid_result(self):
        engine=self.fake_engine(self.archive());script=engine/'experiments/full22/harness.py'
        with script.open('a') as f:f.write("with open(__file__,'a') as changed: changed.write('# modified during fitting\\n')\n")
        result=self.run_command(engine=engine);self.assertNotEqual(result.returncode,0)
        self.assertIn('input/backend changed',result.stderr);self.assertFalse(self.output.exists())

    def test_destination_created_during_fitting_is_preserved(self):
        engine=self.fake_engine(self.archive());script=engine/'experiments/full22/harness.py'
        with script.open('a') as f:f.write(f"Path({str(self.output)!r}).write_bytes(b'preserve concurrent output')\n")
        result=self.run_command(engine=engine);self.assertNotEqual(result.returncode,0)
        self.assertEqual(self.output.read_bytes(),b'preserve concurrent output')
        self.assertFalse(list(self.root.glob('.*.functional.tmp')))

    def test_false_admission_claim_cannot_bypass_independent_native_fidelity(self):
        original=self.archive();changed=self.root/'bad-weights.a22'
        self.rewrite(original,changed,record_change=lambda r:r.arrays.update(codes=np.zeros_like(r.arrays['codes'])))
        result=self.run_command(engine=self.fake_engine(changed));self.assertNotEqual(result.returncode,0)
        self.assertIn('native fidelity view',result.stderr);self.assertFalse(self.output.exists())
        self.assertTrue((self.work/'native.gguf').exists());self.assertTrue((self.work/'native-fit-0.json').exists())

    def test_bad_decoded_commitment_fails_before_native_admission(self):
        original=self.archive();changed=self.root/'bad-commitment.a22'
        self.rewrite(original,changed,lambda m:m['tensors'][0].update(decoded_sha256='0'*64))
        result=self.run_command(engine=self.fake_engine(changed));self.assertNotEqual(result.returncode,0)
        self.assertIn('decoded commitment mismatch',result.stderr);self.assertFalse(self.output.exists())
        self.assertFalse((self.work/'native-fit-0.json').exists());self.assertFalse((self.work/'native.gguf').exists())


if __name__=='__main__':unittest.main()
