import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import gguf
import numpy as np

from experiments.full22 import archive, codec, harness
from test_full22_integrity import fixture


class DecodedCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.names, _ = fixture(self.root)
        # Include an exact positive zero so the signed-zero test changes only its sign bit.
        offset = gguf.GGUFReader(self.source).tensors[0].data_offset
        data = bytearray(self.source.read_bytes())
        data[offset:offset+4] = bytes(4)
        self.source.write_bytes(data)
        for name in ("fit.acal", "selection.acal"):
            path = self.root/name
            data = bytearray(path.read_bytes())
            data[4:36] = hashlib.sha256(self.source.read_bytes()).digest()
            data[-32:] = hashlib.sha256(data[:-32]).digest()
            path.write_bytes(data)
        self.model = self.root/"model.a22"
        with contextlib.redirect_stdout(io.StringIO()):
            harness.build_model({"family":"lossless"},self.model,self.source,self.root/"fit.acal")

    def rewrite_manifest(self, change, filename="altered.a22"):
        destination = self.root/filename
        with zipfile.ZipFile(self.model) as source, zipfile.ZipFile(destination,"w") as target:
            for name in source.namelist():
                data = source.read(name)
                if name == "manifest.json":
                    manifest = json.loads(data)
                    change(manifest)
                    data = json.dumps(manifest).encode()
                target.writestr(name,data)
        return destination

    def test_builder_commits_actual_source_bytes_for_lossless_tensors(self):
        tensors = {t.name:t for t in gguf.GGUFReader(self.source).tensors}
        with zipfile.ZipFile(self.model) as source:
            manifest = json.loads(source.read("manifest.json"))
        self.assertEqual(manifest["format"],"A22-2")
        for entry in manifest["tensors"]:
            self.assertEqual(entry["decoded_sha256"],hashlib.sha256(tensors[entry["name"]].data.tobytes()).hexdigest())
        result = archive.export_model(self.model,self.root/"decoded.gguf")
        self.assertEqual(result["decoded_verified_tensors"],2)
        self.assertEqual((self.root/"decoded.gguf").read_bytes(),self.source.read_bytes())

    def test_required_commitments_are_validated_before_decoding(self):
        for index, value in enumerate((None,False,"0"*63,"0"*65,"z"*64)):
            def mutate(manifest):
                if value is None:
                    manifest["tensors"][0].pop("decoded_sha256")
                else:
                    manifest["tensors"][0]["decoded_sha256"] = value
            model = self.rewrite_manifest(mutate,f"invalid-{index}.a22")
            with self.subTest(value=value), patch.object(codec,"decode") as decode:
                with self.assertRaisesRegex(ValueError,"decoded tensor commitment"):
                    archive.export_model(model,self.root/f"invalid-{index}.gguf")
                decode.assert_not_called()
            self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_bit_faults_fail_before_publication_including_signed_zero(self):
        original = codec.decode
        for name, index, mask in (("ulp",1,1),("signed-zero",0,0x80000000)):
            def inject(*args, **kwargs):
                values = original(*args,**kwargs).copy()
                values.view(np.uint32).flat[index] ^= np.uint32(mask)
                return values
            destination = self.root/(name+".gguf")
            with self.subTest(name=name), patch.object(codec,"decode",side_effect=inject):
                with self.assertRaises(archive.DecodedChecksumError) as raised:
                    archive.export_model(self.model,destination)
            self.assertEqual(raised.exception.tensor,self.names[0])
            self.assertNotEqual(raised.exception.actual,raised.exception.expected)
            self.assertFalse(destination.exists())
            self.assertFalse(list(self.root.glob(".*.tmp")))

    def test_legacy_archive_remains_compatible_without_false_verification_credit(self):
        def legacy(manifest):
            manifest["format"] = "A22-1"
            for entry in manifest["tensors"]:
                entry.pop("decoded_sha256")
        model = self.rewrite_manifest(legacy,"legacy.a22")
        result = archive.export_model(model,self.root/"legacy.gguf")
        self.assertEqual(result["format"],"A22-1")
        self.assertEqual(result["decoded_verified_tensors"],0)
        self.assertEqual((self.root/"legacy.gguf").read_bytes(),self.source.read_bytes())

    def test_bad_commitment_preserves_existing_destination(self):
        def mutate(manifest):
            manifest["tensors"][0]["decoded_sha256"] = "0"*64
        model = self.rewrite_manifest(mutate)
        destination = self.root/"existing.gguf"
        destination.write_bytes(b"keep this output")
        with self.assertRaisesRegex(ValueError,"destination exists"):
            archive.export_model(model,destination)
        self.assertEqual(destination.read_bytes(),b"keep this output")
        with self.assertRaises(archive.DecodedChecksumError):
            archive.export_model(model,self.root/"new.gguf")
        self.assertFalse((self.root/"new.gguf").exists())
        self.assertFalse(list(self.root.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
