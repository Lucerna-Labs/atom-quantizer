import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from experiments.full22 import archive, codec, harness, native, precision
from experiments.full22.data import ROOT, sha256
from test_full22_integrity import fixture

ASSESSOR = ROOT/"target/native-admission-current/release/atom-quantizer"


class AdaptivePrecisionTests(unittest.TestCase):
    def test_starting_precision_and_geometry_are_explicit_and_validated(self):
        for start in (2,3,16,True):
            with self.assertRaises(ValueError):
                precision.run("unused","unused",None,[],"unused","unused",start_bits=start)
        with self.assertRaises(ValueError):
            precision.run("unused","unused",None,[],"unused","unused",rank_geometry="diagonal")
        for geometry in ("unknown","",[]):
            with self.assertRaisesRegex(ValueError,"invalid correction geometry"):
                precision.run("unused","unused","unused",[],"unused","unused",rank_geometry=geometry)

    def test_six_bit_diagonal_cli_uses_the_declared_global_configuration(self):
        self.assertTrue(ASSESSOR.is_file())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,_,_=fixture(root); out=root/"six"
            command=[sys.executable,"-m","experiments.full22.harness","build-adaptive","--source",str(source),
                "--base-calibration",str(root/"fit.acal"),"--residual-calibration",str(root/"selection.acal"),
                "--gate-calibration",str(root/"fit.acal"),"--gate-calibration",str(root/"selection.acal"),
                "--assessor",str(ASSESSOR),"--out",str(out),"--start-bits","6","--rank-geometry","diagonal"]
            subprocess.run(command,check=True,capture_output=True,text=True)
            result=json.loads((out/"result.json").read_text())
            self.assertEqual(result["start_bits"],6)
            self.assertEqual(result["rank_geometry"],"diagonal")
            self.assertTrue(all(level==1 for level in result["stages"][0]["levels"].values()))
            self.assertLessEqual(len(result["stages"]),3)
            with zipfile.ZipFile(result["stages"][0]["model"]["archive"]) as archive:
                manifest=json.loads(archive.read("manifest.json"))
                for row in manifest["tensors"]:
                    self.assertEqual(row["config"],{"bits":6,"rank":2,"rank_geometry":"diagonal"})

    def test_union_and_monotone_bounded_escalation(self):
        levels = {"first":0,"second":1,"passed":0,"exact":3}
        views = [{"report":{"tensors":[{"name":n,"passed":n!="first"} for n in levels]}},
                 {"report":{"tensors":[{"name":n,"passed":n!="second"} for n in levels]}}]
        failures = precision.failure_union(views)
        self.assertEqual(failures,{"first","second"})
        self.assertEqual(precision.advance(levels,failures),{"first":1,"second":2,"passed":0,"exact":3})
        self.assertEqual(levels["first"],0)
        for bad in ({"exact"},{"missing"}):
            with self.assertRaises(ValueError): precision.advance(levels,bad)
        with self.assertRaises(ValueError): precision.failure_union([])

    def test_passing_records_are_reused_without_calling_the_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,names,_=fixture(root)
            config={"bits":4,"rank":2,"rank_geometry":"activation"}
            first,second=root/"first.a22",root/"second.a22"
            with contextlib.redirect_stdout(io.StringIO()):
                harness.build_model(config,first,source,root/"fit.acal",residual_calibration=root/"selection.acal")
                with patch.object(codec,"encode",wraps=codec.encode) as encode:
                    harness.build_model(config,second,source,root/"fit.acal",
                        tensor_configs={names[0]:dict(config,bits=6)},residual_calibration=root/"selection.acal",
                        reuse_archive=first,reuse_sha256=sha256(first))
                    self.assertEqual(encode.call_count,1)
            with zipfile.ZipFile(first) as a,zipfile.ZipFile(second) as b:
                before=json.loads(a.read("manifest.json")); after=json.loads(b.read("manifest.json"))
                self.assertFalse(after["tensors"][0]["reused"])
                self.assertTrue(after["tensors"][1]["reused"])
                self.assertEqual(a.read(before["tensors"][1]["entry"]),b.read(after["tensors"][1]["entry"]))
            source.unlink(); (root/"fit.acal").unlink(); (root/"selection.acal").unlink()
            self.assertEqual(archive.export_model(second,root/"decoded.gguf")["decoded_verified_tensors"],2)

    def test_reuse_rejects_changed_calibration_corruption_and_external_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,_,_=fixture(root); first=root/"first.a22"
            with contextlib.redirect_stdout(io.StringIO()):
                harness.build_model({"bits":4},first,source,root/"fit.acal")
            digest=sha256(first)
            for name,cal,config,entropy in [
                ("cal",root/"selection.acal",{"bits":4},True),
                ("external",root/"fit.acal",{"bits":4,"basis_path":"unused.npy"},True),
                ("entropy",root/"fit.acal",{"bits":4},False),
            ]:
                destination=root/(name+".a22")
                with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
                    harness.build_model(config,destination,source,cal,entropy=entropy,reuse_archive=first,reuse_sha256=digest)
                self.assertFalse(destination.exists())
            damaged=root/"damaged.a22"; data=bytearray(first.read_bytes()); data[len(data)//2]^=1; damaged.write_bytes(data)
            with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
                harness.build_model({"bits":4},root/"bad.a22",source,root/"fit.acal",reuse_archive=damaged,reuse_sha256=digest)
            self.assertEqual(sha256(first),digest)
            self.assertFalse((root/"bad.a22").exists())

    def test_native_reports_cannot_lie_about_identity_coverage_or_policy(self):
        self.assertTrue(ASSESSOR.is_file(),"build the N1 native assessor before these integration tests")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,_,_=fixture(root); cal=root/"fit.acal"
            completed=subprocess.run([str(ASSESSOR),"assess",str(source),str(source),"--calibration",str(cal)],check=True,capture_output=True,text=True)
            report=json.loads(completed.stdout)
            self.assertEqual(native.validate(report,source,source,cal,0),set())
            bad=[]
            value=copy.deepcopy(report); value["candidate_sha256"]="0"*64; bad.append(value)
            value=copy.deepcopy(report); value["tensors"].pop(); bad.append(value)
            value=copy.deepcopy(report); value["tensors"][1]=value["tensors"][0]; bad.append(value)
            value=copy.deepcopy(report); value["tensors"][0]["passed"]=False; bad.append(value)
            value=copy.deepcopy(report); value["tensors"][0]["limits"]["output_mse"]=1; bad.append(value)
            value=copy.deepcopy(report); value["failed_count"]=False; bad.append(value)
            for value in bad:
                with self.assertRaises(ValueError): native.validate(value,source,source,cal,0)
            with self.assertRaises(ValueError): native.validate(report,source,source,cal,2)

    def test_real_adaptive_cli_publishes_only_a_native_admitted_model(self):
        self.assertTrue(ASSESSOR.is_file(),"build the N1 native assessor before these integration tests")
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source,_,_=fixture(root); out=root/"adaptive"
            command=[sys.executable,"-m","experiments.full22.harness","build-adaptive","--source",str(source),
                "--base-calibration",str(root/"fit.acal"),"--residual-calibration",str(root/"selection.acal"),
                "--gate-calibration",str(root/"fit.acal"),"--gate-calibration",str(root/"selection.acal"),
                "--assessor",str(ASSESSOR),"--out",str(out)]
            subprocess.run(command,check=True,capture_output=True,text=True)
            result=json.loads((out/"result.json").read_text())
            self.assertEqual(result["status"],"FIT_ADMITTED")
            self.assertLessEqual(len(result["stages"]),4)
            self.assertEqual(result["stages"][-1]["failed_tensors"],[])
            before=sha256(out/"model.a22")
            failed=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(sha256(out/"model.a22"),before)
            source.unlink(); (root/"fit.acal").unlink(); (root/"selection.acal").unlink()
            report=archive.export_model(out/"model.a22",root/"again.gguf")
            self.assertEqual(report["sha256"],result["model"]["decoded_sha256"])

    def test_failed_publication_never_leaves_a_partial_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/"source"; source.write_bytes(b"measured")
            destination=root/"final"
            with self.assertRaises(ValueError): precision.publish_copy(source,destination,"0"*64)
            self.assertFalse(destination.exists()); self.assertFalse(list(root.glob(".publish-*.tmp")))
            destination.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError): precision.publish_copy(source,destination,sha256(source))
            self.assertEqual(destination.read_bytes(),b"preserve")
            self.assertFalse(list(root.glob(".publish-*.tmp")))

    def test_missing_publication_source_closes_the_staging_descriptor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); descriptors=[]
            original=precision.tempfile.mkstemp
            def capture(*args,**kwargs):
                result=original(*args,**kwargs); descriptors.append(result[0]); return result
            with patch.object(precision.tempfile,"mkstemp",side_effect=capture):
                with self.assertRaises(FileNotFoundError):
                    precision.publish_copy(root/"missing",root/"final","0"*64)
            with self.assertRaises(OSError): os.fstat(descriptors[0])
            self.assertFalse((root/"final").exists())
            self.assertFalse(list(root.glob(".publish-*.tmp")))


if __name__=="__main__": unittest.main()
