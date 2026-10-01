"""R1: capture exact decoder failures under controlled storage and thread layouts."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import platform
import sys
import zipfile

import gguf
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .full22 import codec
from .full22.data import ROOT, sha256
from .full22.records import Record

D03 = ROOT/"artifacts/calibration-roles-2026-10-01"
MODELS = ("q4-IE-r1", "q3-IE-r1", "q4-IE-r2")
OFFSETS = (0, 1, 31, 63)
THREADS = (1, 4)
LIBM = ctypes.CDLL("libm.so.6")
LIBM.fegetround.restype = ctypes.c_int
LIBM.fegetround.argtypes = []


def save(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def positioned(values, offset):
    storage = np.empty(values.nbytes+64,dtype=np.uint8)
    start = (offset-int(storage.ctypes.data)%64)%64
    result = np.ndarray(values.shape,dtype=values.dtype,buffer=storage,offset=start)
    result.view(np.uint8).reshape(-1)[:] = np.frombuffer(values.tobytes(),dtype=np.uint8)
    if result.size and result.ctypes.data%64 != offset:
        raise ValueError("alignment fixture did not use its declared offset")
    return result


def finite_half_bits():
    half = np.array([n for n in range(65536) if (n>>10)&31 != 31],dtype=np.uint16)
    full = []
    for item in half:
        n = int(item)
        sign, exponent, fraction = (n&32768)<<16,(n>>10)&31,n&1023
        if exponent:
            bits = sign|((exponent+112)<<23)|(fraction<<13)
        elif fraction:
            top = fraction.bit_length()-1
            bits = sign|((top+103)<<23)|((fraction-(1<<top))<<(23-top))
        else:
            bits = sign
        full.append(bits)
    return half, np.array(full,dtype=np.uint32)


def cast_probe():
    half, expected = finite_half_bits()
    for offset in range(64):
        actual = positioned(half.view(np.float16),offset).astype(np.float32).view(np.uint32)
        if actual.tobytes() != expected.tobytes():
            different = np.flatnonzero(actual!=expected)
            return {"status":"FAILED","offset":offset,"count":len(different),
                "examples":[[int(half[i]),int(expected[i]),int(actual[i])] for i in different[:16]]}
    return {"status":"PASSED","finite_values":len(half),"offsets":64,"comparisons":len(half)*64}


def layout(record):
    return {name:{"dtype":value.dtype.str,"shape":list(value.shape),"strides":list(value.strides),
        "offset_mod64":value.ctypes.data%64,"aligned":bool(value.flags.aligned)} for name,value in record.arrays.items()}


def capture_failure(out, identity, record, expected, actual, error=None):
    directory = out/"failure"
    directory.mkdir()
    (directory/"record.ar").write_bytes(record.dumps(True))
    np.save(directory/"expected.npy",expected,allow_pickle=False)
    report = dict(identity,layout=layout(record),rounding_mode=LIBM.fegetround(),error=error)
    if actual is not None:
        np.save(directory/"actual.npy",actual,allow_pickle=False)
        mask = expected.view(np.uint32)!=actual.view(np.uint32)
        a,b = expected[mask],actual[mask]
        report.update(differing_values=int(mask.sum()),numerical_equality=bool(np.array_equal(expected,actual)),
            signed_zero_only=bool(np.all((a==0)&(b==0))),finite=bool(np.isfinite(actual).all()),
            max_abs_difference=float(np.max(np.abs(expected.astype(float)-actual))),
            examples=[{"index":index.tolist(),"expected_bits":int(x),"actual_bits":int(y)}
                      for index,x,y in zip(np.argwhere(mask)[:16],a.view(np.uint32)[:16],b.view(np.uint32)[:16])])
    report["files"] = {p.name:sha256(p) for p in directory.iterdir()}
    save(directory/"diagnostic.json",report)
    return report


def run(out):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources = [Path(__file__),ROOT/"research/decoder-reproducibility-2026-10-01/PROTOCOL.md"]
    sources += [ROOT/"experiments/full22"/(name+".py") for name in ("codec","residual","scalar","records","transforms","data")]
    dependencies = [Path(importlib.util.find_spec("numpy._core._multiarray_umath").origin)]
    dependencies += [Path(item["filepath"]) for item in threadpool_info()]
    frozen = {str(p):sha256(p) for p in [*sources,*dependencies]}
    save(out/"runtime.json",{"numpy":np.__version__,"python":sys.version,"platform":platform.platform(),
        "blas":threadpool_info(),"rounding_mode":LIBM.fegetround(),"sha256":frozen})
    conversion = cast_probe()
    save(out/"casts.json",conversion)
    if conversion["status"] != "PASSED":
        raise ValueError("binary16 conversion failed")
    summary = json.loads((D03/"summary.json").read_text())
    results, outer_count = [], 0
    for name in MODELS:
        model = summary["models"][name]
        if sha256(model["archive"])!=model["archive_sha256"] or sha256(model["decoded"])!=model["decoded_sha256"]:
            raise ValueError("D03 model identity changed")
        reader = gguf.GGUFReader(model["decoded"])
        tensors = {t.name:t for t in reader.tensors}
        with zipfile.ZipFile(model["archive"]) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            for threads in THREADS:
                threadpool_limits(threads)
                for offset in OFFSETS:
                    count = 0
                    for entry in manifest["tensors"]:
                        original = Record.loads(archive.read(entry["entry"]))
                        record = Record(dict(original.meta),{k:positioned(v,offset) for k,v in original.arrays.items()})
                        before = record.dumps()
                        expected = np.array(tensors[entry["name"]].data,dtype=np.float32,copy=True)
                        identity = {"model":name,"tensor":entry["name"],"threads":threads,"offset":offset,
                            "archive_sha256":model["archive_sha256"],"decoded_sha256":model["decoded_sha256"],
                            "rounding_mode_before":LIBM.fegetround()}
                        try:
                            actual = codec.decode(record)
                        except Exception as error:
                            capture_failure(out,identity,record,expected,None,repr(error))
                            raise
                        if actual.tobytes()!=expected.tobytes() or LIBM.fegetround()!=identity["rounding_mode_before"]:
                            capture_failure(out,identity,record,expected,actual)
                            raise ValueError("exact decoder mismatch captured")
                        if before != record.dumps():
                            capture_failure(out,identity,record,expected,actual,"stored state mutated during decode")
                            raise ValueError("stored decoder state changed")
                        left = record.arrays.get("lowrank_u")
                        if left is not None and left.shape[1] == 1:
                            arrays = {k:v for k,v in record.arrays.items() if k not in ("lowrank_u","lowrank_v")}
                            base = codec.decode(Record(dict(record.meta),arrays))
                            product = left.astype(np.float32)[:,0,None]*record.arrays["lowrank_v"].astype(np.float32)[None,0,:]
                            outer = base+product
                            if outer.tobytes() != actual.tobytes():
                                capture_failure(out,dict(identity,comparison="elementwise outer product"),record,outer,actual)
                                raise ValueError("rank-one implementations differ")
                            outer_count += 1
                        count += 1
                    if count != 272:
                        raise ValueError("incomplete model pass")
                    results.append({"model":name,"threads":threads,"offset":offset,"tensors":count})
                    save(out/"progress.json",{"passes":results,"rank_one_outer_comparisons":outer_count})
                    print(json.dumps(results[-1]),flush=True)
        if sha256(model["archive"])!=model["archive_sha256"] or sha256(model["decoded"])!=model["decoded_sha256"]:
            raise ValueError("model changed during probe")
    if any(sha256(p)!=h for p,h in frozen.items()):
        raise ValueError("probe implementation or numerical runtime changed")
    result = {"status":"PASSED","passes":results,"tensors":sum(x["tensors"] for x in results),
        "rank_one_outer_comparisons":outer_count,"casts":conversion,"original_failure_cause":"UNDETERMINED"}
    save(out/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="passes"}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
