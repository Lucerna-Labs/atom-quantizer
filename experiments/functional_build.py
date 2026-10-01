"""I2 end-to-end main-app fitting and publication verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import gguf

from .basis_discovery import save_json
from .full22.data import ROOT,OUT,SOURCE,sha256

BINARY=ROOT/'target/functional-build-current/release/atom-quantizer'
D05=ROOT/'artifacts/six-bit-frontier-2026-10-01'
BASE=ROOT/'artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal'


def run(root):
    root=Path(root).resolve(); output=root/'functional.a22';work=root/'fit-work'
    if output.exists() or work.exists():raise ValueError('I2 output/work already exists')
    sources=[*ROOT.glob('src/**/*.rs'),*ROOT.glob('experiments/full22/*.py'),ROOT/'Cargo.toml',ROOT/'Cargo.lock',Path(__file__),ROOT/'research/functional-build-2026-10-01/PROTOCOL.md',BINARY,SOURCE,BASE,OUT/'fit.acal']
    frozen={str(p):sha256(p) for p in sources};save_json(root/'manifest.json',{'sha256':frozen})
    command=[str(BINARY),'build-functional',str(SOURCE),'--base-calibration',str(BASE),'--residual-calibration',str(OUT/'fit.acal'),'--out',str(output),'--work-dir',str(work)]
    started=time.perf_counter()
    with (root/'build.stdout').open('x') as stdout,(root/'build.stderr').open('x') as stderr:
        completed=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,timeout=3600)
    if completed.returncode!=0:raise ValueError(f'main build failed with exit{completed.returncode}')
    receipt=json.loads((root/'build.stdout').read_text())
    saved=json.loads((work/'result.json').read_text())
    if receipt!=saved or receipt['status']!='PUBLISHED' or receipt['fitting_backend']!='python' or not receipt['native_consumer']:
        raise ValueError('main workflow did not publish the declared result')
    reference=json.loads((D05/'summary.json').read_text())['models']['six-activation']
    if (sha256(output)!=reference['archive_sha256'] or output.stat().st_size!=reference['archive_bytes']
            or receipt['archive_sha256']!=reference['archive_sha256'] or receipt['decoded_sha256']!=reference['decoded_sha256']):
        raise ValueError('main build changed the validated D05 representation')
    decoded=root/'standalone.gguf'
    exported=subprocess.run([str(BINARY),'decode',str(output),'--out',str(decoded)],env=dict(os.environ,PATH='/no-python-or-tools-here'),capture_output=True,text=True,check=True)
    (root/'decode.log').write_text(exported.stdout+exported.stderr)
    if sha256(decoded)!=reference['decoded_sha256']:raise ValueError('main native export differs')
    old={t.name:t for t in gguf.GGUFReader(reference['decoded']).tensors};new={t.name:t for t in gguf.GGUFReader(decoded).tensors}
    if len(new)!=272 or old.keys()!=new.keys():raise ValueError('incomplete native model')
    for name,tensor in new.items():
        if tensor.data.tobytes()!=old[name].data.tobytes():raise ValueError('native tensor bytes differ')
    reports=[]
    for item in receipt['native_reports']:
        path=Path(item['path']);report=json.loads(path.read_text())
        if (sha256(path)!=item['sha256'] or not report['all_passed'] or report['tensor_count']!=272
                or report['candidate']!=str(work/'native.gguf') or report['candidate_sha256']!=reference['decoded_sha256']):
            raise ValueError('independent parent native report is incomplete')
        reports.append({'path':str(path),'calibration_sha256':report['calibration_sha256'],'passed':report['passed_count']})
    if len(reports)!=2:raise ValueError('missing fitting view')
    verified=json.loads((work/'verified.json').read_text())
    if verified['status']!='NATIVE_VERIFIED' or verified['archive_sha256']!=reference['archive_sha256']:
        raise ValueError('pre-publication verification receipt missing')
    if any(sha256(p)!=h for p,h in frozen.items()):raise ValueError('I2 implementation or inputs changed')
    result={'status':'PASSED','command':command,'returncode':0,'seconds':time.perf_counter()-started,
        'receipt':receipt,'native_reports':reports,'tensor_bytes_compared':272,'archive_identical_to_D05':True,
        'native_export_identical_to_D05':True,'backend_stdout_sha256':sha256(work/'backend.stdout'),
        'backend_stderr_sha256':sha256(work/'backend.stderr'),'new_inference_runs':0}
    save_json(root/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='receipt'}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);run(parser.parse_args().out)
