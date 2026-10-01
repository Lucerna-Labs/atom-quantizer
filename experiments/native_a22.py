"""I1 real native-consumer and isolated-export verification."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

from .basis_discovery import save_json
from .full22.data import ROOT,sha256

D05=ROOT/"artifacts/six-bit-frontier-2026-10-01"
BINARY=ROOT/"target/native-a22-current/release/atom-quantizer"


def native_model(root,name,model):
    if sha256(model["archive"])!=model["archive_sha256"] or sha256(model["decoded"])!=model["decoded_sha256"]:
        raise ValueError("D05 source artifact changed")
    expected=model["decoded_sha256"]
    destination=root/"exports"/(name+".gguf")
    commands=[[str(BINARY),"verify",model["archive"]],[str(BINARY),"decode",model["archive"],"--out",str(destination)]]
    runs=[]
    for index,command in enumerate(commands):
        started=time.perf_counter()
        run=subprocess.run(command,cwd=root,env=dict(os.environ,PATH='/no-python-or-tools-here'),capture_output=True,text=True,timeout=1800)
        log=root/"exports"/f"{name}-{index}.log";log.write_text(run.stdout+run.stderr)
        if run.returncode!=0: raise ValueError(f"native {name} failed: {run.stderr}")
        if '272' not in run.stdout or 'A22-2' not in run.stdout: raise ValueError("native confirmation incomplete")
        runs.append({"command":command,"returncode":run.returncode,"log_sha256":sha256(log),"seconds":time.perf_counter()-started})
    actual=sha256(destination)
    if actual!=expected: raise ValueError("native complete GGUF differs from measured D05 file")
    row={"name":name,"status":"PASSED","archive":model["archive"],"archive_sha256":model["archive_sha256"],
        "archive_bytes":Path(model["archive"]).stat().st_size,"decoded":str(destination),"decoded_sha256":actual,
        "verified_tensors":272,"runs":runs}
    save_json(root/"exports"/(name+".json"),row)
    print(json.dumps({"name":name,"status":"PASSED","decoded_sha256":actual}),flush=True)
    return row


def regression(root):
    baseline=ROOT/"artifacts/native-admission-2026-10-01"
    cases=json.loads((baseline/"baseline.json").read_text())["cases"]
    directory=root/"oq-regression";directory.mkdir()
    source=baseline/"regression/source.gguf"
    results=[]
    for case in cases:
        oq=directory/(case["name"]+".oq");gguf=directory/(case["name"]+".gguf")
        command=[str(BINARY),str(source),*case["options"],"--out",str(oq),"--out-gguf",str(gguf)]
        run=subprocess.run(command,capture_output=True,text=True,check=True)
        (directory/(case["name"]+".log")).write_text(run.stdout+run.stderr)
        if sha256(oq)!=case["oq_sha256"] or sha256(gguf)!=case["gguf_sha256"] or [s for s in run.stdout.splitlines() if 'cos=' in s]!=case['score_lines']:
            raise ValueError("old quantizer output changed")
        results.append({"name":case["name"],"oq_identical":True,"gguf_identical":True,"scores_identical":True})
    return results


def isolated(root,model):
    directory=root/"isolated";directory.mkdir()
    executable=directory/"atom-quantizer";archive=directory/"model.a22";output=directory/"model.gguf"
    shutil.copy2(BINARY,executable);shutil.copyfile(model["archive"],archive)
    if sha256(executable)!=sha256(BINARY) or sha256(archive)!=model["archive_sha256"]: raise ValueError("isolation copy changed")
    tracer=shutil.which('strace')
    if tracer is None: raise ValueError("strace is required for the declared no-fitting-input audit")
    traces=[]
    for operation,args in [('verify',['verify','model.a22']),('decode',['decode','model.a22','--out','model.gguf'])]:
        trace=directory/(operation+'.trace')
        command=[tracer,'-f','-e','trace=%file,%process','-o',str(trace),'./atom-quantizer',*args]
        run=subprocess.run(command,cwd=directory,env={'PATH':'/no-python-or-tools-here','LANG':'C'},capture_output=True,text=True,timeout=1800)
        (directory/(operation+'.log')).write_text(run.stdout+run.stderr)
        if run.returncode!=0: raise ValueError("isolated native operation failed")
        text=trace.read_text()
        if any(token in text for token in ('.acal','SmolLM2','experiments/','python')): raise ValueError("native consumption accessed fitting/research inputs")
        executions=re.findall(r'execve\(',text)
        if len(executions)!=1: raise ValueError("native consumer launched another program")
        traces.append({'operation':operation,'trace_sha256':sha256(trace),'execve_count':len(executions)})
    if sha256(output)!=model['decoded_sha256']: raise ValueError("isolated native output differs")
    return {'status':'PASSED','directory':str(directory),'only_supplied_inputs':['atom-quantizer','model.a22'],
        'python_unavailable_on_path':True,'decoded_sha256':sha256(output),'traces':traces}


def late_failure(root,model):
    archive=root/'wrong-final-commitment.a22';destination=root/'must-not-publish.gguf'
    with zipfile.ZipFile(model['archive']) as source,zipfile.ZipFile(archive,'x') as target:
        for name in source.namelist():
            data=source.read(name)
            if name=='manifest.json':
                value=json.loads(data);tensor=value['tensors'][-1]['name'];value['tensors'][-1]['decoded_sha256']='0'*64
                data=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
            target.writestr(name,data)
    result=subprocess.run([str(BINARY),'decode',str(archive),'--out',str(destination)],capture_output=True,text=True)
    (root/'late-failure.log').write_text(result.stdout+result.stderr)
    if result.returncode==0 or tensor not in result.stderr or 'decoded commitment mismatch' not in result.stderr:
        raise ValueError("native late failure did not reach the expected check")
    if destination.exists() or list(root.glob('.*.a22.tmp')): raise ValueError("native failure published or retained output")
    return {'status':'PASSED','injected_fault':True,'tensor':tensor,'returncode':result.returncode,'destination_absent':True,'staging_absent':True,'archive_sha256':sha256(archive)}


def run(root):
    root=Path(root).resolve();(root/'exports').mkdir(exist_ok=False)
    prior=json.loads((D05/'summary.json').read_text())
    sources=[*ROOT.glob('src/**/*.rs'),ROOT/'Cargo.toml',ROOT/'Cargo.lock',Path(__file__),ROOT/'research/native-a22-2026-10-01/PROTOCOL.md',BINARY]
    frozen={str(p):sha256(p) for p in sources}
    selected=prior['models']['six-activation']
    work=[]
    for geometry,build in prior['builds'].items():
        for stage in build['stages']:
            model=stage['model'];work.append((root,f"{geometry}-{stage['iteration']}",model))
            frozen[model['archive']]=model['archive_sha256'];frozen[model['decoded']]=model['decoded_sha256']
    save_json(root/'manifest.json',{'sha256':frozen,'model_cases':len(work),'binary_bytes':BINARY.stat().st_size})
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(native_model,*args) for args in work]):
            rows.append(future.result());save_json(root/'progress.json',rows)
    old=regression(root)
    isolation=isolated(root,selected)
    bad=late_failure(root,selected)
    if any(sha256(p)!=h for p,h in frozen.items()): raise ValueError("I1 source, binary or input changed during verification")
    result={'status':'PASSED','models':rows,'native_tensors_verified':sum(r['verified_tensors'] for r in rows),
        'oq_regression':old,'isolation':isolation,'late_failure':bad,'native_decode_without_python':True,
        'new_inference_runs':0,'selected_decoded_sha256':selected['decoded_sha256'],'source_fitting_unchanged':True}
    save_json(root/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','oq_regression')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);run(parser.parse_args().out)
