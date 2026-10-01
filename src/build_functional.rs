//! Real Python fitting followed by independent native verification/publication.
use super::assess;
use atom_quantizer::{a22, calibration, gguf};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Arguments {
    source: PathBuf,
    base: PathBuf,
    residual: PathBuf,
    output: PathBuf,
    work: PathBuf,
    python: PathBuf,
    engine: PathBuf,
}

fn string(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| "non-UTF8 path is unsupported by this CLI".into())
}

fn location(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path.file_name().ok_or("invalid file/directory path")?;
    Ok(parent.canonicalize().map_err(|e| e.to_string())?.join(name))
}

fn existing_file(path: &Path) -> Result<PathBuf, String> {
    // Preserve an interpreter's venv symlink rather than launching its system target.
    let path = location(path)?;
    if !path.is_file() {
        return Err(format!("file is unavailable: {}", path.display()));
    }
    Ok(path)
}

fn absent(path: &Path) -> Result<(), String> {
    if path.symlink_metadata().is_ok() {
        return Err(format!("destination already exists: {}", path.display()));
    }
    Ok(())
}

fn arguments(args: &[String]) -> Result<Arguments, String> {
    let source=args.first().ok_or("usage: atom-quantizer build-functional SOURCE.gguf --base-calibration BASE.acal --residual-calibration RESIDUAL.acal --out NEW.a22 [--work-dir NEW_DIR] [--python PATH] [--engine-dir DIR]")?;
    let mut options = BTreeMap::new();
    let mut index = 1;
    while index < args.len() {
        let key = &args[index];
        if ![
            "--base-calibration",
            "--residual-calibration",
            "--out",
            "--work-dir",
            "--python",
            "--engine-dir",
        ]
        .contains(&key.as_str())
            || options.contains_key(key)
        {
            return Err(format!(
                "unknown or repeated functional-build argument {key}"
            ));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{key} needs a value"))?;
        options.insert(key.clone(), value.clone());
        index += 2;
    }
    let required = |key: &str| {
        options
            .get(key)
            .map(PathBuf::from)
            .ok_or_else(|| format!("functional build requires {key}"))
    };
    let source = existing_file(Path::new(source))?;
    let base = existing_file(&required("--base-calibration")?)?;
    let residual = existing_file(&required("--residual-calibration")?)?;
    let output = location(&required("--out")?)?;
    absent(&output)?;
    let default_work = PathBuf::from(format!("{}.work", output.display()));
    let work = location(
        &options
            .get("--work-dir")
            .map(PathBuf::from)
            .unwrap_or(default_work),
    )?;
    if work == output {
        return Err("work directory and archive destination must differ".into());
    }
    absent(&work)?;
    let engine = options
        .get("--engine-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
        .canonicalize()
        .map_err(|e| format!("fitting engine directory unavailable: {e}"))?;
    if !engine.join("experiments/full22/harness.py").is_file() {
        return Err("fitting engine is missing experiments/full22/harness.py".into());
    }
    let python = existing_file(
        &options
            .get("--python")
            .map(PathBuf::from)
            .unwrap_or_else(|| engine.join(".venv/bin/python")),
    )
    .map_err(|e| format!("functional fitting needs its Python environment (--python PATH): {e}"))?;
    Ok(Arguments {
        source,
        base,
        residual,
        output,
        work,
        python,
        engine,
    })
}

fn checksum(path: &Path) -> Result<String, String> {
    Ok(calibration::file_sha256(string(path)?)?
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn save(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn publish(source: &Path, destination: &Path, expected: &str) -> Result<(), String> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    absent(destination)?;
    let name = destination
        .file_name()
        .ok_or("invalid archive destination")?
        .to_string_lossy();
    let path = destination.with_file_name(format!(
        ".{name}.{}-{}.functional.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    let temporary = Temporary(path);
    let mut input = File::open(source).map_err(|e| e.to_string())?;
    std::io::copy(&mut input, &mut file).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);
    if checksum(&temporary.0)? != expected {
        return Err("archive changed during publication".into());
    }
    std::fs::hard_link(&temporary.0, destination).map_err(|e| e.to_string())?;
    Ok(())
}

fn execute(args: &Arguments) -> Result<Value, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut identities = BTreeMap::new();
    for path in [
        &args.source,
        &args.base,
        &args.residual,
        &args.python,
        &executable,
    ] {
        identities.insert(path.clone(), checksum(path)?);
    }
    if identities.get(&args.base) == identities.get(&args.residual) {
        return Err("base and residual observations must have distinct contents".into());
    }
    for path in [&args.base, &args.residual] {
        calibration::Calibration::load(string(path)?, string(&args.source)?)?;
    }
    let model = gguf::open(string(&args.source)?)?;
    if model.tensors.is_empty() || model.tensors.iter().any(|t| t.ggml_type != gguf::GGML_F32) {
        return Err("functional fitting requires a complete F32 source GGUF".into());
    }
    let directory = args.engine.join("experiments/full22");
    for entry in std::fs::read_dir(&directory).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|s| s == "py") {
            identities.insert(path.clone(), checksum(&path)?);
        }
    }
    let package = args.engine.join("experiments/__init__.py");
    identities.insert(package.clone(), checksum(&package)?);
    let expected_source = checksum(&args.source)?;
    let expected_base = checksum(&args.base)?;
    let expected_residual = checksum(&args.residual)?;
    let fitting = args.work.join("engine");
    let engine_args = vec![
        "build-adaptive".to_string(),
        "--source".into(),
        string(&args.source)?.into(),
        "--base-calibration".into(),
        string(&args.base)?.into(),
        "--residual-calibration".into(),
        string(&args.residual)?.into(),
        "--gate-calibration".into(),
        string(&args.base)?.into(),
        "--gate-calibration".into(),
        string(&args.residual)?.into(),
        "--assessor".into(),
        string(&executable)?.into(),
        "--start-bits".into(),
        "6".into(),
        "--rank-geometry".into(),
        "activation".into(),
        "--out".into(),
        string(&fitting)?.into(),
    ];
    save(
        &args.work.join("request.json"),
        &json!({"profile":"functional-six-v1","source":args.source,"base_calibration":args.base,
        "residual_calibration":args.residual,"output":args.output,"python":args.python,"engine":args.engine,"native_executable":executable,
        "engine_arguments":engine_args,"identities":identities}),
    )?;
    eprintln!(
        "Fitting with the Python engine; logs and stages: {}",
        args.work.display()
    );
    let launcher="import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); sys.argv[0]='functional-fitting'; runpy.run_module('experiments.full22.harness',run_name='__main__')";
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.work.join("backend.stdout"))
        .map_err(|e| e.to_string())?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.work.join("backend.stderr"))
        .map_err(|e| e.to_string())?;
    let status = Command::new(&args.python)
        .args(["-I", "-c", launcher])
        .arg(&args.engine)
        .args(&engine_args)
        .current_dir(&args.engine)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .status()
        .map_err(|e| format!("cannot launch fitting engine: {e}"))?;
    if !status.success() {
        return Err(format!("fitting engine failed with {status}"));
    }
    let archive = fitting.join("model.a22");
    let archive_hash = checksum(&archive)?;
    a22::validate_functional_profile(
        string(&archive)?,
        &expected_source,
        &expected_base,
        &expected_residual,
    )?;
    let native = args.work.join("native.gguf");
    eprintln!("Independently reconstructing and checking A22-2 with the native decoder");
    let stats = a22::decode_to_gguf(string(&archive)?, string(&native)?)?;
    let decoded_hash = checksum(&native)?;
    let mut reports = Vec::new();
    for (index, calibration) in [&args.base, &args.residual].iter().enumerate() {
        eprintln!("Independent native fidelity view {}", index + 1);
        let report = assess::run(&[
            string(&args.source)?.into(),
            string(&native)?.into(),
            "--calibration".into(),
            string(calibration)?.into(),
        ])?;
        let value: Value = serde_json::from_str(&report.json).map_err(|e| e.to_string())?;
        let report_path = args.work.join(format!("native-fit-{index}.json"));
        save(&report_path, &value)?;
        if value.get("candidate_sha256").and_then(Value::as_str) != Some(decoded_hash.as_str()) {
            return Err("native assessment used different decoded weights".into());
        }
        if !report.accepted {
            return Err(format!(
                "native fidelity view {} rejected the fitted model",
                index + 1
            ));
        }
        reports
            .push(json!({"path":report_path,"sha256":checksum(&report_path)?,"all_passed":true}));
    }
    for (path, expected) in &identities {
        if checksum(path)? != *expected {
            return Err(format!("build input/backend changed: {}", path.display()));
        }
    }
    if checksum(&archive)? != archive_hash || checksum(&native)? != decoded_hash {
        return Err("verified artifact changed before publication".into());
    }
    let result = json!({"format":"functional-build-1","status":"NATIVE_VERIFIED","profile":"functional-six-v1",
        "fitting_backend":"python","native_consumer":true,"archive":args.output,"archive_sha256":archive_hash,
        "archive_bytes":stats.artifact_bytes,"decoded":native,"decoded_sha256":decoded_hash,"verified_tensors":stats.tensors,
        "source_sha256":expected_source,"base_calibration_sha256":expected_base,"residual_calibration_sha256":expected_residual,
        "work_directory":args.work,"python":args.python,"native_executable":executable,"native_reports":reports});
    save(&args.work.join("verified.json"), &result)?;
    publish(&archive, &args.output, &archive_hash)?;
    Ok(result)
}

pub(super) fn run(raw: &[String]) -> Result<Value, String> {
    let args = arguments(raw)?;
    std::fs::create_dir(&args.work).map_err(|e| e.to_string())?;
    match execute(&args) {
        Ok(mut result) => {
            result["status"] = Value::String("PUBLISHED".into());
            if let Err(error) = save(&args.work.join("result.json"), &result) {
                return Err(format!("archive published at {}, but result receipt failed: {error}; verified receipt retained in {}",args.output.display(),args.work.display()));
            }
            Ok(result)
        }
        Err(error) => {
            let _ = save(
                &args.work.join("failure.json"),
                &json!({"status":"FAILED","error":error}),
            );
            Err(format!("{error}; retained work: {}", args.work.display()))
        }
    }
}
