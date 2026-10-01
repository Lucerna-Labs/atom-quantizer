//! K1 measurement driver. Timing and resident-memory processes are separate.
use atom_quantizer::{a22, calibration, metrics};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn checksum(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn input(path: &str, cols: usize, batch: usize) -> Result<Vec<f32>, String> {
    let data = fs::read(path).map_err(|e| e.to_string())?;
    if data.len()
        != cols
            .checked_mul(batch)
            .and_then(|n| n.checked_mul(4))
            .ok_or("input overflow")?
    {
        return Err("incorrect input extent".into());
    }
    let values: Vec<_> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite probe inputs".into());
    }
    Ok(values)
}
fn equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn rss() -> Result<Value, String> {
    let status = fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?;
    let mut result = serde_json::Map::new();
    for key in ["VmRSS:", "VmHWM:"] {
        let value = status
            .lines()
            .find(|s| s.starts_with(key))
            .ok_or("missing Linux memory field")?
            .split_whitespace()
            .nth(1)
            .ok_or("invalid Linux memory field")?
            .parse::<u64>()
            .map_err(|e| e.to_string())?;
        result.insert(key.trim_end_matches(':').into(), json!(value));
    }
    Ok(Value::Object(result))
}
fn write_values(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn operator(args: &[String]) -> Result<Value, String> {
    if args.len() != 5 {
        return Err("operator ARCHIVE TENSOR INPUT BATCH NEW_DIRECTORY".into());
    }
    let batch = args[3].parse::<usize>().map_err(|e| e.to_string())?;
    let out = Path::new(&args[4]);
    fs::create_dir(out).map_err(|e| e.to_string())?;
    let start = Instant::now();
    let packed = a22::load_tensor(&args[0], &args[1])?;
    let packed_load = start.elapsed().as_secs_f64();
    if packed.shape().len() != 2 {
        return Err("operator requires matrix".into());
    }
    let (rows, cols) = (packed.shape()[0], packed.shape()[1]);
    let start = Instant::now();
    let dense = packed.to_dense()?;
    let dense_expansion = start.elapsed().as_secs_f64();
    let x = input(&args[2], cols, batch)?;
    let xs: Vec<Vec<f32>> = x.chunks_exact(cols).map(|v| v.to_vec()).collect();
    let expected = metrics::cpu_matmul_batch(dense.values(), &xs, rows, cols).concat();
    let labels = ["fused", "row", "dense_shared", "dense_existing"];
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut order = Vec::new();
    let mut seed = 2201u64;
    for round in 0..13 {
        let mut modes = [0usize, 1, 2, 3];
        for index in (1..4).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            modes.swap(index, seed as usize % (index + 1));
        }
        order.push(modes);
        for mode in modes {
            let start = Instant::now();
            let output;
            let nested;
            if mode == 3 {
                nested = metrics::cpu_matmul_batch(
                    black_box(dense.values()),
                    black_box(&xs),
                    rows,
                    cols,
                );
                let elapsed = start.elapsed().as_secs_f64();
                output = black_box(nested).concat();
                if round >= 2 {
                    samples[mode].push(elapsed);
                }
            } else {
                output = match mode {
                    0 => packed.matmul(black_box(&x), batch, a22::Kernel::Fused)?,
                    1 => packed.matmul(black_box(&x), batch, a22::Kernel::Row)?,
                    _ => dense.matmul(black_box(&x), batch)?,
                };
                let elapsed = start.elapsed().as_secs_f64();
                black_box(&output);
                if round >= 2 {
                    samples[mode].push(elapsed);
                }
            }
            if !equal(&output, &expected) {
                return Err(format!(
                    "{} differs from existing CPU dense output",
                    labels[mode]
                ));
            }
            if round == 0 {
                write_values(&out.join(format!("{}.f32", labels[mode])), &output)?;
            }
        }
    }
    let timings: serde_json::Map<String, Value> = labels
        .iter()
        .enumerate()
        .map(|(i, name)| ((*name).into(), json!(samples[i])))
        .collect();
    Ok(
        json!({"status":"MEASURED", "tensor":args[1], "rows":rows,"columns":cols,"batch":batch,
        "archive_sha256":hex(&calibration::file_sha256(&args[0])?), "input_sha256":hex(&calibration::file_sha256(&args[2])?),
        "output_sha256":checksum(&expected), "packed_owned_bytes":packed.owned_bytes(), "dense_owned_bytes":dense.owned_bytes(),
        "load_and_verification_seconds":packed_load,"dense_expansion_seconds":dense_expansion,
        "warmups":2,"repetitions":11,"method_order":order,"timings_seconds":timings}),
    )
}

fn correctness(args: &[String]) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("correctness ARCHIVE CALIBRATION SOURCE_GGUF".into());
    }
    let observations = calibration::Calibration::load(&args[1], &args[2])?;
    let model = a22::load_packed_model(&args[0])?;
    let mut records = Vec::new();
    let mut matrices = 0usize;
    for (name, packed) in &model.tensors {
        if packed.shape().len() != 2 {
            continue;
        }
        let key = if name == "token_embd.weight" {
            "token_embd.weight::output"
        } else {
            name
        };
        let calibration::Samples::Linear(xs) = observations
            .tensors
            .get(key)
            .ok_or("missing actual operator input")?
        else {
            return Err("not a real linear input role".into());
        };
        let (rows, cols) = (packed.shape()[0], packed.shape()[1]);
        let dense = packed.to_dense()?;
        for batch in [1usize, 8, 32] {
            if xs.len() < batch || xs[..batch].iter().any(|x| x.len() != cols) {
                return Err("operator input scope differs".into());
            }
            let x: Vec<f32> = xs[..batch].iter().flatten().copied().collect();
            let reference =
                metrics::cpu_matmul_batch(dense.values(), &xs[..batch], rows, cols).concat();
            let fused = packed.matmul(&x, batch, a22::Kernel::Fused)?;
            let row = packed.matmul(&x, batch, a22::Kernel::Row)?;
            let shared = dense.matmul(&x, batch)?;
            if !equal(&reference, &fused) || !equal(&reference, &row) || !equal(&reference, &shared)
            {
                return Err(format!("actual operator mismatch {name}, batch {batch}"));
            }
            records.push(json!({"tensor":name,"batch":batch,"rows":rows,"columns":cols,"output_sha256":checksum(&reference),"bit_identical":true}));
        }
        matrices += 1;
        eprintln!("checked matrix {matrices}: {name}");
    }
    Ok(
        json!({"status":"MEASURED","tensors_verified":model.tensors.len(),"matrices":matrices,"cases":records,
        "archive_sha256":hex(&model.artifact_sha256),"calibration_sha256":hex(&calibration::file_sha256(&args[1])?)}),
    )
}

fn resident(args: &[String]) -> Result<Value, String> {
    if args.len() != 5 {
        return Err("resident packed|dense ARCHIVE TENSOR INPUT BATCH".into());
    }
    let batch = args[4].parse::<usize>().map_err(|e| e.to_string())?;
    let before = rss()?;
    let start = Instant::now();
    let report = match args[0].as_str() {
        "packed" => {
            let model = a22::load_packed_model(&args[1])?;
            let load = start.elapsed().as_secs_f64();
            let loaded = rss()?;
            let (_, matrix) = model
                .tensors
                .iter()
                .find(|(name, _)| name == &args[2])
                .ok_or("missing matrix")?;
            if matrix.shape().len() != 2 {
                return Err("resident operation requires matrix".into());
            }
            let x = input(&args[3], matrix.shape()[1], batch)?;
            let y = matrix.matmul(&x, batch, a22::Kernel::Fused)?;
            let after = rss()?;
            let owned: usize = model.tensors.iter().map(|(_, t)| t.owned_bytes()).sum();
            let decoded: usize = model.tensors.iter().map(|(_, t)| t.elements() * 4).sum();
            black_box(&model);
            black_box(&y);
            json!({"tensors":model.tensors.len(),"prefix_bytes":model.prefix.len(),"owned_tensor_bytes":owned,
                "decoded_weight_bytes":decoded,"loaded":loaded,"after_operation":after,"load_seconds":load,
                "archive_sha256":hex(&model.artifact_sha256),"output_sha256":checksum(&y),"output_bytes":y.len()*4})
        }
        "dense" => {
            let model = a22::load_dense_model(&args[1])?;
            let load = start.elapsed().as_secs_f64();
            let loaded = rss()?;
            let (_, matrix) = model
                .tensors
                .iter()
                .find(|(name, _)| name == &args[2])
                .ok_or("missing matrix")?;
            if matrix.shape().len() != 2 {
                return Err("resident operation requires matrix".into());
            }
            let x = input(&args[3], matrix.shape()[1], batch)?;
            let y = matrix.matmul(&x, batch)?;
            let after = rss()?;
            let owned: usize = model.tensors.iter().map(|(_, t)| t.owned_bytes()).sum();
            let decoded: usize = model
                .tensors
                .iter()
                .map(|(_, t)| t.values().len() * 4)
                .sum();
            black_box(&model);
            black_box(&y);
            json!({"tensors":model.tensors.len(),"prefix_bytes":model.prefix.len(),"owned_tensor_bytes":owned,
                "decoded_weight_bytes":decoded,"loaded":loaded,"after_operation":after,"load_seconds":load,
                "archive_sha256":hex(&model.artifact_sha256),"output_sha256":checksum(&y),"output_bytes":y.len()*4})
        }
        _ => return Err("unknown resident mode".into()),
    };
    Ok(
        json!({"status":"MEASURED","mode":args[0],"before":before,"measurements":report,
        "input_sha256":hex(&calibration::file_sha256(&args[3])?),"batch":batch,"tensor":args[2]}),
    )
}

fn run() -> Result<Value, String> {
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("operator") => operator(&args[2..]),
        Some("correctness") => correctness(&args[2..]),
        Some("resident") => resident(&args[2..]),
        _ => Err("choose operator, correctness or resident".into()),
    }
}
fn main() {
    match run() {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("packed probe failed: {error}");
            std::process::exit(1);
        }
    }
}
