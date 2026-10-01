//! Read-only decoded-weight assessment using the quantizer's actual gates.
use super::{is_protected, make_reference, measure_fidelity, FidelityLimits, QuantizeOptions};
use atom_quantizer::{calibration, gguf};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

pub(super) struct Report {
    pub json: String,
    pub accepted: bool,
}

fn quoted(text: &str) -> String {
    let mut out = String::from("\"");
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{20}' => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".into()
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}

struct Arguments<'a> {
    source: &'a str,
    candidate: &'a str,
    calibration: &'a str,
    max_output_mse: f64,
}

fn arguments(args: &[String]) -> Result<Arguments<'_>, String> {
    if args.len() < 2 {
        return Err("usage: atom-quantizer assess SOURCE.gguf CANDIDATE.gguf --calibration INPUT.acal [--max-output-mse VALUE]".into());
    }
    let mut calibration = None;
    let mut output_ceiling = None;
    let mut index = 2;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{} needs a value", args[index]))?;
        match args[index].as_str() {
            "--calibration" if calibration.is_none() => calibration = Some(value.as_str()),
            "--max-output-mse" if output_ceiling.is_none() => {
                output_ceiling = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && *v > 0.0)
                        .ok_or("--max-output-mse needs a finite positive value")?,
                );
            }
            option => return Err(format!("unknown or repeated assessment argument {option}")),
        }
        index += 2;
    }
    Ok(Arguments {
        source: &args[0],
        candidate: &args[1],
        calibration: calibration
            .ok_or("assessment requires --calibration with real operator samples")?,
        max_output_mse: output_ceiling.unwrap_or(QuantizeOptions::default().max_output_mse),
    })
}

pub(super) fn run(args: &[String]) -> Result<Report, String> {
    let args = arguments(args)?;
    let source_hash = calibration::file_sha256(args.source)?;
    let candidate_hash = calibration::file_sha256(args.candidate)?;
    let calibration_hash = calibration::file_sha256(args.calibration)?;
    let mut source = gguf::open(args.source)?;
    let mut candidate = gguf::open(args.candidate)?;
    if source.tensors.is_empty() || source.tensors.len() != candidate.tensors.len() {
        return Err("source and candidate must have the same nonempty tensor set".into());
    }
    if source.arch != candidate.arch {
        return Err("source and candidate architecture identifiers differ".into());
    }
    let mut indices = BTreeMap::new();
    for (index, tensor) in candidate.tensors.iter().enumerate() {
        if indices.insert(tensor.name.clone(), index).is_some() {
            return Err(format!("duplicate candidate tensor {}", tensor.name));
        }
    }
    let mut names = BTreeSet::new();
    for meta in &source.tensors {
        if !names.insert(meta.name.clone()) {
            return Err(format!("duplicate source tensor {}", meta.name));
        }
        let index = *indices
            .get(&meta.name)
            .ok_or_else(|| format!("candidate is missing tensor {}", meta.name))?;
        if candidate.tensors[index].dims != meta.dims {
            return Err(format!("tensor dimensions differ for {}", meta.name));
        }
        if !gguf::dequantizable(meta.ggml_type)
            || !gguf::dequantizable(candidate.tensors[index].ggml_type)
        {
            return Err(format!(
                "unsupported assessment tensor type for {}",
                meta.name
            ));
        }
    }
    let source_header = hex(&Sha256::digest(gguf::model_header(&mut source)?));
    let candidate_header = hex(&Sha256::digest(gguf::model_header(&mut candidate)?));
    let cal = calibration::Calibration::load(args.calibration, args.source)?;
    let mut used = BTreeSet::new();
    let metas = source.tensors.clone();
    let mut rows = Vec::new();
    let mut passed = 0;
    let mut failures = [0usize; 3];
    for (index, meta) in metas.iter().enumerate() {
        let candidate_index = *indices
            .get(&meta.name)
            .ok_or_else(|| format!("candidate is missing tensor {}", meta.name))?;
        let values = gguf::read_tensor_f32(&mut source, index)?;
        let reconstructed = gguf::read_tensor_f32(&mut candidate, candidate_index)?;
        let reference = make_reference(
            &values,
            meta,
            index,
            Some(&cal),
            #[cfg(feature = "cuda")]
            None,
        )?;
        let (kind, samples) = match &reference.samples {
            Some(calibration::Samples::Linear(xs)) => ("linear", xs.len()),
            Some(calibration::Samples::Embedding(ids)) => ("embedding_lookup", ids.len()),
            None => ("histogram", 0),
        };
        if reference.samples.is_some() {
            used.insert(meta.name.clone());
        }
        let fidelity = measure_fidelity(
            &values,
            &reconstructed,
            meta,
            &reference,
            #[cfg(feature = "cuda")]
            None,
        )?;
        let protected = is_protected(&meta.name);
        let limits = FidelityLimits::new(protected, args.max_output_mse);
        let gates = fidelity.gates(reference.calibrated, limits);
        passed += usize::from(gates.accepted());
        failures[0] += usize::from(!gates.kl);
        failures[1] += usize::from(!gates.cosine);
        failures[2] += usize::from(!gates.output_mse);
        let mse_state = match fidelity.output_mse {
            None => "not_applicable",
            Some(value) if value.is_finite() => "finite",
            Some(value) if value.is_infinite() => "unbounded",
            Some(_) => "undefined",
        };
        let dimensions = meta
            .dims
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        rows.push(format!(
            "{{\"name\":{},\"dimensions\":[{}],\"source_type\":{},\"candidate_type\":{},\"protected\":{},\"observation\":{},\"samples\":{},\"output_gate_required\":{},\"kl_forward\":{},\"kl_reverse\":{},\"cosine\":{},\"output_mse\":{},\"output_mse_state\":{},\"limits\":{{\"kl\":{},\"cosine\":{},\"output_mse\":{}}},\"gates\":{{\"kl\":{},\"cosine\":{},\"output_mse\":{}}},\"passed\":{}}}",
            quoted(&meta.name), dimensions, meta.ggml_type, candidate.tensors[candidate_index].ggml_type,
            protected, quoted(kind), samples, reference.calibrated,
            number(fidelity.fwd), number(fidelity.rev), number(fidelity.cosine as f64),
            fidelity.output_mse.map(number).unwrap_or_else(|| "null".into()), quoted(mse_state),
            number(limits.kl), number(limits.cosine as f64), number(limits.output_mse),
            gates.kl, gates.cosine, gates.output_mse, gates.accepted(),
        ));
        if (index + 1) % 20 == 0 || index + 1 == metas.len() {
            eprintln!("assessed {}/{} tensors", index + 1, metas.len());
        }
    }
    if calibration::file_sha256(args.source)? != source_hash
        || calibration::file_sha256(args.candidate)? != candidate_hash
        || calibration::file_sha256(args.calibration)? != calibration_hash
    {
        return Err("assessment inputs changed during measurement".into());
    }
    let mut unused: Vec<_> = cal.tensors.keys().filter(|n| !used.contains(*n)).collect();
    unused.sort();
    let unused = unused
        .iter()
        .map(|name| quoted(name))
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\"format\":\"native-fidelity-1\",\"version\":{},\"backend\":\"cpu\",\"scope\":\"decoded_tensor_fidelity\",\"dimension_order\":\"gguf_innermost_first\",\"source\":{},\"candidate\":{},\"calibration\":{},\"source_sha256\":{},\"candidate_sha256\":{},\"calibration_sha256\":{},\"source_header_sha256\":{},\"candidate_header_sha256\":{},\"headers_identical\":{},\"max_output_mse\":{},\"tensor_count\":{},\"passed_count\":{},\"failed_count\":{},\"gate_failures\":{{\"kl\":{},\"cosine\":{},\"output_mse\":{}}},\"unused_calibration_entries\":[{}],\"all_passed\":{},\"tensors\":[{}]}}",
        quoted(env!("CARGO_PKG_VERSION")), quoted(args.source), quoted(args.candidate), quoted(args.calibration),
        quoted(&hex(&source_hash)), quoted(&hex(&candidate_hash)), quoted(&hex(&calibration_hash)),
        quoted(&source_header), quoted(&candidate_header), source_header == candidate_header,
        number(args.max_output_mse), metas.len(), passed, metas.len()-passed,
        failures[0], failures[1], failures[2], unused, passed == metas.len(), rows.join(","),
    );
    Ok(Report {
        json,
        accepted: passed == metas.len(),
    })
}
