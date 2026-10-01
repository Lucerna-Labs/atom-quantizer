//! Batch conventional I/O for independent native block fields; no quality routing.
use std::io::Read;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: discover_resources ORDER SEED DISABLED_MASK < block-fields".into());
    }
    let order = args[0].parse::<usize>().map_err(|e| e.to_string())?;
    let seed = args[1].parse::<u32>().map_err(|e| e.to_string())?;
    let disabled = args[2].parse::<u16>().map_err(|e| e.to_string())?;
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let values = input
        .split_whitespace()
        .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() || !values.len().is_multiple_of(1056) {
        return Err("each block needs 32 stimulus and 1024 coupling values".into());
    }
    for (index, block) in values.as_chunks::<1056>().0.iter().enumerate() {
        let field = basis_engine::run_local(basis_engine::LocalRequest {
            input: basis_engine::Request {
                signal: std::array::from_fn(|i| block[i]),
                order,
                seed,
                disabled,
            },
            coupling: std::array::from_fn(|i| std::array::from_fn(|j| block[32 + i * 32 + j])),
        })
        .map_err(|e| format!("block {index}: {e}"))?;
        println!(
            "{}",
            field
                .resources()
                .iter()
                .chain(field.diagnostics().iter())
                .map(|x| format!("{x:.17e}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    Ok(())
}
