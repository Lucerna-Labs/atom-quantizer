//! Conventional text boundary for the B02 component entry point.
use std::io::Read;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: discover_local_basis ORDER SEED DISABLED_MASK < stimulus-and-coupling".into(),
        );
    }
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let values = input
        .split_whitespace()
        .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 32 + 32 * 32 {
        return Err("need 32 stimulus and 1024 coupling values".into());
    }
    let result = basis_engine::run_local(basis_engine::LocalRequest {
        input: basis_engine::Request {
            signal: std::array::from_fn(|i| values[i]),
            order: args[0].parse::<usize>().map_err(|e| e.to_string())?,
            seed: args[1].parse::<u32>().map_err(|e| e.to_string())?,
            disabled: args[2].parse::<u16>().map_err(|e| e.to_string())?,
        },
        coupling: std::array::from_fn(|i| std::array::from_fn(|j| values[32 + i * 32 + j])),
    })?;
    println!(
        "{}",
        result
            .matrix()
            .iter()
            .flatten()
            .map(|x| format!("{x:.17e}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "{}",
        result
            .diagnostics()
            .iter()
            .map(|x| format!("{x:.17e}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    Ok(())
}
