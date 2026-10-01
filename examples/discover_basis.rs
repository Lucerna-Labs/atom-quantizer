//! Conventional text I/O boundary; mechanics live in the two owning Atom crates.
use std::io::Read;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: discover_basis ORDER SEED DISABLED_MASK < 32 stimulus numbers".into());
    }
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let values: Vec<f64> = input
        .split_whitespace()
        .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let result = basis_engine::run(basis_engine::Request {
        signal: values
            .try_into()
            .map_err(|_| "need exactly 32 stimulus values")?,
        order: args[0].parse::<usize>().map_err(|e| e.to_string())?,
        seed: args[1].parse::<u32>().map_err(|e| e.to_string())?,
        disabled: args[2].parse::<u16>().map_err(|e| e.to_string())?,
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
