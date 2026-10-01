//! The public orchestrator delegates the explicit, content-independent selection.
mod engine;
pub use basis_primitives::{Field, Matrix, WIDTH};

pub struct Request {
    pub signal: [f64; WIDTH],
    pub seed: u32,
    pub order: usize,
    pub disabled: u16,
}

pub fn run(request: Request) -> Result<Field, String> {
    engine::evolve(request)
}

pub struct LocalRequest {
    pub input: Request,
    pub coupling: Matrix,
}

pub fn run_local(request: LocalRequest) -> Result<Field, String> {
    engine::evolve_local(request)
}
