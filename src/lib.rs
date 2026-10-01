//! Shared codec primitives for Atom Quantizer.
//!
//! The public modules are shared by the main adaptive codec and the separate
//! Q2 composer research binary. OQ03 includes complete decoder recipes while
//! the reader retains explicitly labeled legacy OQ02 support.

pub mod a22;
pub mod allocation;
pub mod calibration;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod encoded;
pub mod gguf;
pub mod metrics;
pub mod oq;
pub mod stacks;
pub mod wq;
