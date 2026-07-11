//! Shared codec primitives for Atom Quantizer.
//!
//! The public modules are shared by the main adaptive codec and the separate
//! Q2 composer research binary. The `.oq` container and its magic remain
//! unchanged so the product rename does not silently break existing artifacts.

#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod gguf;
pub mod metrics;
pub mod oq;
pub mod stacks;
pub mod wq;
