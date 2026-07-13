//! Feature-gated Metal backend plumbing.
//!
//! This module adapts the vendored ggml Metal kernels to
//! `greppy-embed-native`. It contains the runtime, argument layouts, tensor
//! descriptors, operation dispatch, weights, and complete Gemma graph used on
//! supported Apple-Silicon devices.

pub mod errors;
pub mod ffi;
pub mod kargs;
pub mod model;
pub mod ops;
pub mod tensor;
pub mod weights;
