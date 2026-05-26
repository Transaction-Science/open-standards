//! Library surface of the jouleclaw-edge-cli binary, exposed so
//! integration tests in `tests/` can drive the pipeline directly
//! instead of forking the binary.

pub mod cli;
pub mod pipeline;
pub mod render;
pub mod server;
pub mod understanding;
