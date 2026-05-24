//! Thin `ort::Session` wrapper. The browser side uses onnxruntime-web;
//! this is the matching native runtime that loads the same .onnx files
//! the browser does. CPU-only by default; users wanting CUDA/CoreML/etc
//! enable extra ort features in their own Cargo.toml.

use std::path::Path;
use ort::session::Session;

use super::DecodeError;

pub(crate) fn load_session(path: &Path) -> Result<Session, DecodeError> {
    let mut b = Session::builder()
        .map_err(|e| DecodeError::Ort(format!("session builder: {e}")))?;
    b.commit_from_file(path)
        .map_err(|e| DecodeError::Ort(format!("load {} failed: {e}", path.display())))
}
