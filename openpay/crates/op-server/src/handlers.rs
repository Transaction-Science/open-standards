//! HTTP handlers, organized by resource.
//!
//! Each handler is a small wrapper that maps JSON in → domain type
//! → store / engine call → JSON out, with `ApiError` carrying the
//! status code translation.

#![allow(missing_docs)] // request/response wire schemas

pub mod audit;
pub mod dispute;
pub mod fx;
pub mod health;
pub mod intent;
pub mod refund;
pub mod settlement;
pub mod subscription;
pub mod webhook;
