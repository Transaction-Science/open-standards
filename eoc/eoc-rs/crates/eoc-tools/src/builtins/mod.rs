//! Reference sandbox tools an operator can register out of the box.
//!
//! Every primitive is allowlist-only: the default configuration refuses
//! all calls. Operators opt in by populating the allowlist explicitly.

pub mod file;
pub mod http;
pub mod search;
pub mod shell;
pub mod sql;

pub use file::{FileReadTool, FileWriteTool};
pub use http::{HttpMethod, HttpTool};
pub use search::{SearchBackend, SearchTool};
pub use shell::ShellTool;
pub use sql::SqlTool;
