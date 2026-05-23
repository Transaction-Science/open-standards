//! `op` binary entry point.
//!
//! Parses the [`op_cli::Cli`] via clap, hands it off to
//! [`op_cli::run`], and translates any [`op_cli::Error`] into a
//! human-readable stderr message + exit code 1. Network failures
//! print the canonical "couldn't reach …" message demanded by the
//! spec; non-2xx responses print the response body so operators can
//! see the server's error envelope verbatim.

use std::process::ExitCode;

use clap::Parser;
use op_cli::{Cli, Error, run};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            match &err {
                Error::Network { url, .. } => {
                    eprintln!("error: couldn't reach {url}");
                    eprintln!("       {err}");
                }
                Error::HttpStatus { status, body } => {
                    eprintln!("error: server returned HTTP {status}");
                    eprintln!("{body}");
                }
                Error::BadPayload(msg) => {
                    eprintln!("error: invalid response payload: {msg}");
                }
            }
            ExitCode::FAILURE
        }
    }
}
