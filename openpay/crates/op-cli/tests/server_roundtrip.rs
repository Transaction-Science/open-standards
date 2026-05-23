//! End-to-end test: spin up an `op-server` on an ephemeral port,
//! drive the CLI's `health` and `readiness` subcommands through the
//! library entry point, assert they succeed and parse the JSON
//! envelope the server actually returns.
//!
//! The test runs the axum server inside a `tokio::spawn` on a
//! multi-thread runtime; the CLI side is fully blocking (it uses
//! `reqwest::blocking`). Mixing the two only works because the
//! reqwest call lives on a *separate* OS thread, so it never blocks
//! the runtime that's hosting axum.

use std::net::SocketAddr;

use op_cli::{Client, Command, dispatch};
use op_server::{AppState, router};

/// Helper: bind to `127.0.0.1:0`, start axum on the returned port
/// inside a `tokio::spawn`, return the bound address. The task is
/// detached — it will be killed when the runtime is dropped at end
/// of test.
async fn start_server() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    let app = router(AppState::new_in_memory());
    tokio::spawn(async move {
        // axum::serve returns a future that resolves only on
        // shutdown; here we just let it run until the runtime exits.
        let _ = axum::serve(listener, app).await;
    });
    addr
}

#[test]
fn cli_health_against_real_server() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio rt");

    let addr = rt.block_on(start_server());

    // Drive the CLI side on a regular OS thread so the blocking
    // reqwest call doesn't stall the tokio reactor.
    let base = format!("http://{addr}");
    let client = Client::new(base, None);

    let health = dispatch(&client, Command::Health).expect("health ok");
    assert_eq!(
        health.get("status").and_then(|v| v.as_str()),
        Some("ok"),
        "health response shape: {health}"
    );

    let readiness = dispatch(&client, Command::Readiness).expect("readiness ok");
    assert_eq!(
        readiness.get("status").and_then(|v| v.as_str()),
        Some("ready"),
        "readiness response shape: {readiness}"
    );
    // The readiness response includes store counts; they all start at 0.
    assert_eq!(
        readiness.get("refunds").and_then(serde_json::Value::as_u64),
        Some(0)
    );
}

#[test]
fn cli_network_error_when_no_server() {
    // Pick a port that almost certainly has nothing listening.
    let client = Client::new("http://127.0.0.1:1", None);
    let err = dispatch(&client, Command::Health).expect_err("must fail");
    match err {
        op_cli::Error::Network { url, .. } => {
            assert!(url.contains("127.0.0.1:1"), "url was {url}");
        }
        other => panic!("expected Network error, got {other:?}"),
    }
}
