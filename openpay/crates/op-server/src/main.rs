//! `op-server` binary entry point.
//!
//! Reads its full deployment configuration from environment variables
//! (typically supplied via `systemd`'s `EnvironmentFile=` or a Docker
//! `--env-file`):
//!
//! | Var | Default | Effect |
//! |-----|---------|--------|
//! | `OP_BIND_ADDR` | `127.0.0.1:8080` | Address to bind. Vendors expose `0.0.0.0:<port>` behind Caddy / nginx. |
//! | `OP_GRAPH_PATH` | (unset) | Path to the `.graph` file. Unset → volatile in-memory. |
//! | `OP_API_KEYS` | (unset) | Comma-separated key list → API-key auth. Unset → no auth (warning). |
//! | `OP_RATE_LIMIT_PER_MINUTE` | (unset) | Per-bucket request budget. Unset → no limit. |
//! | `OP_BASE_RPC_URL` | (unset) | EVM JSON-RPC endpoint for Base. Pair with the key var below. |
//! | `OP_USDC_BASE_PRIVATE_KEY` | (unset) | Hot-wallet hex key. Combined with `OP_BASE_RPC_URL` registers a `usdc-base` rail. |
//! | `OP_LOG_FORMAT` | `pretty` | `json` switches the tracing subscriber to JSON. |
//! | `RUST_LOG` | `info,op_server=debug` | Standard `tracing` env filter. |
//!
//! TLS is terminated upstream (Caddy / nginx / load balancer); this
//! binary always speaks plain HTTP.

use std::net::SocketAddr;

use op_server::{
    EnvMiddleware, build_middleware_from_env, build_state_from_env, router, router_with_middleware,
};
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();

    fn env_reader(k: &str) -> Option<String> {
        std::env::var(k).ok()
    }

    let state = build_state_from_env(env_reader)?;
    let EnvMiddleware { auth, rate_limit } = build_middleware_from_env(env_reader);

    let app = match (auth, rate_limit) {
        (Some(a), Some(r)) => router_with_middleware(state, a, r),
        (Some(a), None) => router(state).layer(a),
        (None, Some(r)) => router(state).layer(r),
        (None, None) => router(state),
    };
    let app = app.layer(TraceLayer::new_for_http());

    let bind = std::env::var("OP_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());
    let addr: SocketAddr = bind.parse()?;
    tracing::info!(%addr, "op-server starting");

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// Install the `tracing_subscriber` formatter. `OP_LOG_FORMAT=json`
/// emits structured JSON (one record per line, machine-parseable);
/// anything else (default) uses the human-readable formatter the
/// development workflow already expects.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,op_server=debug"));
    match std::env::var("OP_LOG_FORMAT").as_deref() {
        Ok("json") => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .json()
                .init();
        }
        _ => {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }
}

async fn shutdown_signal() {
    use tokio::signal;
    let ctrl_c = async {
        signal::ctrl_c().await.expect("install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    tracing::info!("shutdown signal received");
}
