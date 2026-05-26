//! `jouleclaw-edge-cli` — answer a question through the Edge-First v6
//! pipeline.
//!
//! Three modes:
//!   - Inline (default): load DeBERTa, run pipeline, exit.
//!   - Serve: long-running daemon over a Unix socket.
//!   - Client (--socket PATH): forward query to a running daemon.
//!
//! See `--help` for usage; tests/end_to_end.rs covers the
//! integration semantics under live load.

use jouleclaw_edge_cli::{cli, pipeline, render, server};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mode = match cli::parse_mode(argv) {
        Ok(m) => m,
        Err(cli::ParseError::Help) => {
            print!("{}", cli::HELP);
            return;
        }
        Err(e) => {
            eprintln!("{e}\n");
            eprint!("{}", cli::HELP);
            std::process::exit(2);
        }
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: cannot build tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    let code = rt.block_on(async {
        match mode {
            cli::Mode::Inline(opts) => run_inline(&opts).await,
            cli::Mode::Serve {
                socket,
                model_dir,
                cache_dir,
                wikidata_endpoint,
                wikipedia_endpoint,
            } => run_server(
                socket,
                model_dir,
                cache_dir,
                wikidata_endpoint,
                wikipedia_endpoint,
            )
            .await,
            cli::Mode::Client { socket, options } => run_client(&socket, &options).await,
        }
    });
    std::process::exit(code);
}

async fn run_inline(opts: &cli::Options) -> i32 {
    let log: Box<dyn Fn(&str) + Send + Sync> = if opts.verbose {
        Box::new(|s: &str| eprintln!("{s}"))
    } else {
        Box::new(|_: &str| {})
    };
    match pipeline::run(opts, log.as_ref()).await {
        Ok(o) => render::render(&o, opts.json),
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

async fn run_server(
    socket: std::path::PathBuf,
    model_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    wikidata_endpoint: Option<String>,
    wikipedia_endpoint: Option<String>,
) -> i32 {
    let cfg = server::ServerConfig {
        socket_path: socket,
        model_dir,
        cache_dir,
        wikidata_endpoint,
        wikipedia_endpoint,
    };
    match server::serve(cfg).await {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("server error: {e}");
            1
        }
    }
}

async fn run_client(socket: &std::path::Path, opts: &cli::Options) -> i32 {
    let req = server::ServeRequest {
        query: opts.query.clone(),
        json: opts.json,
        no_verify: opts.no_verify,
        no_cache: opts.no_cache,
        verbose: opts.verbose,
        wikidata_endpoint: opts.wikidata_endpoint.clone(),
        wikipedia_endpoint: opts.wikipedia_endpoint.clone(),
    };
    match server::send_query(socket, &req).await {
        Ok(body) => {
            use std::io::Write;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            let _ = lock.write_all(body.as_bytes());
            if !body.ends_with('\n') {
                let _ = lock.write_all(b"\n");
            }
            0
        }
        Err(e) => {
            eprintln!("client error: {e}");
            1
        }
    }
}
