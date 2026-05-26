//! Argv parser.
//!
//! Small enough to roll without `clap`. Supports:
//!
//!   jouleclaw-edge-cli [OPTIONS] <QUERY>...
//!
//! OPTIONS:
//!     --json                emit Answer/Refusal as JSON
//!     --no-verify           skip DeBERTa entailment (fast dev mode)
//!     --model-dir <PATH>    DeBERTa model directory
//!                           (default: $HOME/.cache/joule-edge/models/deberta-v3-large-mnli,
//!                            then ./models/deberta-v3-large-mnli)
//!     --verbose             print per-stage progress to stderr
//!     -h, --help            show this help
//!
//! All non-option arguments are concatenated with spaces and used
//! as the query text.

use std::path::PathBuf;

pub const HELP: &str = "\
jouleclaw-edge-cli — verified-by-design retrieval-augmented answering

USAGE
    jouleclaw-edge-cli [OPTIONS] <QUERY>...
    jouleclaw-edge-cli serve [--socket PATH] [--model-dir PATH] [--cache-dir PATH]
    jouleclaw-edge-cli --socket PATH [OPTIONS] <QUERY>...    (client mode)

OPTIONS
    --json              emit Answer/Refusal as JSON on stdout
    --no-verify         skip DeBERTa entailment (fast dev mode)
    --model-dir <PATH>  DeBERTa model directory
                        (default: ./models/deberta-v3-large-mnli)
    --cache-dir <PATH>  cache directory for repeat queries
                        (default: $HOME/.cache/joule-edge)
    --no-cache          skip cache lookup/store for this query
    --socket <PATH>     speak to a running 'jouleclaw-edge-cli serve'
                        instance instead of running the pipeline
                        inline. Server amortizes DeBERTa load.
    --verbose           print per-stage progress to stderr
    -h, --help          show this help and exit

MODES
    inline (default): load DeBERTa, run pipeline, exit. ~3 s.
    serve:            load DeBERTa once, listen on a Unix socket,
                      answer queries from clients. First query ~3 s,
                      subsequent queries ~600 ms (cached: 6 ms).
    client (--socket): connect to a running server. ~600 ms cold,
                      6 ms warm.

QUERY
    All non-option arguments are joined into a single query. Wrap
    the query in quotes for shell safety:

        jouleclaw-edge-cli \"what is the capital of France?\"

OUTPUT
    Default: pretty-text Answer with citations.
    --json:  the full Answer (or Refusal) object as JSON.

CACHE
    Verified answers are cached by query fingerprint with a 24-hour
    TTL. Repeat queries return in ~10 ms. Use --no-cache to force a
    fresh run, or --cache-dir to point at a shared location.

    A second cache layer at $HOME/.cache/joule-edge/http/ stores
    raw Wikidata SPARQL and Wikipedia REST responses for 24 h.
    Different user queries that hit the same SPARQL/REST call share
    this cache, so even --no-cache benefits when underlying calls
    overlap.

LOCAL MIRRORS
    --wikidata-endpoint URL    e.g. http://localhost:7878/sparql for a
                               local Qlever/Oxigraph mirror against the
                               Wikidata RDF dump. Sub-100ms queries.
    --wikipedia-endpoint URL   e.g. http://localhost:8080/api/rest_v1/page/summary
                               for a local Kiwix or mirror service.
";

#[derive(Debug, Clone)]
pub struct Options {
    pub query: String,
    pub json: bool,
    pub no_verify: bool,
    pub model_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub no_cache: bool,
    pub verbose: bool,
    /// Override the Wikidata SPARQL endpoint. Default: public
    /// `https://query.wikidata.org/sparql`. Point at a local
    /// Qlever/Oxigraph mirror for sub-100ms queries.
    pub wikidata_endpoint: Option<String>,
    /// Override the Wikipedia REST summary endpoint. Default:
    /// public `https://en.wikipedia.org/api/rest_v1/page/summary`.
    pub wikipedia_endpoint: Option<String>,
}

/// Top-level mode the CLI is operating in.
#[derive(Debug, Clone)]
pub enum Mode {
    /// Run a single query inline (default).
    Inline(Options),
    /// Run as a long-lived server.
    Serve {
        socket: PathBuf,
        model_dir: PathBuf,
        cache_dir: PathBuf,
        /// Optional endpoint overrides applied to every request
        /// the server handles. Useful for backing the server with
        /// a local Wikidata / Wikipedia mirror.
        wikidata_endpoint: Option<String>,
        wikipedia_endpoint: Option<String>,
    },
    /// Connect to a running server and forward the query.
    Client {
        socket: PathBuf,
        options: Options,
    },
}

#[derive(Debug)]
pub enum ParseError {
    Help,
    NoQuery,
    UnknownFlag(String),
    MissingValue(&'static str),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Help => write!(f, "(help requested)"),
            Self::NoQuery => write!(
                f,
                "no query supplied — pass at least one non-option argument"
            ),
            Self::UnknownFlag(s) => write!(f, "unknown flag: {s}"),
            Self::MissingValue(s) => write!(f, "{s} requires a value"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Decide between inline / serve / client mode based on argv.
/// `serve` is a positional subcommand; client mode is selected by
/// the `--socket` flag combined with a query.
pub fn parse_mode(args: Vec<String>) -> Result<Mode, ParseError> {
    if args.first().map(String::as_str) == Some("serve") {
        return parse_serve(args.into_iter().skip(1).collect());
    }
    // Inspect the args for --socket so we can route to client mode.
    let socket_override = args.iter().enumerate().find_map(|(i, a)| {
        if a == "--socket" {
            args.get(i + 1).cloned()
        } else {
            None
        }
    });
    // Strip the --socket pair out of the args before the standard
    // parser sees it (the parser doesn't know about --socket).
    let inline_args: Vec<String> = strip_pair_flag(&args, "--socket");
    let opts = parse(inline_args)?;
    match socket_override {
        Some(s) => Ok(Mode::Client {
            socket: PathBuf::from(s),
            options: opts,
        }),
        None => Ok(Mode::Inline(opts)),
    }
}

fn strip_pair_flag(args: &[String], flag: &str) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            iter.next(); // drop value
            continue;
        }
        out.push(arg.clone());
    }
    out
}

fn parse_serve(args: Vec<String>) -> Result<Mode, ParseError> {
    let mut socket = default_socket_path();
    let mut model_dir = default_model_dir();
    let mut cache_dir = default_cache_dir();
    let mut wikidata_endpoint: Option<String> = None;
    let mut wikipedia_endpoint: Option<String> = None;
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(ParseError::Help),
            "--socket" => match iter.next() {
                Some(v) => socket = PathBuf::from(v),
                None => return Err(ParseError::MissingValue("--socket")),
            },
            "--model-dir" => match iter.next() {
                Some(v) => model_dir = PathBuf::from(v),
                None => return Err(ParseError::MissingValue("--model-dir")),
            },
            "--cache-dir" => match iter.next() {
                Some(v) => cache_dir = PathBuf::from(v),
                None => return Err(ParseError::MissingValue("--cache-dir")),
            },
            "--wikidata-endpoint" => match iter.next() {
                Some(v) => wikidata_endpoint = Some(v),
                None => return Err(ParseError::MissingValue("--wikidata-endpoint")),
            },
            "--wikipedia-endpoint" => match iter.next() {
                Some(v) => wikipedia_endpoint = Some(v),
                None => return Err(ParseError::MissingValue("--wikipedia-endpoint")),
            },
            s if s.starts_with("--") => {
                return Err(ParseError::UnknownFlag(s.to_string()));
            }
            other => {
                return Err(ParseError::UnknownFlag(format!(
                    "unexpected positional arg {other:?} after `serve`"
                )));
            }
        }
    }
    Ok(Mode::Serve {
        socket,
        model_dir,
        cache_dir,
        wikidata_endpoint,
        wikipedia_endpoint,
    })
}

fn default_socket_path() -> PathBuf {
    default_cache_dir().join("server.sock")
}

pub fn parse(args: Vec<String>) -> Result<Options, ParseError> {
    let mut opts = Options {
        query: String::new(),
        json: false,
        no_verify: false,
        model_dir: default_model_dir(),
        cache_dir: default_cache_dir(),
        no_cache: false,
        verbose: false,
        wikidata_endpoint: None,
        wikipedia_endpoint: None,
    };
    let mut tokens: Vec<String> = Vec::new();

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(ParseError::Help),
            "--json" => opts.json = true,
            "--no-verify" => opts.no_verify = true,
            "--no-cache" => opts.no_cache = true,
            "--verbose" => opts.verbose = true,
            "--model-dir" => match iter.next() {
                Some(v) => opts.model_dir = PathBuf::from(v),
                None => return Err(ParseError::MissingValue("--model-dir")),
            },
            "--cache-dir" => match iter.next() {
                Some(v) => opts.cache_dir = PathBuf::from(v),
                None => return Err(ParseError::MissingValue("--cache-dir")),
            },
            "--wikidata-endpoint" => match iter.next() {
                Some(v) => opts.wikidata_endpoint = Some(v),
                None => return Err(ParseError::MissingValue("--wikidata-endpoint")),
            },
            "--wikipedia-endpoint" => match iter.next() {
                Some(v) => opts.wikipedia_endpoint = Some(v),
                None => return Err(ParseError::MissingValue("--wikipedia-endpoint")),
            },
            s if s.starts_with("--") => {
                return Err(ParseError::UnknownFlag(s.to_string()));
            }
            _ => tokens.push(arg),
        }
    }

    if tokens.is_empty() {
        return Err(ParseError::NoQuery);
    }
    opts.query = tokens.join(" ");
    Ok(opts)
}

fn default_model_dir() -> PathBuf {
    PathBuf::from("./models/deberta-v3-large-mnli")
}

fn default_cache_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("joule-edge");
    }
    PathBuf::from(".joule-edge-cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_query_argument() {
        let opts = parse(vec!["what is the capital of France?".into()]).unwrap();
        assert_eq!(opts.query, "what is the capital of France?");
        assert!(!opts.json);
        assert!(!opts.no_verify);
    }

    #[test]
    fn multi_word_query_joins_with_spaces() {
        let opts = parse(vec!["what".into(), "is".into(), "Paris".into()]).unwrap();
        assert_eq!(opts.query, "what is Paris");
    }

    #[test]
    fn json_flag_recognized() {
        let opts = parse(vec!["--json".into(), "q".into()]).unwrap();
        assert!(opts.json);
    }

    #[test]
    fn no_verify_flag_recognized() {
        let opts = parse(vec!["--no-verify".into(), "q".into()]).unwrap();
        assert!(opts.no_verify);
    }

    #[test]
    fn model_dir_takes_path() {
        let opts = parse(vec![
            "--model-dir".into(),
            "/opt/deberta".into(),
            "q".into(),
        ])
        .unwrap();
        assert_eq!(opts.model_dir, PathBuf::from("/opt/deberta"));
    }

    #[test]
    fn missing_value_for_model_dir() {
        let err = parse(vec!["--model-dir".into()]).unwrap_err();
        assert!(matches!(err, ParseError::MissingValue("--model-dir")));
    }

    #[test]
    fn no_query_errors() {
        let err = parse(vec![]).unwrap_err();
        assert!(matches!(err, ParseError::NoQuery));
        let err = parse(vec!["--json".into()]).unwrap_err();
        assert!(matches!(err, ParseError::NoQuery));
    }

    #[test]
    fn help_short_and_long() {
        assert!(matches!(parse(vec!["-h".into()]), Err(ParseError::Help)));
        assert!(matches!(parse(vec!["--help".into()]), Err(ParseError::Help)));
    }

    #[test]
    fn unknown_flag_errors() {
        let err = parse(vec!["--what".into(), "q".into()]).unwrap_err();
        assert!(matches!(err, ParseError::UnknownFlag(_)));
    }

    #[test]
    fn flags_can_come_after_query_tokens() {
        let opts = parse(vec!["q".into(), "--json".into()]).unwrap();
        assert_eq!(opts.query, "q");
        assert!(opts.json);
    }
}
