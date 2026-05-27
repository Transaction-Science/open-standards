//! Deterministic query-to-tool classifier.
//!
//! Maps a natural-language query to a [`DeterministicToolKind`] invocation
//! using rule-based pattern matching. Zero LLM, zero energy: the classifier
//! itself is deterministic, so a "no match" is just as honest as a match.
//!
//! Matchers are tried in order; the first hit wins. Specific patterns
//! (subnet, percentage, hash) precede the broad math matcher to avoid
//! false positives from numeric-only inputs.
//!
//! Confidence is a `u16` in `[0, 10_000]`. The tier above only attempts
//! execution when `confidence >= MIN_MATCH_CONFIDENCE` (8000).

use std::sync::LazyLock;

use jouleclaw_tools::{DeterministicToolKind, HashOp, PercentageOp, TextOp};
use regex::Regex;

/// Minimum router confidence required for the tier to attempt execution.
pub const MIN_MATCH_CONFIDENCE: u16 = 8000;

/// A matched tool invocation with confidence score.
#[derive(Debug, Clone)]
pub struct ToolMatch {
    /// The tool to execute.
    pub tool: DeterministicToolKind,
    /// Confidence that this query is tool-answerable, in `[0, 10_000]`.
    pub confidence: u16,
}

/// Deterministic router that maps queries to [`DeterministicToolKind`]
/// invocations.
///
/// Stateless by default. Downstream crates may push extra matchers via
/// [`ToolRouter::register_matcher`]; these run before the built-in cascade
/// so they can override built-in behaviour for domain-specific tools.
#[derive(Default)]
pub struct ToolRouter {
    // Boxed Fn so the router can be extended at runtime. Held in a Vec so
    // registration order is the dispatch order.
    extra_matchers: Vec<Box<dyn Fn(&str, &str) -> Option<ToolMatch> + Send + Sync>>,
}

impl ToolRouter {
    /// Construct a router with only the built-in matchers.
    pub fn new() -> Self {
        Self {
            extra_matchers: Vec::new(),
        }
    }

    /// Register an extra matcher. The closure receives `(lower, original)`
    /// and may return `Some(ToolMatch)` to claim the query. Extra matchers
    /// run before the built-in cascade.
    pub fn register_matcher<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&str, &str) -> Option<ToolMatch> + Send + Sync + 'static,
    {
        self.extra_matchers.push(Box::new(f));
        self
    }

    /// Register a tool directly under an exact (case-insensitive,
    /// whitespace-trimmed) query string. Convenience over
    /// [`Self::register_matcher`] for static one-shots like `"meet"` →
    /// [`DeterministicToolKind::MeetRoom`].
    pub fn register_tool(&mut self, query: impl Into<String>, tool: DeterministicToolKind) {
        let q = query.into().trim().to_lowercase();
        let tool = tool;
        self.register_matcher(move |lower, _| {
            if lower == q {
                Some(ToolMatch {
                    tool: tool.clone(),
                    confidence: 9500,
                })
            } else {
                None
            }
        });
    }

    /// Try to match a query to a deterministic tool invocation. Returns
    /// `None` if the query doesn't look like a computation request.
    pub fn match_query(&self, query: &str) -> Option<ToolMatch> {
        let q = query.trim();
        if q.is_empty() {
            return None;
        }
        let lower = q.to_lowercase();

        // Extra matchers first — domain-specific overrides win.
        for f in &self.extra_matchers {
            if let Some(m) = f(&lower, q) {
                return Some(m);
            }
        }

        // Built-in cascade. Specific patterns before broad math.
        try_match_uuid(&lower)
            .or_else(|| try_match_hash(&lower))
            .or_else(|| try_match_percentage(&lower))
            .or_else(|| try_match_unit_conversion(&lower))
            .or_else(|| try_match_data_size(&lower))
            .or_else(|| try_match_text_transform(&lower))
            .or_else(|| try_match_statistics(&lower))
            .or_else(|| try_match_day_of_week(&lower))
            .or_else(|| try_match_now(&lower))
            .or_else(|| try_match_math(&lower, q))
    }
}

// ─── UUID ────────────────────────────────────────────────────────────────

fn try_match_uuid(lower: &str) -> Option<ToolMatch> {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(?:generate|create|new|make|give me|get)\s+(?:a\s+)?(?:uuid|guid)(?:\s+v?4)?$")
            .ok()
    });
    if let Some(re) = RE.as_ref()
        && re.is_match(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::Uuid,
            confidence: 9500,
        });
    }
    if lower == "uuid" || lower == "guid" {
        return Some(ToolMatch {
            tool: DeterministicToolKind::Uuid,
            confidence: 9000,
        });
    }
    None
}

// ─── Hash ────────────────────────────────────────────────────────────────

fn try_match_hash(lower: &str) -> Option<ToolMatch> {
    static RE_SHA256: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^(?:sha-?256|sha256)\s+(?:of|hash|for)\s+(.+)$").ok());
    static RE_MD5: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^(?:md5)\s+(?:of|hash|for)\s+(.+)$").ok());
    static RE_HASH_OF: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^(?:hash|checksum)\s+(?:of|for)\s+(.+)$").ok());

    if let Some(re) = RE_SHA256.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let input = strip_quotes(caps[1].trim());
        return Some(ToolMatch {
            tool: DeterministicToolKind::Hash {
                operation: HashOp::Sha256 { input },
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_MD5.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let input = strip_quotes(caps[1].trim());
        return Some(ToolMatch {
            tool: DeterministicToolKind::Hash {
                operation: HashOp::Md5 { input },
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_HASH_OF.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let input = strip_quotes(caps[1].trim());
        return Some(ToolMatch {
            tool: DeterministicToolKind::Hash {
                operation: HashOp::Sha256 { input },
            },
            confidence: 8500,
        });
    }
    None
}

fn strip_quotes(s: &str) -> String {
    s.trim_matches('"').trim_matches('\'').to_string()
}

// ─── Math ────────────────────────────────────────────────────────────────

fn try_match_math(lower: &str, original: &str) -> Option<ToolMatch> {
    static RE_EXPLICIT: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(?:calculate|compute|eval|evaluate|solve)\s+(.+)$").ok()
    });
    static RE_WHATIS: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^(?:what\s+is|what's|whats)\s+(.+?)(?:\s*\?)?$").ok());
    static RE_PURE_EXPR: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^[\d\s\+\-\*/\^%\(\)\.,]+$").ok());
    static RE_FUNC_EXPR: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"(?:sqrt|sin|cos|tan|asin|acos|atan|log|ln|abs|ceil|floor|round|exp|factorial|pi|tau)\s*\(",
        )
        .ok()
    });

    if let Some(re) = RE_EXPLICIT.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let expr = caps[1].trim().to_string();
        if looks_mathematical(&expr) {
            return Some(ToolMatch {
                tool: DeterministicToolKind::Math { expression: expr },
                confidence: 9500,
            });
        }
    }
    if let Some(re) = RE_WHATIS.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let expr = caps[1].trim().to_string();
        if looks_mathematical(&expr) {
            return Some(ToolMatch {
                tool: DeterministicToolKind::Math { expression: expr },
                confidence: 9000,
            });
        }
    }
    let trimmed = original.trim();
    if let Some(re) = RE_PURE_EXPR.as_ref()
        && re.is_match(trimmed)
        && trimmed.chars().any(|c| "+-*/^%".contains(c))
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::Math {
                expression: trimmed.to_string(),
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_FUNC_EXPR.as_ref()
        && re.is_match(lower)
        && looks_mathematical(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::Math {
                expression: original.trim().to_string(),
            },
            confidence: 9000,
        });
    }
    None
}

/// Heuristic: does this string look like a math expression rather than English?
fn looks_mathematical(s: &str) -> bool {
    let has_operator = s.chars().any(|c| "+-*/^%".contains(c));
    let has_digit = s.chars().any(|c| c.is_ascii_digit());
    let math_funcs = [
        "sqrt", "sin", "cos", "tan", "log", "ln", "abs", "exp", "pi", "factorial",
    ];
    let has_math_func = math_funcs.iter().any(|f| s.contains(f));

    // Reject if the string contains English words other than known math identifiers.
    let allowed = [
        "pi", "e", "tau", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "ln", "abs",
        "ceil", "floor", "round", "exp", "factorial", "mod",
    ];
    let has_english_words = s.split_whitespace().any(|w| {
        w.chars().all(|c| c.is_alphabetic()) && w.len() > 1 && !allowed.contains(&w)
    });
    if has_english_words {
        return false;
    }
    (has_digit && has_operator) || has_math_func
}

// ─── Unit conversion ─────────────────────────────────────────────────────

fn try_match_unit_conversion(lower: &str) -> Option<ToolMatch> {
    static RE_CONVERT: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^(?:convert\s+)?(\d+(?:\.\d+)?)\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:to|in|into|as)\s+([a-z]+(?:\s+[a-z]+)?)$",
        )
        .ok()
    });
    static RE_HOWMANY: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^how\s+many\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:in|are\s+in)\s+(\d+(?:\.\d+)?)\s+([a-z]+(?:\s+[a-z]+)?)(?:\s*\?)?$",
        )
        .ok()
    });

    if let Some(re) = RE_CONVERT.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let value: f64 = caps[1].parse().ok()?;
        let from = normalize_unit(&caps[2]);
        let to = normalize_unit(&caps[3]);
        if is_known_unit(&from) && is_known_unit(&to) {
            return Some(ToolMatch {
                tool: DeterministicToolKind::UnitConversion {
                    value,
                    from_unit: from,
                    to_unit: to,
                },
                confidence: 9500,
            });
        }
    }
    if let Some(re) = RE_HOWMANY.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let value: f64 = caps[2].parse().ok()?;
        let to = normalize_unit(&caps[1]);
        let from = normalize_unit(&caps[3]);
        if is_known_unit(&from) && is_known_unit(&to) {
            return Some(ToolMatch {
                tool: DeterministicToolKind::UnitConversion {
                    value,
                    from_unit: from,
                    to_unit: to,
                },
                confidence: 9000,
            });
        }
    }
    None
}

/// Normalise common unit aliases to a canonical form understood by
/// [`jouleclaw_tools::convert_units`].
fn normalize_unit(unit: &str) -> String {
    match unit.trim() {
        // Length
        "km" | "kms" | "kilometers" | "kilometres" | "kilometer" | "kilometre" => "km".into(),
        "m" | "meters" | "metres" | "meter" | "metre" => "m".into(),
        "cm" | "centimeters" | "centimetres" | "centimeter" | "centimetre" => "cm".into(),
        "mm" | "millimeters" | "millimetres" | "millimeter" | "millimetre" => "mm".into(),
        "mi" | "miles" | "mile" => "mi".into(),
        "ft" | "feet" | "foot" => "ft".into(),
        "in" | "inches" | "inch" => "in".into(),
        // Mass
        "kg" | "kgs" | "kilograms" | "kilogram" => "kg".into(),
        "g" | "grams" | "gram" => "g".into(),
        "lb" | "lbs" | "pounds" | "pound" => "lb".into(),
        "oz" | "ounces" | "ounce" => "oz".into(),
        // Temperature
        "c" | "celsius" => "C".into(),
        "f" | "fahrenheit" => "F".into(),
        "k" | "kelvin" => "K".into(),
        // Data
        "b" | "bytes" | "byte" => "B".into(),
        "kb" | "kilobytes" | "kilobyte" => "KB".into(),
        "mb" | "megabytes" | "megabyte" => "MB".into(),
        "gb" | "gigabytes" | "gigabyte" => "GB".into(),
        "tb" | "terabytes" | "terabyte" => "TB".into(),
        // Time
        "s" | "sec" | "secs" | "seconds" | "second" => "s".into(),
        "min" | "mins" | "minutes" | "minute" => "min".into(),
        "h" | "hr" | "hrs" | "hours" | "hour" => "h".into(),
        // Energy
        "j" | "joules" | "joule" => "J".into(),
        "kj" | "kilojoules" | "kilojoule" => "kJ".into(),
        "wh" | "watt hours" | "watt hour" => "Wh".into(),
        "kwh" | "kilowatt hours" | "kilowatt hour" => "kWh".into(),
        other => other.to_string(),
    }
}

fn is_known_unit(unit: &str) -> bool {
    matches!(
        unit,
        "km" | "m"
            | "cm"
            | "mm"
            | "mi"
            | "ft"
            | "in"
            | "kg"
            | "g"
            | "lb"
            | "oz"
            | "C"
            | "F"
            | "K"
            | "B"
            | "KB"
            | "MB"
            | "GB"
            | "TB"
            | "s"
            | "min"
            | "h"
            | "J"
            | "kJ"
            | "Wh"
            | "kWh"
    )
}

// ─── Percentage ──────────────────────────────────────────────────────────

fn try_match_percentage(lower: &str) -> Option<ToolMatch> {
    static RE_OF: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)(?:\s*\?)?$").ok()
    });
    static RE_CHANGE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^(?:percent(?:age)?\s+change|change)\s+from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)(?:\s*\?)?$",
        )
        .ok()
    });

    if let Some(re) = RE_OF.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let percent: f64 = caps[1].parse().ok()?;
        let value: f64 = caps[2].parse().ok()?;
        return Some(ToolMatch {
            tool: DeterministicToolKind::Percentage {
                operation: PercentageOp::Of { percent, value },
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_CHANGE.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let old: f64 = caps[1].parse().ok()?;
        let new: f64 = caps[2].parse().ok()?;
        return Some(ToolMatch {
            tool: DeterministicToolKind::Percentage {
                operation: PercentageOp::Change { old, new },
            },
            confidence: 9500,
        });
    }
    None
}

// ─── Data size ───────────────────────────────────────────────────────────

fn try_match_data_size(lower: &str) -> Option<ToolMatch> {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(?:format\s+)?(\d+)\s+bytes(?:\s+(?:in\s+)?(?:human|readable))?$").ok()
    });
    if let Some(re) = RE.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let bytes: u64 = caps[1].parse().ok()?;
        return Some(ToolMatch {
            tool: DeterministicToolKind::DataSize { bytes },
            confidence: 9000,
        });
    }
    None
}

// ─── Text transforms ─────────────────────────────────────────────────────

fn try_match_text_transform(lower: &str) -> Option<ToolMatch> {
    static RE_B64_ENC: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^base64\s+encode\s+(.+)$").ok());
    static RE_B64_DEC: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^base64\s+decode\s+(.+)$").ok());
    static RE_URL_ENC: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^url\s+encode\s+(.+)$").ok());
    static RE_URL_DEC: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^url\s+decode\s+(.+)$").ok());
    static RE_WORD_COUNT: LazyLock<Option<Regex>> =
        LazyLock::new(|| Regex::new(r"^word\s+count\s+(.+)$").ok());
    static RE_CASE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(r"^(uppercase|lowercase|titlecase|title\s+case|reverse)\s+(.+)$").ok()
    });

    if let Some(re) = RE_B64_ENC.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: TextOp::Base64Encode,
                input: caps[1].trim().to_string(),
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_B64_DEC.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: TextOp::Base64Decode,
                input: caps[1].trim().to_string(),
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_URL_ENC.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: TextOp::UrlEncode,
                input: caps[1].trim().to_string(),
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_URL_DEC.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: TextOp::UrlDecode,
                input: caps[1].trim().to_string(),
            },
            confidence: 9500,
        });
    }
    if let Some(re) = RE_WORD_COUNT.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: TextOp::WordCount,
                input: caps[1].trim().to_string(),
            },
            confidence: 9000,
        });
    }
    if let Some(re) = RE_CASE.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let op = match &caps[1] {
            "uppercase" => TextOp::Uppercase,
            "lowercase" => TextOp::Lowercase,
            "titlecase" | "title case" => TextOp::TitleCase,
            "reverse" => TextOp::Reverse,
            _ => return None,
        };
        return Some(ToolMatch {
            tool: DeterministicToolKind::TextTransform {
                operation: op,
                input: caps[2].trim().to_string(),
            },
            confidence: 9000,
        });
    }
    None
}

// ─── Statistics ──────────────────────────────────────────────────────────

fn try_match_statistics(lower: &str) -> Option<ToolMatch> {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^(?:mean|average|median|mode|std\s*dev(?:iation)?|standard\s+deviation|variance|stats?|statistics)\s+(?:of\s+)?(.+)$",
        )
        .ok()
    });
    if let Some(re) = RE.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        let data = caps[1].trim().to_string();
        if data
            .split([',', ' ', ';'])
            .filter(|s| !s.is_empty())
            .all(|s| s.trim().parse::<f64>().is_ok())
        {
            return Some(ToolMatch {
                tool: DeterministicToolKind::Statistics { input: data },
                confidence: 9000,
            });
        }
    }
    None
}

// ─── Day of week ─────────────────────────────────────────────────────────

fn try_match_day_of_week(lower: &str) -> Option<ToolMatch> {
    static RE: LazyLock<Option<Regex>> = LazyLock::new(|| {
        Regex::new(
            r"^(?:what\s+day\s+(?:is|was)|day\s+of\s+(?:the\s+)?week\s+(?:for|of|on)?)\s+(.+?)(?:\s*\?)?$",
        )
        .ok()
    });
    if let Some(re) = RE.as_ref()
        && let Some(caps) = re.captures(lower)
    {
        return Some(ToolMatch {
            tool: DeterministicToolKind::DateCalc {
                operation: jouleclaw_tools::DateOp::DayOfWeek {
                    date: caps[1].trim().to_string(),
                },
            },
            confidence: 9000,
        });
    }
    None
}

// ─── Now / current time ──────────────────────────────────────────────────

fn try_match_now(lower: &str) -> Option<ToolMatch> {
    if matches!(
        lower,
        "now" | "current time" | "what time is it" | "what time is it?" | "time now"
    ) {
        return Some(ToolMatch {
            tool: DeterministicToolKind::CurrentTime { timezone: None },
            confidence: 9000,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> ToolRouter {
        ToolRouter::new()
    }

    #[test]
    fn empty_query_no_match() {
        assert!(router().match_query("").is_none());
        assert!(router().match_query("   ").is_none());
    }

    #[test]
    fn search_query_no_match() {
        // Natural-language search queries must NOT trip the matcher.
        assert!(
            router()
                .match_query("best programming language 2026")
                .is_none()
        );
    }

    #[test]
    fn math_calculate_form() {
        let m = router().match_query("calculate 2 + 2").expect("match");
        assert!(m.confidence >= MIN_MATCH_CONFIDENCE);
        match m.tool {
            DeterministicToolKind::Math { expression } => assert_eq!(expression, "2 + 2"),
            other => panic!("expected Math, got {other:?}"),
        }
    }

    #[test]
    fn math_pure_expression() {
        let m = router().match_query("5 * (3 + 2)").expect("match");
        match m.tool {
            DeterministicToolKind::Math { .. } => {}
            other => panic!("expected Math, got {other:?}"),
        }
    }

    #[test]
    fn uuid_command_forms() {
        for q in ["uuid", "guid", "generate uuid", "give me a uuid v4"] {
            let m = router().match_query(q).unwrap_or_else(|| panic!("{q}"));
            assert!(matches!(m.tool, DeterministicToolKind::Uuid));
        }
    }

    #[test]
    fn sha256_hash_form() {
        let m = router().match_query("sha256 of hello").expect("match");
        match m.tool {
            DeterministicToolKind::Hash {
                operation: HashOp::Sha256 { input },
            } => assert_eq!(input, "hello"),
            other => panic!("expected sha256, got {other:?}"),
        }
    }

    #[test]
    fn md5_form() {
        let m = router().match_query("md5 of hello").expect("match");
        assert!(matches!(
            m.tool,
            DeterministicToolKind::Hash {
                operation: HashOp::Md5 { .. }
            }
        ));
    }

    #[test]
    fn unit_conversion_miles_to_km() {
        let m = router().match_query("convert 5 miles to km").expect("match");
        match m.tool {
            DeterministicToolKind::UnitConversion {
                value,
                from_unit,
                to_unit,
            } => {
                assert_eq!(value, 5.0);
                assert_eq!(from_unit, "mi");
                assert_eq!(to_unit, "km");
            }
            other => panic!("expected UnitConversion, got {other:?}"),
        }
    }

    #[test]
    fn percentage_of_form() {
        let m = router().match_query("what is 25% of 200").expect("match");
        match m.tool {
            DeterministicToolKind::Percentage {
                operation: PercentageOp::Of { percent, value },
            } => {
                assert_eq!(percent, 25.0);
                assert_eq!(value, 200.0);
            }
            other => panic!("expected Percentage::Of, got {other:?}"),
        }
    }

    #[test]
    fn base64_encode_form() {
        let m = router().match_query("base64 encode hello").expect("match");
        match m.tool {
            DeterministicToolKind::TextTransform {
                operation: TextOp::Base64Encode,
                input,
            } => assert_eq!(input, "hello"),
            other => panic!("expected base64 encode, got {other:?}"),
        }
    }

    #[test]
    fn data_size_form() {
        let m = router().match_query("1073741824 bytes").expect("match");
        assert!(matches!(m.tool, DeterministicToolKind::DataSize { bytes } if bytes == 1_073_741_824));
    }

    #[test]
    fn extra_matcher_wins_over_builtin() {
        let mut r = ToolRouter::new();
        r.register_tool("uuid", DeterministicToolKind::MeetRoom);
        let m = r.match_query("uuid").expect("match");
        // Extra matcher fired first → returned MeetRoom, not Uuid.
        assert!(matches!(m.tool, DeterministicToolKind::MeetRoom));
    }

    #[test]
    fn confidence_meets_floor() {
        // Every built-in match should clear MIN_MATCH_CONFIDENCE.
        for q in [
            "calculate 2 + 2",
            "uuid",
            "sha256 of hello",
            "convert 5 miles to km",
            "what is 25% of 200",
            "base64 encode hello",
        ] {
            let m = router().match_query(q).unwrap_or_else(|| panic!("{q}"));
            assert!(
                m.confidence >= MIN_MATCH_CONFIDENCE,
                "{q} confidence {} < {}",
                m.confidence,
                MIN_MATCH_CONFIDENCE
            );
        }
    }
}
