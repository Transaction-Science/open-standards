//! StatsD / DogStatsD line format.
//!
//! Line shape:
//!
//! ```text
//! <metric_name>:<value>|<type>[|@<sample_rate>][|#tag1:val1,tag2:val2]
//! ```
//!
//! - `<type>` is one of `c` (counter), `g` (gauge), `h` (histogram), `ms`
//!   (timing), `s` (set).
//! - Sample rate (`@0.1`) is optional.
//! - Tags (`#a:b,c:d`) follow the DogStatsD extension.

use crate::span::AttrValue;

/// One StatsD/DogStatsD line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsdLine {
    rendered: String,
}

impl StatsdLine {
    /// Counter line: `name:value|c`.
    pub fn counter(name: &str, value: u64, tags: &[(String, AttrValue)]) -> Self {
        Self::build(name, &value.to_string(), "c", None, tags)
    }

    /// Gauge line: `name:value|g`.
    pub fn gauge(name: &str, value: f64, tags: &[(String, AttrValue)]) -> Self {
        Self::build(name, &format_float(value), "g", None, tags)
    }

    /// Histogram line: `name:value|h`.
    pub fn histogram(name: &str, value: f64, tags: &[(String, AttrValue)]) -> Self {
        Self::build(name, &format_float(value), "h", None, tags)
    }

    /// Timing line: `name:value|ms`.
    pub fn timing_ms(name: &str, value_ms: f64, tags: &[(String, AttrValue)]) -> Self {
        Self::build(name, &format_float(value_ms), "ms", None, tags)
    }

    /// Sampled counter: `name:value|c|@rate`.
    pub fn counter_sampled(
        name: &str,
        value: u64,
        sample_rate: f64,
        tags: &[(String, AttrValue)],
    ) -> Self {
        Self::build(name, &value.to_string(), "c", Some(sample_rate), tags)
    }

    /// As a UDP-friendly string.
    pub fn as_str(&self) -> &str {
        &self.rendered
    }

    /// Take ownership of the rendered string.
    pub fn into_string(self) -> String {
        self.rendered
    }

    fn build(
        name: &str,
        value: &str,
        ty: &str,
        sample_rate: Option<f64>,
        tags: &[(String, AttrValue)],
    ) -> Self {
        let mut s = String::with_capacity(64);
        s.push_str(&sanitize_name(name));
        s.push(':');
        s.push_str(value);
        s.push('|');
        s.push_str(ty);
        if let Some(rate) = sample_rate {
            s.push('|');
            s.push('@');
            s.push_str(&format_float(rate));
        }
        if !tags.is_empty() {
            s.push('|');
            s.push('#');
            for (i, (k, v)) in tags.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&sanitize_tag(k));
                s.push(':');
                s.push_str(&sanitize_tag(&attr_value_to_string(v)));
            }
        }
        Self { rendered: s }
    }
}

fn sanitize_name(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ':' | '|' | '#' => '_',
            other => other,
        })
        .collect()
}

fn sanitize_tag(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ',' | '|' | '#' => '_',
            other => other,
        })
        .collect()
}

fn attr_value_to_string(v: &AttrValue) -> String {
    match v {
        AttrValue::String(s) => s.clone(),
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => format_float(*f),
        AttrValue::Bool(b) => b.to_string(),
    }
}

fn format_float(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e16 {
        format!("{}", f as i64)
    } else {
        f.to_string()
    }
}
