//! Prometheus text-format exposition.
//!
//! Implements the [text-based exposition format] used by Prometheus 2.x,
//! including `# HELP`, `# TYPE`, `_bucket{le="..."}`, `_sum`, `_count`.
//!
//! [text-based exposition format]: https://prometheus.io/docs/instrumenting/exposition_formats/

use crate::metric::{Counter, Gauge, Histogram};
use crate::span::AttrValue;

/// Aggregator for Prometheus exposition.
#[derive(Debug, Default)]
pub struct PrometheusExposer {
    buf: String,
}

impl PrometheusExposer {
    /// New empty exposer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Expose a counter.
    pub fn add_counter(&mut self, c: &Counter) {
        self.write_help(c.name(), c.description());
        self.write_type(c.name(), "counter");
        self.write_sample(c.name(), c.attributes(), c.get() as f64);
    }

    /// Expose a gauge.
    pub fn add_gauge(&mut self, g: &Gauge) {
        self.write_help(g.name(), g.description());
        self.write_type(g.name(), "gauge");
        self.write_sample(g.name(), g.attributes(), g.get());
    }

    /// Expose a histogram with cumulative buckets, sum, count.
    pub fn add_histogram(&mut self, h: &Histogram) {
        self.write_help(h.name(), h.description());
        self.write_type(h.name(), "histogram");
        let snap = h.snapshot();
        let attrs = h.attributes();
        let bucket_name = format!("{}_bucket", h.name());
        for (i, b) in snap.boundaries.iter().enumerate() {
            let mut labels = render_attrs(attrs);
            let le_entry = format!("le=\"{}\"", format_float(*b));
            push_label(&mut labels, &le_entry);
            self.buf
                .push_str(&format!("{bucket_name}{{{labels}}} {}\n", snap.cumulative[i]));
        }
        // +Inf bucket
        let mut labels = render_attrs(attrs);
        push_label(&mut labels, "le=\"+Inf\"");
        let last = *snap.cumulative.last().unwrap_or(&0);
        self.buf
            .push_str(&format!("{bucket_name}{{{labels}}} {last}\n"));

        let attr_labels = render_attrs(attrs);
        let attr_section = if attr_labels.is_empty() {
            String::new()
        } else {
            format!("{{{attr_labels}}}")
        };
        self.buf
            .push_str(&format!("{}_sum{} {}\n", h.name(), attr_section, snap.sum));
        self.buf
            .push_str(&format!("{}_count{} {}\n", h.name(), attr_section, snap.count));
    }

    /// Consume and return the exposition text.
    pub fn finish(self) -> String {
        self.buf
    }

    fn write_help(&mut self, name: &str, description: &str) {
        if !description.is_empty() {
            self.buf
                .push_str(&format!("# HELP {} {}\n", name, escape_help(description)));
        }
    }

    fn write_type(&mut self, name: &str, ty: &str) {
        self.buf.push_str(&format!("# TYPE {name} {ty}\n"));
    }

    fn write_sample(&mut self, name: &str, attrs: &[(String, AttrValue)], value: f64) {
        let labels = render_attrs(attrs);
        if labels.is_empty() {
            self.buf.push_str(&format!("{name} {value}\n"));
        } else {
            self.buf.push_str(&format!("{name}{{{labels}}} {value}\n"));
        }
    }
}

fn render_attrs(attrs: &[(String, AttrValue)]) -> String {
    let mut out = String::new();
    for (k, v) in attrs.iter() {
        if !out.is_empty() {
            out.push(',');
        }
        out.push_str(k);
        out.push_str("=\"");
        out.push_str(&escape_label(&attr_to_string(v)));
        out.push('"');
    }
    out
}

fn push_label(labels: &mut String, kv: &str) {
    if !labels.is_empty() {
        labels.push(',');
    }
    labels.push_str(kv);
}

fn attr_to_string(v: &AttrValue) -> String {
    match v {
        AttrValue::String(s) => s.clone(),
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => f.to_string(),
        AttrValue::Bool(b) => b.to_string(),
    }
}

fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

fn escape_help(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

fn format_float(f: f64) -> String {
    if f.is_infinite() {
        if f.is_sign_positive() {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else if f == f.trunc() && f.abs() < 1e16 {
        format!("{}", f as i64)
    } else {
        f.to_string()
    }
}
