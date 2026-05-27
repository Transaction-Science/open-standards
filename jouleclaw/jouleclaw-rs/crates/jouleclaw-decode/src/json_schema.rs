//! JSON-schema subset → regex lowering.
//!
//! We compile a useful subset of JSON-Schema (`type`, `enum`, `const`,
//! `properties`, `required`, `items`, `minItems`, `maxItems`,
//! `pattern`, `minimum`, `maximum`) into a regex that matches the
//! exact JSON string the model would emit.
//!
//! ## What we cover
//!
//! - `"type": "string"` — matches a JSON string literal, optionally
//!   bounded by `minLength` / `maxLength` and optionally constrained
//!   by `pattern`.
//! - `"type": "number"` and `"type": "integer"` — matches a JSON
//!   number literal.
//! - `"type": "boolean"` — `true|false`.
//! - `"type": "null"` — `null`.
//! - `"type": "array"`, with `items` (homogeneous), `minItems`, `maxItems`.
//! - `"type": "object"`, with `properties` and `required`. Properties
//!   are emitted in `required` order; non-required keys are not
//!   currently emitted (a documented v0.1 compromise).
//! - `"enum"` — alternation over JSON literals.
//! - `"const"` — exact JSON literal.
//!
//! ## Compromises (v0.1)
//!
//! - Optional object properties are omitted (only required keys
//!   appear in the emitted object). This guarantees a single canonical
//!   ordering and keeps the regex closed-form.
//! - `additionalProperties` is ignored.
//! - `oneOf`, `anyOf`, `allOf`, `not` are not yet supported.
//! - JSON-string escapes are limited to `\" \\ \/ \b \f \n \r \t`
//!   plus `\uXXXX`. Arbitrary control bytes are excluded.
//! - Numeric constraints (`minimum`, `maximum`, `multipleOf`) are not
//!   enforced inside the regex; only the *shape* is enforced.

use serde_json::Value;

use crate::error::DecodeError;

/// Lower a `serde_json::Value` schema into a regex string.
pub fn schema_to_regex(schema: &Value) -> Result<String, DecodeError> {
    let mut out = String::new();
    emit_value_with_ws(schema, &mut out)?;
    Ok(out)
}

fn emit_value_with_ws(schema: &Value, out: &mut String) -> Result<(), DecodeError> {
    emit_value(schema, out)
}

fn emit_value(schema: &Value, out: &mut String) -> Result<(), DecodeError> {
    // `const` wins over everything else.
    if let Some(c) = schema.get("const") {
        let lit = json_value_literal(c)?;
        out.push_str(&escape_regex(&lit));
        return Ok(());
    }
    // `enum` is an alternation over canonical literals.
    if let Some(Value::Array(opts)) = schema.get("enum") {
        if opts.is_empty() {
            return Err(DecodeError::JsonSchema("empty enum".into()));
        }
        out.push_str("(?:");
        for (i, opt) in opts.iter().enumerate() {
            if i > 0 {
                out.push('|');
            }
            let lit = json_value_literal(opt)?;
            out.push_str(&escape_regex(&lit));
        }
        out.push(')');
        return Ok(());
    }
    let ty = schema
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| DecodeError::JsonSchema("schema missing `type`".into()))?;
    match ty {
        "string" => emit_string(schema, out),
        "integer" => {
            out.push_str("-?(?:0|[1-9][0-9]*)");
            Ok(())
        }
        "number" => {
            // signed JSON number with optional fraction + exponent.
            out.push_str(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?");
            Ok(())
        }
        "boolean" => {
            out.push_str("(?:true|false)");
            Ok(())
        }
        "null" => {
            out.push_str("null");
            Ok(())
        }
        "array" => emit_array(schema, out),
        "object" => emit_object(schema, out),
        other => Err(DecodeError::JsonSchema(format!(
            "unsupported type `{other}`"
        ))),
    }
}

fn emit_string(schema: &Value, out: &mut String) -> Result<(), DecodeError> {
    if let Some(Value::String(pat)) = schema.get("pattern") {
        // Embed the user's pattern as the *contents* of a JSON string.
        // We trust the caller's pattern but wrap it in quotes.
        out.push('"');
        out.push_str(pat);
        out.push('"');
        return Ok(());
    }
    let min = schema.get("minLength").and_then(|v| v.as_u64()).unwrap_or(0);
    let max = schema.get("maxLength").and_then(|v| v.as_u64());
    // A JSON string char: printable ASCII excluding `"` and `\`, plus
    // escape sequences. We keep it ASCII for v0.1.
    let chr = r#"(?:[ !#-\[\]-~]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})"#;
    out.push('"');
    out.push_str(chr);
    match (min, max) {
        (0, None) => out.push('*'),
        (n, None) => out.push_str(&format!("{{{n},}}", n = n)),
        (n, Some(m)) => out.push_str(&format!("{{{n},{m}}}")),
    }
    out.push('"');
    Ok(())
}

fn emit_array(schema: &Value, out: &mut String) -> Result<(), DecodeError> {
    let items = schema.get("items");
    let min = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);
    let max = schema.get("maxItems").and_then(|v| v.as_u64());

    let mut item_re = String::new();
    if let Some(items) = items {
        emit_value(items, &mut item_re)?;
    } else {
        // Untyped items — accept any JSON scalar.
        item_re.push_str(&any_scalar_regex());
    }

    out.push('\\');
    out.push('[');
    match (min, max) {
        (0, Some(0)) => { /* empty array */ }
        (0, _) => {
            // Optional: [] or [a,a,...]
            let upper = max.map(|m| m as usize);
            let inner = repeat_with_separator(&item_re, 1, upper, ",");
            out.push_str("(?:");
            out.push_str(&inner);
            out.push_str(")?");
        }
        (n, _) => {
            let lower = n as usize;
            let upper = max.map(|m| m as usize);
            let inner = repeat_with_separator(&item_re, lower, upper, ",");
            out.push_str(&inner);
        }
    }
    out.push('\\');
    out.push(']');
    Ok(())
}

fn emit_object(schema: &Value, out: &mut String) -> Result<(), DecodeError> {
    let props = schema
        .get("properties")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let required: Vec<String> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    if required.is_empty() && props.is_empty() {
        out.push_str("\\{\\}");
        return Ok(());
    }
    out.push_str("\\{");
    for (i, key) in required.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&escape_regex(key));
        out.push('"');
        out.push(':');
        let prop_schema = props
            .get(key)
            .ok_or_else(|| DecodeError::JsonSchema(format!("required key `{key}` has no schema")))?;
        emit_value(prop_schema, out)?;
    }
    out.push_str("\\}");
    Ok(())
}

fn any_scalar_regex() -> String {
    // Loose union over JSON scalars.
    String::from(
        r#"(?:null|true|false|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|"(?:[ !#-\[\]-~]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*")"#,
    )
}

fn repeat_with_separator(
    item: &str,
    min: usize,
    max: Option<usize>,
    sep: &str,
) -> String {
    // First item, then (min-1) or more `,item` groups.
    let mut s = String::new();
    s.push_str("(?:");
    s.push_str(item);
    let lo_after = min.saturating_sub(1);
    let hi_after = max.map(|m| m.saturating_sub(1));
    s.push_str("(?:");
    s.push_str(sep);
    s.push_str(item);
    s.push(')');
    match (lo_after, hi_after) {
        (0, None) => s.push('*'),
        (n, None) => s.push_str(&format!("{{{n},}}")),
        (n, Some(m)) => s.push_str(&format!("{{{n},{m}}}")),
    }
    s.push(')');
    s
}

/// Format a `serde_json::Value` as a canonical JSON literal (no whitespace).
fn json_value_literal(v: &Value) -> Result<String, DecodeError> {
    serde_json::to_string(v)
        .map_err(|e| DecodeError::JsonSchema(format!("cannot serialise literal: {e}")))
}

/// Escape a string so it matches itself literally in our regex syntax.
fn escape_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '.' | '\\' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' => {
                out.push('\\');
                out.push(ch);
            }
            c if (c as u32) < 0x20 => {
                // control byte — use \xHH
                out.push_str(&format!("\\x{:02X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::Nfa;
    use crate::grammar::{Compiled, Grammar};
    use serde_json::json;

    fn nfa_of(g: &Grammar) -> &Nfa {
        match g.compiled() {
            Compiled::Nfa(n) => &n.nfa,
            Compiled::Cfg(_) => panic!("expected nfa-form grammar"),
        }
    }

    #[test]
    fn integer_schema() {
        let re = schema_to_regex(&json!({"type":"integer"})).unwrap();
        let g = Grammar::from_regex(&re).unwrap();
        let nfa = nfa_of(&g);
        let s = nfa.start_set();
        let end = nfa.run_bytes(&s, b"42").unwrap();
        assert!(nfa.any_accept(&end));
        let end = nfa.run_bytes(&s, b"-7").unwrap();
        assert!(nfa.any_accept(&end));
        // 03 is not a valid JSON integer.
        let end = nfa.run_bytes(&s, b"03");
        assert!(end.is_none() || !nfa.any_accept(&end.unwrap()));
    }

    #[test]
    fn boolean_schema() {
        let re = schema_to_regex(&json!({"type":"boolean"})).unwrap();
        let g = Grammar::from_regex(&re).unwrap();
        let nfa = nfa_of(&g);
        let s = nfa.start_set();
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"true").unwrap()));
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"false").unwrap()));
        assert!(nfa.run_bytes(&s, b"yes").is_none());
    }

    #[test]
    fn enum_schema() {
        let re = schema_to_regex(&json!({"enum":["red","green","blue"]})).unwrap();
        let g = Grammar::from_regex(&re).unwrap();
        let nfa = nfa_of(&g);
        let s = nfa.start_set();
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"\"red\"").unwrap()));
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"\"blue\"").unwrap()));
        assert!(nfa.run_bytes(&s, b"\"yellow\"").is_none());
    }

    #[test]
    fn object_schema_required_only() {
        let re = schema_to_regex(&json!({
            "type":"object",
            "properties": {
                "name": {"type":"string"},
                "age": {"type":"integer"}
            },
            "required": ["name","age"]
        }))
        .unwrap();
        let g = Grammar::from_regex(&re).unwrap();
        let nfa = nfa_of(&g);
        let s = nfa.start_set();
        let ok = b"{\"name\":\"Ada\",\"age\":42}";
        let end = nfa.run_bytes(&s, ok).unwrap();
        assert!(nfa.any_accept(&end));
        // Wrong order → reject.
        let bad = b"{\"age\":42,\"name\":\"Ada\"}";
        let end = nfa.run_bytes(&s, bad);
        assert!(end.is_none() || !nfa.any_accept(&end.unwrap()));
    }

    #[test]
    fn array_schema_bounds() {
        let re = schema_to_regex(&json!({
            "type":"array",
            "items": {"type":"integer"},
            "minItems": 1,
            "maxItems": 3
        }))
        .unwrap();
        let g = Grammar::from_regex(&re).unwrap();
        let nfa = nfa_of(&g);
        let s = nfa.start_set();
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"[1]").unwrap()));
        assert!(nfa.any_accept(&nfa.run_bytes(&s, b"[1,2,3]").unwrap()));
        // empty array fails minItems=1.
        let e = nfa.run_bytes(&s, b"[]");
        assert!(e.is_none() || !nfa.any_accept(&e.unwrap()));
    }
}
