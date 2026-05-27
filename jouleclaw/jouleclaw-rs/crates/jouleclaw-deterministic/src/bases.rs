//! Base-conversion primitives. Decimal works with `i64`; hex output is
//! lowercase with a `0x` prefix; binary and octal are bare digits.

use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(DecToHex),
        Arc::new(HexToDec),
        Arc::new(DecToBin),
        Arc::new(BinToDec),
        Arc::new(DecToOct),
        Arc::new(OctToDec),
    ]
}

fn strip_prefix_ci<'a>(q: &'a str, prefix: &str) -> Option<&'a str> {
    let q = q.trim();
    if q.len() < prefix.len() {
        return None;
    }
    let (head, tail) = q.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest.trim())
}

pub struct DecToHex;
impl LawfulPrimitive for DecToHex {
    fn id(&self) -> &str {
        "lawful:bases:dec-to-hex"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "dec to hex")?;
        let n: i64 = rest.parse().ok()?;
        if n < 0 {
            Some(format!("-0x{:x}", -n))
        } else {
            Some(format!("0x{n:x}"))
        }
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct HexToDec;
impl LawfulPrimitive for HexToDec {
    fn id(&self) -> &str {
        "lawful:bases:hex-to-dec"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "hex to dec")?;
        let s = rest.trim();
        let (sign, body): (i64, &str) = if let Some(b) = s.strip_prefix('-') {
            (-1, b)
        } else {
            (1, s)
        };
        let body = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")).unwrap_or(body);
        let n = i64::from_str_radix(body, 16).ok()?;
        Some((sign * n).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct DecToBin;
impl LawfulPrimitive for DecToBin {
    fn id(&self) -> &str {
        "lawful:bases:dec-to-bin"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "dec to bin")?;
        let n: i64 = rest.parse().ok()?;
        if n < 0 {
            Some(format!("-{:b}", -n))
        } else {
            Some(format!("{n:b}"))
        }
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct BinToDec;
impl LawfulPrimitive for BinToDec {
    fn id(&self) -> &str {
        "lawful:bases:bin-to-dec"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "bin to dec")?;
        let s = rest.trim();
        let (sign, body): (i64, &str) = if let Some(b) = s.strip_prefix('-') {
            (-1, b)
        } else {
            (1, s)
        };
        let body = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")).unwrap_or(body);
        let n = i64::from_str_radix(body, 2).ok()?;
        Some((sign * n).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct DecToOct;
impl LawfulPrimitive for DecToOct {
    fn id(&self) -> &str {
        "lawful:bases:dec-to-oct"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "dec to oct")?;
        let n: i64 = rest.parse().ok()?;
        if n < 0 {
            Some(format!("-{:o}", -n))
        } else {
            Some(format!("{n:o}"))
        }
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct OctToDec;
impl LawfulPrimitive for OctToDec {
    fn id(&self) -> &str {
        "lawful:bases:oct-to-dec"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "oct to dec")?;
        let s = rest.trim();
        let (sign, body): (i64, &str) = if let Some(b) = s.strip_prefix('-') {
            (-1, b)
        } else {
            (1, s)
        };
        let body = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")).unwrap_or(body);
        let n = i64::from_str_radix(body, 8).ok()?;
        Some((sign * n).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dec_hex() {
        assert_eq!(DecToHex.try_resolve("dec to hex 255").as_deref(), Some("0xff"));
        assert_eq!(HexToDec.try_resolve("hex to dec 0xff").as_deref(), Some("255"));
        assert_eq!(HexToDec.try_resolve("hex to dec ff").as_deref(), Some("255"));
        assert_eq!(HexToDec.try_resolve("hex to dec FF").as_deref(), Some("255"));
    }

    #[test]
    fn dec_bin() {
        assert_eq!(DecToBin.try_resolve("dec to bin 10").as_deref(), Some("1010"));
        assert_eq!(BinToDec.try_resolve("bin to dec 1010").as_deref(), Some("10"));
        assert_eq!(BinToDec.try_resolve("bin to dec 0b1010").as_deref(), Some("10"));
    }

    #[test]
    fn dec_oct() {
        assert_eq!(DecToOct.try_resolve("dec to oct 8").as_deref(), Some("10"));
        assert_eq!(OctToDec.try_resolve("oct to dec 10").as_deref(), Some("8"));
    }

    #[test]
    fn malformed_returns_none() {
        assert!(DecToHex.try_resolve("dec to hex notanumber").is_none());
        assert!(BinToDec.try_resolve("bin to dec 12").is_none()); // 2 not binary
        assert!(HexToDec.try_resolve("nothing here").is_none());
    }

    #[test]
    fn category_count() {
        assert_eq!(primitives().len(), 6);
    }
}
