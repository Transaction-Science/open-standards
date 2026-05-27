//! Arithmetic primitives. Pure integer math; division rounds toward
//! zero; everything saturates inside i64 — overflowing inputs return
//! `None`.

use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

/// Build the arithmetic primitive set.
pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(GcdPrim),
        Arc::new(LcmPrim),
        Arc::new(FactorialPrim),
        Arc::new(FibonacciPrim),
        Arc::new(IsPrimePrim),
        Arc::new(FactorPrim),
        Arc::new(EvalPrim),
        Arc::new(AbsPrim),
        Arc::new(SignPrim),
        Arc::new(SquarePrim),
        Arc::new(CubePrim),
    ]
}

/// Strip a leading keyword, case-insensitive on the keyword only.
/// Returns `Some(remainder_trimmed)` if the query (after trim) starts
/// with the keyword followed by whitespace.
fn strip_keyword<'a>(query: &'a str, keyword: &str) -> Option<&'a str> {
    let q = query.trim();
    if q.len() < keyword.len() {
        return None;
    }
    let (head, tail) = q.split_at(keyword.len());
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    // Must be followed by whitespace (or be the whole string).
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest.trim())
}

fn parse_i64(s: &str) -> Option<i64> {
    s.trim().parse::<i64>().ok()
}

// ---- gcd ----------------------------------------------------------------

pub struct GcdPrim;
impl LawfulPrimitive for GcdPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:gcd"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "gcd")?;
        let mut parts = rest.split_whitespace();
        let a = parse_i64(parts.next()?)?;
        let b = parse_i64(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }
        Some(gcd_i64(a, b).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        80
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut x, mut y) = (a.unsigned_abs(), b.unsigned_abs());
    while y != 0 {
        let t = y;
        y = x % y;
        x = t;
    }
    x as i64
}

// ---- lcm ----------------------------------------------------------------

pub struct LcmPrim;
impl LawfulPrimitive for LcmPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:lcm"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "lcm")?;
        let mut parts = rest.split_whitespace();
        let a = parse_i64(parts.next()?)?;
        let b = parse_i64(parts.next()?)?;
        if parts.next().is_some() {
            return None;
        }
        if a == 0 || b == 0 {
            return Some("0".to_string());
        }
        let g = gcd_i64(a, b);
        let prod = (a / g).checked_mul(b)?;
        Some(prod.abs().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        90
    }
}

// ---- factorial ----------------------------------------------------------

pub struct FactorialPrim;
impl LawfulPrimitive for FactorialPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:factorial"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let q = query.trim();
        // "<n>!" form
        if let Some(stripped) = q.strip_suffix('!') {
            let n = parse_i64(stripped.trim())?;
            return factorial(n);
        }
        let rest = strip_keyword(query, "factorial")?;
        let n = parse_i64(rest)?;
        factorial(n)
    }
    fn declared_cost_uj(&self) -> u64 {
        120
    }
}

fn factorial(n: i64) -> Option<String> {
    if !(0..=20).contains(&n) {
        return None;
    }
    let mut acc: i64 = 1;
    for k in 1..=n {
        acc = acc.checked_mul(k)?;
    }
    Some(acc.to_string())
}

// ---- fibonacci ----------------------------------------------------------

pub struct FibonacciPrim;
impl LawfulPrimitive for FibonacciPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:fibonacci"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "fibonacci")
            .or_else(|| strip_keyword(query, "fib"))?;
        let n = parse_i64(rest)?;
        if !(0..=92).contains(&n) {
            return None;
        }
        let (mut a, mut b): (i64, i64) = (0, 1);
        for _ in 0..n {
            let next = a.checked_add(b)?;
            a = b;
            b = next;
        }
        Some(a.to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        110
    }
}

// ---- is_prime -----------------------------------------------------------

pub struct IsPrimePrim;
impl LawfulPrimitive for IsPrimePrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:is-prime"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "is_prime")
            .or_else(|| strip_keyword(query, "prime?"))?;
        let n = parse_i64(rest)?;
        Some(if is_prime(n) { "true".into() } else { "false".into() })
    }
    fn declared_cost_uj(&self) -> u64 {
        180
    }
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i: i64 = 3;
    while let Some(sq) = i.checked_mul(i) {
        if sq > n {
            return true;
        }
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

// ---- factor -------------------------------------------------------------

pub struct FactorPrim;
impl LawfulPrimitive for FactorPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:factor"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "factor")?;
        let mut n = parse_i64(rest)?;
        if n < 2 {
            return None;
        }
        let mut out = Vec::new();
        let mut p: i64 = 2;
        while p.checked_mul(p).map(|sq| sq <= n).unwrap_or(false) {
            while n % p == 0 {
                out.push(p.to_string());
                n /= p;
            }
            p += if p == 2 { 1 } else { 2 };
        }
        if n > 1 {
            out.push(n.to_string());
        }
        Some(out.join(" "))
    }
    fn declared_cost_uj(&self) -> u64 {
        200
    }
}

// ---- eval ---------------------------------------------------------------
//
// Tiny recursive-descent parser for integer arithmetic with +, -, *, /,
// unary minus, parentheses. Division rounds toward zero. Returns None
// for malformed expressions or overflow. Recognises ANY non-empty query
// after trim, so register it AFTER the more specific keyword primitives.

pub struct EvalPrim;
impl LawfulPrimitive for EvalPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:eval"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let q = query.trim();
        if q.is_empty() {
            return None;
        }
        // Only operate on strings that contain only digits, whitespace,
        // and the operators we know — defends against accidental matches
        // on natural-language queries.
        if !q.chars().all(|c| {
            c.is_ascii_digit() || c.is_whitespace() || matches!(c, '+' | '-' | '*' | '/' | '(' | ')')
        }) {
            return None;
        }
        // Require at least one operator so a bare number passes through to
        // the next primitive (or fails — eval would otherwise eat them).
        if !q.chars().any(|c| matches!(c, '+' | '-' | '*' | '/')) {
            return None;
        }
        let mut p = Parser::new(q);
        let v = p.expr()?;
        p.skip_ws();
        if !p.eof() {
            return None;
        }
        Some(v.to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        150
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }
    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Option<i64> {
        let mut lhs = self.term()?;
        loop {
            self.skip_ws();
            let op = match self.peek() {
                Some(b'+') => b'+',
                Some(b'-') => b'-',
                _ => break,
            };
            self.pos += 1;
            let rhs = self.term()?;
            lhs = if op == b'+' {
                lhs.checked_add(rhs)?
            } else {
                lhs.checked_sub(rhs)?
            };
        }
        Some(lhs)
    }
    fn term(&mut self) -> Option<i64> {
        let mut lhs = self.factor()?;
        loop {
            self.skip_ws();
            let op = match self.peek() {
                Some(b'*') => b'*',
                Some(b'/') => b'/',
                _ => break,
            };
            self.pos += 1;
            let rhs = self.factor()?;
            lhs = if op == b'*' {
                lhs.checked_mul(rhs)?
            } else {
                if rhs == 0 {
                    return None;
                }
                lhs.checked_div(rhs)?
            };
        }
        Some(lhs)
    }
    fn factor(&mut self) -> Option<i64> {
        self.skip_ws();
        if self.eat(b'-') {
            let v = self.factor()?;
            return v.checked_neg();
        }
        if self.eat(b'+') {
            return self.factor();
        }
        if self.eat(b'(') {
            let v = self.expr()?;
            if !self.eat(b')') {
                return None;
            }
            return Some(v);
        }
        // Number.
        self.skip_ws();
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return None;
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).ok()?;
        s.parse::<i64>().ok()
    }
}

// ---- abs ----------------------------------------------------------------

pub struct AbsPrim;
impl LawfulPrimitive for AbsPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:abs"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "abs")?;
        let n = parse_i64(rest)?;
        n.checked_abs().map(|v| v.to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        50
    }
}

// ---- sign ---------------------------------------------------------------

pub struct SignPrim;
impl LawfulPrimitive for SignPrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:sign"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "sign")?;
        let n = parse_i64(rest)?;
        Some(n.signum().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        50
    }
}

// ---- square / cube ------------------------------------------------------

pub struct SquarePrim;
impl LawfulPrimitive for SquarePrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:square"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "square")?;
        let n = parse_i64(rest)?;
        n.checked_mul(n).map(|v| v.to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

pub struct CubePrim;
impl LawfulPrimitive for CubePrim {
    fn id(&self) -> &str {
        "lawful:arithmetic:cube"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_keyword(query, "cube")?;
        let n = parse_i64(rest)?;
        n.checked_mul(n).and_then(|s| s.checked_mul(n)).map(|v| v.to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_works() {
        assert_eq!(GcdPrim.try_resolve("gcd 12 8").as_deref(), Some("4"));
        assert_eq!(GcdPrim.try_resolve("  GCD  100  75 ").as_deref(), Some("25"));
        assert_eq!(GcdPrim.try_resolve("gcd 0 0").as_deref(), Some("0"));
        assert!(GcdPrim.try_resolve("gcd 12").is_none());
        assert!(GcdPrim.try_resolve("gcd foo bar").is_none());
        assert!(GcdPrim.try_resolve("not gcd").is_none());
    }

    #[test]
    fn lcm_works() {
        assert_eq!(LcmPrim.try_resolve("lcm 4 6").as_deref(), Some("12"));
        assert_eq!(LcmPrim.try_resolve("lcm 0 5").as_deref(), Some("0"));
        assert!(LcmPrim.try_resolve("lcm 4").is_none());
    }

    #[test]
    fn factorial_works() {
        assert_eq!(FactorialPrim.try_resolve("factorial 5").as_deref(), Some("120"));
        assert_eq!(FactorialPrim.try_resolve("5!").as_deref(), Some("120"));
        assert_eq!(FactorialPrim.try_resolve("0!").as_deref(), Some("1"));
        assert!(FactorialPrim.try_resolve("factorial -1").is_none());
        assert!(FactorialPrim.try_resolve("factorial 21").is_none()); // overflow guard
    }

    #[test]
    fn fibonacci_works() {
        assert_eq!(FibonacciPrim.try_resolve("fib 0").as_deref(), Some("0"));
        assert_eq!(FibonacciPrim.try_resolve("fib 1").as_deref(), Some("1"));
        assert_eq!(FibonacciPrim.try_resolve("fib 10").as_deref(), Some("55"));
        assert_eq!(FibonacciPrim.try_resolve("fibonacci 10").as_deref(), Some("55"));
        assert!(FibonacciPrim.try_resolve("fib -1").is_none());
    }

    #[test]
    fn is_prime_works() {
        assert_eq!(IsPrimePrim.try_resolve("is_prime 7").as_deref(), Some("true"));
        assert_eq!(IsPrimePrim.try_resolve("is_prime 9").as_deref(), Some("false"));
        assert_eq!(IsPrimePrim.try_resolve("prime? 2").as_deref(), Some("true"));
        assert_eq!(IsPrimePrim.try_resolve("is_prime 1").as_deref(), Some("false"));
        assert!(IsPrimePrim.try_resolve("is_prime").is_none());
    }

    #[test]
    fn factor_works() {
        assert_eq!(FactorPrim.try_resolve("factor 12").as_deref(), Some("2 2 3"));
        assert_eq!(FactorPrim.try_resolve("factor 7").as_deref(), Some("7"));
        assert!(FactorPrim.try_resolve("factor 1").is_none());
    }

    #[test]
    fn eval_works() {
        assert_eq!(EvalPrim.try_resolve("5 + 3 * 2").as_deref(), Some("11"));
        assert_eq!(EvalPrim.try_resolve("(8-2)/3").as_deref(), Some("2"));
        assert_eq!(EvalPrim.try_resolve("-5 + 3").as_deref(), Some("-2"));
        assert_eq!(EvalPrim.try_resolve("10/3").as_deref(), Some("3")); // toward zero
        assert!(EvalPrim.try_resolve("hello").is_none());
        assert!(EvalPrim.try_resolve("42").is_none()); // no operator
        assert!(EvalPrim.try_resolve("5/0").is_none());
    }

    #[test]
    fn abs_and_sign() {
        assert_eq!(AbsPrim.try_resolve("abs -5").as_deref(), Some("5"));
        assert_eq!(AbsPrim.try_resolve("abs 5").as_deref(), Some("5"));
        assert_eq!(SignPrim.try_resolve("sign -5").as_deref(), Some("-1"));
        assert_eq!(SignPrim.try_resolve("sign 0").as_deref(), Some("0"));
        assert_eq!(SignPrim.try_resolve("sign 42").as_deref(), Some("1"));
    }

    #[test]
    fn square_and_cube() {
        assert_eq!(SquarePrim.try_resolve("square 4").as_deref(), Some("16"));
        assert_eq!(CubePrim.try_resolve("cube 3").as_deref(), Some("27"));
    }

    #[test]
    fn category_count() {
        assert!(primitives().len() >= 10);
    }
}
