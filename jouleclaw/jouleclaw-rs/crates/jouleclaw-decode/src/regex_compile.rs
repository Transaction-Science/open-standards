//! Regex parser + Thompson NFA construction over bytes.
//!
//! We accept a useful subset of the `regex` crate's syntax — enough for
//! the patterns that show up in real schemas (digits, identifiers,
//! delimited literals, alternations, ranges, optional/star/plus,
//! bounded repetitions, escape sequences, character classes). Anchors
//! are implicit: every grammar matches the *entire* generated string.
//!
//! ## Subset supported
//!
//! - Literals: ASCII and arbitrary bytes via `\xHH`.
//! - Escapes: `\d \D \s \S \w \W \n \r \t \\ \. \( \) \[ \] \{ \} \| \+ \* \? \^ \$ \/ \"`.
//! - Character classes: `[...]`, `[^...]`, including ranges `a-z` and named escapes.
//! - Alternation: `a|b`.
//! - Grouping: `(...)`, non-capturing `(?:...)`.
//! - Quantifiers: `*`, `+`, `?`, `{n}`, `{n,}`, `{n,m}`.
//! - Wildcard: `.` (matches any byte 0..=255 — we treat input as bytes).
//!
//! ## Not supported (yet)
//!
//! Lookaround, backreferences, named groups, Unicode classes beyond
//! ASCII shortcuts. These are not needed for v0.1 constrained decoding.

use crate::automaton::{ByteClass, Edge, Nfa, StateId};
use crate::error::DecodeError;

/// Parse `pattern` and build a byte-NFA matching the entire string.
pub fn compile(pattern: &str) -> Result<Nfa, DecodeError> {
    let ast = Parser::new(pattern).parse_top()?;
    let mut nfa = Nfa::new();
    let start = nfa.add_state();
    nfa.start = start;
    let (in_s, out_s) = build(&mut nfa, &ast);
    nfa.add_edge(start, Edge::Epsilon(in_s));
    nfa.accepts.insert(out_s);
    Ok(nfa)
}

/// Parsed regex AST.
#[derive(Clone, Debug)]
pub enum Node {
    /// Empty (matches the empty string).
    Empty,
    /// Single byte class.
    Class(ByteClass),
    /// Concatenation.
    Concat(Vec<Node>),
    /// Alternation.
    Alt(Vec<Node>),
    /// `*` — zero or more.
    Star(Box<Node>),
    /// `+` — one or more.
    Plus(Box<Node>),
    /// `?` — zero or one.
    Opt(Box<Node>),
    /// `{n,m}` — bounded repetition (m=None means unbounded).
    Repeat(Box<Node>, usize, Option<usize>),
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            src: s.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, msg: impl Into<String>) -> DecodeError {
        DecodeError::RegexParse {
            pos: self.pos,
            msg: msg.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn parse_top(&mut self) -> Result<Node, DecodeError> {
        let node = self.parse_alt()?;
        if self.pos != self.src.len() {
            return Err(self.err("trailing input after pattern"));
        }
        Ok(node)
    }

    fn parse_alt(&mut self) -> Result<Node, DecodeError> {
        let mut alts = vec![self.parse_concat()?];
        while self.peek() == Some(b'|') {
            self.bump();
            alts.push(self.parse_concat()?);
        }
        if alts.len() == 1 {
            Ok(alts.pop().unwrap_or(Node::Empty))
        } else {
            Ok(Node::Alt(alts))
        }
    }

    fn parse_concat(&mut self) -> Result<Node, DecodeError> {
        let mut parts = Vec::new();
        while let Some(b) = self.peek() {
            if b == b'|' || b == b')' {
                break;
            }
            parts.push(self.parse_quantified()?);
        }
        if parts.is_empty() {
            Ok(Node::Empty)
        } else if parts.len() == 1 {
            Ok(parts.pop().unwrap_or(Node::Empty))
        } else {
            Ok(Node::Concat(parts))
        }
    }

    fn parse_quantified(&mut self) -> Result<Node, DecodeError> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some(b'*') => {
                self.bump();
                Ok(Node::Star(Box::new(atom)))
            }
            Some(b'+') => {
                self.bump();
                Ok(Node::Plus(Box::new(atom)))
            }
            Some(b'?') => {
                self.bump();
                Ok(Node::Opt(Box::new(atom)))
            }
            Some(b'{') => {
                self.bump();
                let (n, m) = self.parse_repeat_bounds()?;
                Ok(Node::Repeat(Box::new(atom), n, m))
            }
            _ => Ok(atom),
        }
    }

    fn parse_repeat_bounds(&mut self) -> Result<(usize, Option<usize>), DecodeError> {
        // We've already consumed the `{`.
        let n = self.parse_uint()?;
        match self.peek() {
            Some(b'}') => {
                self.bump();
                Ok((n, Some(n)))
            }
            Some(b',') => {
                self.bump();
                if self.peek() == Some(b'}') {
                    self.bump();
                    Ok((n, None))
                } else {
                    let m = self.parse_uint()?;
                    if self.peek() != Some(b'}') {
                        return Err(self.err("expected `}` in repeat"));
                    }
                    self.bump();
                    if m < n {
                        return Err(self.err("repeat upper bound smaller than lower"));
                    }
                    Ok((n, Some(m)))
                }
            }
            _ => Err(self.err("malformed repeat bounds")),
        }
    }

    fn parse_uint(&mut self) -> Result<usize, DecodeError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.bump();
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.err("expected number"));
        }
        let s = core::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.err("non-ascii in number"))?;
        s.parse::<usize>()
            .map_err(|_| self.err("number out of range"))
    }

    fn parse_atom(&mut self) -> Result<Node, DecodeError> {
        let b = self.peek().ok_or_else(|| self.err("unexpected end"))?;
        match b {
            b'(' => {
                self.bump();
                // Optional non-capturing `?:`.
                if self.peek() == Some(b'?')
                    && self.src.get(self.pos + 1) == Some(&b':')
                {
                    self.pos += 2;
                }
                let inner = self.parse_alt()?;
                if self.peek() != Some(b')') {
                    return Err(self.err("unmatched `(`"));
                }
                self.bump();
                Ok(inner)
            }
            b'[' => {
                self.bump();
                Ok(Node::Class(self.parse_char_class()?))
            }
            b'.' => {
                self.bump();
                Ok(Node::Class(ByteClass::full()))
            }
            b'\\' => {
                self.bump();
                Ok(Node::Class(self.parse_escape()?))
            }
            b')' | b'|' | b'*' | b'+' | b'?' | b'{' | b'}' | b']' => {
                Err(self.err(format!("unexpected `{}`", b as char)))
            }
            _ => {
                self.bump();
                Ok(Node::Class(ByteClass::single(b)))
            }
        }
    }

    fn parse_char_class(&mut self) -> Result<ByteClass, DecodeError> {
        let mut negate = false;
        if self.peek() == Some(b'^') {
            negate = true;
            self.bump();
        }
        let mut class = ByteClass::empty();
        let mut first = true;
        loop {
            let b = self
                .peek()
                .ok_or_else(|| self.err("unterminated character class"))?;
            if b == b']' && !first {
                self.bump();
                break;
            }
            first = false;
            let lo = if b == b'\\' {
                self.bump();
                // Escapes inside class can be character classes themselves
                // (e.g. \d). We union those in directly.
                let esc = self.parse_escape()?;
                // If the class has a single byte, keep parsing range syntax;
                // if it has multiple bytes, treat it as an atomic class.
                let single = single_byte(&esc);
                if let Some(b) = single {
                    b
                } else {
                    class.union_with(&esc);
                    continue;
                }
            } else {
                self.bump();
                b
            };
            if self.peek() == Some(b'-')
                && self.src.get(self.pos + 1) != Some(&b']')
                && self.src.get(self.pos + 1).is_some()
            {
                self.bump(); // consume `-`
                let hi_b = self
                    .peek()
                    .ok_or_else(|| self.err("unterminated range"))?;
                let hi = if hi_b == b'\\' {
                    self.bump();
                    let esc = self.parse_escape()?;
                    single_byte(&esc)
                        .ok_or_else(|| self.err("escape in range upper bound is not single byte"))?
                } else {
                    self.bump();
                    hi_b
                };
                if hi < lo {
                    return Err(self.err("inverted range in character class"));
                }
                class.union_with(&ByteClass::range(lo, hi));
            } else {
                class.add(lo);
            }
        }
        Ok(if negate { class.complement() } else { class })
    }

    fn parse_escape(&mut self) -> Result<ByteClass, DecodeError> {
        let b = self
            .bump()
            .ok_or_else(|| self.err("dangling backslash"))?;
        Ok(match b {
            b'd' => ByteClass::range(b'0', b'9'),
            b'D' => ByteClass::range(b'0', b'9').complement(),
            b'w' => {
                let mut c = ByteClass::range(b'a', b'z');
                c.union_with(&ByteClass::range(b'A', b'Z'));
                c.union_with(&ByteClass::range(b'0', b'9'));
                c.add(b'_');
                c
            }
            b'W' => {
                let mut c = ByteClass::range(b'a', b'z');
                c.union_with(&ByteClass::range(b'A', b'Z'));
                c.union_with(&ByteClass::range(b'0', b'9'));
                c.add(b'_');
                c.complement()
            }
            b's' => {
                let mut c = ByteClass::empty();
                for byte in [b' ', b'\t', b'\n', b'\r', 0x0Cu8, 0x0Bu8] {
                    c.add(byte);
                }
                c
            }
            b'S' => {
                let mut c = ByteClass::empty();
                for byte in [b' ', b'\t', b'\n', b'\r', 0x0Cu8, 0x0Bu8] {
                    c.add(byte);
                }
                c.complement()
            }
            b'n' => ByteClass::single(b'\n'),
            b'r' => ByteClass::single(b'\r'),
            b't' => ByteClass::single(b'\t'),
            b'0' => ByteClass::single(0),
            b'x' => {
                let hi = self
                    .bump()
                    .ok_or_else(|| self.err("\\x needs two hex digits"))?;
                let lo = self
                    .bump()
                    .ok_or_else(|| self.err("\\x needs two hex digits"))?;
                let v = ((hex_val(hi)? as u16) << 4) | hex_val(lo)? as u16;
                ByteClass::single(v as u8)
            }
            // Punctuation escapes — match the byte literally.
            b'\\' | b'/' | b'"' | b'\'' | b'.' | b'(' | b')' | b'[' | b']'
            | b'{' | b'}' | b'|' | b'+' | b'*' | b'?' | b'^' | b'$' | b'-' => {
                ByteClass::single(b)
            }
            other => return Err(self.err(format!("unknown escape `\\{}`", other as char))),
        })
    }
}

fn hex_val(b: u8) -> Result<u8, DecodeError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(DecodeError::RegexParse {
            pos: 0,
            msg: format!("not a hex digit: `{}`", b as char),
        }),
    }
}

fn single_byte(c: &ByteClass) -> Option<u8> {
    let mut found = None;
    for b in 0u8..=255 {
        if c.contains(b) {
            if found.is_some() {
                return None;
            }
            found = Some(b);
        }
    }
    found
}

/// Thompson construction.
fn build(nfa: &mut Nfa, node: &Node) -> (StateId, StateId) {
    match node {
        Node::Empty => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            nfa.add_edge(a, Edge::Epsilon(b));
            (a, b)
        }
        Node::Class(c) => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            nfa.add_edge(a, Edge::Class(c.clone(), b));
            (a, b)
        }
        Node::Concat(parts) => {
            if parts.is_empty() {
                let a = nfa.add_state();
                let b = nfa.add_state();
                nfa.add_edge(a, Edge::Epsilon(b));
                return (a, b);
            }
            let (first_in, mut last_out) = build(nfa, &parts[0]);
            for part in &parts[1..] {
                let (pin, pout) = build(nfa, part);
                nfa.add_edge(last_out, Edge::Epsilon(pin));
                last_out = pout;
            }
            (first_in, last_out)
        }
        Node::Alt(alts) => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            for alt in alts {
                let (ain, aout) = build(nfa, alt);
                nfa.add_edge(a, Edge::Epsilon(ain));
                nfa.add_edge(aout, Edge::Epsilon(b));
            }
            (a, b)
        }
        Node::Star(inner) => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            let (iin, iout) = build(nfa, inner);
            nfa.add_edge(a, Edge::Epsilon(iin));
            nfa.add_edge(a, Edge::Epsilon(b));
            nfa.add_edge(iout, Edge::Epsilon(iin));
            nfa.add_edge(iout, Edge::Epsilon(b));
            (a, b)
        }
        Node::Plus(inner) => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            let (iin, iout) = build(nfa, inner);
            nfa.add_edge(a, Edge::Epsilon(iin));
            nfa.add_edge(iout, Edge::Epsilon(iin));
            nfa.add_edge(iout, Edge::Epsilon(b));
            (a, b)
        }
        Node::Opt(inner) => {
            let a = nfa.add_state();
            let b = nfa.add_state();
            let (iin, iout) = build(nfa, inner);
            nfa.add_edge(a, Edge::Epsilon(iin));
            nfa.add_edge(a, Edge::Epsilon(b));
            nfa.add_edge(iout, Edge::Epsilon(b));
            (a, b)
        }
        Node::Repeat(inner, n, m) => {
            // Lower {n,m} into a concat of n copies + (m-n) optional copies,
            // or n copies + a Star copy when m is None.
            let mut parts: Vec<Node> = Vec::new();
            for _ in 0..*n {
                parts.push((**inner).clone());
            }
            match m {
                Some(mm) => {
                    let extra = mm.saturating_sub(*n);
                    for _ in 0..extra {
                        parts.push(Node::Opt(inner.clone()));
                    }
                }
                None => {
                    parts.push(Node::Star(inner.clone()));
                }
            }
            build(nfa, &Node::Concat(parts))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn matches(pattern: &str, input: &str) -> bool {
        let nfa = compile(pattern).expect("compile");
        let start = nfa.start_set();
        match nfa.run_bytes(&start, input.as_bytes()) {
            Some(end) => nfa.any_accept(&end),
            None => false,
        }
    }

    #[test]
    fn literal_match() {
        assert!(matches("abc", "abc"));
        assert!(!matches("abc", "ab"));
        assert!(!matches("abc", "abcd"));
    }

    #[test]
    fn star_match() {
        assert!(matches("a*", ""));
        assert!(matches("a*", "a"));
        assert!(matches("a*", "aaaa"));
        assert!(!matches("a*", "ab"));
    }

    #[test]
    fn plus_and_opt() {
        assert!(matches("a+", "a"));
        assert!(matches("a+", "aaa"));
        assert!(!matches("a+", ""));
        assert!(matches("ab?", "a"));
        assert!(matches("ab?", "ab"));
        assert!(!matches("ab?", "abb"));
    }

    #[test]
    fn alternation_and_grouping() {
        assert!(matches("a|b", "a"));
        assert!(matches("a|b", "b"));
        assert!(!matches("a|b", "c"));
        assert!(matches("(ab|cd)+", "abcdab"));
    }

    #[test]
    fn classes() {
        assert!(matches("[abc]", "a"));
        assert!(matches("[abc]", "b"));
        assert!(!matches("[abc]", "d"));
        assert!(matches("[^abc]", "d"));
        assert!(!matches("[^abc]", "a"));
        assert!(matches("[0-9]+", "12345"));
    }

    #[test]
    fn escapes() {
        assert!(matches(r"\d+", "42"));
        assert!(!matches(r"\d+", "4a"));
        assert!(matches(r"\w+", "Foo_bar9"));
        assert!(matches(r"\.", "."));
        assert!(!matches(r"\.", "x"));
    }

    #[test]
    fn repeat_bounds() {
        assert!(matches("a{3}", "aaa"));
        assert!(!matches("a{3}", "aa"));
        assert!(!matches("a{3}", "aaaa"));
        assert!(matches("a{2,4}", "aa"));
        assert!(matches("a{2,4}", "aaaa"));
        assert!(!matches("a{2,4}", "a"));
        assert!(!matches("a{2,4}", "aaaaa"));
        assert!(matches("a{2,}", "aaaaaa"));
    }

    #[test]
    fn dot_matches_any() {
        assert!(matches(".+", "anything"));
        assert!(matches("a.c", "abc"));
        assert!(matches("a.c", "azc"));
    }

    #[test]
    fn nfa_step_partial_then_accept() {
        let nfa = compile(r"[0-9]+").unwrap();
        let mut cur = nfa.start_set();
        cur = nfa.step(&cur, b'1');
        assert!(!cur.is_empty());
        assert!(nfa.any_accept(&cur));
        cur = nfa.step(&cur, b'2');
        assert!(nfa.any_accept(&cur));
    }

    #[test]
    fn dead_set_means_unmatched() {
        let nfa = compile(r"hello").unwrap();
        let mut cur = nfa.start_set();
        cur = nfa.step(&cur, b'h');
        cur = nfa.step(&cur, b'x');
        assert!(cur.is_empty());
        let _: BTreeSet<_> = cur;
    }

    #[test]
    fn hex_escape() {
        // \x41 is 'A'.
        assert!(matches(r"\x41+", "AAA"));
        assert!(!matches(r"\x41+", "B"));
    }

    #[test]
    fn non_capturing_group() {
        assert!(matches("(?:ab)+", "abab"));
        assert!(!matches("(?:ab)+", "abc"));
    }
}
