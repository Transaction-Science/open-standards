//! ILP address parser and validator.
//!
//! ILP addresses are dotted strings rooted at a top-level allocator
//! scheme:
//!
//! * `g.` — global / production ledgers (`g.us.bank.alice`).
//! * `private.` — locally administered private networks.
//! * `example.` — documentation / examples.
//! * `peer.` — link-local routing between two connectors.
//! * `self.` — loopback / connector-internal.
//! * `test.`, `test1.`, `test2.`, `test3.` — testnets.
//! * `local.` — host-local services (extension used by some operators).
//!
//! Each segment must be 1..=128 characters of `[A-Za-z0-9_~-]`. The
//! total address length is bounded at 1023 octets per ILP-RFC-15.

use crate::error::{IlpError, Result};

/// The top-level allocator scheme parsed off the head of an address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressScheme {
    /// `g.` — global production allocator.
    Global,
    /// `private.` — operator-private allocator.
    Private,
    /// `example.` — documentation only.
    Example,
    /// `peer.` — link-local between two BTP peers.
    Peer,
    /// `self.` — connector-local loopback.
    SelfScheme,
    /// `test.` family — any of `test`, `test1`, `test2`, `test3`.
    Test,
    /// `local.` — host-local services.
    Local,
}

impl AddressScheme {
    /// Return the canonical string form of the scheme (e.g. `"g"`).
    pub fn as_str(self) -> &'static str {
        match self {
            AddressScheme::Global => "g",
            AddressScheme::Private => "private",
            AddressScheme::Example => "example",
            AddressScheme::Peer => "peer",
            AddressScheme::SelfScheme => "self",
            AddressScheme::Test => "test",
            AddressScheme::Local => "local",
        }
    }
}

/// A validated ILP address, owned by the caller.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    raw: String,
    scheme: AddressScheme,
}

impl Address {
    /// Parse and validate an ILP address.
    pub fn parse(input: &str) -> Result<Self> {
        if input.is_empty() {
            return Err(IlpError::InvalidAddress("empty".into()));
        }
        if input.len() > 1023 {
            return Err(IlpError::InvalidAddress(format!(
                "length {} exceeds 1023",
                input.len()
            )));
        }
        let head = input.split('.').next().unwrap_or("");
        let scheme = match head {
            "g" => AddressScheme::Global,
            "private" => AddressScheme::Private,
            "example" => AddressScheme::Example,
            "peer" => AddressScheme::Peer,
            "self" => AddressScheme::SelfScheme,
            "test" | "test1" | "test2" | "test3" => AddressScheme::Test,
            "local" => AddressScheme::Local,
            other => {
                return Err(IlpError::InvalidAddress(format!(
                    "unknown scheme: {other}"
                )))
            }
        };
        for (i, seg) in input.split('.').enumerate() {
            if seg.is_empty() {
                return Err(IlpError::InvalidAddress(format!(
                    "empty segment at index {i}"
                )));
            }
            if seg.len() > 128 {
                return Err(IlpError::InvalidAddress(format!(
                    "segment {i} exceeds 128 chars"
                )));
            }
            for ch in seg.chars() {
                if !is_address_char(ch) {
                    return Err(IlpError::InvalidAddress(format!(
                        "illegal char {ch:?} in segment {i}"
                    )));
                }
            }
        }
        Ok(Self {
            raw: input.to_string(),
            scheme,
        })
    }

    /// Borrow the address as a `&str`.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Return the allocator scheme.
    pub fn scheme(&self) -> AddressScheme {
        self.scheme
    }

    /// Iterate over the dotted segments.
    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.raw.split('.')
    }

    /// Test whether this address starts with the supplied prefix
    /// (segment-aligned).
    ///
    /// `"g.us.bank"` is a prefix of `"g.us.bank.alice"` but *not* of
    /// `"g.us.bankhq"`. Used by the route table for longest-prefix
    /// matching.
    pub fn starts_with_prefix(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true;
        }
        if !self.raw.starts_with(prefix) {
            return false;
        }
        let next = self.raw.as_bytes().get(prefix.len()).copied();
        matches!(next, None | Some(b'.'))
    }
}

fn is_address_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '~' | '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_global() {
        let a = Address::parse("g.us.bank.alice").unwrap();
        assert_eq!(a.scheme(), AddressScheme::Global);
        let segs: Vec<_> = a.segments().collect();
        assert_eq!(segs, ["g", "us", "bank", "alice"]);
    }

    #[test]
    fn rejects_unknown_scheme() {
        let err = Address::parse("foo.bar").unwrap_err();
        assert!(matches!(err, IlpError::InvalidAddress(_)));
    }

    #[test]
    fn segment_prefix_match() {
        let a = Address::parse("g.us.bank.alice").unwrap();
        assert!(a.starts_with_prefix("g.us.bank"));
        assert!(!a.starts_with_prefix("g.us.bankhq"));
        assert!(a.starts_with_prefix(""));
    }
}
