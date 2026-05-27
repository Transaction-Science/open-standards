//! Bitset over token ids.

use serde::{Deserialize, Serialize};

/// A bitset over token ids. Compact, cache-friendly, copyable in chunks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenMask {
    /// One bit per token id, packed LSB-first into u64 words.
    bits: Vec<u64>,
    /// Logical length in token ids.
    len: usize,
}

impl TokenMask {
    /// Allocate a mask with all bits cleared.
    pub fn new(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            len,
        }
    }

    /// Allocate a mask with every token allowed.
    pub fn all_allowed(len: usize) -> Self {
        let mut m = Self::new(len);
        for tok in 0..len {
            m.allow(tok as u32);
        }
        m
    }

    /// Number of token ids the mask covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the mask covers zero token ids.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Mark a token id as allowed.
    pub fn allow(&mut self, tok: u32) {
        let tok = tok as usize;
        if tok >= self.len {
            return;
        }
        let w = tok / 64;
        let b = tok % 64;
        self.bits[w] |= 1u64 << b;
    }

    /// Mark a token id as forbidden.
    pub fn forbid(&mut self, tok: u32) {
        let tok = tok as usize;
        if tok >= self.len {
            return;
        }
        let w = tok / 64;
        let b = tok % 64;
        self.bits[w] &= !(1u64 << b);
    }

    /// True when the token id is allowed in the current state.
    pub fn allowed(&self, tok: u32) -> bool {
        let tok = tok as usize;
        if tok >= self.len {
            return false;
        }
        let w = tok / 64;
        let b = tok % 64;
        (self.bits[w] >> b) & 1 == 1
    }

    /// Count of allowed token ids.
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// True when at least one token id is allowed.
    pub fn any(&self) -> bool {
        self.bits.iter().any(|w| *w != 0)
    }

    /// Iterate over allowed token ids in ascending order.
    pub fn iter_allowed(&self) -> AllowedIter<'_> {
        AllowedIter {
            mask: self,
            next: 0,
        }
    }
}

/// Iterator over allowed token ids.
#[derive(Debug)]
pub struct AllowedIter<'a> {
    mask: &'a TokenMask,
    next: usize,
}

impl<'a> Iterator for AllowedIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        while self.next < self.mask.len {
            let cur = self.next;
            self.next += 1;
            if self.mask.allowed(cur as u32) {
                return Some(cur as u32);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_allows_nothing() {
        let m = TokenMask::new(10);
        assert_eq!(m.popcount(), 0);
        assert!(!m.allowed(0));
        assert!(!m.allowed(9));
    }

    #[test]
    fn allow_then_check() {
        let mut m = TokenMask::new(200);
        m.allow(5);
        m.allow(64);
        m.allow(199);
        assert!(m.allowed(5));
        assert!(m.allowed(64));
        assert!(m.allowed(199));
        assert!(!m.allowed(4));
        assert!(!m.allowed(63));
        assert_eq!(m.popcount(), 3);
    }

    #[test]
    fn all_allowed_works() {
        let m = TokenMask::all_allowed(130);
        assert_eq!(m.popcount(), 130);
        for i in 0..130 {
            assert!(m.allowed(i));
        }
        assert!(!m.allowed(130));
    }

    #[test]
    fn iter_allowed_in_order() {
        let mut m = TokenMask::new(20);
        m.allow(3);
        m.allow(7);
        m.allow(19);
        let v: Vec<u32> = m.iter_allowed().collect();
        assert_eq!(v, vec![3, 7, 19]);
    }

    #[test]
    fn out_of_range_is_silent_or_false() {
        let mut m = TokenMask::new(8);
        m.allow(100); // ignored
        m.forbid(100); // ignored
        assert!(!m.allowed(100));
    }
}
