//! Property-based tests.
//!
//! Two invariants we check on arbitrary inputs:
//!
//! 1. **No panics.** The parser must reject malformed input cleanly,
//!    never crash. This is the most important property — a panic on
//!    untrusted bytes is a denial-of-service vulnerability in the
//!    secure-element interface.
//!
//! 2. **Round-trip stability.** For any well-formed TLV tree, encoding
//!    it and re-parsing must produce the same tree. (Encoder is part
//!    of a later phase; for now we test the parser against
//!    hand-constructed valid inputs.)

use op_emv::stream::TlvIter;
use op_emv::tree::Tlv;
use proptest::prelude::*;

proptest! {
    /// The parser must never panic on arbitrary bytes — only return Err.
    #[test]
    fn parser_never_panics_on_random_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
        // Drain the iterator. Whether or not it errors is fine; what
        // matters is that we don't panic.
        for _ in TlvIter::new(&bytes) {}
    }

    /// The tree builder must also be panic-safe.
    #[test]
    fn tree_builder_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
        let _ = Tlv::parse_all(&bytes);
    }

    /// Streaming and tree views must agree on top-level TLV count
    /// whenever both succeed.
    #[test]
    fn stream_and_tree_agree_on_count(bytes in proptest::collection::vec(any::<u8>(), 0..1024)) {
        let stream_result: Result<Vec<_>, _> = TlvIter::new(&bytes).collect();
        let tree_result = Tlv::parse_all(&bytes);
        match (stream_result, tree_result) {
            (Ok(s), Ok(t)) => prop_assert_eq!(s.len(), t.len()),
            (Err(_), Err(_)) => {} // both failed: fine
            // It's possible for the tree builder to fail while the
            // streaming top-level iterator succeeds, because the tree
            // recurses into constructed TLVs. The reverse should NOT
            // happen: if streaming says no, tree must also say no.
            (Err(_), Ok(_)) => prop_assert!(false, "tree succeeded where stream failed"),
            (Ok(_), Err(_)) => {} // tree-recursion found an inner error
        }
    }
}
