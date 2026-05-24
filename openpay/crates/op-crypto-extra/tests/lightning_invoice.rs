//! Integration: BOLT-11 + LNURL structural parsing.

use op_crypto_extra::lightning::{Bolt11Network, lnurl_decode, parse_bolt11};

#[test]
fn empty_or_garbage_does_not_panic() {
    for s in ["", "x", "1", "abc1xy", "hello world"] {
        let _ = parse_bolt11(s); // must not panic; returns Err
    }
}

#[test]
fn networks_have_distinct_prefixes() {
    let prefixes: Vec<&str> = [
        Bolt11Network::Mainnet,
        Bolt11Network::Testnet,
        Bolt11Network::Signet,
        Bolt11Network::Regtest,
    ]
    .iter()
    .map(|n| n.hrp_prefix())
    .collect();
    // All distinct.
    for i in 0..prefixes.len() {
        for j in (i + 1)..prefixes.len() {
            assert_ne!(prefixes[i], prefixes[j]);
        }
    }
}

#[test]
fn lnurl_garbage_rejected() {
    // Various malformed lnurl strings should all error (not panic).
    for s in ["", "lnurl", "lnurl1", "lnurl1xxx", "lnurl1!!!"] {
        let res = lnurl_decode(s);
        assert!(res.is_err(), "{s} should not decode");
    }
}
