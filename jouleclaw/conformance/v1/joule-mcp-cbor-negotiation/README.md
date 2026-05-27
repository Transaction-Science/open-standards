# Conformance vector — joule-mcp CBOR negotiation

Exercises the `x-jouleclaw/joule-mcp@1` capability handshake. Two
endpoints that both advertise the capability MUST elect the CBOR wire;
when either side omits it the wire MUST stay JSON-RPC.

## The matrix

| Side A advertises | Side B advertises | Negotiated wire |
|---|---|---|
| (none)                          | (none)                          | JSON-RPC |
| (none)                          | `x-jouleclaw/joule-mcp@1`       | JSON-RPC |
| `x-jouleclaw/joule-mcp@1`       | (none)                          | JSON-RPC |
| `x-jouleclaw/joule-mcp@1`       | `x-jouleclaw/joule-mcp@1`       | **CBOR** |

A conforming runtime MUST pass `cargo test -p jouleclaw-mcp` with the
`negotiate_*` tests green, which encode this matrix verbatim. The
canonical capability tag is `x-jouleclaw/joule-mcp@1` — implementations
MUST advertise that exact string. Tags with different versions or
namespaces do NOT trigger CBOR.

## Acceptance bounds

For each row of the matrix the function call

```rust
negotiate(&local_caps, &remote_caps)
```

MUST return the indicated `WireEncoding` value. There is no platform
variation here — this is a pure protocol-level test, not a measurement.

## Files

- `matrix.json` — the four-row matrix above in machine-readable form
- `README.md` — this file

This vector has no `receipt.json` — it tests the protocol handshake,
not a cascade walk.
