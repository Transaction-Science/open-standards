//! Materialise the bundled rate-table snapshot to `data/rate_table_v1.cbor`.
//!
//! Run from the repo root:
//!
//! ```text
//! cargo run -p op-tax --example gen_snapshot
//! ```
//!
//! The output is a CBOR file consumed by `RateTable::load_cbor`. Tests
//! that need the snapshot read it via the path in
//! `tests/integration.rs`; production deployments typically replace
//! this with a daily-refreshed file from a vendor feed.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = op_tax::RateTable::bundled();
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rate_table_v1.cbor");
    let f = std::fs::File::create(&out_path)?;
    t.write_cbor(f)?;
    println!(
        "Wrote {} entries to {}",
        t.entries.len(),
        out_path.display()
    );
    Ok(())
}
