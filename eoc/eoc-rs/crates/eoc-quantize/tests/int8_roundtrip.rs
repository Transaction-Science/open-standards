//! Int8 round-trip tolerance test.

use eoc_quantize::int8::{Int8Asymmetric, Int8Symmetric};
use eoc_quantize::scheme::QuantizationScheme;

#[test]
fn symmetric_roundtrip_within_scale() {
    let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let q = Int8Symmetric::new();
    let enc = q.quantize(&x);
    let xr = q.dequantize(&enc);
    assert_eq!(xr.len(), x.len());
    let mut max_err = 0.0_f32;
    for (a, b) in x.iter().zip(xr.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    // Symmetric int8 rounding error must be at most ~half a scale unit.
    assert!(
        max_err <= enc.scale + 1e-5,
        "max err {max_err} exceeds scale {}",
        enc.scale
    );
}

#[test]
fn asymmetric_roundtrip_within_scale() {
    let x: Vec<f32> = (0..256).map(|i| 3.0 + i as f32 * 0.01).collect();
    let q = Int8Asymmetric::new();
    let enc = q.quantize(&x);
    let xr = q.dequantize(&enc);
    let mut max_err = 0.0_f32;
    for (a, b) in x.iter().zip(xr.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err <= enc.scale + 1e-5,
        "asym max err {max_err} exceeds scale {}",
        enc.scale
    );
}
