//! Sanity check: the int8 joules-per-token estimate must be strictly
//! lower than the fp16 estimate (and well below fp32).

use eoc_quantize::energy_tradeoff::EnergyModel;
use eoc_quantize::scheme::Numeric;

#[test]
fn int8_cheaper_than_fp16() {
    let model = EnergyModel::default();
    let j_int8 = model.microjoules_per_token(Numeric::Int8);
    let j_fp16 = model.microjoules_per_token(Numeric::Fp16);
    assert!(
        j_int8 < j_fp16,
        "int8={j_int8} should be less than fp16={j_fp16}"
    );
}

#[test]
fn fp32_baseline_is_highest() {
    let model = EnergyModel::default();
    let j_fp32 = model.microjoules_per_token(Numeric::Fp32);
    for n in [
        Numeric::Fp16,
        Numeric::Bf16,
        Numeric::Fp8E4m3,
        Numeric::Fp8E5m2,
        Numeric::Int8,
        Numeric::Int4,
        Numeric::Nf4,
        Numeric::Int2,
    ] {
        assert!(
            model.microjoules_per_token(n) < j_fp32,
            "{n:?} should be cheaper than fp32"
        );
    }
}
