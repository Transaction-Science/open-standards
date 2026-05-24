//! Ratio sampler determinism + parent-based fallback.

use eoc_observability::{
    AlwaysOffSampler, AlwaysOnSampler, ParentBased, Sampler, SamplingDecision, SpanContext,
    SpanId, TraceFlags, TraceIdRatioBased, TraceId,
};

#[test]
fn ratio_zero_drops_all() {
    let s = TraceIdRatioBased::new(0.0);
    let tid = TraceId([0xff; 16]);
    assert_eq!(
        s.should_sample(None, &tid, "op"),
        SamplingDecision::Drop
    );
}

#[test]
fn ratio_one_keeps_all() {
    let s = TraceIdRatioBased::new(1.0);
    let tid = TraceId([0; 16]);
    assert_eq!(
        s.should_sample(None, &tid, "op"),
        SamplingDecision::RecordAndSample
    );
}

#[test]
fn ratio_half_splits_on_high_bit() {
    let s = TraceIdRatioBased::new(0.5);
    // Threshold = u64::MAX / 2 (approximately). A trace_id whose high
    // 8 bytes start with 0x00 must pass; one starting with 0xff must drop.
    let low = TraceId([0x00; 16]);
    let high = {
        let mut b = [0u8; 16];
        b[0] = 0xff;
        TraceId(b)
    };
    assert_eq!(
        s.should_sample(None, &low, "op"),
        SamplingDecision::RecordAndSample
    );
    assert_eq!(
        s.should_sample(None, &high, "op"),
        SamplingDecision::Drop
    );
}

#[test]
fn ratio_is_clamped() {
    let s = TraceIdRatioBased::new(-0.5);
    assert_eq!(s.ratio(), 0.0);
    let s = TraceIdRatioBased::new(2.0);
    assert_eq!(s.ratio(), 1.0);
}

#[test]
fn parent_based_follows_sampled_parent() {
    let pb = ParentBased::new(Box::new(AlwaysOffSampler));
    let parent = SpanContext::new(
        TraceId([1; 16]),
        SpanId([2; 8]),
        TraceFlags::SAMPLED,
    );
    assert_eq!(
        pb.should_sample(Some(&parent), &parent.trace_id, "op"),
        SamplingDecision::RecordAndSample
    );
}

#[test]
fn parent_based_follows_unsampled_parent() {
    let pb = ParentBased::new(Box::new(AlwaysOnSampler));
    let parent = SpanContext::new(
        TraceId([1; 16]),
        SpanId([2; 8]),
        TraceFlags::default(), // not sampled
    );
    assert_eq!(
        pb.should_sample(Some(&parent), &parent.trace_id, "op"),
        SamplingDecision::Drop
    );
}

#[test]
fn parent_based_defers_to_root_when_no_parent() {
    let pb = ParentBased::new(Box::new(AlwaysOnSampler));
    let tid = TraceId([0; 16]);
    // Even invalid trace_id with no parent should defer to root.
    assert_eq!(
        pb.should_sample(None, &tid, "op"),
        SamplingDecision::RecordAndSample
    );
}
