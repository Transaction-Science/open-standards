//! End-to-end span lifecycle: start, attribute, end, batch-export.

use eoc_observability::{
    AlwaysOnSampler, BatchSpanProcessor, GenAiAttributes, InMemoryExporter, Sampler,
    SamplingDecision, Span, SpanContext, SpanId, SpanKind, Status, TraceFlags, TraceId,
};
use std::sync::Arc;

#[test]
fn full_lifecycle_emits_one_span() {
    let exporter = Arc::new(InMemoryExporter::new());
    let processor = BatchSpanProcessor::new(exporter.clone());

    // Sampler says yes.
    let sampler = AlwaysOnSampler;
    let trace_id = TraceId([1u8; 16]);
    let decision = sampler.should_sample(None, &trace_id, "llm.generate");
    assert_eq!(decision, SamplingDecision::RecordAndSample);

    // Build a sampled context.
    let span_id = SpanId([2u8; 8]);
    let ctx = SpanContext::new(trace_id, span_id, TraceFlags::SAMPLED);

    // Start the span.
    let mut span = Span::new(
        &ctx,
        None,
        "llm.generate",
        SpanKind::Client,
        1_000_000_000,
    );

    // GenAI attributes.
    let attrs = GenAiAttributes {
        operation_name: Some("chat".to_string()),
        system: Some("openai".to_string()),
        request_model: Some("gpt-4o".to_string()),
        response_model: Some("gpt-4o-2024-08-06".to_string()),
        prompt_tokens: Some(100),
        completion_tokens: Some(200),
        total_tokens: Some(300),
        finish_reasons: Some("stop".to_string()),
        ..Default::default()
    }
    .to_attributes();
    for (k, v) in attrs {
        span.set_attribute(k, v);
    }

    // Joule cost.
    span.set_joule_cost(eoc_core::JouleCost::measured(12_345_000));

    span.set_status(Status::ok());
    span.end(2_000_000_000);
    assert!(span.is_ended());
    assert_eq!(span.duration_nanos(), 1_000_000_000);

    processor.on_end(span).expect("on_end");
    processor.force_flush().expect("flush");

    let collected = exporter.finished_spans();
    assert_eq!(collected.len(), 1);
    let exported = &collected[0];
    assert_eq!(exported.name, "llm.generate");
    assert!(exported.joule_cost.is_some());
    let joule = exported.joule_cost.expect("joule_cost present");
    assert_eq!(joule.microjoules, 12_345_000);

    // Verify gen_ai.system attribute round-tripped.
    let has_system = exported
        .attributes
        .iter()
        .any(|(k, _)| k == "gen_ai.system");
    assert!(has_system, "gen_ai.system attribute missing");
}

#[test]
fn batch_processor_respects_max_batch_size() {
    let exporter = Arc::new(InMemoryExporter::new());
    let cfg = eoc_observability::BatchConfig {
        max_batch_size: 2,
        max_queue_size: 10,
    };
    let processor = BatchSpanProcessor::with_config(exporter.clone(), cfg);

    let trace_id = TraceId([1u8; 16]);
    for i in 0..3u8 {
        let span_id = SpanId([i + 1; 8]);
        let ctx = SpanContext::new(trace_id, span_id, TraceFlags::SAMPLED);
        let mut s = Span::new(&ctx, None, "op", SpanKind::Internal, 0);
        s.end(1);
        processor.on_end(s).expect("on_end");
    }
    // After 2 spans the queue auto-flushed; the third remains pending.
    assert_eq!(processor.pending_count(), 1);
    assert_eq!(exporter.finished_spans().len(), 2);

    processor.shutdown().expect("shutdown");
    assert_eq!(exporter.finished_spans().len(), 3);
}
