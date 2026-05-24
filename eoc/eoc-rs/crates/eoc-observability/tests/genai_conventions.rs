//! GenAI semconv attribute serialization + LangSmith / LangFuse mapping.

use eoc_observability::{
    AttrValue, GenAiAttributes, LangFuseExporter, LangSmithExporter, Span, SpanContext,
    SpanExporter, SpanId, SpanKind, TraceFlags, TraceId, span_to_observation, span_to_run,
};

fn make_span_with_genai() -> Span {
    let ctx = SpanContext::new(
        TraceId([0x10; 16]),
        SpanId([0x20; 8]),
        TraceFlags::SAMPLED,
    );
    let mut s = Span::new(&ctx, None, "llm.chat", SpanKind::Client, 1_000);
    let attrs = GenAiAttributes {
        operation_name: Some("chat".to_string()),
        system: Some("anthropic".to_string()),
        request_model: Some("claude-opus-4-7".to_string()),
        temperature: Some(0.7),
        max_tokens: Some(2048),
        response_model: Some("claude-opus-4-7-2026-05-01".to_string()),
        finish_reasons: Some("stop".to_string()),
        prompt_tokens: Some(123),
        completion_tokens: Some(456),
        total_tokens: Some(579),
        ..Default::default()
    }
    .to_attributes();
    for (k, v) in attrs {
        s.set_attribute(k, v);
    }
    s.set_joule_cost(eoc_core::JouleCost::measured(987_654));
    s.end(2_000);
    s
}

#[test]
fn genai_attrs_render_canonical_keys() {
    let attrs = GenAiAttributes {
        request_model: Some("gpt-4o".into()),
        prompt_tokens: Some(10),
        completion_tokens: Some(20),
        total_tokens: Some(30),
        ..Default::default()
    }
    .to_attributes();

    // gen_ai.request.model -> "gpt-4o"
    let model = attrs
        .iter()
        .find(|(k, _)| k == "gen_ai.request.model")
        .expect("model attr present");
    match &model.1 {
        AttrValue::String(s) => assert_eq!(s, "gpt-4o"),
        other => panic!("expected String, got {other:?}"),
    }

    // gen_ai.usage.total_tokens -> 30
    let total = attrs
        .iter()
        .find(|(k, _)| k == "gen_ai.usage.total_tokens")
        .expect("total tokens present");
    match &total.1 {
        AttrValue::Int(i) => assert_eq!(*i, 30),
        other => panic!("expected Int, got {other:?}"),
    }
}

#[test]
fn langsmith_export_emits_run_type_llm_for_client_span() {
    let span = make_span_with_genai();
    let v = span_to_run(&span);
    assert_eq!(v["run_type"], "llm");
    assert_eq!(v["name"], "llm.chat");
    assert_eq!(v["extra"]["gen_ai.system"], "anthropic");
    assert_eq!(v["extra"]["eoc.joules.microjoules"], 987_654);
    assert_eq!(v["extra"]["eoc.joules.source"], "measured");
}

#[test]
fn langfuse_export_emits_generation_when_genai_attrs_present() {
    let span = make_span_with_genai();
    let v = span_to_observation(&span);
    assert_eq!(v["type"], "GENERATION");
    assert_eq!(v["model"], "claude-opus-4-7-2026-05-01");
    assert_eq!(v["usage"]["input"], 123);
    assert_eq!(v["usage"]["output"], 456);
    assert_eq!(v["usage"]["total"], 579);
    assert_eq!(v["usage"]["eoc_microjoules"], 987_654);
}

#[test]
fn langsmith_exporter_collects_runs() {
    let exp = LangSmithExporter::new();
    let span = make_span_with_genai();
    exp.export(&[span]).expect("export");
    let runs = exp.finished_runs();
    assert_eq!(runs.len(), 1);
}

#[test]
fn langfuse_exporter_collects_observations() {
    let exp = LangFuseExporter::new();
    let span = make_span_with_genai();
    exp.export(&[span]).expect("export");
    let obs = exp.finished_observations();
    assert_eq!(obs.len(), 1);
}
