//! Output rendering — pretty text for humans, JSON for tools.

use std::io::Write;

use jouleclaw_compose::AnswerOrRefusal;
use jouleclaw_schema::{Answer, AnswerSegment, AnswerStatus, Refusal, VerificationAction};

use crate::pipeline::{EntailerKind, PipelineOutput};

/// Emit either pretty text (default) or JSON to stdout depending
/// on the `json` flag. Returns the exit code: 0 for Answer, 1 for
/// Refusal.
pub fn render(out: &PipelineOutput, json: bool) -> i32 {
    if json {
        render_json(out);
    } else {
        render_text(out);
    }
    match &out.result {
        AnswerOrRefusal::Answer(_) => 0,
        AnswerOrRefusal::Refusal(_) => 1,
    }
}

/// Render the JSON envelope as a String (used by the server to
/// send over the socket; the local CLI also calls this and prints).
pub fn render_json_to_string(out: &PipelineOutput) -> String {
    serde_json::to_string_pretty(&build_json_envelope(out))
        .unwrap_or_else(|e| format!("{{\"error\":\"json serialize: {e}\"}}"))
        + "\n"
}

/// Render the pretty text output as a String (server-side variant
/// of the stdout-printer).
pub fn render_text_to_string(out: &PipelineOutput) -> String {
    let mut buf = Vec::with_capacity(2048);
    write_text(&mut buf, out);
    String::from_utf8_lossy(&buf).into_owned()
}

fn build_json_envelope(out: &PipelineOutput) -> serde_json::Value {
    let draft_view: Vec<_> = out
        .draft
        .iter()
        .map(|s| {
            serde_json::json!({
                "segment_id": s.segment_id,
                "text": s.text,
                "cited_item_ids": s.cited_item_ids,
            })
        })
        .collect();
    serde_json::json!({
        "query": out.query,
        "verdict": out.report.verdict,
        "reroute_passes": out.reroute_passes,
        "cache_hit": out.cache_hit,
        "entailer": match out.entailer_kind {
            EntailerKind::Deberta => "deberta-v3",
            EntailerKind::NoVerify => "no-verify",
        },
        "stages_ms": {
            "understanding": out.stages.understanding_ms,
            "plan": out.stages.plan_ms,
            "execute": out.stages.execute_ms,
            "draft": out.stages.draft_ms,
            "atomize": out.stages.atomize_ms,
            "verify": out.stages.verify_ms,
            "compose": out.stages.compose_ms,
            "total": out.stages.total_ms,
        },
        "plan": out.plan,
        "items": out.items,
        "draft": draft_view,
        "claims": out.claims,
        "report": out.report,
        "result": match &out.result {
            AnswerOrRefusal::Answer(a) => serde_json::json!({ "answer": a }),
            AnswerOrRefusal::Refusal(r) => serde_json::json!({ "refusal": r }),
        },
    })
}

fn render_json(out: &PipelineOutput) {
    let s = render_json_to_string(out);
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(s.as_bytes());
}

fn render_text(out: &PipelineOutput) {
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    write_text(&mut w, out);
}

fn write_text<W: Write>(w: &mut W, out: &PipelineOutput) {
    let _ = writeln!(w, "");
    let _ = writeln!(w, "Q: {}", out.query);
    let _ = writeln!(w, "");

    match &out.result {
        AnswerOrRefusal::Answer(a) => render_answer(w, a, out),
        AnswerOrRefusal::Refusal(r) => render_refusal(w, r, out),
    }
}

fn render_answer<W: Write>(w: &mut W, ans: &Answer, out: &PipelineOutput) {
    let body = compose_answer_body(&ans.segments);
    let _ = writeln!(w, "{body}");
    let _ = writeln!(w, "");

    if !ans.segments.is_empty() {
        let _ = writeln!(w, "Sources:");
        let mut seen = std::collections::BTreeSet::<String>::new();
        for seg in &ans.segments {
            for id in &seg.cited_item_ids {
                if let Some(item) = out.items.iter().find(|it| it.item_id == *id) {
                    let label = item
                        .source_url
                        .clone()
                        .or_else(|| Some(format!("{} ({})", item.source_id, retriever_id(item))))
                        .unwrap_or_else(|| item.source_id.clone());
                    if seen.insert(label.clone()) {
                        let _ = writeln!(w, "  • {label}");
                    }
                }
            }
        }
        let _ = writeln!(w, "");
    }

    if !ans.caveats.is_empty() {
        let _ = writeln!(w, "Caveats:");
        for c in &ans.caveats {
            let _ = writeln!(w, "  • {c}");
        }
        let _ = writeln!(w, "");
    }

    let status_label = match ans.status {
        AnswerStatus::Verified => "Verified",
        AnswerStatus::Degraded => "Degraded (with caveats)",
        AnswerStatus::Partial => "Partial (anytime interruption)",
    };
    let verdict_label = match out.report.verdict {
        VerificationAction::ProceedToComposition => "proceed",
        VerificationAction::ReRouteWithRefinement => "re-route",
        VerificationAction::DegradeGracefully => "degrade gracefully",
        VerificationAction::Refuse => "refuse",
    };
    let entailer_label = match out.entailer_kind {
        EntailerKind::Deberta => "DeBERTa-v3-large-mnli",
        EntailerKind::NoVerify => "no-verify",
    };

    let reroute_note = if out.reroute_passes > 0 {
        format!("  ·  Reroutes: {}", out.reroute_passes)
    } else {
        String::new()
    };
    let cache_note = if out.cache_hit { "  ·  cache hit" } else { "" };
    let _ = writeln!(
        w,
        "Status: {status_label}  ·  Verdict: {verdict_label}{reroute_note}{cache_note}"
    );
    let _ = writeln!(
        w,
        "Invariants verified: {}",
        ans.invariants_verified.join(", ")
    );
    let _ = writeln!(
        w,
        "Entailer: {entailer_label}  ·  Joules: {:.1}  ·  Latency: {}",
        ans.joules_spent_total,
        format_ms(ans.latency_ms)
    );

    let s = out.stages;
    let _ = writeln!(
        w,
        "Stages (ms): understanding={} plan={} execute={} draft={} atomize={} verify={} compose={} total={}",
        s.understanding_ms,
        s.plan_ms,
        s.execute_ms,
        s.draft_ms,
        s.atomize_ms,
        s.verify_ms,
        s.compose_ms,
        s.total_ms,
    );
}

fn render_refusal<W: Write>(w: &mut W, r: &Refusal, out: &PipelineOutput) {
    let _ = writeln!(w, "Refused: {}", r.reason_message);
    let _ = writeln!(w, "Reason code: {}", r.reason_code);
    if !r.blocking_violations.is_empty() {
        let _ = writeln!(w, "Blocking violations: {}", r.blocking_violations.join(", "));
    }
    let _ = writeln!(w, "");
    let _ = writeln!(
        w,
        "Verdict: refuse  ·  Latency: {}",
        format_ms(out.stages.total_ms)
    );
}

fn compose_answer_body(segments: &[AnswerSegment]) -> String {
    segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn retriever_id(item: &jouleclaw_schema::RetrievedItem) -> &str {
    item.retrieval_context.retriever_id.as_str()
}

fn format_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        let secs = ms / 1000;
        format!("{}m{}s", secs / 60, secs % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_ms_uses_friendly_units() {
        assert_eq!(format_ms(50), "50ms");
        assert_eq!(format_ms(1_500), "1.50s");
        assert_eq!(format_ms(65_000), "1m5s");
    }
}
