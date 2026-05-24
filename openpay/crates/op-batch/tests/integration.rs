//! Cross-module integration: build → name → submit → reconcile.

use chrono::TimeZone;
use op_batch::{
    BatchRail, FileNaming, Submission, SubmissionSink,
    file_io::MemorySink,
    nacha::{self, SecCode},
    reconciliation::{self, Expected, ReconcileSource, StatementLine},
};

#[test]
fn nacha_build_then_submit_then_reconcile() {
    let now = chrono::Utc.with_ymd_and_hms(2026, 6, 1, 12, 0, 0).unwrap();
    let profile = nacha::NachaProfile {
        odfi_routing: "121000248".into(),
        immediate_origin: "1234567890".into(),
        immediate_destination: "0210000211".into(),
        company_name: "OPENPAY".into(),
        company_id: "9876543210".into(),
        company_entry_description: "SETTLEMNT".into(),
    };
    let entries = vec![nacha::EntryDetail {
        transaction_code: 22,
        rdfi_routing: "021000021".into(),
        account_number: "1111".into(),
        amount_cents: 100_000,
        individual_id: "IND-1".into(),
        receiver_name: "ALICE".into(),
        discretionary: String::new(),
        addenda_indicator: '0',
        trace_number: "121000240000001".into(),
    }];
    let file = nacha::build_file(&profile, SecCode::Ppd, entries, "260602", now).unwrap();
    let bytes = file.encode().unwrap();
    let filename = FileNaming::nacha(&profile.odfi_routing, now, 1);
    assert_eq!(filename, "121000248.20260601.001");

    let sink = MemorySink::new();
    sink.submit(Submission {
        rail: BatchRail::Nacha,
        contents: bytes,
        filename: filename.clone(),
    })
    .unwrap();
    let stored = sink.get(&filename).unwrap().unwrap();
    let parsed = nacha::NachaFile::decode(&stored.contents).unwrap();
    assert_eq!(parsed.batches[0].entries.len(), 1);
    assert_eq!(parsed.batches[0].entries[0].amount_cents, 100_000);

    // Reconcile a synthetic bank statement line back against the entry.
    let report = reconciliation::reconcile(
        BatchRail::Nacha,
        ReconcileSource::NachaPrenote,
        &[Expected {
            payment_id: "pay-1".into(),
            amount_minor: 100_000,
            reference: "121000240000001".into(),
        }],
        &[StatementLine {
            statement_ref: "BANKREF-7".into(),
            echoed_reference: "121000240000001".into(),
            amount_minor: 100_000,
            value_date: now,
        }],
    )
    .unwrap();
    assert!(report.is_clean());
}
