//! End-to-end settlement test:
//!
//! 1. Post three transactions into the ledger.
//! 2. Open a batch, add all three entries.
//! 3. Tick a daily cutoff past 07:00 UTC — batch closes.
//! 4. Compute holdback (50 bps).
//! 5. Render NACHA file.
//! 6. Submit-for-payout, mark settled.
//!
//! No mocks for the financial primitives; we walk the same surfaces
//! a production operator would.

use op_core::{Currency, Money};
use op_ledger::{
    Account, AccountClass, Entry, InMemoryLedgerStore, Ledger, LedgerStore, Transaction,
};
use op_settlement::nacha::{NachaCredit, NachaProfile, SecCode};
use op_settlement::{
    Cutoff, HoldbackPolicy, InMemorySettlementStore, PayoutRail, SettlementEngine, SettlementStore,
    nacha_file,
};

#[test]
fn ledger_to_nacha_full_pipeline() {
    // ---- Ledger setup ----
    let ledger_store = InMemoryLedgerStore::new();
    let l = Ledger::new("Acme").unwrap();
    let lid = l.id;
    ledger_store.create_ledger(l).unwrap();
    let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
    let rev = Account::new(lid, "rev", AccountClass::Revenue, Currency::USD);
    let cid = cash.id;
    let rid = rev.id;
    ledger_store.create_account(cash).unwrap();
    ledger_store.create_account(rev).unwrap();

    let amounts = [750_000_i64, 250_000, 100_000];
    let mut tx_ids = Vec::new();
    for (i, amt) in amounts.iter().enumerate() {
        let tx = Transaction::new_posted(
            lid,
            1_700_000_000 + i as u64, // monotonic effective time
            vec![
                Entry::debit(cid, Money::from_minor(*amt, Currency::USD)),
                Entry::credit(rid, Money::from_minor(*amt, Currency::USD)),
            ],
        )
        .unwrap();
        let tx_id = ledger_store.post_transaction(tx).unwrap();
        tx_ids.push(tx_id);
    }

    // ---- Settlement engine ----
    let settlement = InMemorySettlementStore::new();
    let engine = SettlementEngine::new(
        Currency::USD,
        PayoutRail::AchNacha,
        Cutoff::daily(7).unwrap(),
        HoldbackPolicy::flat(50), // 50 bp
    );

    // Open a batch at Nov 14 22:13:20 UTC.
    let opened_at = 1_700_000_000_u64;
    let batch_id = engine.open_batch(&settlement, opened_at).unwrap();

    // Pull each posted tx into the batch.
    for (i, tx_id) in tx_ids.iter().enumerate() {
        engine
            .add_entry(
                &settlement,
                batch_id,
                *tx_id,
                Money::from_minor(amounts[i], Currency::USD),
                Some(format!("ord-{i}")),
            )
            .unwrap();
    }

    // Tick past the Nov 15 07:00 UTC cutoff — should close.
    let next_day_after_cutoff = 1_700_032_000_u64;
    let closed = engine
        .tick(&settlement, opened_at, next_day_after_cutoff)
        .unwrap();
    assert_eq!(closed, Some(batch_id));
    let batch_after_close = settlement.get_batch(batch_id).unwrap();
    assert_eq!(batch_after_close.status.code(), "closed");
    let hb = batch_after_close.holdback.expect("holdback set");
    // Gross = $11,000.00 = 1_100_000 cents. 50bp = 5_500 cents.
    assert_eq!(hb.gross, Money::from_minor(1_100_000, Currency::USD));
    assert_eq!(hb.reserve, Money::from_minor(5_500, Currency::USD));

    // ---- NACHA payout ----
    let profile = NachaProfile {
        odfi_routing: "121000248".into(),
        immediate_origin: "1234567890".into(),
        immediate_destination: "0210000211".into(),
        company_name: "OPENPAY VENDOR INC".into(),
        company_id: "9876543210".into(),
        company_entry_description: "SETTLEMNT".into(),
        effective_entry_date: "261122".into(),
    };
    let credits: Vec<NachaCredit> = amounts
        .iter()
        .enumerate()
        .map(|(i, amt)| NachaCredit {
            rdfi_routing: "021000021".into(),
            account_number: format!("11111111{i:02}"),
            receiver_name: format!("RECEIVER {i}"),
            amount_cents: u64::try_from(*amt).unwrap(),
            individual_id: format!("RCV-{i}"),
            sec: SecCode::Ccd,
        })
        .collect();

    let nacha = nacha_file(&batch_after_close, &profile, &credits).unwrap();
    // 1 file_header + 1 batch_header + 3 details + 1 batch_ctl +
    // 1 file_ctl = 7 records, padded to 10.
    assert_eq!(nacha.lines().count(), 10);
    for line in nacha.lines() {
        assert_eq!(line.len(), 94);
    }

    // ---- Submit and settle ----
    engine
        .submit_for_payout(&settlement, batch_id, "trace-abc-123", 1_700_032_500)
        .unwrap();
    let final_batch = engine
        .mark_settled(&settlement, batch_id, 1_700_032_600)
        .unwrap();
    assert!(final_batch.status.is_terminal());
    assert_eq!(final_batch.status.code(), "paid");
}

#[test]
fn empty_batch_cannot_be_paid() {
    let settlement = InMemorySettlementStore::new();
    let engine = SettlementEngine::new(
        Currency::USD,
        PayoutRail::AchNacha,
        Cutoff::Manual,
        HoldbackPolicy::none(),
    );
    let bid = engine.open_batch(&settlement, 1_000).unwrap();
    // Close with no entries — legal (zero-amount batch).
    let b = engine.close_batch(&settlement, bid, 0, 2_000).unwrap();
    assert_eq!(b.status.code(), "closed");
    // But NACHA generation rejects empty credit lists.
    let profile = NachaProfile {
        odfi_routing: "121000248".into(),
        immediate_origin: "1234567890".into(),
        immediate_destination: "0210000211".into(),
        company_name: "OPENPAY VENDOR INC".into(),
        company_id: "9876543210".into(),
        company_entry_description: "SETTLEMNT".into(),
        effective_entry_date: "261122".into(),
    };
    let err = nacha_file(&b, &profile, &[]).unwrap_err();
    assert!(matches!(err, op_settlement::Error::EmptyBatch));
}
