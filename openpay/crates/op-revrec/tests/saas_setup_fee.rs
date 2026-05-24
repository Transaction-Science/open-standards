//! Multi-element arrangement: SaaS subscription ($120/year, 12-month
//! straight-line) + one-time setup fee ($300, recognized at go-live).
//!
//! The combined contract is priced at $400 (a $20 discount off the
//! $420 list). Under ASC 606-10-32-31 the discount allocates by
//! relative SSP, NOT by line.

use chrono::NaiveDate;
use op_core::{Currency, Money};
use op_revrec::{
    Contract, ContractId, InMemoryLedger, DeferredRevenueLedger, ObligationId,
    PerformanceObligation, Presentation, RecognitionPattern, TransactionPrice,
    allocate_transaction_price, generate,
};

fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
}

fn usd(n: i64) -> Money {
    Money::from_minor(n, Currency::USD)
}

#[tokio::test]
async fn saas_plus_setup_fee_full_lifecycle() {
    let saas = PerformanceObligation {
        id: ObligationId::new("saas-2026"),
        standalone_selling_price: usd(120_00),
        pattern: RecognitionPattern::StraightLine {
            start: ymd(2026, 1, 1),
            end: ymd(2026, 12, 31),
        },
        presentation: Presentation::Gross,
    };
    let setup = PerformanceObligation {
        id: ObligationId::new("setup-fee"),
        standalone_selling_price: usd(300_00),
        pattern: RecognitionPattern::PointInTime { date: ymd(2026, 1, 15) },
        presentation: Presentation::Gross,
    };

    let contract = Contract {
        id: ContractId::new(),
        customer_ref: "customer-1".into(),
        effective_date: ymd(2026, 1, 1),
        obligations: vec![saas.clone(), setup.clone()],
        transaction_price: TransactionPrice::fixed(usd(400_00)),
    };

    // Step 4: allocate. SSPs: saas 12000, setup 30000 -> total 42000.
    // saas alloc = 40000 * 12000 / 42000 = 11428.57 -> 11428
    // setup alloc = plug = 40000 - 11428 = 28572
    let alloc = allocate_transaction_price(&contract).expect("alloc");
    let map: std::collections::HashMap<_, _> = alloc.into_iter().collect();
    let saas_alloc = map[&ObligationId::new("saas-2026")];
    let setup_alloc = map[&ObligationId::new("setup-fee")];
    assert_eq!(saas_alloc + setup_alloc, 40_000);

    // Step 5: schedules.
    let saas_sched = generate(&saas, saas_alloc, Currency::USD).expect("saas sched");
    let setup_sched = generate(&setup, setup_alloc, Currency::USD).expect("setup sched");
    assert_eq!(saas_sched.entries.len(), 12);
    assert_eq!(setup_sched.entries.len(), 1);
    assert_eq!(saas_sched.total_minor(), saas_alloc);
    assert_eq!(setup_sched.total_minor(), setup_alloc);

    // Open deferral at contract inception and walk the schedule.
    let ledger = InMemoryLedger::new();
    ledger
        .open_deferral(
            contract.id,
            None,
            usd(400_00),
            contract.effective_date,
            "contract inception",
        )
        .await
        .expect("open");

    // Post setup recognition (the one entry on 2026-01-15).
    ledger
        .post_recognition(contract.id, &setup_sched.entries[0], "go-live")
        .await
        .expect("setup rec");

    // Post the first 3 months of SaaS.
    for e in saas_sched.entries.iter().take(3) {
        ledger
            .post_recognition(contract.id, e, "saas monthly")
            .await
            .expect("monthly rec");
    }

    let bal = ledger.balances(contract.id).await.expect("bal");
    let three_months_saas: i64 = saas_sched.entries.iter().take(3).map(|e| e.amount_minor).sum();
    assert_eq!(bal.recognized_minor, setup_alloc + three_months_saas);
    assert_eq!(
        bal.deferred_minor,
        40_000 - setup_alloc - three_months_saas
    );
}
