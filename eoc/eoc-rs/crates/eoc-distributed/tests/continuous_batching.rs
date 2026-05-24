//! Continuous batching admits, decodes, and evicts requests.

use eoc_distributed::{BatchConfig, BatchRequest, ContinuousBatcher};

fn req(id: &str, max_new: u32, slots: u32) -> BatchRequest {
    BatchRequest {
        id: id.into(),
        generated_tokens: 0,
        max_new_tokens: max_new,
        prefill_kv_slots: slots,
    }
}

#[test]
fn admit_then_decode_to_completion() {
    let mut b = ContinuousBatcher::new(BatchConfig {
        max_running: 4,
        kv_budget: 100,
        max_queue_tokens: 32,
    });
    b.enqueue(req("a", 3, 20)).expect("ok");
    b.enqueue(req("b", 2, 20)).expect("ok");
    b.enqueue(req("c", 5, 20)).expect("ok");

    let admitted = b.admit();
    assert_eq!(admitted, 3);
    assert_eq!(b.running_len(), 3);
    assert_eq!(b.used_kv(), 60);

    // Step 1 — none done yet.
    assert_eq!(b.step(), 0);
    // Step 2 — "b" finishes (max_new=2).
    assert_eq!(b.step(), 1);
    assert_eq!(b.used_kv(), 40);
    // Step 3 — "a" finishes (max_new=3).
    assert_eq!(b.step(), 1);
    // Steps 4, 5 — "c" finishes.
    b.step();
    let done = b.step();
    assert_eq!(done, 1);
    assert_eq!(b.running_len(), 0);
    assert_eq!(b.used_kv(), 0);
}

#[test]
fn admit_stops_at_kv_budget() {
    let mut b = ContinuousBatcher::new(BatchConfig {
        max_running: 32,
        kv_budget: 50,
        max_queue_tokens: 32,
    });
    b.enqueue(req("a", 1, 30)).expect("ok");
    b.enqueue(req("b", 1, 30)).expect("ok"); // would overflow
    b.enqueue(req("c", 1, 10)).expect("ok"); // could fit but is blocked
                                              // by FIFO order behind b.
    assert_eq!(b.admit(), 1);
    assert_eq!(b.running_len(), 1);
    assert_eq!(b.waiting_len(), 2);
}

#[test]
fn iteration_counter_advances() {
    let mut b = ContinuousBatcher::new(BatchConfig::default());
    assert_eq!(b.iteration(), 0);
    b.step();
    b.step();
    b.step();
    assert_eq!(b.iteration(), 3);
}
