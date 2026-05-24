//! Cancellation: a client disconnect (modeled as a `CancelToken::cancel`)
//! propagates to upstream awaiters via `select!`.

use std::time::Duration;

use eoc_streaming::cancel::CancelToken;
use eoc_streaming::stream::{Event, Role, Sink, TokenStream};

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn cancel_wakes_awaiters() {
    let tok = CancelToken::new();
    let tok2 = tok.clone();
    let h = tokio::spawn(async move {
        tok2.cancelled().await;
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(!h.is_finished());
    tok.cancel();
    h.await.unwrap();
    assert!(tok.is_cancelled());
}

#[tokio::test(flavor = "current_thread")]
async fn cancel_short_circuits_select_against_stream() {
    let tok = CancelToken::new();
    let mut s = TokenStream::bounded(4);
    let sink = s.sink();

    // Push one event then sit idle.
    sink.send(Event::MessageStart {
        id: None,
        role: Role::Assistant,
    })
    .await
    .unwrap();

    // Cancel after a short delay.
    let tok2 = tok.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        tok2.cancel();
    });

    // Consume the queued event, then a second `recv` should be
    // cancelled rather than blocking forever.
    let first = tokio::select! {
        ev = s.recv() => ev,
        _ = tok.cancelled() => panic!("cancel fired before first event"),
    };
    assert!(matches!(first, Some(Event::MessageStart { .. })));

    let outcome = tokio::select! {
        _ = s.recv() => "got_event",
        _ = tok.cancelled() => "cancelled",
    };
    assert_eq!(outcome, "cancelled");
}

#[tokio::test]
async fn cancel_is_observable_from_clones() {
    let tok = CancelToken::new();
    let a = tok.clone();
    let b = tok.clone();
    assert!(!a.is_cancelled());
    assert!(!b.is_cancelled());
    tok.cancel();
    assert!(a.is_cancelled());
    assert!(b.is_cancelled());
}
