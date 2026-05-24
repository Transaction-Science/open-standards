//! Backpressure: producer awaits when the channel is full and the
//! receiver is slow.

use std::time::Duration;

use eoc_streaming::backpressure::{BoundedStream, Watermarks};
use eoc_streaming::stream::{Event, Role};

fn ev() -> Event {
    Event::MessageStart {
        id: None,
        role: Role::Assistant,
    }
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn try_send_returns_backpressure_when_full() {
    let mut s = BoundedStream::new(Watermarks {
        capacity: 2,
        high: 2,
        low: 1,
    });
    let p = s.producer();
    p.try_send(ev()).unwrap();
    p.try_send(ev()).unwrap();
    let err = p.try_send(ev()).unwrap_err();
    assert!(matches!(
        err,
        eoc_streaming::error::StreamError::Backpressure
    ));
    // Drain one slot; should accept again.
    let _ = s.recv().await;
    p.try_send(ev()).unwrap();
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn awaiting_send_pauses_until_receiver_drains() {
    let mut s = BoundedStream::new(Watermarks {
        capacity: 1,
        high: 1,
        low: 0,
    });
    let p = s.producer();
    p.send(ev()).await.unwrap();
    // Now full; spawn a send that should park until we recv.
    let p2 = p.clone();
    let h = tokio::spawn(async move {
        p2.send(ev()).await.unwrap();
        true
    });
    // Time has not advanced — task should still be pending.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!h.is_finished(), "producer should be parked");
    // Drain one slot — task should now complete.
    let _ = s.recv().await;
    assert!(h.await.unwrap());
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn watermark_signals_pause_and_resume() {
    let wm = Watermarks {
        capacity: 4,
        high: 3,
        low: 1,
    };
    let mut s = BoundedStream::new(wm);
    let p = s.producer();
    p.send(ev()).await.unwrap();
    assert!(!p.should_pause());
    p.send(ev()).await.unwrap();
    p.send(ev()).await.unwrap();
    assert!(p.should_pause());
    for _ in 0..3 {
        let _ = s.recv().await;
    }
    assert!(p.should_resume());
}
