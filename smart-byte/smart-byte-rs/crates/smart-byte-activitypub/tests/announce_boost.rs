//! Announce (boost) + Like + Undo.

use smart_byte_activitypub::{Activity, ActivityObject, InboxEffect, InboxState, Object, Result};

#[test]
fn announce_records_boost() -> Result<()> {
    let mut state = InboxState::new();
    let note = Object::note(
        "https://a.test/notes/1",
        "https://a.test/users/alice",
        "boost me",
    );
    state.handle(&Activity::create(
        "https://a.test/activities/c",
        "https://a.test/users/alice",
        note,
    ))?;

    let boost = Activity::announce(
        "https://b.test/activities/boost-1",
        "https://b.test/users/bob",
        "https://a.test/notes/1",
    );
    let effect = state.handle(&boost)?;
    assert_eq!(
        effect,
        InboxEffect::Boosted {
            object: "https://a.test/notes/1".into(),
            activity: "https://b.test/activities/boost-1".into(),
        }
    );
    let boosts = state
        .announces
        .get("https://a.test/notes/1")
        .expect("boosts recorded");
    assert!(boosts.contains("https://b.test/activities/boost-1"));
    Ok(())
}

#[test]
fn like_then_undo() -> Result<()> {
    let mut state = InboxState::new();
    let note = Object::note(
        "https://a.test/notes/2",
        "https://a.test/users/alice",
        "like me",
    );
    state.handle(&Activity::create(
        "https://a.test/activities/c2",
        "https://a.test/users/alice",
        note,
    ))?;

    let like = Activity::like(
        "https://b.test/activities/like-1",
        "https://b.test/users/bob",
        "https://a.test/notes/2",
    );
    state.handle(&like)?;
    assert!(state.likes["https://a.test/notes/2"].contains("https://b.test/users/bob"));

    // Bob unlikes by Undo of the embedded Like, with in_reply_to pointing
    // at the target.
    let mut embedded = Object::note(
        "https://b.test/activities/like-1",
        "https://b.test/users/bob",
        "",
    );
    embedded.type_field = "Like".into();
    embedded.in_reply_to = Some("https://a.test/notes/2".into());
    let undo = Activity::undo(
        "https://b.test/activities/undo-1",
        "https://b.test/users/bob",
        ActivityObject::embedded(embedded),
    );
    let effect = state.handle(&undo)?;
    assert_eq!(effect, InboxEffect::Undone);
    assert!(!state.likes["https://a.test/notes/2"].contains("https://b.test/users/bob"));
    Ok(())
}
