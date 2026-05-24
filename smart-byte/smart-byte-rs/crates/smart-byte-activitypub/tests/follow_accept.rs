//! Follow → Accept round-trip via the inbox state.

use smart_byte_activitypub::{
    Activity, ActivityObject, InboxEffect, InboxState, Object, Result,
};

#[test]
fn follow_then_accept_records_follower() -> Result<()> {
    let mut state = InboxState::new();

    // Alice sends a Follow to Bob. Bob's server receives it.
    let follow = Activity::follow(
        "https://a.test/activities/follow-1",
        "https://a.test/users/alice",
        "https://b.test/users/bob",
    );
    let effect = state.handle(&follow)?;
    assert_eq!(
        effect,
        InboxEffect::FollowPending("https://a.test/activities/follow-1".into())
    );

    // Bob's server sends back an Accept referencing the Follow IRI.
    let accept = Activity::accept(
        "https://b.test/activities/accept-1",
        "https://b.test/users/bob",
        ActivityObject::iri("https://a.test/activities/follow-1"),
    );
    let effect = state.handle(&accept)?;
    assert_eq!(
        effect,
        InboxEffect::FollowAccepted {
            target: "https://b.test/users/bob".into(),
            follower: "https://a.test/users/alice".into(),
        }
    );

    // Bob now has Alice as a follower.
    let followers = state
        .followers
        .get("https://b.test/users/bob")
        .expect("bob has followers");
    assert!(followers.contains("https://a.test/users/alice"));
    Ok(())
}

#[test]
fn accept_with_embedded_follow_object() -> Result<()> {
    let mut state = InboxState::new();
    let mut embedded = Object::note(
        "https://a.test/activities/follow-2",
        "https://a.test/users/alice",
        "",
    );
    embedded.type_field = "Follow".into();
    let accept = Activity::accept(
        "https://b.test/activities/accept-2",
        "https://b.test/users/bob",
        ActivityObject::embedded(embedded),
    );
    let effect = state.handle(&accept)?;
    assert!(matches!(effect, InboxEffect::FollowAccepted { .. }));
    Ok(())
}

#[test]
fn reject_clears_pending_follow() -> Result<()> {
    let mut state = InboxState::new();
    let follow = Activity::follow(
        "https://a.test/activities/follow-x",
        "https://a.test/users/alice",
        "https://b.test/users/bob",
    );
    state.handle(&follow)?;
    assert!(state
        .pending_follows
        .contains_key("https://a.test/activities/follow-x"));

    let reject = Activity::reject(
        "https://b.test/activities/reject-x",
        "https://b.test/users/bob",
        ActivityObject::iri("https://a.test/activities/follow-x"),
    );
    state.handle(&reject)?;
    assert!(!state
        .pending_follows
        .contains_key("https://a.test/activities/follow-x"));
    Ok(())
}
