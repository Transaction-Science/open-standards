//! Publish a Note via the Outbox, resolve its audience, and replay it
//! through an Inbox on the other side.

use smart_byte_activitypub::{
    Activity, InboxEffect, InboxState, Object, Outbox, Result, StaticResolver, PUBLIC,
};

#[test]
fn create_note_round_trip() -> Result<()> {
    // Alice composes a public Note that ccs her followers.
    let note = Object::note(
        "https://a.test/notes/1",
        "https://a.test/users/alice",
        "<p>hello fediverse</p>",
    )
    .cc("https://a.test/users/alice/followers");

    let create = Activity::create(
        "https://a.test/activities/c-1",
        "https://a.test/users/alice",
        note.clone(),
    );

    // Alice's followers collection contains Bob and Carol.
    let mut resolver = StaticResolver::new();
    resolver.insert(
        "https://a.test/users/alice/followers",
        vec![
            "https://b.test/users/bob".to_string(),
            "https://c.test/users/carol".to_string(),
        ],
    );

    // Publish via the outbox.
    let mut outbox = Outbox::new();
    let audience = outbox.publish(create.clone(), &resolver, &|actor| match actor {
        "https://b.test/users/bob" => Some("https://b.test/users/bob/inbox".into()),
        "https://c.test/users/carol" => Some("https://c.test/users/carol/inbox".into()),
        _ => None,
    })?;
    assert!(audience.is_public);
    assert_eq!(audience.actors.len(), 2);
    assert_eq!(outbox.total_items(), 1);

    let delivery = outbox.next().expect("one pending delivery");
    assert_eq!(delivery.inboxes.len(), 2);
    assert!(delivery.body.to.iter().any(|t| t == PUBLIC));

    // Bob's inbox applies the Create.
    let mut bob = InboxState::new();
    let effect = bob.handle(&delivery.body)?;
    assert_eq!(effect, InboxEffect::ObjectStored("https://a.test/notes/1".into()));
    let stored = bob
        .objects
        .get("https://a.test/notes/1")
        .expect("note stored");
    assert_eq!(stored.content.as_deref(), Some("<p>hello fediverse</p>"));
    Ok(())
}

#[test]
fn delete_replaces_with_tombstone() -> Result<()> {
    let mut state = InboxState::new();
    let note = Object::note(
        "https://a.test/notes/9",
        "https://a.test/users/alice",
        "to be removed",
    );
    let create = Activity::create(
        "https://a.test/activities/c-9",
        "https://a.test/users/alice",
        note,
    );
    state.handle(&create)?;
    let delete = Activity::delete(
        "https://a.test/activities/d-9",
        "https://a.test/users/alice",
        "https://a.test/notes/9",
    );
    let effect = state.handle(&delete)?;
    assert_eq!(effect, InboxEffect::ObjectDeleted("https://a.test/notes/9".into()));
    let after = state
        .objects
        .get("https://a.test/notes/9")
        .expect("tombstone present");
    assert_eq!(after.type_field, "Tombstone");
    assert_eq!(after.former_type.as_deref(), Some("Note"));
    Ok(())
}
