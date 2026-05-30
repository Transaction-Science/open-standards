//! The joule-ui protocol — typed declarative component specs against a
//! published widget registry. Pure Rust, no IO; the renderer is the
//! host's.
//!
//! ## Why declarative over code
//!
//! Agents are far more reliable emitting structured data than emitting
//! code, and renderers are far easier to harden when they own their own
//! widgets. The field has converged on the same shape — A2UI, MCP Apps
//! (SEP-1865), CopilotKit's declarative middle. joule-ui pins it as a
//! protocol: the agent emits a [`Widget`] tree; the host renders it
//! through a [`Registry`] it published.
//!
//! ## What the protocol is
//!
//! A [`Widget`] is a name + a set of typed [`PropValue`] props + a list
//! of child widgets. A [`Registry`] is the host's published map from
//! widget name to [`WidgetSchema`] — what props the widget takes, what
//! types those props are, what children it accepts. [`validate`]
//! recursively checks a widget tree against the registry and returns a
//! list of every [`ValidationError`] it finds (not just the first).
//!
//! ## Honest scope (v1)
//!
//! - **Whole-tree, not streaming.** Streaming-delta extensions slot in
//!   later without changing the spec shape.
//! - **No transport.** A validated widget tree round-trips through
//!   `serde_json`; the consumer chooses MCP / REST / SSE / gRPC.
//! - **No renderer.** The host owns the registry and the rendering
//!   surface. joule-ui only attests to the spec's well-formedness.
//! - **Action references, not callbacks.** A [`PropValue::Action`]
//!   carries an opaque action id the host binds at render time. No
//!   closures, no code emission.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

// ─────────────────────────────────────────────────────────────────────
// Widget tree
// ─────────────────────────────────────────────────────────────────────

/// A node in the agent-emitted UI tree. Recursive by design — a widget
/// is `(name, props, children)`. The wire form is plain JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Widget {
    /// Registry key — the widget's named type (e.g. `"Card"`,
    /// `"Button"`, `"TextInput"`).
    pub name: String,
    /// Caller-supplied props. `BTreeMap` so the wire JSON has stable
    /// key ordering — important for content-addressing further up the
    /// stack (e.g. `jouleclaw-prov` receipts).
    #[serde(default)]
    pub props: BTreeMap<String, PropValue>,
    /// Child widgets, in render order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Widget>,
}

impl Widget {
    /// Construct a leaf widget (no children).
    pub fn leaf(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            props: BTreeMap::new(),
            children: Vec::new(),
        }
    }
    /// Attach a prop. Builder-style; `value` is anything convertible to
    /// [`PropValue`] via [`Into`].
    pub fn prop(mut self, key: impl Into<String>, value: impl Into<PropValue>) -> Self {
        self.props.insert(key.into(), value.into());
        self
    }
    /// Append a child.
    pub fn child(mut self, child: Widget) -> Self {
        self.children.push(child);
        self
    }
    /// Append several children.
    pub fn children(mut self, children: impl IntoIterator<Item = Widget>) -> Self {
        self.children.extend(children);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// Prop values
// ─────────────────────────────────────────────────────────────────────

/// A typed prop value. A subset of JSON plus an explicit `Action` variant
/// the host binds at render time. No closures, no expressions, no code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum PropValue {
    Null,
    Bool(bool),
    Number(f64),
    Text(String),
    List(Vec<PropValue>),
    Object(BTreeMap<String, PropValue>),
    /// An opaque action identifier. The host wires `id` to a handler
    /// at render time; the spec never carries the handler itself.
    Action {
        id: String,
        /// Optional payload — pure data the host receives when the
        /// action fires.
        #[serde(default)]
        payload: Option<Box<PropValue>>,
    },
}

impl PropValue {
    /// The variant tag — useful for error messages and for matching
    /// against a [`PropType`] expectation.
    pub fn ty(&self) -> PropType {
        match self {
            PropValue::Null => PropType::Null,
            PropValue::Bool(_) => PropType::Bool,
            PropValue::Number(_) => PropType::Number,
            PropValue::Text(_) => PropType::Text,
            PropValue::List(_) => PropType::List,
            PropValue::Object(_) => PropType::Object,
            PropValue::Action { .. } => PropType::Action,
        }
    }
}

impl From<bool> for PropValue {
    fn from(b: bool) -> Self {
        PropValue::Bool(b)
    }
}
impl From<i64> for PropValue {
    fn from(n: i64) -> Self {
        PropValue::Number(n as f64)
    }
}
impl From<f64> for PropValue {
    fn from(n: f64) -> Self {
        PropValue::Number(n)
    }
}
impl From<&str> for PropValue {
    fn from(s: &str) -> Self {
        PropValue::Text(s.to_string())
    }
}
impl From<String> for PropValue {
    fn from(s: String) -> Self {
        PropValue::Text(s)
    }
}

/// The discriminator for [`PropValue`] — used in [`PropSchema`] to
/// declare what shape a prop takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropType {
    Null,
    Bool,
    Number,
    Text,
    List,
    Object,
    Action,
    /// Accept any [`PropValue`] variant. Useful for free-form slots
    /// (e.g. arbitrary metadata) but discouraged for typed widgets.
    Any,
}

// ─────────────────────────────────────────────────────────────────────
// Schema + registry
// ─────────────────────────────────────────────────────────────────────

/// What a single prop slot accepts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropSchema {
    /// Expected variant.
    pub ty: PropType,
    /// `true` if the prop must be present on every instance.
    #[serde(default)]
    pub required: bool,
}

impl PropSchema {
    pub fn required(ty: PropType) -> Self {
        Self { ty, required: true }
    }
    pub fn optional(ty: PropType) -> Self {
        Self {
            ty,
            required: false,
        }
    }
}

/// What kinds of children a widget accepts, by name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum AllowedChildren {
    /// No children permitted (e.g. `Text`, `Image`).
    None,
    /// Any registered widget.
    Any,
    /// Only widgets whose names appear in this list. Useful for
    /// structural containers (e.g. `Tabs` only accepts `Tab`).
    OnlyOf(Vec<String>),
}

/// One widget's schema in the registry: name, prop slots, child rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WidgetSchema {
    pub name: String,
    /// Short human-readable description — appears in registry catalogs
    /// the host might publish for agents to read.
    pub description: String,
    /// Prop slots, keyed by name. `BTreeMap` for stable wire ordering.
    #[serde(default)]
    pub props: BTreeMap<String, PropSchema>,
    /// Child-acceptance policy.
    pub children: AllowedChildren,
    /// Whether unknown props (props not in `props`) are tolerated. The
    /// strict mode rejects them; the relaxed mode permits them so
    /// hosts can extend a widget with attributes the spec doesn't yet
    /// name. Default is strict (`false`).
    #[serde(default)]
    pub allow_extra_props: bool,
}

impl WidgetSchema {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            props: BTreeMap::new(),
            children: AllowedChildren::None,
            allow_extra_props: false,
        }
    }
    pub fn prop(mut self, name: impl Into<String>, schema: PropSchema) -> Self {
        self.props.insert(name.into(), schema);
        self
    }
    pub fn children(mut self, allowed: AllowedChildren) -> Self {
        self.children = allowed;
        self
    }
    pub fn permit_extra_props(mut self) -> Self {
        self.allow_extra_props = true;
        self
    }
}

/// The host's published widget catalog. Validation looks every widget
/// name up here; an unknown name is a [`ValidationError::UnknownWidget`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    widgets: HashMap<String, WidgetSchema>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, schema: WidgetSchema) -> Self {
        self.widgets.insert(schema.name.clone(), schema);
        self
    }
    pub fn add(&mut self, schema: WidgetSchema) {
        self.widgets.insert(schema.name.clone(), schema);
    }
    pub fn get(&self, name: &str) -> Option<&WidgetSchema> {
        self.widgets.get(name)
    }
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.widgets.keys().map(String::as_str)
    }
    pub fn len(&self) -> usize {
        self.widgets.len()
    }
    pub fn is_empty(&self) -> bool {
        self.widgets.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────

/// One specific thing wrong with a widget tree. Validation collects
/// every error rather than stopping at the first, so an authoring loop
/// can fix them in one pass.
#[derive(Debug, Clone, PartialEq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum ValidationError {
    #[error("unknown widget `{name}` at {path}")]
    UnknownWidget { name: String, path: String },
    #[error("widget `{widget}` is missing required prop `{prop}` at {path}")]
    MissingProp {
        widget: String,
        prop: String,
        path: String,
    },
    #[error(
        "widget `{widget}` prop `{prop}` has wrong type at {path}: expected {expected:?}, got {got:?}"
    )]
    WrongPropType {
        widget: String,
        prop: String,
        expected: PropType,
        got: PropType,
        path: String,
    },
    #[error("widget `{widget}` does not declare prop `{prop}` at {path}")]
    ExtraProp {
        widget: String,
        prop: String,
        path: String,
    },
    #[error(
        "widget `{parent}` does not accept child `{child}` at {path}"
    )]
    DisallowedChild {
        parent: String,
        child: String,
        path: String,
    },
}

/// Validate `widget` against `registry`. Returns every error in the
/// tree, in pre-order traversal. Empty result = valid spec.
pub fn validate(widget: &Widget, registry: &Registry) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    validate_inner(widget, registry, "$", &mut errors);
    errors
}

fn validate_inner(
    widget: &Widget,
    registry: &Registry,
    path: &str,
    errors: &mut Vec<ValidationError>,
) {
    let Some(schema) = registry.get(&widget.name) else {
        errors.push(ValidationError::UnknownWidget {
            name: widget.name.clone(),
            path: path.to_string(),
        });
        // Stop descending — without a schema we can't say what's wrong
        // with the props or children; reporting their unknowns would
        // duplicate this finding.
        return;
    };

    // Required props present?
    for (prop_name, prop_schema) in &schema.props {
        if prop_schema.required && !widget.props.contains_key(prop_name) {
            errors.push(ValidationError::MissingProp {
                widget: widget.name.clone(),
                prop: prop_name.clone(),
                path: path.to_string(),
            });
        }
    }

    // Each present prop conforms to its declared type?
    for (prop_name, prop_value) in &widget.props {
        match schema.props.get(prop_name) {
            Some(prop_schema) => {
                if !type_matches(prop_schema.ty, prop_value.ty()) {
                    errors.push(ValidationError::WrongPropType {
                        widget: widget.name.clone(),
                        prop: prop_name.clone(),
                        expected: prop_schema.ty,
                        got: prop_value.ty(),
                        path: path.to_string(),
                    });
                }
            }
            None if schema.allow_extra_props => {
                // permitted
            }
            None => {
                errors.push(ValidationError::ExtraProp {
                    widget: widget.name.clone(),
                    prop: prop_name.clone(),
                    path: path.to_string(),
                });
            }
        }
    }

    // Children: name-allowed + recursion.
    let allowed = &schema.children;
    for (i, child) in widget.children.iter().enumerate() {
        let child_path = format!("{path}.children[{i}]");
        let allowed_here = match allowed {
            AllowedChildren::None => false,
            AllowedChildren::Any => true,
            AllowedChildren::OnlyOf(list) => list.iter().any(|n| n == &child.name),
        };
        if !allowed_here {
            errors.push(ValidationError::DisallowedChild {
                parent: widget.name.clone(),
                child: child.name.clone(),
                path: child_path.clone(),
            });
            // Still descend to report errors inside the child — that's
            // more useful for authoring than masking them behind one
            // disallowed-child error.
        }
        validate_inner(child, registry, &child_path, errors);
    }
}

fn type_matches(expected: PropType, got: PropType) -> bool {
    matches!(expected, PropType::Any) || expected == got
}

/// Convenience: `Ok(())` iff `validate(widget, registry)` is empty.
pub fn is_valid(widget: &Widget, registry: &Registry) -> Result<(), Vec<ValidationError>> {
    let errs = validate(widget, registry);
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_minimal() -> Registry {
        Registry::new()
            .with(
                WidgetSchema::new("Card", "rectangular surface with optional title")
                    .prop("title", PropSchema::required(PropType::Text))
                    .prop("subtitle", PropSchema::optional(PropType::Text))
                    .children(AllowedChildren::Any),
            )
            .with(
                WidgetSchema::new("Text", "a run of text")
                    .prop("body", PropSchema::required(PropType::Text)),
            )
            .with(
                WidgetSchema::new("Button", "an interactive button")
                    .prop("label", PropSchema::required(PropType::Text))
                    .prop("on_click", PropSchema::required(PropType::Action)),
            )
            .with(
                WidgetSchema::new("Tabs", "horizontal tab strip")
                    .children(AllowedChildren::OnlyOf(vec!["Tab".into()])),
            )
            .with(
                WidgetSchema::new("Tab", "one tab inside a Tabs")
                    .prop("title", PropSchema::required(PropType::Text))
                    .children(AllowedChildren::Any),
            )
    }

    #[test]
    fn well_formed_spec_validates_clean() {
        let r = registry_minimal();
        let spec = Widget::leaf("Card")
            .prop("title", "Welcome")
            .prop("subtitle", "Get started")
            .child(Widget::leaf("Text").prop("body", "Hello, world."))
            .child(
                Widget::leaf("Button")
                    .prop("label", "Click me")
                    .prop(
                        "on_click",
                        PropValue::Action {
                            id: "submit".into(),
                            payload: None,
                        },
                    ),
            );
        assert!(is_valid(&spec, &r).is_ok());
    }

    #[test]
    fn unknown_widget_is_reported_and_descent_stops() {
        let r = registry_minimal();
        let spec = Widget::leaf("Card").prop("title", "ok").child(
            Widget::leaf("MysteryBox")
                .child(Widget::leaf("AnotherUnknown")),
        );
        let errs = validate(&spec, &r);
        // One error for MysteryBox; the deeper unknown is not reported
        // because we stop descending when the schema is unknown.
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0],
            ValidationError::UnknownWidget { ref name, .. } if name == "MysteryBox"
        ));
    }

    #[test]
    fn missing_required_prop_is_reported() {
        let r = registry_minimal();
        let spec = Widget::leaf("Card"); // missing `title`
        let errs = validate(&spec, &r);
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0],
            ValidationError::MissingProp { ref widget, ref prop, .. }
                if widget == "Card" && prop == "title"
        ));
    }

    #[test]
    fn wrong_prop_type_is_reported() {
        let r = registry_minimal();
        let spec = Widget::leaf("Text").prop("body", 42_i64);
        let errs = validate(&spec, &r);
        assert_eq!(errs.len(), 1);
        match &errs[0] {
            ValidationError::WrongPropType {
                widget,
                prop,
                expected,
                got,
                ..
            } => {
                assert_eq!(widget, "Text");
                assert_eq!(prop, "body");
                assert_eq!(*expected, PropType::Text);
                assert_eq!(*got, PropType::Number);
            }
            other => panic!("expected WrongPropType, got {other:?}"),
        }
    }

    #[test]
    fn extra_prop_is_reported_in_strict_mode() {
        let r = registry_minimal();
        let spec = Widget::leaf("Text")
            .prop("body", "hi")
            .prop("color", "red"); // not declared on Text
        let errs = validate(&spec, &r);
        assert_eq!(errs.len(), 1);
        assert!(matches!(
            errs[0],
            ValidationError::ExtraProp { ref prop, .. } if prop == "color"
        ));
    }

    #[test]
    fn allow_extra_props_relaxes_the_strict_check() {
        let r = Registry::new().with(
            WidgetSchema::new("Box", "flexible container")
                .prop("title", PropSchema::optional(PropType::Text))
                .permit_extra_props(),
        );
        let spec = Widget::leaf("Box").prop("data-test", "x");
        assert!(is_valid(&spec, &r).is_ok());
    }

    #[test]
    fn disallowed_child_is_reported() {
        let r = registry_minimal();
        // Tabs only accepts Tab children.
        let spec = Widget::leaf("Tabs").child(Widget::leaf("Text").prop("body", "oops"));
        let errs = validate(&spec, &r);
        assert!(errs.iter().any(|e| matches!(
            e,
            ValidationError::DisallowedChild { parent, child, .. }
                if parent == "Tabs" && child == "Text"
        )));
    }

    #[test]
    fn child_errors_descend_even_after_disallowed() {
        // A disallowed child still gets its own errors reported, so the
        // author sees the whole picture in one pass.
        let r = registry_minimal();
        let bad_text_inside_tabs = Widget::leaf("Tabs").child(Widget::leaf("Text")); // also missing 'body'
        let errs = validate(&bad_text_inside_tabs, &r);
        assert_eq!(errs.len(), 2);
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::DisallowedChild { .. })));
        assert!(errs
            .iter()
            .any(|e| matches!(e, ValidationError::MissingProp { prop, .. } if prop == "body")));
    }

    #[test]
    fn any_type_accepts_anything() {
        let r = Registry::new().with(
            WidgetSchema::new("Meta", "free-form metadata")
                .prop("payload", PropSchema::required(PropType::Any)),
        );
        for value in [
            PropValue::Text("hi".into()),
            PropValue::Number(3.14),
            PropValue::Bool(true),
            PropValue::Null,
        ] {
            let spec = Widget::leaf("Meta").prop("payload", PropValue::clone(&value));
            assert!(is_valid(&spec, &r).is_ok(), "value {value:?} should pass Any");
        }
    }

    #[test]
    fn pre_order_traversal_finds_all_errors_in_one_pass() {
        let r = registry_minimal();
        let spec = Widget::leaf("Card")
            // missing title (1)
            .child(Widget::leaf("Text")) // missing body (2)
            .child(Widget::leaf("Button").prop("label", "x")); // missing on_click (3)
        let errs = validate(&spec, &r);
        assert_eq!(errs.len(), 3, "expected three errors, got {errs:?}");
    }

    #[test]
    fn widget_round_trips_through_json() {
        let spec = Widget::leaf("Card")
            .prop("title", "hi")
            .child(
                Widget::leaf("Button").prop("label", "go").prop(
                    "on_click",
                    PropValue::Action {
                        id: "submit".into(),
                        payload: Some(Box::new(PropValue::Object({
                            let mut o = BTreeMap::new();
                            o.insert("draft".into(), PropValue::Bool(true));
                            o
                        }))),
                    },
                ),
            );
        let json = serde_json::to_string(&spec).expect("ser");
        let back: Widget = serde_json::from_str(&json).expect("deser");
        assert_eq!(back, spec);
    }

    #[test]
    fn empty_children_field_does_not_serialise() {
        // skip_serializing_if = "Vec::is_empty" means a leaf widget's
        // wire form has no `children` key — old readers see the
        // expected shape.
        let spec = Widget::leaf("Text").prop("body", "hi");
        let json = serde_json::to_string(&spec).expect("ser");
        assert!(!json.contains("\"children\""), "got: {json}");
    }

    #[test]
    fn registry_round_trips_through_json() {
        let r = registry_minimal();
        let bytes = serde_json::to_vec(&r).expect("ser");
        let back: Registry = serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back.len(), r.len());
        assert!(back.get("Card").is_some());
    }

    #[test]
    fn path_is_pre_order_and_human_readable() {
        let r = Registry::new().with(
            WidgetSchema::new("Box", "container")
                .children(AllowedChildren::Any)
                .prop("title", PropSchema::required(PropType::Text)),
        );
        let spec = Widget::leaf("Box")
            .prop("title", "ok")
            .child(Widget::leaf("Box")); // missing 'title' at nested path
        let errs = validate(&spec, &r);
        assert_eq!(errs.len(), 1);
        if let ValidationError::MissingProp { path, .. } = &errs[0] {
            assert_eq!(path, "$.children[0]");
        } else {
            panic!("unexpected: {errs:?}");
        }
    }

    #[test]
    fn action_carries_optional_payload() {
        let a = PropValue::Action {
            id: "submit".into(),
            payload: None,
        };
        let j = serde_json::to_value(&a).expect("ser");
        // Should serialise as { "type": "action", "value": { "id": "submit", "payload": null } }
        assert_eq!(j["type"], "action");
        assert_eq!(j["value"]["id"], "submit");
    }
}
