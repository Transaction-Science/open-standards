# joule-ui

A typed declarative protocol for **agent-emitted user interfaces**.

joule-ui is the sibling to [JouleClaw](../jouleclaw/) for the moment when an
agent's resolution has to become something a human can see and interact
with. The agent emits a **widget tree** — `(name, props, children)`,
recursively — against a **widget registry** the host has published. The
host renders. No React in the spec, no HTML over the wire, no agent
holding the renderer.

The shape the field has converged on for this is *declarative components
against a registry*: agents are far more reliable emitting structured
data than emitting code, and renderers are far easier to harden when they
own their own widgets. joule-ui pins that shape as a protocol; the
renderer is product surface, owned by the consumer.

## What ships in v1

- `joule-ui-rs/crates/joule-ui-core/` — the protocol types (`Widget`,
  `PropValue`, `WidgetSchema`, `Registry`) and the recursive validator.
  Pure Rust, no IO, JSON-serialisable.

What is **not** here in v1:

- A renderer. The renderer is owned by the host (browser app, native
  shell, terminal UI). joule-ui only validates the spec.
- A transport. The validated spec can ride MCP, REST, SSE, gRPC — any
  bytes-out protocol. v1 is transport-agnostic.
- A "model emits React" path. The agent emits a declarative tree against
  the registry; arbitrary code emission is out of scope.
- Streaming deltas. v1 is whole-tree; streaming primitives slot in
  later without changing the protocol shape.

## Why a sibling, not inside JouleClaw

The spec layer is protocol — small, stable, conformant. The renderer
registry is product surface — many of them, evolving fast. Splitting
them protects the *tools for use* line: JouleClaw stays the runtime
substrate; joule-ui is the UI protocol; consumers ship their own
widgets.

## Licence

Specification text and protocol document: CC-BY-4.0 (see [`LICENSE`](LICENSE)).
Reference implementation in `joule-ui-rs/`: Apache-2.0.
