//! Body synthesizers — `I=Bodies` on the Synthesis I axis.
//!
//! Most Joule tiers return tokens or signals — outputs that exist
//! only in the runtime's memory until the caller acts on them. A
//! body tier is different: it touches the world directly. Sending
//! an email, executing a payment, deploying code, actuating a
//! servo, writing to a file the user can see.
//!
//! Two properties make body tiers structurally distinct:
//!
//!   1. **Irreversibility.** Once committed, the action cannot be
//!      undone by the runtime alone. A wire transfer is sent. A
//!      packet is in flight. A robot arm has moved.
//!
//!   2. **External cost.** The joule cost of a body action is not
//!      the cost of the inference that produced the plan — it's the
//!      cost of the *action itself* (network, electricity, the
//!      downstream service's resource consumption).
//!
//! The Joule mechanism:
//!
//!   * A `BodyTier` produces a `Plan` (what would be done) rather
//!     than acting immediately.
//!   * The caller decides whether to `commit(plan)`. Commit is the
//!     irreversible step.
//!   * Every commit produces a `VerificationToken` (since body
//!     outcomes are always V=Delayed by nature).
//!
//! The plan/commit split makes body tiers safe-by-default: dispatching
//! a query through a body tier doesn't yet touch the world. The
//! caller has to make an explicit commit call. The cascade can use
//! body tiers in "dry-run" mode for testing or validation without
//! the runtime ever causing a side effect.

use crate::coord::Coord;
use crate::types::{Answer, AnswerError, AnswerOutput, Query, TierId};
use crate::verification::{
    VerificationLedger, VerificationStatus, VerificationToken,
};

/// A plan the tier proposes. Calling `commit` makes it real.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Human-readable description of what would happen.
    pub description: String,
    /// Opaque payload the body tier needs to execute the plan.
    /// Format is tier-specific; the runtime treats it as bytes.
    pub payload: Vec<u8>,
    /// Estimated joule cost of *the action itself* — not the inference
    /// that produced the plan. For an email send: SMTP traffic +
    /// downstream server cost. For a wire transfer: settlement
    /// computation + network. For a robot arm: motor electricity.
    pub action_joules: f64,
    /// Whether this plan is reversible in principle. Most body
    /// actions are not. A few (revocable email sends, undo-able file
    /// writes) are. The flag is informational; the runtime treats
    /// every commit as irreversible from its own perspective.
    pub reversible: bool,
}

/// The trait body synthesizers implement.
///
/// A body tier extends the `Tier` contract: it doesn't act on
/// `try_answer`, it just *plans*. The caller commits explicitly.
pub trait BodyTier: Send {
    fn id(&self) -> TierId;
    fn coord(&self) -> Coord;

    /// Produce a plan for a query, without acting. This is what the
    /// cascade calls when routing through the tier.
    fn plan(&mut self, q: &Query) -> Result<Plan, AnswerError>;

    /// Commit a plan. Performs the irreversible action. Returns the
    /// observed cost of *the action itself* (which may differ from
    /// `plan.action_joules` — that's why we have calibration).
    ///
    /// A real-world implementation calls out to the network, the
    /// filesystem, a motor controller, etc. This trait keeps the
    /// signature synchronous; long-running commits can be backed by
    /// the `ActiveTier` machinery if they need to be observed over
    /// time.
    fn commit(&mut self, plan: &Plan) -> Result<f64, BodyError>;

    /// Whether this tier has been put in dry-run mode. In dry-run,
    /// `commit` returns success without touching the world — useful
    /// for testing the cascade end-to-end without side effects.
    fn is_dry_run(&self) -> bool;
}

/// Failure mode for a body commit.
#[derive(Debug, Clone)]
pub enum BodyError {
    /// The action was attempted and failed (network error, recipient
    /// unreachable, permission denied).
    Failed { reason: String },
    /// Commit was refused before the action — preconditions not met
    /// (no API key, budget exceeded for the action class, etc.).
    Refused { reason: String },
    /// Dry-run mode rejected the commit (a no-op response to the
    /// commit call, used in tests).
    DryRunBlocked,
}

impl std::fmt::Display for BodyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { reason } => write!(f, "body action failed: {}", reason),
            Self::Refused { reason } => write!(f, "body commit refused: {}", reason),
            Self::DryRunBlocked => write!(f, "dry-run mode: commit blocked"),
        }
    }
}

impl std::error::Error for BodyError {}

/// A wrapper that turns a body tier into a cascade-compatible
/// `(plan, token)` answer. The plan is returned as the answer's
/// output; the verification token tracks the eventual commit
/// outcome.
///
/// Callers can:
///   1. Inspect the plan via `answer.output`
///   2. Decide whether to commit
///   3. Call `commit_plan(token, ...)` to perform the action
///   4. The verification ledger records the outcome
pub struct BodyDispatch {
    pub body: Box<dyn BodyTier>,
    /// Ledger to issue verification tokens against.
    ledger: VerificationLedger,
    /// Pending plans keyed by token, so `commit_plan` can find them.
    pending_plans: std::collections::HashMap<VerificationToken, Plan>,
    pub commits_succeeded: u64,
    pub commits_failed: u64,
    /// Safety policy gating commits. Default: permissive.
    pub policy: crate::body_safety::SafetyPolicy,
    /// Rolling state used to enforce the policy (recent commits,
    /// recent dry-runs, recent budget usage).
    safety_state: crate::body_safety::SafetyState,
}

impl BodyDispatch {
    pub fn new(body: Box<dyn BodyTier>) -> Self {
        Self {
            body,
            ledger: VerificationLedger::new(),
            pending_plans: std::collections::HashMap::new(),
            commits_succeeded: 0,
            commits_failed: 0,
            policy: crate::body_safety::SafetyPolicy::permissive(),
            safety_state: crate::body_safety::SafetyState::new(),
        }
    }

    /// Configure the safety policy. Returns self for chaining.
    pub fn with_policy(mut self, policy: crate::body_safety::SafetyPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Record a successful dry-run for a plan. The caller is
    /// responsible for actually running the body tier in dry-run
    /// mode (`is_dry_run() = true`); this just notes that the run
    /// happened so a future commit of the same plan can pass the
    /// dry-run gate.
    pub fn record_dry_run(&mut self, plan: &Plan) {
        let hash = crate::body_safety::hash_plan(plan);
        self.safety_state.record_dry_run(hash);
    }

    pub fn safety_state(&self) -> &crate::body_safety::SafetyState {
        &self.safety_state
    }

    /// Produce an answer carrying the plan. The answer is `Pending`
    /// until the caller calls `commit_plan(token)`.
    pub fn plan_for(&mut self, q: &Query) -> Result<(VerificationToken, Answer), AnswerError> {
        let plan = self.body.plan(q)?;
        let tier_id = self.body.id();
        let coord = Some(self.body.coord());

        // Issue verification token with the *estimated* action cost.
        let token = self.ledger.issue(
            tier_id, coord.clone(),
            plan.action_joules, plan.action_joules,
        );
        self.pending_plans.insert(token, plan.clone());

        let answer = Answer {
            output: AnswerOutput::Text(format!("[plan] {}", plan.description)),
            tier_used: tier_id,
            // Joules spent so far = planning cost. The action cost
            // lands when commit is called.
            joules_spent: 0.0,
            confidence: 0.9,
            trace: crate::types::ExecutionTrace::default(),
            verification: VerificationStatus::Pending(token),
        };
        Ok((token, answer))
    }

    /// Commit a previously-planned action. Performs the irreversible
    /// step. Returns the actual joule cost of the action.
    ///
    /// After commit, the verification ledger has resolved the token;
    /// calibration can read the (estimated, actual) pair.
    pub fn commit_plan(&mut self, token: VerificationToken)
        -> Result<f64, BodyError>
    {
        let plan = self.pending_plans.remove(&token)
            .ok_or(BodyError::Refused {
                reason: "unknown token".into()
            })?;

        // Policy gate. If the policy denies, return Refused without
        // touching the body tier. The ledger entry is resolved as a
        // failure so calibration sees the policy denial.
        let plan_hash = crate::body_safety::hash_plan(&plan);
        if let Err(deny) = self.safety_state.check_commit(
            &self.policy, plan_hash, plan.action_joules,
        ) {
            self.commits_failed += 1;
            let reason = format!("{}", deny);
            let _ = self.ledger.resolve(
                token,
                &crate::verification::VerificationOutcome::Failure {
                    actual_joules: 0.0,
                    reason: reason.clone(),
                },
            );
            return Err(BodyError::Refused { reason });
        }

        let result = self.body.commit(&plan);

        // Record outcome to ledger.
        let outcome = match &result {
            Ok(actual) => {
                self.commits_succeeded += 1;
                // Record the commit in safety state so rate/budget
                // accounting tracks reality.
                self.safety_state.record_commit(*actual);
                crate::verification::VerificationOutcome::Success {
                    actual_joules: *actual,
                }
            }
            Err(BodyError::Failed { reason }) => {
                self.commits_failed += 1;
                crate::verification::VerificationOutcome::Failure {
                    actual_joules: plan.action_joules,  // estimate
                    reason: reason.clone(),
                }
            }
            Err(other) => {
                self.commits_failed += 1;
                crate::verification::VerificationOutcome::Failure {
                    actual_joules: 0.0,
                    reason: format!("{}", other),
                }
            }
        };
        let _ = self.ledger.resolve(token, &outcome);

        result
    }

    /// Cancel a pending plan without committing.
    pub fn cancel_plan(&mut self, token: VerificationToken) -> bool {
        if self.pending_plans.remove(&token).is_some() {
            let _ = self.ledger.resolve(
                token,
                &crate::verification::VerificationOutcome::Timeout,
            );
            true
        } else {
            false
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending_plans.len()
    }

    pub fn ledger(&self) -> &VerificationLedger {
        &self.ledger
    }
}

// ============================================================
// Example body tier — file write
// ============================================================

/// A body tier that writes text to a file. Concrete, simple,
/// real-world example of `I=Bodies` — a side effect that touches
/// the host system.
pub struct FileWriter {
    pub directory: std::path::PathBuf,
    pub id: TierId,
    pub coord: Coord,
    dry_run: bool,
    pub writes: u64,
}

impl FileWriter {
    pub fn new(directory: impl AsRef<std::path::Path>, id: TierId, coord: Coord) -> Self {
        Self {
            directory: directory.as_ref().to_path_buf(),
            id, coord,
            dry_run: false,
            writes: 0,
        }
    }

    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }
}

impl BodyTier for FileWriter {
    fn id(&self) -> TierId { self.id }
    fn coord(&self) -> Coord { self.coord.clone() }
    fn is_dry_run(&self) -> bool { self.dry_run }

    fn plan(&mut self, q: &Query) -> Result<Plan, AnswerError> {
        let text = match &q.input {
            crate::types::QueryInput::Text(s) => s.clone(),
            _ => return Err(AnswerError::TierFailed {
                tier: self.id,
                cause: "file writer only accepts text input".into(),
            }),
        };
        // Parse "write <text> to <filename>" format.
        let (content, filename) = parse_write_query(&text)
            .ok_or_else(|| AnswerError::TierFailed {
                tier: self.id,
                cause: "expected: write <text> to <file.ext>".into(),
            })?;

        // Build the full path (under self.directory).
        let path = self.directory.join(&filename);
        let payload = format!("{}\n{}", path.display(), content).into_bytes();

        Ok(Plan {
            description: format!(
                "write {} bytes to {}",
                content.len(), path.display()),
            payload,
            // Cost model: file IO ~50 nJ + per-byte cost.
            action_joules: 5e-8 + (content.len() as f64) * 1e-11,
            reversible: false,  // truncation; previous contents lost.
        })
    }

    fn commit(&mut self, plan: &Plan) -> Result<f64, BodyError> {
        if self.dry_run {
            return Err(BodyError::DryRunBlocked);
        }
        let payload_str = std::str::from_utf8(&plan.payload)
            .map_err(|e| BodyError::Failed {
                reason: format!("non-utf8 payload: {}", e)
            })?;
        let mut parts = payload_str.splitn(2, '\n');
        let path_str = parts.next()
            .ok_or_else(|| BodyError::Failed {
                reason: "malformed payload".into()
            })?;
        let content = parts.next().unwrap_or("");
        std::fs::write(path_str, content)
            .map_err(|e| BodyError::Failed {
                reason: format!("write failed: {}", e)
            })?;
        self.writes += 1;
        Ok(plan.action_joules)
    }
}

/// Parse "write <text> to <filename>" — returns (text, filename).
fn parse_write_query(s: &str) -> Option<(String, String)> {
    let lower = s.to_ascii_lowercase();
    if !lower.starts_with("write ") { return None; }
    // Find " to " separator.
    let rest = &s["write ".len()..];
    let lower_rest = &lower["write ".len()..];
    let pos = lower_rest.rfind(" to ")?;
    let text = rest[..pos].trim().to_string();
    let filename = rest[pos + " to ".len()..].trim().to_string();
    if text.is_empty() || filename.is_empty() { return None; }
    Some((text, filename))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{Zone, Entity, Thermo, Interface, Verify, Encoding, NamedPrimitive, PrimitiveSet};
    use crate::types::{L1Primitive, QueryInput, JouleBudget, QualityFloor, ContextRef};

    fn body_coord() -> Coord {
        Coord::new(
            Zone::Z1, Entity::Reactive, Thermo::L1_Measure,
            Interface::Bodies,        // ← the load-bearing axis
            Verify::Delayed,           // outcome lands in the world
            Encoding::None,
        ).with_primitives(PrimitiveSet::of(&[NamedPrimitive::ToolCall]))
    }

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn tmpdir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "joule-body-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos(),
        ))
    }

    #[test]
    fn body_tier_has_bodies_interface() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let writer = FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord());
        let c = writer.coord();
        assert_eq!(c.interface, Interface::Bodies);
        assert_eq!(c.verify, Verify::Delayed);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_does_not_touch_world() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord());
        let q = text_query("write hello to test.txt");
        let plan = writer.plan(&q).unwrap();

        // The file should NOT exist after planning.
        let path = dir.join("test.txt");
        assert!(!path.exists(),
            "plan() must not touch the world; file exists at {:?}", path);
        assert!(plan.description.contains("test.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn commit_writes_file() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord());
        let q = text_query("write hello-world to greeting.txt");
        let plan = writer.plan(&q).unwrap();
        let cost = writer.commit(&plan).unwrap();

        let path = dir.join("greeting.txt");
        assert!(path.exists(), "commit must touch the world");
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "hello-world");
        assert!(cost > 0.0);
        assert_eq!(writer.writes, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dry_run_refuses_commit() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord())
            .dry_run();
        let q = text_query("write x to y.txt");
        let plan = writer.plan(&q).unwrap();
        let result = writer.commit(&plan);
        assert!(matches!(result, Err(BodyError::DryRunBlocked)));

        let path = dir.join("y.txt");
        assert!(!path.exists(), "dry-run must not touch the world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn body_dispatch_lifecycle() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let writer = Box::new(FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord()));
        let mut dispatch = BodyDispatch::new(writer);

        // Plan.
        let q = text_query("write data to output.txt");
        let (token, answer) = dispatch.plan_for(&q).unwrap();
        assert_eq!(answer.verification, VerificationStatus::Pending(token));
        assert!(answer.output.to_string().contains("output.txt") ||
            matches!(&answer.output, AnswerOutput::Text(s) if s.contains("output.txt")));
        assert_eq!(dispatch.pending_count(), 1);

        // File doesn't exist yet.
        let path = dir.join("output.txt");
        assert!(!path.exists());

        // Commit.
        let actual_cost = dispatch.commit_plan(token).unwrap();
        assert!(actual_cost > 0.0);
        assert!(path.exists(), "after commit, file must exist");
        assert_eq!(dispatch.commits_succeeded, 1);
        assert_eq!(dispatch.pending_count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancel_pending_plan_does_not_commit() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let writer = Box::new(FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord()));
        let mut dispatch = BodyDispatch::new(writer);
        let q = text_query("write x to cancel-me.txt");
        let (token, _) = dispatch.plan_for(&q).unwrap();
        assert_eq!(dispatch.pending_count(), 1);

        let cancelled = dispatch.cancel_plan(token);
        assert!(cancelled);
        assert_eq!(dispatch.pending_count(), 0);
        let path = dir.join("cancel-me.txt");
        assert!(!path.exists(), "cancelled plan must not touch the world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plan_rejects_malformed_query() {
        let dir = tmpdir();
        std::fs::create_dir_all(&dir).unwrap();
        let mut writer = FileWriter::new(&dir,
            TierId::L1(L1Primitive::Execute), body_coord());
        let q = text_query("this is not a write instruction");
        let result = writer.plan(&q);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_write_query_recognizes_formats() {
        assert_eq!(parse_write_query("write hello to foo.txt"),
            Some(("hello".into(), "foo.txt".into())));
        assert_eq!(parse_write_query("write the quick brown fox to /tmp/test"),
            Some(("the quick brown fox".into(), "/tmp/test".into())));
        // Missing parts.
        assert_eq!(parse_write_query("write to foo.txt"), None);
        assert_eq!(parse_write_query("write hello to"), None);
        assert_eq!(parse_write_query("hello to foo.txt"), None);
    }
}

impl std::fmt::Display for AnswerOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnswerOutput::Text(s) => write!(f, "{}", s),
            AnswerOutput::Structured(b) => write!(f, "<{} structured bytes>", b.len()),
            AnswerOutput::Refused(r) => write!(f, "<refused: {:?}>", r),
        }
    }
}
