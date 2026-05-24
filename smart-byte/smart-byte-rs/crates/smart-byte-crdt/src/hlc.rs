//! Hybrid Logical Clock (Kulkarni et al., 2014).
//!
//! HLC combines a physical wall-clock component with a logical counter
//! so that timestamps are:
//!
//! * Monotonically non-decreasing on each replica.
//! * Causally consistent across replicas (if event A happened-before
//!   event B then `hlc(A) < hlc(B)`).
//! * Approximately equal to wall-clock time, bounded by clock skew.
//!
//! The wall component is stored as milliseconds since the Unix epoch.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// 128-bit identifier for a participating replica. A `ReplicaId` is
/// opaque to the CRDT engine — only equality and ordering are used.
///
/// Wire format: big-endian 16-byte array (so it round-trips through
/// CBOR cleanly; `serde_cbor 0.11` does not natively encode `u128`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(from = "[u8; 16]", into = "[u8; 16]")]
pub struct ReplicaId(pub u128);

impl From<[u8; 16]> for ReplicaId {
    fn from(b: [u8; 16]) -> Self {
        Self(u128::from_be_bytes(b))
    }
}
impl From<ReplicaId> for [u8; 16] {
    fn from(r: ReplicaId) -> Self {
        r.0.to_be_bytes()
    }
}



impl ReplicaId {
    /// Construct a `ReplicaId` from a raw 128-bit value.
    pub const fn new(raw: u128) -> Self {
        Self(raw)
    }

    /// Derive a deterministic `ReplicaId` by hashing arbitrary bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let h = blake3::hash(bytes);
        let b = h.as_bytes();
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&b[..16]);
        Self(u128::from_be_bytes(raw))
    }
}

/// A point-in-time HLC timestamp.
///
/// `wall` is milliseconds since the Unix epoch. `logical` is the
/// per-millisecond logical counter used to disambiguate events that
/// share a wall reading. `node` ties the timestamp to its emitter,
/// providing a total order even for fully concurrent events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HybridLogicalClock {
    pub wall: u64,
    pub logical: u32,
    pub node: ReplicaId,
}

impl HybridLogicalClock {
    /// Zero timestamp on the given replica. Useful as a sentinel "no
    /// events seen yet" value.
    pub const fn zero(node: ReplicaId) -> Self {
        Self {
            wall: 0,
            logical: 0,
            node,
        }
    }
}

impl PartialOrd for HybridLogicalClock {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HybridLogicalClock {
    fn cmp(&self, other: &Self) -> Ordering {
        // Total order: (wall, logical, node).
        self.wall
            .cmp(&other.wall)
            .then_with(|| self.logical.cmp(&other.logical))
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Mutable HLC state owned by a replica.
#[derive(Clone, Debug)]
pub struct HlcClock {
    last: HybridLogicalClock,
    node: ReplicaId,
    /// Optional wall-clock source override; defaults to `std::time::SystemTime`.
    wall_source: WallSource,
}

#[derive(Clone, Debug)]
enum WallSource {
    System,
    Manual(u64),
}

impl HlcClock {
    /// Construct an HLC clock for the given replica, using the system
    /// wall clock.
    pub fn new(node: ReplicaId) -> Self {
        Self {
            last: HybridLogicalClock::zero(node),
            node,
            wall_source: WallSource::System,
        }
    }

    /// Construct an HLC clock with a manually-controlled wall source
    /// (useful in tests and for deterministic replay).
    pub fn with_manual_wall(node: ReplicaId, wall_ms: u64) -> Self {
        Self {
            last: HybridLogicalClock::zero(node),
            node,
            wall_source: WallSource::Manual(wall_ms),
        }
    }

    /// Advance a manual wall clock. Has no effect when the system clock
    /// is in use.
    pub fn set_manual_wall(&mut self, wall_ms: u64) {
        self.wall_source = WallSource::Manual(wall_ms);
    }

    fn read_wall(&self) -> u64 {
        match self.wall_source {
            WallSource::System => {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(self.last.wall)
            }
            WallSource::Manual(ms) => ms,
        }
    }

    /// Emit a new HLC timestamp for a locally generated event.
    pub fn now(&mut self) -> HybridLogicalClock {
        let wall = self.read_wall();
        let next = if wall > self.last.wall {
            HybridLogicalClock {
                wall,
                logical: 0,
                node: self.node,
            }
        } else {
            HybridLogicalClock {
                wall: self.last.wall,
                logical: self.last.logical.saturating_add(1),
                node: self.node,
            }
        };
        self.last = next;
        next
    }

    /// Update the clock after observing a remote timestamp.
    ///
    /// Returns a new HLC timestamp that strictly succeeds both the
    /// previously-emitted timestamp and the observed one.
    pub fn update(&mut self, observed: HybridLogicalClock) -> HybridLogicalClock {
        let wall = self.read_wall();
        let max_wall = wall.max(self.last.wall).max(observed.wall);

        let logical = if max_wall == self.last.wall && max_wall == observed.wall {
            self.last.logical.max(observed.logical).saturating_add(1)
        } else if max_wall == self.last.wall {
            self.last.logical.saturating_add(1)
        } else if max_wall == observed.wall {
            observed.logical.saturating_add(1)
        } else {
            0
        };

        let next = HybridLogicalClock {
            wall: max_wall,
            logical,
            node: self.node,
        };
        self.last = next;
        next
    }

    /// Replica id owning this clock.
    pub fn node(&self) -> ReplicaId {
        self.node
    }

    /// Last emitted timestamp.
    pub fn last(&self) -> HybridLogicalClock {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_monotonic() {
        let mut clock = HlcClock::with_manual_wall(ReplicaId::new(1), 100);
        let t0 = clock.now();
        let t1 = clock.now();
        let t2 = clock.now();
        assert!(t0 < t1);
        assert!(t1 < t2);
    }

    #[test]
    fn update_succeeds_remote_observation() {
        let mut clock = HlcClock::with_manual_wall(ReplicaId::new(1), 100);
        let local = clock.now();
        let remote = HybridLogicalClock {
            wall: 500,
            logical: 3,
            node: ReplicaId::new(2),
        };
        let merged = clock.update(remote);
        assert!(merged > local);
        assert!(merged > remote);
    }

    #[test]
    fn wall_clock_advance_resets_logical() {
        let mut clock = HlcClock::with_manual_wall(ReplicaId::new(1), 100);
        let _ = clock.now();
        let _ = clock.now();
        clock.set_manual_wall(200);
        let t = clock.now();
        assert_eq!(t.wall, 200);
        assert_eq!(t.logical, 0);
    }

    #[test]
    fn total_order_breaks_ties_by_node() {
        let a = HybridLogicalClock {
            wall: 1,
            logical: 1,
            node: ReplicaId::new(1),
        };
        let b = HybridLogicalClock {
            wall: 1,
            logical: 1,
            node: ReplicaId::new(2),
        };
        assert!(a < b);
    }

    #[test]
    fn replica_id_from_bytes_is_stable() {
        let r1 = ReplicaId::from_bytes(b"replica-A");
        let r2 = ReplicaId::from_bytes(b"replica-A");
        assert_eq!(r1, r2);
    }
}
