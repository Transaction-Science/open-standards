//! NIP-13 proof of work.
//!
//! The PoW metric is the number of leading zero bits on the event id.
//! Miners increment a `["nonce", "<n>", "<target>"]` tag until they
//! produce an id with the required difficulty.

use crate::error::NostrError;
use crate::event::{Event, UnsignedEvent};
use crate::keys::{NostrSecretKey, hex_decode};

/// Count leading zero bits of `id` (big-endian).
pub fn leading_zero_bits(id: &[u8; 32]) -> u32 {
    let mut count = 0u32;
    for b in id {
        if *b == 0 {
            count += 8;
            continue;
        }
        count += b.leading_zeros();
        break;
    }
    count
}

/// Compute the PoW difficulty of a signed event's id.
pub fn event_difficulty(event: &Event) -> Result<u32, NostrError> {
    let id = hex_decode(&event.id)?;
    if id.len() != 32 {
        return Err(NostrError::InvalidEvent("id is not 32 bytes".into()));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&id);
    Ok(leading_zero_bits(&arr))
}

/// Verify that an event satisfies the requested `target` leading-zero bits.
///
/// Per NIP-13 the event MUST also declare the target via a
/// `["nonce", <n>, <target>]` tag, but we leave that check to callers
/// who may want to be lenient on the target advertisement.
pub fn verify(event: &Event, target: u32) -> Result<(), NostrError> {
    let have = event_difficulty(event)?;
    if have < target {
        return Err(NostrError::InsufficientPow {
            have,
            want: target,
        });
    }
    Ok(())
}

/// Mine an event until its id satisfies `target` leading-zero bits, or
/// the iteration budget is exhausted. The `created_at` field is left
/// unchanged; only the nonce tag is mutated.
///
/// `max_iter` bounds how long mining can run for. A typical target of
/// 20 bits requires ~1M iterations.
pub fn mine(
    base: UnsignedEvent,
    sk: &NostrSecretKey,
    target: u32,
    max_iter: u64,
) -> Result<Event, NostrError> {
    if target > 256 {
        return Err(NostrError::Crypto("target exceeds 256 bits".into()));
    }
    let mut nonce: u64 = 0;
    while nonce < max_iter {
        let mut candidate = base.clone();
        candidate.tags.push(vec![
            "nonce".to_string(),
            nonce.to_string(),
            target.to_string(),
        ]);
        let id = candidate.id();
        if leading_zero_bits(&id) >= target {
            return candidate.sign(sk);
        }
        nonce += 1;
    }
    Err(NostrError::InsufficientPow {
        have: 0,
        want: target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_zero_counts() {
        let mut id = [0u8; 32];
        assert_eq!(leading_zero_bits(&id), 256);
        id[0] = 0x0f; // 4 leading zero bits
        assert_eq!(leading_zero_bits(&id), 4);
        id[0] = 0x80; // 0 leading zero bits
        assert_eq!(leading_zero_bits(&id), 0);
    }

    #[test]
    fn mine_low_target() {
        let sk = NostrSecretKey::generate();
        let base = UnsignedEvent::new(sk.public_key(), 1, "mined", 1_700_000_000);
        let event = mine(base, &sk, 4, 1024).expect("mine");
        verify(&event, 4).expect("verify");
    }
}
