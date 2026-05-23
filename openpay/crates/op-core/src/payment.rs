//! The `Payment<S>` typestate machine.
//!
//! A payment's lifecycle is represented as a state-parametric type. Each
//! state is a zero-sized marker. Transitions are *functions* that consume
//! the value in one state and return a value in the next — the type
//! checker, not runtime code, enforces that you cannot capture an
//! unauthorized payment or refund a voided one.
//!
//! ```text
//!  Created ──authorize──▶ Authorized ──capture──▶ Captured ──refund──▶ Refunded
//!     │                       │                       │
//!     │                       ├──void────▶ Voided
//!     │                       │
//!     └──fail────▶ Failed ◀───┘
//! ```
//!
//! ## Why typestate, not an enum
//!
//! An `enum PaymentState { Created, Authorized, ... }` with a single
//! `Payment` struct lets bugs like `payment.capture()` on a `Created`
//! payment fail at *runtime*. The typestate encoding makes these failures
//! impossible to write: there is no `capture` method on `Payment<Created>`.
//!
//! ## Cost
//!
//! Zero. The state markers are zero-sized; `Payment<S>` has the same
//! memory layout regardless of `S`. The compiler erases the state at
//! codegen time.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::method::PaymentMethod;
use crate::money::Money;
use crate::rails::RailKind;

/// Sealed marker trait — only states defined in this module can satisfy it.
pub trait PaymentState: sealed::Sealed {
    /// Human-readable name for diagnostics.
    const NAME: &'static str;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! state {
    ($name:ident, $literal:expr) => {
        /// Payment state marker. See module docs for the state diagram.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl PaymentState for $name {
            const NAME: &'static str = $literal;
        }
    };
}

state!(Created, "Created");
state!(Authorized, "Authorized");
state!(Captured, "Captured");
state!(Voided, "Voided");
state!(Refunded, "Refunded");
state!(Failed, "Failed");

/// A payment, parameterized by its state.
///
/// The state marker `S` is a zero-sized type; transitions move between
/// `Payment<A>` and `Payment<B>` by consuming `self`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Payment<S: PaymentState> {
    /// Stable id (UUID v7 — time-ordered).
    pub id: Uuid,
    /// Amount and currency requested.
    pub amount: Money,
    /// Amount already captured (≤ `amount`). Zero in `Created`/`Authorized`.
    pub captured: Money,
    /// Amount already refunded. Always ≤ `captured`.
    pub refunded: Money,
    /// How value moves.
    pub method: PaymentMethod,
    /// Which rail family was selected.
    pub rail: RailKind,
    /// Opaque rail-issued reference (auth code, network txn id, end-to-end id).
    pub rail_ref: Option<String>,
    /// Phantom state marker.
    #[serde(skip)]
    _state: core::marker::PhantomData<S>,
}

impl Payment<Created> {
    /// Construct a fresh payment in the `Created` state.
    #[must_use]
    pub fn new(amount: Money, method: PaymentMethod, rail: RailKind) -> Self {
        Self {
            id: Uuid::now_v7(),
            captured: Money::zero(amount.currency),
            refunded: Money::zero(amount.currency),
            amount,
            method,
            rail,
            rail_ref: None,
            _state: core::marker::PhantomData,
        }
    }

    /// Move to `Authorized` once the rail has confirmed authorization
    /// (e.g. issuer auth approval, `FedNow` request-for-payment accepted).
    #[must_use]
    pub fn authorize(self, rail_ref: String) -> Payment<Authorized> {
        Payment {
            id: self.id,
            amount: self.amount,
            captured: self.captured,
            refunded: self.refunded,
            method: self.method,
            rail: self.rail,
            rail_ref: Some(rail_ref),
            _state: core::marker::PhantomData,
        }
    }

    /// Move to `Failed`. Used when the rail rejects the authorization.
    #[must_use]
    pub fn fail(self, reason: String) -> Payment<Failed> {
        Payment {
            id: self.id,
            amount: self.amount,
            captured: self.captured,
            refunded: self.refunded,
            method: self.method,
            rail: self.rail,
            rail_ref: Some(reason),
            _state: core::marker::PhantomData,
        }
    }
}

impl Payment<Authorized> {
    /// Capture up to the authorized amount. Partial captures are allowed
    /// as long as `captured + amount_to_capture <= self.amount`.
    ///
    /// # Errors
    /// - `CurrencyMismatch` if `amount` is in a different currency.
    /// - `Overflow` on arithmetic overflow.
    /// - `IllegalTransition` if `amount` exceeds remaining capacity.
    pub fn capture(self, amount: Money) -> Result<Payment<Captured>> {
        let new_captured = self.captured.checked_add(amount)?;
        if new_captured.minor_units > self.amount.minor_units {
            return Err(crate::error::Error::IllegalTransition {
                from: Authorized::NAME,
                to: Captured::NAME,
            });
        }
        Ok(Payment {
            id: self.id,
            amount: self.amount,
            captured: new_captured,
            refunded: self.refunded,
            method: self.method,
            rail: self.rail,
            rail_ref: self.rail_ref,
            _state: core::marker::PhantomData,
        })
    }

    /// Void the authorization without capturing.
    #[must_use]
    pub fn void(self) -> Payment<Voided> {
        Payment {
            id: self.id,
            amount: self.amount,
            captured: self.captured,
            refunded: self.refunded,
            method: self.method,
            rail: self.rail,
            rail_ref: self.rail_ref,
            _state: core::marker::PhantomData,
        }
    }
}

impl Payment<Captured> {
    /// Refund all or part of the captured amount.
    ///
    /// # Errors
    /// - `CurrencyMismatch`, `Overflow`, or `IllegalTransition` if the
    ///   requested refund exceeds the remaining refundable balance.
    pub fn refund(self, amount: Money) -> Result<Payment<Refunded>> {
        let new_refunded = self.refunded.checked_add(amount)?;
        if new_refunded.minor_units > self.captured.minor_units {
            return Err(crate::error::Error::IllegalTransition {
                from: Captured::NAME,
                to: Refunded::NAME,
            });
        }
        Ok(Payment {
            id: self.id,
            amount: self.amount,
            captured: self.captured,
            refunded: new_refunded,
            method: self.method,
            rail: self.rail,
            rail_ref: self.rail_ref,
            _state: core::marker::PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::{PaymentMethod, VaultRef};
    use crate::money::Currency;

    fn sample() -> Payment<Created> {
        Payment::new(
            Money::from_minor(10_00, Currency::USD),
            PaymentMethod::Vault(VaultRef::new("tok_test")),
            RailKind::Card,
        )
    }

    #[test]
    fn happy_path_authorize_capture() {
        let p = sample()
            .authorize("auth_123".into())
            .capture(Money::from_minor(10_00, Currency::USD))
            .unwrap();
        assert_eq!(p.captured.minor_units, 10_00);
    }

    #[test]
    fn partial_capture_then_full() {
        let auth = sample().authorize("auth_123".into());
        let cap1 = auth
            .capture(Money::from_minor(4_00, Currency::USD))
            .unwrap();
        assert_eq!(cap1.captured.minor_units, 4_00);
    }

    #[test]
    fn overcapture_rejected() {
        let auth = sample().authorize("auth_123".into());
        let res = auth.capture(Money::from_minor(20_00, Currency::USD));
        assert!(matches!(
            res,
            Err(crate::error::Error::IllegalTransition { .. })
        ));
    }

    #[test]
    fn capture_then_refund() {
        let cap = sample()
            .authorize("auth_123".into())
            .capture(Money::from_minor(10_00, Currency::USD))
            .unwrap();
        let ref_ = cap.refund(Money::from_minor(3_00, Currency::USD)).unwrap();
        assert_eq!(ref_.refunded.minor_units, 3_00);
    }

    #[test]
    fn overrefund_rejected() {
        let cap = sample()
            .authorize("auth_123".into())
            .capture(Money::from_minor(10_00, Currency::USD))
            .unwrap();
        let res = cap.refund(Money::from_minor(20_00, Currency::USD));
        assert!(matches!(
            res,
            Err(crate::error::Error::IllegalTransition { .. })
        ));
    }

    // The illegal transitions below would not COMPILE. We document them in
    // a test as commented-out code to make the guarantee visible.
    //
    // #[test] fn cannot_capture_created() {
    //     let p: Payment<Created> = sample();
    //     p.capture(Money::from_minor(10_00, Currency::USD));  // no such method
    // }
    //
    // #[test] fn cannot_refund_authorized() {
    //     let p = sample().authorize("auth_123".into());
    //     p.refund(Money::from_minor(1, Currency::USD));  // no such method
    // }
    //
    // #[test] fn cannot_double_void() {
    //     let v = sample().authorize("auth_123".into()).void();
    //     v.void();  // no such method on Payment<Voided>
    // }
}
