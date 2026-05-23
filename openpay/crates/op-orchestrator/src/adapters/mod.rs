//! Concrete [`RailAdapter`](crate::RailAdapter) implementations.
//!
//! Three are shipped:
//!
//! - [`CardAdapter`] wraps any
//!   [`op_rails_card::CardAcquirer`] (Hyperswitch, Stripe, Adyen,
//!   Finix — anything that implements the trait).
//! - [`A2aAdapter`] wraps any
//!   [`op_rails_a2a::A2aAcquirer`] (`FedNow`, PIX, SEPA Instant —
//!   same pattern).
//! - [`CryptoAdapter`] wraps any
//!   [`op_rails_crypto::CryptoGateway`] (USDC on Solana / Base /
//!   Ethereum, EURC, PYUSD).
//!
//! Operators write more adapters as new rails come online; the
//! orchestrator never needs changes.

pub mod a2a;
pub mod card;
pub mod crypto;

pub use a2a::{A2aAdapter, MerchantBankProfile};
pub use card::CardAdapter;
pub use crypto::CryptoAdapter;
