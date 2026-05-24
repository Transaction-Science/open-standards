//! ERC-4337 account abstraction: `UserOperation` v0.7 layout.
//!
//! v0.7 introduced the `PackedUserOperation` struct: the bundler
//! still sees fields as separate values, but the EntryPoint receives
//! them packed (with gas fields concatenated into single 32-byte
//! slots) for calldata-size reduction.
//!
//! This module models both the unpacked `UserOperation` (what
//! application code constructs) and the `PackedUserOperation`
//! (what gets handed to the EntryPoint). It does **not** compute
//! the user-op hash — that requires the EntryPoint address, chain
//! id, and EIP-712 typed-data hashing, which depends on operator
//! choice of EntryPoint deployment.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// EntryPoint version the user-op targets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntryPointVersion {
    /// EntryPoint v0.6 (legacy).
    V0_6,
    /// EntryPoint v0.7 (current). Uses `PackedUserOperation`.
    V0_7,
    /// EntryPoint v0.8 (forthcoming).
    V0_8,
}

impl EntryPointVersion {
    /// Canonical deployed contract address for the version on
    /// Ethereum mainnet (and most L2s — the EntryPoint is deployed
    /// at the same address everywhere via CREATE2).
    #[must_use]
    pub const fn canonical_address(self) -> &'static str {
        match self {
            Self::V0_6 => "0x5ff137d4b0fdcd49dca30c7cf57e578a026d2789",
            Self::V0_7 => "0x0000000071727de22e5e9d8baf0edac6f37da032",
            Self::V0_8 => "0x4337084d9e255ff0702461cf8895ce9e3b5ff108",
        }
    }
}

/// ERC-4337 v0.7 unpacked UserOperation.
///
/// Field names and order match the EntryPoint v0.7 spec
/// (`PackedUserOperation` Solidity struct, unpacked form). Amounts
/// in wei. All addresses lowercase hex with `0x` prefix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserOperation {
    /// The smart account the op targets.
    pub sender: String,
    /// Anti-replay nonce. The EntryPoint partitions the nonce space
    /// by the top 192 bits ("key"); same key = strictly increasing
    /// sequence.
    pub nonce: u128,
    /// Account deployment calldata. Empty for already-deployed
    /// accounts; non-empty triggers a CREATE2 via the factory whose
    /// address is the first 20 bytes.
    pub init_code: Vec<u8>,
    /// The actual call the account executes.
    pub call_data: Vec<u8>,
    /// Gas budget for the account's execution phase.
    pub call_gas_limit: u128,
    /// Gas budget for the EntryPoint's verification phase.
    pub verification_gas_limit: u128,
    /// Gas burnt before the EntryPoint is even entered (calldata
    /// cost + intrinsic).
    pub pre_verification_gas: u128,
    /// EIP-1559 max-fee-per-gas.
    pub max_fee_per_gas: u128,
    /// EIP-1559 max-priority-fee-per-gas.
    pub max_priority_fee_per_gas: u128,
    /// Paymaster + paymaster gas limits + paymaster data. Empty for
    /// self-funded ops.
    pub paymaster_and_data: Vec<u8>,
    /// Account-side signature (EOA sig / multisig blob / passkey
    /// assertion — depends on the account's `validateUserOp`).
    pub signature: Vec<u8>,
}

impl UserOperation {
    /// Construct an empty self-funded op for `sender`. Operator
    /// fills in the rest before submission.
    #[must_use]
    pub fn new(sender: impl Into<String>) -> Self {
        Self {
            sender: sender.into(),
            nonce: 0,
            init_code: Vec::new(),
            call_data: Vec::new(),
            call_gas_limit: 0,
            verification_gas_limit: 0,
            pre_verification_gas: 0,
            max_fee_per_gas: 0,
            max_priority_fee_per_gas: 0,
            paymaster_and_data: Vec::new(),
            signature: Vec::new(),
        }
    }

    /// Validate enough of the structural envelope that a bundler
    /// won't reject the op on a trivial mistake. Does not validate
    /// the signature.
    ///
    /// # Errors
    /// Returns [`Error::Constraint`] when fields violate the spec.
    pub fn validate_structure(&self) -> Result<()> {
        if !self.sender.starts_with("0x") || self.sender.len() != 42 {
            return Err(Error::Constraint {
                field: "sender",
                reason: "must be 0x + 40 hex chars".into(),
            });
        }
        if self.call_gas_limit == 0 {
            return Err(Error::Constraint {
                field: "call_gas_limit",
                reason: "must be positive".into(),
            });
        }
        if self.verification_gas_limit == 0 {
            return Err(Error::Constraint {
                field: "verification_gas_limit",
                reason: "must be positive".into(),
            });
        }
        if self.max_fee_per_gas < self.max_priority_fee_per_gas {
            return Err(Error::Constraint {
                field: "max_fee_per_gas",
                reason: "must be >= max_priority_fee_per_gas".into(),
            });
        }
        if !self.paymaster_and_data.is_empty() && self.paymaster_and_data.len() < 20 {
            return Err(Error::Constraint {
                field: "paymaster_and_data",
                reason: "must be empty or >= 20 bytes (paymaster addr prefix)".into(),
            });
        }
        Ok(())
    }
}

/// ERC-4337 v0.7 `PackedUserOperation` — the calldata layout the
/// EntryPoint actually receives.
///
/// Two fields are packed into single 32-byte slots:
/// - `account_gas_limits` = (verificationGasLimit << 128) |
///   callGasLimit
/// - `gas_fees` = (maxPriorityFeePerGas << 128) | maxFeePerGas
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackedUserOperation {
    /// Sender (smart account).
    pub sender: String,
    /// Nonce (192-bit key || 64-bit sequence — packed externally).
    pub nonce: u128,
    /// Init code (factory_addr ++ factory_calldata).
    pub init_code: Vec<u8>,
    /// Account call data.
    pub call_data: Vec<u8>,
    /// Packed gas limits — 32 bytes.
    pub account_gas_limits: [u8; 32],
    /// Pre-verification gas (kept unpacked).
    pub pre_verification_gas: u128,
    /// Packed gas fees — 32 bytes.
    pub gas_fees: [u8; 32],
    /// Paymaster blob.
    pub paymaster_and_data: Vec<u8>,
    /// Signature.
    pub signature: Vec<u8>,
}

/// Pack a [`UserOperation`] into the v0.7 `PackedUserOperation`
/// layout (the bytes the EntryPoint consumes).
///
/// Each of the two packed slots is `(high_field << 128) |
/// low_field` encoded as 32-byte big-endian.
#[must_use]
pub fn pack_user_op(op: &UserOperation) -> PackedUserOperation {
    let account_gas_limits = pack_two_u128(op.verification_gas_limit, op.call_gas_limit);
    let gas_fees = pack_two_u128(op.max_priority_fee_per_gas, op.max_fee_per_gas);
    PackedUserOperation {
        sender: op.sender.clone(),
        nonce: op.nonce,
        init_code: op.init_code.clone(),
        call_data: op.call_data.clone(),
        account_gas_limits,
        pre_verification_gas: op.pre_verification_gas,
        gas_fees,
        paymaster_and_data: op.paymaster_and_data.clone(),
        signature: op.signature.clone(),
    }
}

fn pack_two_u128(high: u128, low: u128) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..16].copy_from_slice(&high.to_be_bytes());
    out[16..].copy_from_slice(&low.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_op() -> UserOperation {
        UserOperation {
            sender: "0x1111111111111111111111111111111111111111".into(),
            nonce: 42,
            init_code: vec![],
            call_data: vec![0xde, 0xad],
            call_gas_limit: 100_000,
            verification_gas_limit: 60_000,
            pre_verification_gas: 21_000,
            max_fee_per_gas: 10_000_000_000,
            max_priority_fee_per_gas: 2_000_000_000,
            paymaster_and_data: vec![],
            signature: vec![0xbe; 65],
        }
    }

    #[test]
    fn entrypoint_v07_canonical_address() {
        assert_eq!(
            EntryPointVersion::V0_7.canonical_address(),
            "0x0000000071727de22e5e9d8baf0edac6f37da032"
        );
    }

    #[test]
    fn validate_accepts_canonical_op() {
        sample_op().validate_structure().unwrap();
    }

    #[test]
    fn validate_rejects_bad_sender() {
        let mut op = sample_op();
        op.sender = "not-an-address".into();
        let err = op.validate_structure().unwrap_err();
        assert!(matches!(err, Error::Constraint { field: "sender", .. }));
    }

    #[test]
    fn validate_rejects_zero_call_gas() {
        let mut op = sample_op();
        op.call_gas_limit = 0;
        assert!(op.validate_structure().is_err());
    }

    #[test]
    fn validate_rejects_priority_higher_than_max() {
        let mut op = sample_op();
        op.max_priority_fee_per_gas = op.max_fee_per_gas + 1;
        let err = op.validate_structure().unwrap_err();
        assert!(matches!(
            err,
            Error::Constraint {
                field: "max_fee_per_gas",
                ..
            }
        ));
    }

    #[test]
    fn pack_layout_high_low_split() {
        let op = sample_op();
        let packed = pack_user_op(&op);

        // account_gas_limits: high = 60_000 (vGas), low = 100_000 (cGas)
        let high_v: u128 = u128::from_be_bytes(packed.account_gas_limits[..16].try_into().unwrap());
        let low_c: u128 = u128::from_be_bytes(packed.account_gas_limits[16..].try_into().unwrap());
        assert_eq!(high_v, 60_000);
        assert_eq!(low_c, 100_000);

        // gas_fees: high = maxPriority, low = maxFee
        let high_p: u128 = u128::from_be_bytes(packed.gas_fees[..16].try_into().unwrap());
        let low_m: u128 = u128::from_be_bytes(packed.gas_fees[16..].try_into().unwrap());
        assert_eq!(high_p, 2_000_000_000);
        assert_eq!(low_m, 10_000_000_000);
    }
}
