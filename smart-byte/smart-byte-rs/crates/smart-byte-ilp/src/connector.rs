//! Connector — the per-hop forwarding decision.
//!
//! An ILP connector forwards `Prepare` packets to the next hop on the
//! way to the destination, just like an IP router forwards IP datagrams.
//! The forwarding rule has four parts:
//!
//! 1. Longest-prefix match of the destination address against a route
//!    table.
//! 2. Currency conversion across the hop using a quote table.
//! 3. Rate-limit + balance check against the next-hop account.
//! 4. Return either `ForwardDecision::Forward` with the rewritten
//!    `Prepare`, or `ForwardDecision::Reject` with a `RejectCode`.

use crate::address::Address;
use crate::error::{IlpError, Result};
use crate::packet::{Prepare, RejectCode};
use std::collections::HashMap;

/// A single route table entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    /// Address prefix this route serves (e.g. `"g.us.bank"`).
    pub prefix: String,
    /// Identifier of the next-hop account / peer.
    pub next_hop: String,
    /// Asset code the next hop is denominated in.
    pub asset_code: String,
    /// Asset scale of the next hop.
    pub asset_scale: u8,
}

/// A simple longest-prefix-match route table.
#[derive(Clone, Debug, Default)]
pub struct RouteTable {
    routes: Vec<Route>,
}

impl RouteTable {
    /// Construct an empty route table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a route. Multiple routes with the same prefix are
    /// preserved in insertion order; only the first match wins.
    pub fn insert(&mut self, route: Route) {
        self.routes.push(route);
    }

    /// Iterate over the underlying routes.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Resolve `destination` to its best route by longest-prefix match.
    pub fn resolve(&self, destination: &Address) -> Option<&Route> {
        let mut best: Option<&Route> = None;
        for r in &self.routes {
            if destination.starts_with_prefix(&r.prefix) {
                match best {
                    None => best = Some(r),
                    Some(prev) if r.prefix.len() > prev.prefix.len() => best = Some(r),
                    _ => {}
                }
            }
        }
        best
    }
}

/// Per-hop account state — balance, prepaid, and a per-second packet
/// rate limit.
#[derive(Clone, Debug)]
pub struct AccountState {
    /// Available balance the connector is willing to extend.
    pub balance: u64,
    /// Maximum value of any single prepare against this account.
    pub max_packet_amount: u64,
    /// Per-second packet quota.
    pub packets_per_sec: u32,
    /// Counter of packets handled this second (caller resets each tick).
    pub packets_this_sec: u32,
}

/// Outcome of a per-hop forwarding decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ForwardDecision {
    /// Forward the rewritten `Prepare` to `next_hop`.
    Forward {
        /// Next-hop identifier resolved by the route table.
        next_hop: String,
        /// Rewritten prepare with the new amount in the next hop's asset.
        prepare: Prepare,
    },
    /// Reject the prepare with a standard reject code.
    Reject {
        /// Reject code per IL-RFC-27.
        code: RejectCode,
        /// Human-readable message.
        message: String,
    },
}

/// A stateful connector: route table, per-account state, and a quote
/// table for cross-asset conversion.
#[derive(Clone, Debug)]
pub struct Connector {
    /// Operator-assigned address of this connector (e.g. `"g.us.bank"`).
    pub address: Address,
    /// Asset details of this connector's source side.
    pub source_asset_code: String,
    /// Asset scale of this connector's source side.
    pub source_asset_scale: u8,
    /// Route table for next-hop selection.
    pub routes: RouteTable,
    /// Quotes keyed by `(from_asset, to_asset)`. Value is a 64-bit
    /// fixed-point rate: out_amount = in_amount * rate / 1_000_000.
    pub quotes: HashMap<(String, String), u64>,
    /// Per-account state keyed by next-hop identifier.
    pub accounts: HashMap<String, AccountState>,
}

impl Connector {
    /// Construct a new connector for the given address + source asset.
    pub fn new(address: Address, source_asset_code: String, source_asset_scale: u8) -> Self {
        Self {
            address,
            source_asset_code,
            source_asset_scale,
            routes: RouteTable::new(),
            quotes: HashMap::new(),
            accounts: HashMap::new(),
        }
    }

    /// Forward a `Prepare`. Returns a typed decision describing whether
    /// the packet went out and at what amount, or why it was rejected.
    pub fn forward(&mut self, prepare: &Prepare) -> Result<ForwardDecision> {
        let route = match self.routes.resolve(&prepare.destination) {
            Some(r) => r.clone(),
            None => {
                return Ok(ForwardDecision::Reject {
                    code: RejectCode::F02Unreachable,
                    message: format!("no route to {}", prepare.destination.as_str()),
                })
            }
        };
        let next_amount = self.convert(
            prepare.amount,
            &self.source_asset_code,
            &route.asset_code,
        )?;
        let Some(account) = self.accounts.get_mut(&route.next_hop) else {
            return Err(IlpError::Balance(format!(
                "no account state for {}",
                route.next_hop
            )));
        };
        if account.max_packet_amount > 0 && next_amount > account.max_packet_amount {
            return Ok(ForwardDecision::Reject {
                code: RejectCode::F08AmountTooLarge,
                message: "exceeds max packet amount".into(),
            });
        }
        if account.packets_this_sec >= account.packets_per_sec {
            return Ok(ForwardDecision::Reject {
                code: RejectCode::T05RateLimited,
                message: "rate limit exceeded".into(),
            });
        }
        if account.balance < next_amount {
            return Ok(ForwardDecision::Reject {
                code: RejectCode::T04InsufficientLiquidity,
                message: "insufficient balance".into(),
            });
        }
        account.balance -= next_amount;
        account.packets_this_sec += 1;
        let new_prepare = Prepare {
            amount: next_amount,
            expires_at: prepare.expires_at,
            condition: prepare.condition,
            destination: prepare.destination.clone(),
            data: prepare.data.clone(),
        };
        Ok(ForwardDecision::Forward {
            next_hop: route.next_hop,
            prepare: new_prepare,
        })
    }

    /// Apply a quote table lookup. If `from == to` the amount passes
    /// through unchanged.
    pub fn convert(&self, amount: u64, from: &str, to: &str) -> Result<u64> {
        if from == to {
            return Ok(amount);
        }
        let rate = self.quotes.get(&(from.to_string(), to.to_string())).copied();
        let rate = rate.ok_or_else(|| IlpError::NoQuote {
            from: from.to_string(),
            to: to.to_string(),
        })?;
        let converted = (amount as u128)
            .saturating_mul(rate as u128)
            / 1_000_000u128;
        Ok(converted.min(u64::MAX as u128) as u64)
    }

    /// Refund the next-hop account on a downstream `Reject` — restores
    /// the prepaid balance the original `forward` deducted.
    pub fn refund(&mut self, next_hop: &str, amount: u64) {
        if let Some(a) = self.accounts.get_mut(next_hop) {
            a.balance = a.balance.saturating_add(amount);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condition::Fulfillment;

    fn fixture() -> (Connector, Prepare) {
        let mut c = Connector::new(
            Address::parse("g.us.bank").unwrap(),
            "USD".into(),
            6,
        );
        c.routes.insert(Route {
            prefix: "g.eu".into(),
            next_hop: "peer-eu".into(),
            asset_code: "EUR".into(),
            asset_scale: 6,
        });
        c.quotes
            .insert(("USD".into(), "EUR".into()), 900_000); // 0.9 EUR per USD
        c.accounts.insert(
            "peer-eu".into(),
            AccountState {
                balance: 1_000_000,
                max_packet_amount: 0,
                packets_per_sec: 100,
                packets_this_sec: 0,
            },
        );
        let f = Fulfillment::new([4u8; 32]);
        let p = Prepare {
            amount: 1_000,
            expires_at: *b"20260524120000000",
            condition: f.condition(),
            destination: Address::parse("g.eu.bank.bob").unwrap(),
            data: vec![],
        };
        (c, p)
    }

    #[test]
    fn forwards_and_converts() {
        let (mut c, p) = fixture();
        let decision = c.forward(&p).unwrap();
        let ForwardDecision::Forward { next_hop, prepare } = decision else {
            panic!("expected forward");
        };
        assert_eq!(next_hop, "peer-eu");
        assert_eq!(prepare.amount, 900); // 1000 USD * 0.9 = 900 EUR
    }

    #[test]
    fn rejects_unknown_destination() {
        let (mut c, mut p) = fixture();
        p.destination = Address::parse("g.us.other").unwrap();
        match c.forward(&p).unwrap() {
            ForwardDecision::Reject { code, .. } => assert_eq!(code, RejectCode::F02Unreachable),
            _ => panic!("expected reject"),
        }
    }

    #[test]
    fn rate_limit_rejects() {
        let (mut c, p) = fixture();
        if let Some(a) = c.accounts.get_mut("peer-eu") {
            a.packets_this_sec = a.packets_per_sec;
        }
        match c.forward(&p).unwrap() {
            ForwardDecision::Reject { code, .. } => assert_eq!(code, RejectCode::T05RateLimited),
            _ => panic!("expected reject"),
        }
    }
}
