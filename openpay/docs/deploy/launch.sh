#!/usr/bin/env bash
#
# OpenPay launch script — single-command go-live on Base mainnet.
#
# Prereqs:
#   1. Rust 1.95+ installed (`rustup default stable`).
#   2. A hot wallet private key with ETH on Base mainnet for gas
#      (~$5 of ETH covers thousands of USDC transfers).
#   3. The wallet pre-funded with whatever USDC float you'll be
#      sending out (refunds, payouts). For receive-only pilots you
#      need only ETH for gas on the eventual sweeper.
#   4. A domain pointed at this server, with Caddy/nginx terminating
#      TLS in front (see docs/deploy/Caddyfile.sample).
#
# This script bootstraps a single-instance deployment with the
# embedded Minigraf .graph file on disk. Multi-instance / HA
# requires a graph daemon in front (out of scope for this script).
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────
# Edit these or pass via environment.

: "${OPENPAY_HOME:=/var/lib/openpay}"
: "${OPENPAY_BIND:=127.0.0.1:8080}"
: "${OPENPAY_RPC:=https://mainnet.base.org}"
# Set these in /etc/openpay/openpay.env (NOT this script) before
# running. They are referenced here as required env vars.
: "${OP_USDC_BASE_PRIVATE_KEY:?OP_USDC_BASE_PRIVATE_KEY must be set — see docs/deploy/README.md §6}"
: "${OP_API_KEYS:?OP_API_KEYS must be set — generate with: openssl rand -hex 32}"

# ─── Build ────────────────────────────────────────────────────────
echo ">>> Building op-server + op-cli with live features..."
cargo build --release \
  -p op-server -p op-cli \
  --features "op-rails-crypto/evm,op-webhook/reqwest-transport"

# ─── State directory ──────────────────────────────────────────────
echo ">>> Ensuring state directory at ${OPENPAY_HOME}..."
mkdir -p "${OPENPAY_HOME}"

# ─── Launch ───────────────────────────────────────────────────────
echo ">>> Starting op-server on ${OPENPAY_BIND}"
echo ">>> RPC: ${OPENPAY_RPC}"
echo ">>> State: ${OPENPAY_HOME}/data.graph"
echo

export OP_BIND_ADDR="${OPENPAY_BIND}"
export OP_GRAPH_PATH="${OPENPAY_HOME}/data.graph"
export OP_BASE_RPC_URL="${OPENPAY_RPC}"
export OP_RATE_LIMIT_PER_MINUTE="${OP_RATE_LIMIT_PER_MINUTE:-600}"
export OP_LOG_FORMAT="${OP_LOG_FORMAT:-json}"
# OP_API_KEYS + OP_USDC_BASE_PRIVATE_KEY are already in the env.

exec target/release/op-server
