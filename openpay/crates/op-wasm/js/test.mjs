// JS-side smoke tests for the wasm-pack bundled output.
//
// Run after `bash scripts/build-wasm.sh --target nodejs`:
//
//   node js/test.mjs
//
// Covers what the Rust-side wasm-bindgen-test cannot: the
// JS-visible OpenPayError shape, the explicit `.free()` ownership
// protocol, and the TypeScript-friendly camelCase property names.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  CardData,
  HeuristicScorer,
  OpenPayError,
  RustVault,
  TokenFormat,
  TokenLifetime,
  TokenizationPolicy,
  VaultRef,
} from "../pkg/op_wasm.js";

const VALID_VISA = "4242424242424242";
const VALID_MC = "5555555555554444";

test("CardData constructs and exposes camelCase getters", () => {
  const card = new CardData(VALID_VISA, 12, 2030);
  assert.equal(card.firstSix, "424242");
  assert.equal(card.lastFour, "4242");
  assert.equal(card.expMonth, 12);
  assert.equal(card.expYear, 2030);
  card.free();
});

test("CardData rejects invalid PAN with OpenPayError.InvalidInput", () => {
  let thrown = null;
  try {
    new CardData("1111111111111111", 12, 2030);
  } catch (e) {
    thrown = e;
  }
  assert.ok(thrown !== null, "expected throw");
  assert.equal(thrown.code, 1);
  assert.equal(thrown.kind, "InvalidInput");
  assert.ok(thrown.message.length > 0);
});

test("CardData rejects bad expiration", () => {
  for (const [m, y] of [[13, 2030], [0, 2030]]) {
    assert.throws(() => new CardData(VALID_VISA, m, y));
  }
});

test("CardData methods throw after .free() (wasm-bindgen null-check)", () => {
  const card = new CardData(VALID_VISA, 12, 2030);
  card.free();
  assert.throws(() => card.firstSix);
});

test("RustVault round-trip via classes", () => {
  const vault = new RustVault("test");
  assert.equal(vault.name, "test");

  const card = new CardData(VALID_VISA, 12, 2030);
  const token = vault.tokenize(card);
  // card was consumed (moved by-value across the boundary); JS-side
  // pointer is now null.
  assert.throws(() => card.firstSix);

  assert.ok(token.asString.startsWith("tok_v7_"));

  const recovered = vault.detokenize(token);
  assert.equal(recovered.lastFour, "4242");
  recovered.free();
  token.free();
  vault.free();
});

test("Unknown token throws VaultLookupFailed", () => {
  const vault = new RustVault("err");
  const fake = VaultRef.fromString("tok_v7_doesnotexist");
  try {
    vault.detokenize(fake);
    assert.fail("expected throw");
  } catch (e) {
    assert.equal(e.code, 2);
    assert.equal(e.kind, "VaultLookupFailed");
  }
  fake.free();
  vault.free();
});

test("Malformed token also collapses to VaultLookupFailed (oracle discipline)", () => {
  const vault = new RustVault("err");
  const bad = VaultRef.fromString("not-a-token");
  try {
    vault.detokenize(bad);
    assert.fail("expected throw");
  } catch (e) {
    assert.equal(e.code, 2);
    assert.equal(e.kind, "VaultLookupFailed");
  }
  bad.free();
  vault.free();
});

test("Single-use token throws TokenAlreadyConsumed on second use", () => {
  const vault = new RustVault("single");
  const card = new CardData(VALID_VISA, 12, 2030);
  const token = vault.tokenize(card, TokenizationPolicy.singleUse(120));

  // First detokenize succeeds.
  vault.detokenize(token).free();

  // Second detokenize fails.
  try {
    vault.detokenize(token);
    assert.fail("expected throw");
  } catch (e) {
    assert.equal(e.code, 4);
    assert.equal(e.kind, "TokenAlreadyConsumed");
  }

  token.free();
  vault.free();
});

test("Distinct tokens for same PAN (random format default)", () => {
  const vault = new RustVault("uniq");
  const t1 = vault.tokenize(new CardData(VALID_VISA, 12, 2030));
  const t2 = vault.tokenize(new CardData(VALID_VISA, 12, 2030));
  assert.notEqual(t1.asString, t2.asString);
  t1.free();
  t2.free();
  vault.free();
});

test("Deterministic format produces same token for same PAN", () => {
  const vault = new RustVault("det");
  const policy = new TokenizationPolicy();
  policy.format = TokenFormat.Deterministic;

  const t1 = vault.tokenize(new CardData(VALID_VISA, 12, 2030), policy);
  const t2 = vault.tokenize(new CardData(VALID_VISA, 12, 2030), policy);
  assert.equal(t1.asString, t2.asString);
  t1.free();
  t2.free();
  vault.free();
});

test("exists() and delete()", () => {
  const vault = new RustVault("del");
  const token = vault.tokenize(new CardData(VALID_VISA, 12, 2030));

  assert.equal(vault.exists(token), true);
  assert.equal(vault.delete(token), true);
  assert.equal(vault.exists(token), false);
  assert.equal(vault.delete(token), false); // idempotent

  token.free();
  vault.free();
});

test("HeuristicScorer.name is stable", () => {
  const s = new HeuristicScorer();
  assert.equal(s.name, "heuristic-v1");
  s.free();
});

test("TokenizationPolicy factory helpers", () => {
  const single = TokenizationPolicy.singleUse(60);
  assert.equal(single.lifetime, TokenLifetime.SingleUse);
  assert.equal(single.ttlSeconds, 60n); // u64 → BigInt on JS side

  const cof = TokenizationPolicy.cardOnFile();
  assert.equal(cof.lifetime, TokenLifetime.Reusable);
  assert.equal(cof.ttlSeconds, 0n);
});

test("PAN never appears in token string (opacity invariant)", () => {
  const vault = new RustVault("opacity");
  const token = vault.tokenize(new CardData(VALID_VISA, 12, 2030));
  const s = token.asString;
  assert.ok(!s.includes(VALID_VISA));
  assert.ok(!s.includes("424242"));
  assert.ok(!s.includes("4242"));
  token.free();
  vault.free();
});

test("Multiple cards coexist", () => {
  const vault = new RustVault("multi");
  const visa = vault.tokenize(new CardData(VALID_VISA, 12, 2030));
  const mc = vault.tokenize(new CardData(VALID_MC, 11, 2028));

  const visaRecovered = vault.detokenize(visa);
  assert.equal(visaRecovered.lastFour, "4242");
  visaRecovered.free();

  const mcRecovered = vault.detokenize(mc);
  assert.equal(mcRecovered.lastFour, "4444");
  assert.equal(mcRecovered.expMonth, 11);
  mcRecovered.free();

  visa.free();
  mc.free();
  vault.free();
});

test("Two vaults with the same name are isolated", () => {
  const a = new RustVault("same");
  const b = new RustVault("same");
  const token = a.tokenize(new CardData(VALID_VISA, 12, 2030));
  assert.throws(() => b.detokenize(token));
  token.free();
  a.free();
  b.free();
});

test("OpenPayError toString includes code and kind", () => {
  const vault = new RustVault("e");
  const fake = VaultRef.fromString("tok_v7_no");
  try {
    vault.detokenize(fake);
  } catch (e) {
    const s = e.toString();
    assert.ok(s.startsWith("OpenPayError ["));
    assert.ok(s.includes("VaultLookupFailed"));
  }
  fake.free();
  vault.free();
});
