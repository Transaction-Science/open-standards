# OpenPay for JavaScript / TypeScript

WebAssembly bridge over the OpenPay Rust core. Same architectural
shape as the iOS (Phase 8) and Android (Phase 9) bridges: opaque
class handles, oracle-discipline error collapsing, and cross-platform
error code alignment.

## What you get

- **`CardData`** — Validated PAN + expiration. Luhn / length /
  date sanity checked on construction. PAN bytes live in wasm
  linear memory and are never readable from JavaScript.
- **`VaultRef`** — Opaque token reference.
- **`RustVault`** — In-memory reference vault (development).
- **`HeuristicScorer`** — Rule-based fraud scorer.
- **`OpenPayError`** — Error class with `.code`, `.kind`, `.message`.
- **`TokenizationPolicy`** — Configurable format / lifetime / TTL.
- **`TokenFormat`** / **`TokenLifetime`** — TypeScript-friendly enums.

## Build the wasm bundle first

```bash
# One-time setup:
cargo install wasm-pack
rustup target add wasm32-unknown-unknown

# From the crate root (`crates/op-wasm/`):
bash scripts/build-wasm.sh --target web      # browsers (ESM)
bash scripts/build-wasm.sh --target nodejs   # Node.js (CommonJS)
bash scripts/build-wasm.sh --target bundler  # webpack / rollup
```

The build produces `pkg/op_wasm.js`, `pkg/op_wasm_bg.wasm`, and
`pkg/op_wasm.d.ts` (the TypeScript types).

## Usage example

### Browser (ESM)

```typescript
import init, {
  CardData,
  RustVault,
  TokenizationPolicy,
  OpenPayError,
} from './pkg/op_wasm.js';

await init();

const vault = new RustVault('checkout');

try {
  const card = new CardData('4242424242424242', 12, 2030);
  const token = vault.tokenize(card, TokenizationPolicy.cardOnFile());
  // `card` was consumed by tokenize() — its JS pointer is now null.

  console.log('Token:', token.asString);

  // Persist token.asString to localStorage or send to server.
  localStorage.setItem('pm:default', token.asString);

  token.free();
} catch (e) {
  if (e instanceof OpenPayError) {
    console.warn(`payment failed: ${e.kind}`, e.message);
  } else {
    throw e;
  }
} finally {
  vault.free();
}
```

### Recovering a token later

```typescript
import { VaultRef } from './pkg/op_wasm.js';

const tokenStr = localStorage.getItem('pm:default');
const ref = VaultRef.fromString(tokenStr);

try {
  const card = vault.detokenize(ref);
  submitToAcquirer(card.firstSix, card.lastFour);
  card.free();
} catch (e) {
  if (e.code === 2 /* VaultLookupFailed */) {
    showError('Card not found');
  } else if (e.code === 3 /* TokenExpired */) {
    showError('Token expired — please re-enter your card');
  } else {
    throw e;
  }
} finally {
  ref.free();
}
```

### Node.js

```javascript
const { RustVault, CardData } = require('./pkg/op_wasm.js');
// No init() call needed for the nodejs target.

const vault = new RustVault('test');
const token = vault.tokenize(new CardData('4242424242424242', 12, 2030));
console.log(token.asString);
token.free();
vault.free();
```

## The .free() protocol

wasm-bindgen does not have a finalizer mechanism — the JS engine
doesn't expose one portably. Every `#[wasm_bindgen]`-exported Rust
struct gets an explicit `.free()` method that **must** be called
when the JS object is no longer needed. Forgetting to call it leaks
wasm linear memory (which is never reclaimed by the JS GC).

Three idiomatic patterns:

1. **Explicit `.free()` in a `try/finally`:**

   ```javascript
   const card = new CardData(pan, m, y);
   try {
     // ...
   } finally {
     card.free();
   }
   ```

2. **Consume-by-value methods** automatically invalidate the JS
   pointer. After `vault.tokenize(card)`, the `card` JS object's
   pointer is set to null and any further method call throws. No
   `.free()` needed.

3. **`using` (ES2026)** if your toolchain targets ES2026 or you
   shim `Symbol.dispose`:

   ```javascript
   using vault = new RustVault('checkout');
   using card = new CardData(pan, m, y);
   // automatic .free() when scope exits
   ```

## Error inspection

`OpenPayError` has three readable properties:

| Property | Type | Description |
|---|---|---|
| `.code` | `number` | i32 discriminant, matches iOS/Android |
| `.kind` | `string` | PascalCase variant name |
| `.message` | `string` | Short human-readable description |

Discriminants (stable across iOS, Android, web):

| Code | Kind | Meaning |
|---|---|---|
| 1 | `InvalidInput` | PAN, expiration, or token format malformed |
| 2 | `VaultLookupFailed` | Token unknown, malformed, or auth-failed (collapsed) |
| 3 | `TokenExpired` | Past TTL |
| 4 | `TokenAlreadyConsumed` | Single-use, already used |
| 5 | `FraudDeclined` | Scorer rejected |
| 6 | `FraudReviewRequired` | Scorer flagged for human review |
| 7 | `Backend` | Vault / rail / scorer opaque failure |
| 8 | `Internal` | FFI-internal bug |
| 9 | `Capacity` | Rate limit or capacity exhaustion |

### Note: `OpenPayError` does not extend `Error`

wasm-bindgen-exported Rust structs become standalone JS classes; they
don't inherit from the native `Error` prototype. So
`e instanceof Error` returns `false` for thrown `OpenPayError`
instances. This is a [known wasm-bindgen
limitation](https://github.com/rustwasm/wasm-bindgen/issues/1787).
Pattern-match on `.code` or `.kind` instead, or wrap at the JS
boundary:

```typescript
import { OpenPayError } from './pkg/op_wasm.js';

export class OpenPayJsError extends Error {
  constructor(public readonly inner: OpenPayError) {
    super(inner.message);
    this.name = 'OpenPayJsError';
  }
  get code()    { return this.inner.code; }
  get kind()    { return this.inner.kind; }
}

export function safeTokenize(vault, card, policy) {
  try {
    return vault.tokenize(card, policy);
  } catch (e) {
    if (e instanceof OpenPayError) throw new OpenPayJsError(e);
    throw e;
  }
}
```

## TypeScript

The `pkg/op_wasm.d.ts` file is generated automatically by wasm-bindgen
from the Rust signatures. Drop `pkg/` into your `tsconfig.json`
`include` and TypeScript will pick it up.

## Testing

Two test suites:

- **Rust-side** (`tests/wasm_bindgen.rs`) runs inside a real wasm
  host:

  ```bash
  wasm-pack test --node              # Node.js
  wasm-pack test --headless --chrome # Chrome
  wasm-pack test --headless --firefox # Firefox
  ```

- **JS-side** (`js/test.mjs`) exercises the consumer-visible
  classes after `wasm-pack build`:

  ```bash
  bash scripts/build-wasm.sh --target nodejs
  node js/test.mjs
  ```

Plain `cargo test -p op-wasm` runs the host-side unit tests (FfiError
discriminant tests, OpenPayError construction, policy mapping).

## Why no Web Crypto API integration?

The Web Crypto API (`SubtleCrypto`) is async-only — every operation
returns a `Promise`. The OpenPay [`Vault`] trait is sync (consistent
across iOS Keychain, Android Keystore, in-memory Rust, etc.).
Reshaping the trait to be async would ripple through every crate;
it's a coherent next step but doesn't fit Phase 10.

A future `WebCryptoVault` class, analogous to Phase 9's
`KeystoreVault`, will use `SubtleCrypto` for hardware-accelerated
AES-GCM and IndexedDB for persistence. That ships as Phase 10.1
once the async-vault story is designed.

For now: `RustVault` uses the same `aes-gcm-siv` cipher as iOS and
Android. The cryptographic posture is identical across platforms.

## Browser support

WebAssembly is supported by:

- Chrome 57+ (March 2017)
- Firefox 52+ (March 2017)
- Safari 11+ (Sept 2017)
- Edge 16+ (Oct 2017)

This covers ~96% of the global web audience per caniuse.com (May 2026).
For older browsers, ship a server-side tokenization fallback.
