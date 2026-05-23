# demo-merchant

The smallest end-to-end OpenPay payment demo. Generates a fresh EVM
wallet, listens for op-server webhooks, polls Base for USDC balance
changes, and exits on the first observed payment.

Whichever signal arrives first — an op-server `intent.approved`
webhook for our wallet, or a non-zero `balanceOf` result on the USDC
contract — wins.

## Run

Terminal 1, op-server:

```bash
cargo run --release -p op-server
```

Terminal 2, demo merchant (testnet first, please):

```bash
cargo run --release -p demo-merchant -- --testnet
```

Output:

```
Merchant wallet: 0xabc123...
Explorer:        https://sepolia.basescan.org/address/0xabc123...
Listening on http://127.0.0.1:9090 for op-server webhooks at /webhook
Polling https://sepolia.base.org every 30s for balanceOf updates.
```

Send USDC (Sepolia testnet faucet recommended) to the printed address.
The demo prints `PAID: ...` and exits 0.

## Flags

- `--rpc-url URL` — EVM JSON-RPC endpoint. Default `https://mainnet.base.org`.
- `--listen ADDR` — webhook bind address. Default `127.0.0.1:9090`.
- `--amount-usdc N` — expected amount, display only. Default `0.10`.
- `--explorer URL` — block-explorer base. Default `https://basescan.org`.
- `--testnet` — switch RPC + explorer to Base Sepolia.
- `--poll-secs N` — `balanceOf` poll interval. Default `30`.

## Caveats

- Real USDC on mainnet costs real money. Use `--testnet` for your
  first run.
- The private key is generated in-process and never persisted; if you
  exit the demo, the funds at that address are effectively burned
  from your control.
- The webhook endpoint has no signature verification — it's a demo,
  not a production receiver.
