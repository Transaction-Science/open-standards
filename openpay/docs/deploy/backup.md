# OpenPay — backup and restore

The entire deployment — refunds, disputes, settlement batches,
ledger, reconciliation, webhooks, subscriptions, idempotency cache
— lives in **one file**: the `.graph` Minigraf database pointed at
by `AppState::with_graph_path`. Backup is "copy that file";
restore is "put it back."

This is the single biggest win of the Phase 26 single-file
persistence model (`docs/26-graph-persistence-progress.md`): no
multi-table dump, no schema-aware dump tool, no per-store backup
job. One file. Atomic.

## What you're backing up

By default the systemd unit puts the file at
`/var/lib/openpay/data.graph`. (See `OP_GRAPH_PATH` in
`openpay.env.sample`.) If you configured a different path, that's
the one.

## The honest caveat: hot copies can miss recent writes

Minigraf holds in-memory state that the on-disk file lags slightly
behind. A `cp` / `rsync` of the file **while op-server is serving
traffic** will get you something close-but-not-equal to the
in-memory truth — the last few seconds of writes may be missing,
and depending on what's mid-flush you may copy a partial page.

Two ways to get a clean snapshot:

### Option A — stop the service briefly (simplest, recommended)

```bash
sudo systemctl stop openpay
sudo rsync -av /var/lib/openpay/data.graph \
    backup-host:/snapshots/openpay-$(date +%F-%H%M).graph
sudo systemctl start openpay
```

The service shutdown handler (SIGTERM → `shutdown_signal` in
`crates/op-server/src/main.rs`) drains in-flight requests cleanly,
and `Drop` on `GraphHandle` flushes pending writes. Typical
service downtime: under five seconds for graphs in the
single-digit-GB range.

### Option B — call `compact()` first, then copy

`GraphHandle::compact()` (see `crates/op-graph/src/graph.rs:239`)
flushes pending writes and rewrites the file in compact form.
Right after `compact()` returns, the on-disk state is consistent
with what was in memory up to that moment. There's no shipped HTTP
endpoint for this — you'd add one to your patched `main.rs` (an
operator-only path on a local-only socket, ideally) or trigger it
from a maintenance binary that opens the same graph path.

This still doesn't capture writes that arrive *between* `compact()`
and the copy. For that you want Option A.

## restic example (recommended for offsite)

```bash
restic -r s3:s3.amazonaws.com/openpay-backups init    # first time only

sudo systemctl stop openpay
sudo restic -r s3:s3.amazonaws.com/openpay-backups \
    --tag openpay --tag $(hostname) \
    backup /var/lib/openpay/data.graph
sudo systemctl start openpay

# Verify periodically:
sudo restic -r s3:s3.amazonaws.com/openpay-backups check
```

restic deduplicates across snapshots — a daily backup of a
mostly-static graph file is cheap. Pair with `restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 12`
to age out old snapshots.

## Restore

Stop the service, drop the file in place, start the service.
Minigraf opens it on startup; every store recovers automatically.

```bash
sudo systemctl stop openpay
sudo cp /snapshots/openpay-2026-05-20-0300.graph \
        /var/lib/openpay/data.graph
sudo chown openpay:openpay /var/lib/openpay/data.graph
sudo systemctl start openpay

./target/release/op readiness
# Stores should all answer ready.
```

No migrations to run. No "reseed the index." The audit-report
endpoint (`GET /v1/audit/report`) will reflect the restored state.

## What about the in-memory graph?

`AppState::new_in_memory()` keeps everything in RAM and drops it on
restart. There is nothing to back up; there is also nothing to
recover. If you're running on the in-memory graph, you accepted
that. (Suitable for: tests, demos, ephemeral kiosks. Not for:
anything you want to see tomorrow.)

## Cross-host restore

The `.graph` file is portable across hosts of the same Rust target
triple. Copy it to a new machine, install the matching `op-server`
binary, point `OP_GRAPH_PATH` at the file, start the service —
done. No external dependencies, no DB cluster to migrate.

## What about bi-temporal history?

The ledger's bi-temporal history (`op_ledger::LedgerHistory`) is
stored in the same `.graph` file. Restoring the file restores the
history. Time-travel queries (`balance_as_of`, the audit report's
`as_of` parameter, etc.) work against the restored state with no
extra work.
