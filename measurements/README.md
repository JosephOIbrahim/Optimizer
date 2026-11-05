# Measurement records

The evidence behind every verdict the harness prints.

These are the raw records, committed alongside the code that produced them, so
that a claim in the operator cards can be checked against the file it came
from rather than taken on trust.

## What is here

```
optimize_*.json   a lever run: both arms measured alternately in one session,
                  with the raw series (baseline_series, candidate_series,
                  paired_diffs) so the effect, the p-value and the drift can
                  all be re-derived from the file alone
levers_*.json    the retired method's sweeps, with the retractions applied
render_*.json     single-frame renders
gemm_*.json       standalone GEMM measurements at a fixed ceiling
```

A record that carries a verdict also carries how it was reached: the drift of
the control, its spread, the contention events, and — for a power-ceiling run
— the prior ceiling, where it came from, and whether the restore was verified
by a read-back.

## Where the live store is

These files are COPIES. The live store is `E:\OptimizerV12\measurements`, and
it is deliberately off OneDrive: many small JSON files syncing through
OneDrive is a bad trade, and E: has the space. The copies here are for
version control and for reading next to the code.

Records cite each other by their absolute path on E:, so a pointer inside one
record resolves against the live store rather than against this directory.

The render output — PNG frames, tens of megabytes — is not committed. It is
output, not evidence.

## Keeping the copies current

Every campaign does this itself, at the end of the run: the record it just
wrote is staged along with anything else that changed on the live store, and
all of it lands in ONE commit whose message says what the campaign found.

```
Sync-Measurements.bat          copy the live records in and commit them
Sync-Measurements.bat --check  report what would change, and change nothing
```

The two are for refreshing the copies without measuring anything.

The sync compares CONTENT, not file names. That matters because a retraction
rewrites a record in place — same name, different bytes — and a sync that
copied "the new files" would never notice.

It commits only `measurements/`, so source edits you have in flight are left
alone — even if you have them *staged*. It will not delete a committed
record that has gone from the live store: that is reported, because deleting
evidence is a decision. And it will not commit a record that does not parse,
so a truncated write cannot land in the history.

## The retractions

Three verdicts from 2026-11-03 were false, and the records that carried them
are here with the retraction applied. The original claim is never deleted: the
live verdict reads `RETRACTED — was: ADMIT (+12.34%) ...`, so a reader who
greps for an admission still finds the original text, marked.

| record | claimed | actually |
|---|---|---|
| `levers_20261103_192945.json` | husk_j48 ADMIT +12.34% | DENY -0.60% |
| `levers_20261103_221724.json` | aov_beauty_only ADMIT +16.91% | ADMIT +9.79% |
| `levers_20261103_222710.json` | aov_no_crypto ADMIT +11.25% | DENY +0.77% |

Each retraction points at the measurement that replaced it, and where a
replacement was itself superseded, that one is in `replacement_history` with
the reason it stopped being current.
