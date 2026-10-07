# AliCCP environment weighting: pre-run identifiability audit

Status: audit protocol frozen before executing the full audit below; no weighting training is authorized by this document. The raw-partition collapse and normalized seed 1/2 label cross-tabs were known before this audit and are disclosed in `docs/experiments/2026-10-07-env-weighting-supervisor-audit.md`. This is not a prospective test of those previously seen facts.

## Question and source

Can a normalized Stage-1 partition support train-internal estimation of **BSI-driven** environment weights for canonical AliCCP short-budget comparisons? Audit the five already-frozen canonical Stage-1 backbones; do not train new backbones or choose seeds by outcome. The training prefix has 2,000,000 rows; official validation and test sets are not read by this audit. The raw seed-1 partition is recorded only as a NO-GO control.

Use `python -m aliccp_benchmark.audit_env_weighting` from this branch. Inputs are Stage-1 `meta.json`/`backbone.pt`/`env_ids.pt` in their original worktrees and the shared training prefix. Compare the normalized rank function to the Git blob `1a68c0b48dad7b30b849a6cb9582fc17041cfbd1`; abort on hash mismatch. Abort on backbone hash mismatch. Write only the specified audit JSON, never overwrite a Stage-1 artifact. The output is descriptive, not a model-performance result.

## Frozen audit gates

For each normalized partition, all must pass before a weighting training proposal can be considered:

1. Each environment has at least 20% of the 2,000,000 training rows.
2. Each environment has at least 1,000 BSI negatives, 100,000 BSI positives and 10,000 CTR positives. CVR positive and negative counts are recorded; **per-environment CVR AUC/loss is not an eligible weighting signal** when one environment has only about 1–2 positives.
3. In five contiguous train-internal folds, each fold × environment cell has at least 200 BSI negatives and 50,000 total examples. Folds are fixed by row order; no official validation/test data may enter these cells.

The script also reports the extreme effective sample size under per-environment weights clipped to `[0.1, 10]` and normalized to sample mean one. This is a diagnostic, not a GO gate. A future learned-weight run must show at least 0.01 absolute deviation from uniform in an explicitly defined statistic before claiming its mechanism activated; that check cannot be evaluated by this pre-run audit.

If any required support gate fails, mark this direction NO-GO and do not run weighting training. If all pass, write a separate treatment preregistration for `pure fair baseline`, `normalized + uniform`, `normalized + learned`, with same model seed, data prefix, Stage-1 identity, Stage-2 budget and head initialization. The last two arms must differ only by weighting. Train-internal holdout/cross-fitting may learn weights; official validation/test are evaluation-only. The supervisor must approve that treatment preregistration before formal runs.
