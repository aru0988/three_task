# AliCCP generalization-driven environment weighting: pre-run NO-GO

Status: **NO-GO_IDENTIFIABILITY**. No learned environment weighting, Stage-2 treatment, or official validation/test evaluation was run in this direction. This is a mechanism/support failure, not an AUC regression or a performance category.

Protocol and code commit: `57af530` (after supervisor audit `3853d63`), both on `exp/aliccp-generalization-env-weighting` from pure fair baseline `8133d32`. The audit used the fixed 2,000,000-row AliCCP training prefix, five already-frozen canonical Stage-1 backbones, rank normalization pinned to blob `1a68c0b48dad7b30b849a6cb9582fc17041cfbd1`, and the thresholds in `docs/superpowers/specs/2026-10-07-aliccp-env-weighting-audit-prereg.md`. Full test suite before audit: 42/42 passed. The raw seed-1 partition and its collapse were known in advance and are not presented as new evidence.

## Independently recomputed from raw audit JSON

| Model seed | Post-hoc normalized env0/env1 counts | Minimum CTR positives in an environment (gate ≥10,000) | Minimum CVR positives in an environment (diagnostic) | Minimum BSI negatives in a fold × environment cell (gate ≥200) | G1 / G2 / G3 |
|---|---:|---:|---:|---:|---|
| 1688723512 | 936,534 / 1,063,466 | 4,754 | 1 | 119 | PASS / FAIL / FAIL |
| 1688723740 | 1,077,778 / 922,222 | 4,481 | 2 | 173 | PASS / FAIL / FAIL |
| 1688738016 | 1,081,317 / 918,683 | 5,172 | 0 | 95 | PASS / FAIL / FAIL |
| 1688749593 | 1,062,736 / 937,264 | 4,380 | 3 | 145 | PASS / FAIL / FAIL |
| 1688762746 | 1,003,144 / 996,856 | 4,602 | 1 | 102 | PASS / FAIL / FAIL |

I independently re-summed each JSON partition: both environment counts total 2,000,000; CTR positives total 92,899, CVR positives 566, and BSI negatives 16,921 for each seed. All five Stage-1 backbone hashes matched their metadata, and the pinned rank function passed bit-equality tests. Balanced environment sizes therefore do **not** solve the rare-label and fold-support problem. The canonical raw seed-1 control is additionally collapsed at 566 / 1,999,434.

Raw outputs (committed under `artifacts/aliccp_bench/audit/`):

- `env-weighting-seed1.json`, SHA-256 `6adb1bf6ab5cc2b89d47fe96f0d50daf310d62d60e090159fc5349ca67a466a1`, audit wall 32.5 s.
- `env-weighting-seeds2-5.json`, SHA-256 `d08b35a64cda931f4ef7aadb943cb73768461d972fb4f6e90ca8514fe1b85a5f`, audit wall 101.6 s.

These outputs are **post-hoc partitions from the canonical raw backbones**, not the separately trained normalized-clustering backbones. Neither prior normalized-clustering utility nor any Stage-2 weighting gain may be inferred from them. The pre-registered G2 and G3 failures terminate this direction without training. Any attempt to replace contiguous folds, relax support thresholds, use a different signal, or redesign the router is a new hypothesis requiring a new independent branch and preregistration; it must not be retrofitted to these results.
