# AliCCP parameter-efficient ensemble: consolidated ledger

This direction has one initial-screen source branch, `exp/aliccp-parameter-efficient-ensemble` @ `54b81683bfb121a67156be3d0cb7e9b071be64e6`, and one fixed seed2–5 expansion branch, `exp/aliccp-parameter-efficient-ensemble-seeds2-5` @ `8da8eb1b175ccec38887fecdbd1b9ca669286a32`. The expansion code was cherry-picked from the initial implementation, but the initial results document is unique to its source commit; this ledger preserves both. Source results: initial branch `docs/superpowers/2026-10-08-aliccp-parameter-efficient-ensemble-screen-results.md` and expansion branch `docs/superpowers/2026-10-08-aliccp-ensemble-five-seed-results.md`.

All three arms share seed, AliCCP 2M/500k/1M prefix, Stage-1 artifact, split, optimizer, and five-epoch budget. B is the original new-task head (8,129 parameters); C adds one rank-16 residual; E averages two rank-8 residual heads. C and E both have 9,169 trainable parameters. Equal parameter count does **not** mean equal compute. The primary comparisons are E−B and E−C BSI test AUC.

| Seed | B test AUC | C test AUC | E test AUC | E−B | E−C | E dual-head mean absolute prediction difference |
|---|---:|---:|---:|---:|---:|---:|
| 1688723512 | 0.598839218 | 0.608008050 | 0.624060414 | +0.025221196 | +0.016052364 | 0.000330829 |
| 1688723740 | 0.597442265 | 0.595664308 | 0.596584070 | −0.000858195 | +0.000919763 | 0.000090643 |
| 1688738016 | 0.616926565 | 0.620586953 | 0.619093004 | +0.002166439 | −0.001493950 | 0.000163193 |
| 1688749593 | 0.679916515 | 0.686589424 | 0.683942753 | +0.004026237 | −0.002646672 | 0.000052093 |
| 1688762746 | 0.657729030 | 0.655494754 | 0.657749111 | +0.000020081 | +0.002254358 | 0.000042185 |

Five-seed mean E−B `+0.006115152` with descriptive 95% t CI `[−0.007355247,+0.019585551]`; mean E−C `+0.003017172` with CI `[−0.006343331,+0.012377676]`. Under the later classification threshold +0.001, E−B is positive in 3/5 and E−C in 2/5, but **both comparisons are simultaneously positive in only 1/5**. Dual-head prediction difference exceeds the preregistered non-collapse threshold in only 2/5. E averaging is worse than the val-selected member on test in all 5 seeds. The historical preregistered terminal decision is **NO-GO: utility and two-head mechanism unstable**. The large first-seed gain is not a five-seed effect claim.

The maintained branch contains the seed2–5 implementation and tracked summaries. Historical raw predictions/checkpoints are Git-ignored local artifacts; the source results state that an independent parser recomputed AUCs from predictions, checked run identity and paired conditions, and did not rely on Claude Code's prose. This consolidation cites that prior audit rather than pretending raw predictions are included on GitHub.
