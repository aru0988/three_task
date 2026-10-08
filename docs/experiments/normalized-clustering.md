# Stage-1 normalized environment clustering: consolidated experiment ledger

This direction changes Stage-1 environment clustering, then evaluates the downstream Stage-2 BSI task. It is **not** a frozen-backbone Stage-2-only intervention. All deltas below compare normalized Stage-1 plus its Stage-2 head against the raw Stage-1 plus its paired Stage-2 head at the same AliCCP seed and budget. The later +0.001 reclassification is separate from each historical preregistration verdict.

| Source step | Original branch tip | Evidence preserved |
|---|---|---|
| First normalized-clustering experiment | `exp/aliccp-stage1-normalized-env-clustering` @ `2058de843282de2dd7113c95cedb7d5f2e7de7a6` | Mechanism repair and first paired utility result. |
| Second seed | `exp/aliccp-stage1-normalized-clustering-seed-replication` @ `7ab445cb4126800c5d5e0decebb923c186780b23` | Mechanism replicated, but utility direction did not. |
| Third seed | `exp/aliccp-stage1-normalized-clustering-seed3` @ `f8ff4cf0735188551f2c4fa7c45b8458b6f56436` | Mechanism repaired and positive paired BSI delta. |
| Fourth/fifth seeds and terminal view | `exp/aliccp-stage1-normalized-clustering-seeds45` @ `3dd9bd333ea05422e5d83f8b5f8e0cfa1856bf22` | Five-seed aggregation and terminal verdict. Seed3 tip is an ancestor of this tip; first/second branches are separately cited. |

Source: terminal five-seed specification §9.5 on `3dd9bd3`, which replays prior-seed evidence rather than relabeling it.

| Seed | Δtest BSI AUC | Δval BSI AUC | Later class | Historical/branch verdict | Mechanism |
|---|---:|---:|---|---|---|
| 1688723512 | +0.00824283986406793 | +0.010059271876175169 | `POSITIVE` | `REPAIRED/NO_MATERIAL_DEGRADATION` | repaired |
| 1688723740 | +0.00045881952817949934 | −0.0014831644646635667 | `NO_CLEAR` | `NOT_SUPPORTED` for utility replication | 10/10 replication checks |
| 1688738016 | +0.02347892658289208 | +0.017557536803150087 | `POSITIVE` | `MECHANISM_REPAIRED_SEED3/SEED3_CONDITION_PASSED` | 14/14 |
| 1688749593 | −0.00317958913081684 | −0.0010362333908835453 | `NO_CLEAR` | mechanism repaired, utility unstable | 14/14 |
| 1688762746 | −0.00016803096046424937 | +0.003790467938876163 | `NO_CLEAR` | mechanism repaired, utility unstable | 14/14 |

Mean paired `Δtest = +0.005766593176771684`, reported 95% CI `[−0.007596229081178292,+0.01912941543472166]`; mean `Δval = +0.0057775757525308615`. The environment-balance mechanism passed **5/5**, but only **2/5** seeds reached +0.001 test improvement, one additional seed was marginally positive, and the CI includes zero. Terminal preregistered verdict: **`MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY`**. The two final no-clear seeds were classified `HEADROOM_NOT_EVIDENT` in the source. No stable downstream utility claim follows from the mechanism repair.

The maintained branch starts from the complete seed4/5 implementation. It does not overwrite prior seed-specific preregistrations or raw artifacts. Individual raw run JSON/checkpoints are Git-ignored local files; GitHub contains committed specifications, summaries, code, and source commit history, not all raw predictions.
