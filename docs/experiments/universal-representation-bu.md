# Universal Representation: Paired B/U Experiments

This branch compares the original MPT-Rec baseline **B**, whose structure contains no Universal Representation module, with one U-based modification at a time. Four-arm B/U/R/G experiments live only on `exp/universal-representation`.

## Protocol and baseline reuse

B is trained once per exact protocol identity and then reused. Reuse requires the same dataset/task order, split fingerprint, seeds, code and hyperparameters, initialization, budgets, patience, validation selection and saved artifact hashes. The machine-readable registry is `results/universal/paired-baselines.json`. Any mismatch creates a new baseline entry rather than silently reusing a result.

Current Census protocol: seed 1685480945, split seed 20260929, 199523/49881/49881 examples, Stage-1 and Stage-2 maximum 20 epochs, patience 5, validation-only selection. The canonical original MPT-Rec B has Education validation/test AUC 0.8606132635/0.8599184895 and Income/Marital test AUC 0.9455806928/0.9909109553.

Classification is appended without changing historical decisions: U−B test AUC >= +0.001 is positive; −0.02 < delta < +0.001 is no clear improvement; delta <= −0.02 is clear decline. Convergence, old-task behavior and cost are reported separately.

## Census no-read, frozen U

Evidence: `results/universal/paired-census-no-read-frozen-seed1685480945/`; clean training commit `b619f19`; independent verifier passed every check.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8592235244 | 0.8573714985 | 19 | 20 |

U−B is −0.0013897391 validation and −0.0025469910 test: no clear improvement. B converged by patience; U reached the budget before patience. Old tasks do not read U, so B and U have identical base hashes and old-task predictions. This variant is closed unless a new mechanism hypothesis is registered.

## Census detached old-task read, frozen U

Evidence: `results/universal/paired-census-read-detached-frozen-seed1685480945/`; clean training commit `427282d`; independent verifier passed every check.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8622726923 | 0.8614862850 | 20 | 20 |

U−B is +0.0016594288 validation and +0.0015677956 test: positive on this single seed. Income changes by −0.0007808833 and Marital by +0.0009854415 relative to B, so neither old task declines by 0.001. U reaches its best validation score at the 20-epoch cap and is not declared converged. Relative to no-read U, detached-read improves Education test AUC by +0.0041147866; this descriptive cross-screen contrast motivates retaining detached read for the next frozen-versus-trainable-U test.

## Reproduction and verification

```powershell
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m unittest universal_experiment.test_convergence universal_experiment.test_paired universal_experiment.test_paired_run universal_experiment.test_verify_paired -v
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-no-read-frozen-seed1685480945
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-read-detached-frozen-seed1685480945
```

The next authorized comparison is detached-read U with frozen versus trainable Stage-2 U. Each U variant is judged against the same canonical B; B is not retrained when the manifest identity matches.

## AliCCP detached old-task read, frozen U (30 epoch protocol)

Evidence: `results/universal/paired-aliccp-read-detached-frozen-seed1688723512/`;
clean training commit `bb00456`; seed1688723512, p2M/v500k/t1M, 30/30 epoch caps,
patience5. Interrupted predecessor had only config and was archived outside this worktree;
the completed run restarted from initialization, not from an optimizer checkpoint.

| Arm | BSI val AUC | BSI test AUC | Stage-2 best/run |
| --- | ---: | ---: | ---: |
| B | 0.6652132051 | 0.7008913094 | 30/30 |
| U | 0.6638495270 | 0.7015955320 | 30/30 |

U−B val −0.0013636781, test +0.0007042227: no clear improvement at this budget.
Both Stage-2 curves still improve at the cap; neither is declared converged.
The test sign alone is not evidence of a stable gain. No-read and trainable-U follow-ups
are planned to isolate read policy and freezing, without test-based tuning.

## Follow-up execution

Code and CC commands: `docs/superpowers/plans/2026-10-11-u-cc-execution.md`.
Trainable-U Stage-2 changes only head/U optimization; no extra SSL objective is added.
Old-task predictions are re-evaluated through the selected Stage-1 wrapper after Stage-2.
Baseline reuse is explicit and checked; longer-budget heads are fresh, not optimizer resume.

## Census detached old-task read, trainable U

Evidence: `results/universal/paired-census-read-detached-trainable-seed1685480945/`; clean training commit `2fe152e`. B and both Stage-1 arms are reused (canonical B from `paired-census-no-read-frozen-seed1685480945`, detached-read U Stage-1 from `paired-census-read-detached-frozen-seed1685480945`); only the U Stage-2 is retrained, 20-epoch cap, patience 5. The independent verifier passed every check, including frozen-base hashes, the U update hash, and the old-task re-evaluation artifacts.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8675493881 | 0.8661897805 | 11 | 16 |

U−B is +0.0069361247 validation and +0.0062712910 test: positive on this single seed. Unlike frozen U (best at the 20-epoch cap), trainable U converges by patience at epoch 11 of 16, and adds +0.0052766958 validation / +0.0047034955 test over the frozen arm at Stage-2.

Old tasks after trainable Stage-2 (U wrapper re-evaluated with the selected Stage-2 U; base hash unchanged):

| Old task | Val AUC | Δ vs own Stage-1 | Δ vs B | Test AUC | Δ vs own Stage-1 | Δ vs B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Income | 0.9427400723 | −0.0010750585 | −0.0023830649 | 0.9441695547 | −0.0006302547 | −0.0014111380 |
| Marital | 0.9922126808 | +0.0000850002 | +0.0009584733 | 0.9919440219 | +0.0000476251 | +0.0010330666 |

Income validation falls by ≥0.001 relative to both its own Stage-1 and B: this variant is a trade-off, not lossless on old tasks. Both Stage-1 records remain not converged (budget exhausted); both Stage-2 arms converge by patience. Cost: U Stage-2 optimizer parameters 133,939 (head 43,448) vs B head 27,063; Stage-2 loops 121.7 s (U) vs 104.7 s (B); Stage-1 reused, whole run ≈2.6 min wall including the cached-feature pass.

## AliCCP detached old-task read, trainable U (30 epoch protocol)

Evidence: `results/universal/paired-aliccp-read-detached-trainable-seed1688723512/`; clean training commit `692cf57` (includes the BSI-category reuse-key normalization without count/hash tolerance); seed 1688723512, 30/30 caps, patience 5. B is fully reused and the U Stage-1 is reused from `paired-aliccp-read-detached-frozen-seed1688723512`; only the U Stage-2 is retrained. The independent verifier passed every check, including frozen-base hashes, the U update hash and old-task re-evaluation.

| Arm | BSI val AUC | BSI test AUC | Stage-2 best/run |
| --- | ---: | ---: | ---: |
| B | 0.6652132051 | 0.7008913094 | 30/30 |
| U | 0.6853973403 | 0.7027656346 | 30/30 |

U−B is +0.0201841351 validation and +0.0018743252 test: positive on this single seed, and unfreezing U is the main source of the gain (+0.0215478133 val / +0.0011701026 test over the frozen arm at Stage-2; frozen U−B test was only +0.0007042227). Both Stage-2 arms still improve at the 30-epoch cap and neither is declared converged.

Old tasks after trainable Stage-2 (U wrapper re-evaluated with the selected Stage-2 U; base hash unchanged):

| Old task | Val AUC | Δ vs own Stage-1 | Δ vs B | Test AUC | Δ vs own Stage-1 | Δ vs B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CTR | 0.5667431706 | −0.0113704359 | −0.0136593839 | 0.5606956119 | −0.0114838402 | −0.0134251376 |
| CVR | 0.5398301896 | −0.0259367964 | −0.0193370346 | 0.5702182912 | −0.0172054198 | −0.0214907508 |

Old-task regressions are far beyond the 0.001 trade-off threshold on validation and test for both tasks: the +0.0019 test gain is not old-task-free and this variant is not lossless. Reused Stage-1 records: U converged by patience (best 24/29); B budget-exhausted (best 26/30). Cost: U Stage-2 optimizer parameters 58,706 (head 12,226) vs B head 8,129; Stage-2 loops 308.7 s (U) vs 211.4 s (B); Stage-1 reused, whole run ≈6.3 min wall. Per the execution plan, AliCCP remains "not sufficiently converged" at 30 epochs; a budget proposal based on the late validation curves is deferred for approval instead of extending now.
