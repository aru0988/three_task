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

## AliCCP no-read, frozen U (30 epoch protocol)

Evidence: `results/universal/paired-aliccp-no-read-frozen-seed1688723512/`;
clean training commit `c5fef0b`; seed 1688723512, 30/30 caps, patience 5.
B is fully reused from `paired-aliccp-read-detached-frozen-seed1688723512`;
the U Stage-1 is trained fresh under the no-read policy rather than reusing the
read-detached weights. The independent verifier passed every check, including
the byte-identical no-read invariance of old-task predictions and the unchanged
frozen U hash.

| Arm | BSI val AUC | BSI test AUC | Stage-2 best/run |
| --- | ---: | ---: | ---: |
| B | 0.6652132051 | 0.7008913094 | 30/30 |
| U | 0.6645898085 | 0.6999214365 | 30/30 |

U−B is −0.0006233967 validation and −0.0009698728 test: no clear improvement.
The no-read U Stage-1 leaves the base hash exactly equal to B
(3949b7aa68ba05541814bcd658f7d8f3f99e53c246d59bee9f5c156d94bcce19), and after
Stage-2 the old-task predictions remain byte-identical to B: CTR
0.5804025544/0.5741207495 and CVR 0.5591672242/0.5917090420 validation/test,
with delta 0.0 against both own Stage-1 and B. This variant is lossless on
old tasks by construction; the frozen read-detached versus no-read contrast
stays descriptive at one seed each.

Stage-1 (U) is budget-exhausted (best 26/30), and both Stage-2 arms still
improve at the 30-epoch cap, so AliCCP remains "not sufficiently converged"
here as well. Cost: fresh U Stage-1 2256.5 s; Stage-2 U 314.3 s with 12,226
optimizer parameters (head only) vs reused B Stage-2 211.4 s with 8,129;
whole run ≈43 min wall. The AliCCP budget proposal from the late validation
curves remains deferred for approval.

## Census no-read, trainable U

Evidence: `results/universal/paired-census-no-read-trainable-seed1685480945/`;
clean training commit `1e365e2`. B and the no-read U Stage-1 are reused from
`paired-census-no-read-frozen-seed1685480945`; only the U Stage-2 is retrained,
20-epoch cap, patience 5. The independent verifier passed every check,
including the byte-identical no-read invariance of old-task predictions.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8663566164 | 0.8650532565 | 11 | 16 |

U−B is +0.0057433529 validation and +0.0051347670 test: positive on this
single seed. Trainable U converges by patience (best 11 of 16), and the old
tasks stay byte-identical to B after Stage-2 (Income 0.9451231372/0.9455806928,
Marital 0.9912542075/0.9909109553 validation/test, delta 0.0 against own
Stage-1 and B); the base hash is unchanged and the no-read invariance holds
even with U updated. This variant is lossless on old tasks, unlike the
trainable read-detached arm, whose Education val is 0.0011928 higher but pays
an Income validation drop. Both Stage-1 records remain not converged (budget
exhausted, best 18/20); both Stage-2 arms converge by patience. Cost: U
Stage-2 optimizer parameters 133,939 (head 43,448) vs B head 27,063; Stage-2
loops 125.2 s (U) vs 104.7 s (B); Stage-1 reused, whole run ≈2.8 min wall
including the cached-feature pass.

## AliCCP no-read, trainable U (30 epoch protocol)

Evidence: `results/universal/paired-aliccp-no-read-trainable-seed1688723512/`;
clean training commit `bc4feaa`. B and the no-read U Stage-1 are reused
(Stage-1 from `paired-aliccp-no-read-frozen-seed1688723512`); only the U
Stage-2 is retrained, 30-epoch cap, patience 5. The independent verifier
passed every check, including the byte-identical no-read invariance of
old-task predictions.

| Arm | BSI val AUC | BSI test AUC | Stage-2 best/run |
| --- | ---: | ---: | ---: |
| B | 0.6652132051 | 0.7008913094 | 30/30 |
| U | 0.6854804161 | 0.7041617112 | 30/30 |

U−B is +0.0202672110 validation and +0.0032704019 test: positive on this
single seed, and unfreezing U is the main source of the gain
(+0.0208906077 val / +0.0042402747 test over the no-read frozen arm; frozen
no-read U−B test was −0.0009698728). Old tasks after Stage-2 are
byte-identical to B (CTR 0.5804025544/0.5741207495 and CVR
0.5591672242/0.5917090420 validation/test, delta 0.0 against both own
Stage-1 and B), so with trainable U the no-read policy keeps the variant
lossless on old tasks, unlike the read-detached trainable arm whose CTR/CVR
fall by 0.011–0.026. The Stage-2 U validation curve is nearly identical to
the read-detached arm (0.68548 vs 0.68540 at the cap), so the old-task
integrity difference is attributable to the read policy rather than to the
new-task objective. Both Stage-2 arms still improve at the 30-epoch cap and
neither Stage-1 is converged (budget exhausted, best 26/30): AliCCP remains
"not sufficiently converged". Cost: U Stage-2 optimizer parameters 58,706
(head 12,226) vs B head 8,129; Stage-2 U loop 337.3 s (B reused, 211.4 s);
Stage-1 reused, whole run ≈7.3 min wall.

## AliCCP no-read, trainable U: Stage-2 40-epoch recheck

Evidence: `results/universal/paired-aliccp-no-read-trainable-s2budget40-seed1688723512/`;
clean training commit `92caecc`; seed 1688723512, Stage-1 30 (reused pair) /
Stage-2 40, patience 5. Triggered by the approved bounded-budget recheck after
both 30-epoch Stage-2 arms still improved at the cap. Both Stage-2 heads are
retrained from the same original initialization (Stage-1 pair reused from
`paired-aliccp-no-read-trainable-seed1688723512`, not the 30-epoch heads).
The independent verifier passed every check.

| Arm | BSI val AUC | BSI test AUC | Stage-2 best/run |
| --- | ---: | ---: | ---: |
| B | 0.6712760007 | 0.7057992926 | 40/40 |
| U | 0.6928681934 | 0.7087584663 | 40/40 |

U−B is +0.0215921928 validation and +0.0029591736 test: positive on this
single seed. Epoch 30 of both arms reproduces the 30-budget run exactly
(B 0.6652132051 / U 0.6854804161), confirming deterministic re-training from
the same initialization. Validation AUC over the extended segment:

| Epoch | B | U |
| ---: | ---: | ---: |
| 30 | 0.6652132051 | 0.6854804161 |
| 32 | 0.6665772430 | 0.6868575489 |
| 34 | 0.6679011954 | 0.6884039730 |
| 36 | 0.6690880391 | 0.6899139648 |
| 38 | 0.6702277971 | 0.6914110226 |
| 40 | 0.6712760007 | 0.6928681934 |

Both curves still rise at the 40-epoch cap (best_epoch_at_cap, +0.0061 B /
+0.0074 U over epochs 31–40), so neither arm is converged: AliCCP remains
"not sufficiently converged" and, per the approved plan, the budget is not
extended further without a new decision. Old tasks after Stage-2 stay
byte-identical to B (CTR 0.5804025544/0.5741207495 and CVR
0.5591672242/0.5917090420 validation/test, delta 0.0 against both own Stage-1
and B, predictions sha 3a82020a…), so no-read losslessness holds at the longer
budget. Reused Stage-1 records: both budget-exhausted (best 26/30). Cost:
Stage-2 B 401.7 s with 8,129 optimizer parameters vs U 469.6 s with 58,706;
whole run ≈14.5 min wall with Stage-1 reused.

## Census read-detached, frozen U: Stage-2 30-epoch recheck

Evidence: `results/universal/paired-census-read-detached-frozen-s2budget30-seed1685480945/`;
clean training commit `5b21e28`; seed 1685480945, Stage-1 20 (reused pair) /
Stage-2 30, patience 5. Triggered by the 20-epoch frozen U best sitting at the
cap. Both Stage-2 heads are retrained from the same original initialization
(Stage-1 pair reused from `paired-census-read-detached-frozen-seed1685480945`,
not the 20-epoch B). The independent verifier passed every check.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8622726923 | 0.8614862850 | 20 | 25 |

Both arms converge by patience under the 30-epoch budget, and the selected U
epoch-20 checkpoint reproduces the 20-epoch run's AUCs exactly, so the earlier
"best at cap" was not a truncated result. Frozen U keeps its read-detached
Stage-1 old-task predictions unchanged (delta 0.0 from Stage-1; Income test
−0.0007808833 and Marital test +0.0009854415 vs B, same as the 20-epoch run).
Cost: Stage-2 loops 124.1 s (B) vs 185.6 s (U); the AliCCP budget proposal
stays deferred for approval instead of extending now.

## Census no-read, trainable U: seed 1685463909

Evidence: `results/universal/paired-census-no-read-trainable-seed1685463909/`;
clean training commit `45b8866`; seed 1685463909, fresh paired training (no
reuse), Stage-1 and Stage-2 caps 20, patience 5. This is one of the two
approved stability seeds for the Census no-read trainable plan. The
independent verifier passed every check, including the byte-identical no-read
invariance of old-task predictions.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8635226747 | 0.8640280017 | 15 | 20 |
| U | 0.8672727177 | 0.8666747099 | 13 | 18 |

U−B is +0.0037500430 validation and +0.0026467082 test: positive on this
single seed. Trainable U converges by patience (best 13 of 18), while the B
Stage-2 arm is budget-exhausted (best 15 of 20) and is not declared converged.
Old tasks after Stage-2 stay byte-identical to B (Income 0.9448110629/0.9457288281
and Marital 0.9914603986/0.9913323056 validation/test, delta 0.0 against both
own Stage-1 and B); the no-read base is unchanged and B/U Stage-1 share the
same base hash (df15f2bd…). Both Stage-1 records are budget-exhausted (best
19/20 each). Cost: Stage-1 B 935.8 s vs U 1145.3 s; Stage-2 B 128.5 s with
27,063 optimizer parameters vs U 136.4 s with 133,939 (head 43,448); whole run
40.6 min wall (fresh, no reuse).

## Census no-read, trainable U: seed 1685477428

Evidence: `results/universal/paired-census-no-read-trainable-seed1685477428/`;
clean training commit `9597cb6`; seed 1685477428, fresh paired training (no
reuse), Stage-1 and Stage-2 caps 20, patience 5. This is the second approved
stability seed for the Census no-read trainable plan. The independent verifier
passed every check, including the byte-identical no-read invariance of
old-task predictions.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8632184334 | 0.8622872589 | 20 | 20 |
| U | 0.8648151095 | 0.8634055180 | 18 | 20 |

U−B is +0.0015966761 validation and +0.0011182592 test: positive on this
single seed, the smallest margin of the three. Both Stage-1 records converge
by patience (best 11/16), while both Stage-2 arms stop without converging
(B best at the 20/20 cap, U budget-exhausted with best 18/20). Old tasks after
Stage-2 stay byte-identical to B (Income 0.9442636743/0.9449610321 and Marital
0.9927782211/0.9926099163 validation/test, delta 0.0 against both own Stage-1
and B); the no-read base is unchanged and B/U Stage-1 share the same base hash
(68d5232d…). Cost: Stage-1 B 688.1 s vs U 868.2 s; Stage-2 B 118.1 s with
27,063 optimizer parameters vs U 143.7 s with 133,939 (head 43,448); whole run
31.6 min wall (fresh, no reuse).

## Census no-read trainable U: cross-seed summary

Scripted reconstruction from the committed `verification.json` records of the
three seeds (independent AUC recomputation; delta = U − B). Seeds 1685463909
and 1685477428 are canonical model seeds, so this varies the base/model seed
only and does not claim independence of every stochastic source. Test values
are reported for the frozen protocol and were not used for tuning.

| Seed | B val | B test | U val | U test | Δval | Δtest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1685480945 | 0.8606132635 | 0.8599184895 | 0.8663566164 | 0.8650532565 | +0.0057433529 | +0.0051347670 |
| 1685463909 | 0.8635226747 | 0.8640280017 | 0.8672727177 | 0.8666747099 | +0.0037500430 | +0.0026467082 |
| 1685477428 | 0.8632184334 | 0.8622872589 | 0.8648151095 | 0.8634055180 | +0.0015966761 | +0.0011182592 |

Paired cross-seed statistics over the three deltas (n = 3, t(0.975, df = 2)
= 4.3027):

| Statistic | Δval | Δtest |
| --- | ---: | ---: |
| mean | +0.0036966907 | +0.0029665781 |
| sample sd | 0.0020738532 | 0.0020272694 |
| 95% paired t interval | [−0.0014550462, +0.0088484276] | [−0.0020694384, +0.0080025946] |
| worst seed (1685477428) | +0.0015966761 | +0.0011182592 |
| positive (> 0) | 3/3 | 3/3 |
| classification-positive (≥ +0.001) | 3/3 | 3/3 |

All three seeds move in the same direction on validation and test, and the
worst seed stays above the +0.001 test threshold, but with n = 3 the paired
t intervals include zero: this is directional evidence, not a resolved
cross-seed effect. Old-task no-read losslessness holds byte-identically in
all three seeds (delta 0.0 against B). U costs more consistently (Stage-2
optimizer parameters 133,939 vs 27,063; Stage-1 and Stage-2 loops longer in
every seed). Convergence is mixed across seeds, and no seed shows an old-task
decline. On this evidence the Census no-read trainable variant is the most
supported candidate for continued validation, but cross-seed stability is not
established and no final plan is declared. AliCCP stability seeds remain
unrun: AliCCP is still "not sufficiently converged" and its budget decision
stays deferred.
