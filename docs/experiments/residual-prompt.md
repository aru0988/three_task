# Stage-2 Residual Prompt: consolidated experiment ledger

This ledger brings **both datasets and all 12 RP source branches** into one direction-level view. It is an index of historical experiments, not a pooled comparison across datasets or budgets. AUC differences are treatment minus the paired same-seed, same-split, same-budget control. Historical preregistered verdicts remain unchanged; the later `Δtest ≥ +0.001` classification is a separate field. `NO_CLEAR` denotes `−0.02 < Δtest < +0.001` and `CLEAR_DEGRADATION` denotes `Δtest ≤ −0.02`.

All source tips below are immutable Git commit identifiers from [the branch inventory](BRANCH_INDEX.md). Source specifications and tracked `SUMMARY.md` files are on GitHub; individual run `metrics.json`/NPZ files are Git-ignored and available only in local run directories unless separately published. Consequently a GitHub-only reader can verify the committed report and source code but **cannot independently recompute every AUC from raw predictions**. Historic independent-verifier outcomes are quoted as historic evidence, not a new verification performed by this consolidation.

## Source coverage and outcomes

| Dataset / experiment | Source branch tip | Historical result; current classification or limit |
|---|---|---|
| CensusIncome initial RP | `exp/stage2-residual-prompt-gate` @ `e0ad6afbf61de037819f64b74021ff7ea95384ca` | `VALID_NEGATIVE`; `NO_CLEAR` under later threshold; mechanism G1–G5 passed. |
| AliCCP initial RP | `exp/aliccp-stage2-residual-prompt-gate` @ `3e2f0833fdb8a4966d5d815b41a8cefcfb0887d6` | `VALID_NEGATIVE` under historical +0.0055 gate; later `POSITIVE`; M0/G1–G8 passed, inherited Stage-1 B1/B4 failures disclosed. |
| AliCCP second seed | `exp/aliccp-stage2-residual-prompt-seed-replication` @ `a40836e3d6a7f73c4943bada98139d414957301a` | `REPLICATION_SUPPORTED`; later `POSITIVE`; mechanism passed. |
| AliCCP 10-epoch pair | `exp/aliccp-stage2-residual-prompt-longer-budget` @ `e46e5d28ea572c1d6d5ace68b8100b848fe374e7` | Historical `PERSISTS`; later `POSITIVE`; best epoch right-censored at 10/10. |
| AliCCP 20-epoch pair | `exp/aliccp-stage2-residual-prompt-20epoch` @ `10e86dc6cbd047b103211ab75dd99c1c68db3526` | Historical `NOT_PERSIST` against +0.0055; later `POSITIVE`; effect narrows with budget, both 20/20 right-censored. |
| AliCCP five-seed expansion | `exp/aliccp-stage2-residual-prompt-five-seed` @ `28aa1feae1fea0762136beebdfe6eb8387edb170` | 5/5 later `POSITIVE`; historic 20-epoch launch condition passed, not a claim that 20-epoch effect persisted. |
| AliCCP shuffled condition | `exp/aliccp-stage2-residual-prompt-shuffled-condition` @ `fbfff0945c77c74866955238f67e9aeaa0774670` | `INVALID/MECHANISM_FAIL`: G5 activity band failed. Its secondary AUC must not be treated as a valid causal result. |
| AliCCP pinned alpha/condition | `exp/aliccp-stage2-residual-prompt-alpha-pinned-condition` @ `f08ae6e459ae9fe048daf11eda43799d19ccc7d1` | `CONDITION_ALIGNMENT_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`; correct condition did not beat shuffled. |
| AliCCP unconditional control | `exp/aliccp-stage2-residual-prompt-unconditional-control` @ `6375bbcb36f5df849b9ae64bced1b434641d1429` | `SAMPLE_CONDITIONING_NOT_SUPPORTED/GAP_BELOW_MATERIALITY`; per-sample and global condition effectively tied under the pinned-alpha control. |
| AliCCP delayed unfreeze | `exp/aliccp-stage2-residual-prompt-alpha-dynamics` @ `2a940b44a54470d6dbf2225669e560617ff36133` | `DYNAMICS_SUPPORTED` in preregistered partial form; half-gap target narrowly missed. |
| CensusIncome second seed | `exp/census-stage2-residual-prompt-seed-recheck` @ `41feb29067dc2dec94e330c6b7393762e8cd2f78` | Later `POSITIVE`; original absolute target 0.8521 still missed; expand decision triggered. |
| CensusIncome third seed / three-seed view | `exp/census-stage2-residual-prompt-3seed-expansion` @ `dac28dd0aa697152584f817c93f951d5f7cabcbf` | Three-seed `UTILITY_SEED_UNSTABLE/CLOSE_LINE`; mechanism 3/3 passed but utility not stable. |

## AliCCP canonical five-seed short-budget pairs

Source: `exp/aliccp-stage2-residual-prompt-five-seed` specification §10.1 and its historical independent verifier §11 (151/151 reported pass). Each row is a *within-seed* paired comparison; numbers must not be compared across different Stage-1 models as though paired.

| Seed | Baseline test / val AUC | RP test / val AUC | Δtest | Δval | Historical U1; later class |
|---|---|---|---:|---:|---|
| 1688723512 | 0.5988392178311113 / 0.5781533414372665 | 0.6042337617134492 / 0.5826350822240143 | +0.005394543882337954 | +0.00448174078674779 | FAIL at +0.0055; `POSITIVE` |
| 1688723740 | 0.5974422649550507 / 0.5809347091990792 | 0.6055453782825184 / 0.5895572066556973 | +0.008103113327467715 | +0.008622497456618139 | PASS; `POSITIVE` |
| 1688738016 | 0.6169265645506535 / 0.5839919245306395 | 0.6183829042203324 / 0.5853617876680048 | +0.001456339669678841 | +0.0013698631373653125 | FAIL; `POSITIVE` |
| 1688749593 | 0.6799165153727622 / 0.629554258588165 | 0.6848685140438336 / 0.6337670282397281 | +0.004951998671071434 | +0.004212769651563031 | FAIL; `POSITIVE` |
| 1688762746 | 0.657729030107454 / 0.601838646234007 | 0.6673750462948658 / 0.6135061036853381 | +0.009646016187411788 | +0.011667457451331131 | PASS; `POSITIVE` |

Five-seed mean `Δtest = +0.005910402347593546`, sample SD `0.0031538117454772883`, reported t-based 95% CI `[+0.0019944278461222166,+0.009826376849064875]`; 5/5 positive at +0.001, 2/5 meet the historical +0.0055 U1 gate. M0/G1–G8/A passed 5/5; inherited B-class failures are separately disclosed in the source. First-run dirty artifacts for affected seeds were excluded from canonical comparison and clean bit-equal reruns are documented in source §10.4.

## AliCCP budget and causal controls

All rows below have their own control and must not be pooled into the five-seed short-budget mean.

| Comparison | Main recorded test effect | Val direction | Historical interpretation / caution |
|---|---:|---|---|
| Seed 1688723740, 10 epochs: RP − paired baseline | +0.0074396653390377265 | +0.0048363447472773435 | `PERSISTS` at historical +0.0055 gate, both best epochs 10/10. Source `e46e5d2` spec. |
| Same seed, 20 epochs: RP − paired baseline | +0.004155967377889924 | +0.0023134736258212385 | `NOT_PERSIST` at historical gate, still later `POSITIVE`; budget trend +0.0081031 → +0.0074397 → +0.0041560. Source `10e86dc` spec. |
| Shuffled learned condition, seed 1688723740 | +0.0014787611000699474 vs baseline | See source | `INVALID/MECHANISM_FAIL` because ratio mean 0.003168 < 0.005 activity floor; no causal interpretation. Source `fbfff09` spec. |
| Pinned-alpha shuffled minus correct condition | +0.001861278011441092 | Shuffled also higher on val | Correct-minus-shuffled gap −0.001861278011441092; `CONDITION_ALIGNMENT_NOT_SUPPORTED`. This is not a learnable-vs-fixed comparison. Source `f08ae6e` spec. |
| Pinned-alpha conditioned minus unconditional | −0.000058180870650015315 | +0.0003818709963828715 | `SAMPLE_CONDITIONING_NOT_SUPPORTED`; equal α and parameter count, but one seed and one global condition. Source `6375bbc` spec §12. |
| Delayed-unfreeze `Δ_D` | +0.002594479377722947 | +0.002774889076071596 | `DYNAMICS_SUPPORTED` partial form; gap closure 0.4890085 < 0.5. Source `2a940b4` spec. |

## CensusIncome Education pairs

Source: original RP spec (`e0ad6af`), second-seed spec (`41feb29`), and three-seed spec (`dac28dd`). All use short 5-epoch paired Stage-2 runs with new task Education.

| Seed | Baseline test / best-val AUC | RP test / best-val AUC | Δtest | Δval | Later class; historical limit |
|---|---|---|---:|---:|---|
| 1685480945 | 0.8500685307175756 / 0.8527881906 | 0.8477376871797389 / 0.8501679359 | −0.0023308435378367465 | −0.00262025464356741 | `NO_CLEAR`; historical `VALID_NEGATIVE` against absolute 0.8521 target. |
| 1685463909 | 0.845256873871603 / 0.8476002975447581 | 0.8482477806320898 / 0.8499928855458697 | +0.0029909067604868556 | +0.0023925880011116396 | `POSITIVE`; historical absolute 0.8521 target still missed. |
| 1685477428 | 0.8427796966516823 / 0.8436747451693688 | 0.8475299750854703 / 0.8492923301160789 | +0.004750278433788058 | +0.0056175849467100525 | `POSITIVE`; historical absolute 0.8521 target still missed. |

Three-seed mean `Δtest = +0.0018034472188127222`, sample SD `0.003686884486472501`, reported t-based 95% CI `[−0.007355281572787369,+0.010962176010412814]`. Mechanism G1–G5 passed 3/3, but 2/3 signs positive, CI crosses zero, and worst seed is −0.0023308435; source terminal verdict is **`UTILITY_SEED_UNSTABLE/CLOSE_LINE`**. Do not infer stable cross-dataset benefit from the AliCCP five-seed result.

## Code and provenance status

- Maintained implementation currently starts from the AliCCP five-seed code (`28aa1fe`). The CensusIncome implementation and AliCCP ablation implementations are **not yet ported** into this branch; source commits remain the exact reproducibility reference. This document consolidates results but does not falsely claim one executable implements every variant yet.
- Code-tree comparison found distinct files `rp_shuffled.py`, `rp_pinned.py`, `rp_uncond.py`, and `rp_delay.py` on their source branches; a wholesale merge would delete the five-seed verifier and rewrite shared `bench.py`/`residual_prompt.py`. Explicit integration tests are required before any such merge.
- The 12 source branches remain live until archive tags, ledger/source checks, and compatible-code tests pass. No branch deletion is justified by the ledger alone.
