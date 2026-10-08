# AliCCP latent cross-task prediction diagnostic: consolidated ledger

Source branches: initial corrected seed1 `exp/aliccp-latent-cross-task-diagnostic` @ `72b13e2e2a8e9de7010a014524cecf62786e3d8d`; fixed seed2–3 extension `exp/aliccp-latent-cross-task-diagnostic-seeds2-3` @ `8900a8bb6e13de3db21cfa8d6cf3eb7a8aff96bd`. The initial branch contains its own result document, not present in the extension tree; both tips are preserved by this ledger. Frozen preregistrations were `aa410cd` and `4c48a3d` respectively.

This is a **diagnostic probe**, not a modified MPT-Rec model or deployed BSI predictor. `base` uses the frozen BSI-head logit; `full` adds aligned frozen old-task CTR/CVR logits; parameter-matched `shuffle` jointly permutes those old-task logits. `Δ = AUC(full) − max(AUC(base),AUC(shuffle))` on the stated split. All three seeds share the AliCCP fair prefix and train/dev/test budgets 2M/500k/1M, a 1.4M/300k/300k internal partition, 5-epoch/patience-2 BSI head, and frozen Stage-1. The original buggy seed1 execution (which permuted full as well as shuffle) was removed before push and contributes no metric.

| Seed | C Δ | Dev Δ | Test base/full/shuffle AUC | Test Δ | Complete per-seed signal |
|---|---:|---:|---|---:|---|
| 1688723512 | +0.052728 | +0.052895 | 0.460926 / 0.519184 / 0.462077 | +0.057107 | PASS |
| 1688723740 | **−0.000236** | +0.033463 | 0.423305 / 0.467284 / 0.423406 | +0.043878 | **FAIL: C direction** |
| 1688738016 | +0.006150 | +0.021382 | 0.421070 / 0.445745 / 0.421482 | +0.024263 | PASS |

Source three-seed result reports mean paired test Δ `+0.041749234`, SD `0.016525035`, descriptive t-based 95% CI `[+0.000698771,+0.082799697]`, complete-signal count **2/3**. This passes the preregistered group rule for a *follow-up information hypothesis*, not an architecture improvement claim. The lower CI limit is below +0.001; absolute test AUCs are modest and base probes below 0.5 on all seeds. The seed2 internal C failure is retained verbatim. The supervising audit recomputed split AUCs and predictions directly from raw NPZ files, with each run's independent verifier 31/31; those raw files remain Git-ignored local artifacts, not included in the GitHub commit tree.

The maintained branch starts from the seed2–3 implementation, which shares the initial corrected diagnostic method. The next potential architectural experiment requires a new preregistration and capacity-matched control; this ledger does not authorize or report such an experiment.
