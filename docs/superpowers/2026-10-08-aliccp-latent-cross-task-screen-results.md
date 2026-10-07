# AliCCP latent cross-task prediction diagnostic: fixed seed-1 screen

Preregistration: `docs/superpowers/specs/2026-10-08-aliccp-latent-cross-task-diagnostic-design.md` at `aa410cd`. This is a diagnostic of information in old-task predictions, not a deployed MPT-Rec improvement or a novelty claim.

- Valid formal run: `20261008-0559-p2M-v500k-t1M-m1688723512-latdiag-de3858b`, implementation commit `de3858b`, `git.dirty=false`; Stage-1 `s1-5c060b9c-m1688723512-e3-3a30e2c0`, fingerprint `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`. A/B/C = 1.4M/300k/300k; official dev/test = 500k/1M; head max 5 epochs/patience 2, best epoch 5; GPU RTX 3060 Laptop; reported wall 132.4 s. The backbone remained byte-identical and gradient-free.
- Controls: one A-trained BSI head; B-only fitted logistic probes; `base` = BSI logit, `full` = aligned BSI+CTR+CVR logits, `shuffle` = BSI plus jointly row-permuted CTR/CVR. Test source inspection was deferred until all probes were frozen. The same old-task marginals and three coefficients are present in full/shuffle; only row alignment differs.

| Split | BSI positives / n | Base AUC | Full AUC | Shuffle AUC | Full − max(Base, Shuffle) |
|---|---:|---:|---:|---:|---:|
| B (probe fit; descriptive) | 297,663 / 300,000 | 0.539477 | 0.582892 | 0.540122 | +0.042771 |
| C (out-of-sample) | 295,260 / 300,000 | 0.497029 | 0.551450 | 0.498721 | +0.052728 |
| Official dev | 496,205 / 500,000 | 0.479178 | 0.532450 | 0.479555 | +0.052895 |
| Official test (once) | 989,976 / 1,000,000 | 0.460926 | 0.519184 | 0.462077 | **+0.057107** |

Independent verification: `verification.json` passed 31/31 gates. The supervising thread separately read `raw.npz`, `probe_probs.npz`, `perms.npz`, `probes.json`, `reported.json` without importing the experiment scorer: sklearn recomputation matched all four split AUCs and deltas to <1e-10; direct logit→standardization→coefficient reconstruction for C and test full/shuffle predictions differed by at most 2.22e-16; full and shuffle vectors were not identical. Raw sample/positive counts, commit, fingerprint, Stage-1 identity, and clean-tree status agreed with the recorded configuration.

Classification under the added threshold: **positive information signal** (`Δtest≥+0.001`, `ΔC≥+0.001`, `Δdev≥0`, mechanism and identity gates passed). Per the frozen rule, proceed to fixed seeds 2–3 in a separate branch; do not tune this seed. Caution: both the baseline and full absolute test AUCs are low, and this probe is not an improved MPT-Rec head. One seed is insufficient for stability or transfer claims.

Quality-control event before the valid run: the first implementation accidentally permuted both `full` and `shuffle`; its independent verifier copied the error and falsely passed. That run and its derived summary/logs were deleted as invalid before the corrected formal run. The unpushed implementation commit was amended, regression tests now differentiate aligned and permuted columns, and the valid result above is the only formal direction-4 result retained.
