# U follow-up execution interfaces

User-approved scope: only B/U comparisons, old two tasks and the new third task;
prepare code now and let CC run formal experiments later on `exp/universal-representation-bu`.

- B remains structurally original MPT-Rec. Frozen-U predictions preserve the existing path.
- Trainable U uses fresh per-batch encoder outputs, no detached residual. Only the new-task
  head and U are optimized. No Stage-1 auxiliary loss is added during Stage-2 (single factor).
- Select head and U jointly by new-task validation AUC. Save both selected states.
- Freeze the whole original base. Re-evaluate both old tasks through the actual Stage-1
  wrapper, including ReadDetached projections/gates. ReadDetached can change old predictions
  even with a frozen base; no-read cannot. Report old-task AUC before/after Stage-2 separately.
- `--reuse-b` copies checked B evidence instead of training B. `--reuse-u-stage1` restores
  the exact selected U Stage-1 wrapper. Both require matching dataset/seed/data/budgets/patience,
  code identity and runtime; source checkpoint/prediction hashes and raw AUC are checked.
  Only the three reviewed historical training commits are admitted without a code-identity
  field. No automatic fallback to training on mismatch.
- `--reuse-stage1-pair` restores both selected Stage-1 arms, but freshly initializes and
  trains BOTH new-task heads under the requested Stage-2 budget. This is not optimizer resume.
- Explicit `--stage1-epochs` / `--stage2-epochs` support bounded budget comparisons.
  Missing optimizer state must never be described as resumed training.
- Existing output directories are never overwritten. Smoke files outside tracked results
  are implementation tests, never scientific evidence. Formal starts require clean commits.
- Independent verification checks raw AUC, selection, matching budgets/seed, checkpoint hashes,
  frozen base, U update/freeze, and old-task AUC deltas. Convergence is a separate field;
  verification passing does NOT mean convergence.

Out of scope: automated selection by test AUC, extra mechanisms, CC launch, formal training,
unbounded epochs, merging master or modifying four-arm history.
