# Universal Representation Branch Split Design

## Goal

Separate the completed four-arm Universal Representation work from the newer B/U-only program without deleting valid evidence or mixing protocols. The old branch remains the historical B/U/R/G record. The new branch becomes the only place for subsequent B/U experiments and reuses a canonical MPT-Rec baseline when its full protocol identity matches.

## Branch boundaries

### `exp/universal-representation`

- Move the branch pointer to `ff178de`, the last verified four-arm-only commit.
- Retain the B/U/R/G implementation, CensusIncome and AliCCP four-arm evidence, short-budget and 20-epoch records, and their existing documentation.
- Exclude every later paired B/U-only commit and result.

### `exp/universal-representation-bu`

- Create the branch from the current verified tip containing the paired results.
- Preserve the paired B/U implementation, tests, raw evidence, verification files, preregistration and convergence rules.
- Remove four-arm-only runners, verifiers, tests, result directories and four-arm narrative from the branch tip when they are not dependencies of the paired implementation.
- Retain shared dataset/model utilities only when the B/U runner imports or tests them.
- Maintain a B/U-only experiment ledger rather than mixing later outcomes into the historical four-arm document.

Git history remains recoverable. No valid raw result is deleted from the repository as a whole: four-arm evidence remains reachable from the old branch and paired evidence remains reachable from the new branch.

## Canonical baseline reuse

A baseline is identified by the tuple:

1. dataset and task ordering;
2. exact split or data-prefix fingerprint;
3. model seed and environment/split seed;
4. MPT-Rec code identity and hyperparameters;
5. Stage-1 and Stage-2 budgets, patience and validation selection rules;
6. model initialization identity;
7. raw prediction, checkpoint and configuration hashes.

If every field matches, later U experiments reference the existing B artifacts and do not retrain B. If any field changes, a new canonical B is trained once for that tuple and may then be reused by all matching U variants.

The current canonical CensusIncome entry is seed `1685480945`, split seed `20260929`, maximum 20 epochs and patience 5. Its Education validation/test AUC is `0.8606132635 / 0.8599184895`; Income/Marital test AUC is `0.9455806928 / 0.9909109553`. Raw predictions and provenance remain versioned in the B/U branch.

## Migration procedure

1. Confirm the current B/U tip and `ff178de` exist locally and remotely and that the worktree is clean.
2. Create `exp/universal-representation-bu` at the current verified tip and push it before changing the old branch pointer.
3. Use a separate isolated worktree for branch cleanup.
4. Inventory file dependencies before removing four-arm-only content. Do not remove a shared adapter imported by paired code.
5. Add a B/U-only ledger and canonical baseline manifest. Remove four-arm-only files from the new branch tip.
6. Run the complete paired test suite, independent verifiers for both paired Census runs, import/static checks and `git diff --check`.
7. Push the cleaned B/U branch.
8. Move `exp/universal-representation` locally and remotely to `ff178de` using an explicit force-with-lease update. Verify that the remote old branch and new branch resolve to their intended commits.

## Safety and failure handling

- The new branch must be pushed before the old branch is moved.
- Use `--force-with-lease`, never an unconditional force push.
- Abort the pointer move if the remote old branch changed after the initial audit.
- Do not rewrite commits or delete Git objects.
- Do not regenerate experimental evidence during this repository-organization task.
- A missing dependency, verifier failure or hash mismatch stops the cleanup before either final push.

## Acceptance criteria

- The old branch tip is `ff178de` and contains only the four-arm program at its tip.
- The new branch contains the paired B/U runner, tests, canonical baseline manifest, B/U ledger and both independently verified Census results.
- The new branch tip does not expose four-arm-only experiment entry points or result directories.
- Paired tests pass and both saved-result verifiers pass from the cleaned branch.
- Both worktrees are clean and both remote refs match their intended local refs.
