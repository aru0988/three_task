# Universal Representation Branch Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a historical four-arm branch and a clean B/U-only branch with a reusable, auditable MPT-Rec baseline.

**Architecture:** Protect the current paired work under a new remote ref before moving the old ref. Clean the new branch by dependency inventory rather than broad deletion, add a machine-readable baseline manifest and B/U ledger, verify saved evidence, then move the old branch with force-with-lease to its frozen four-arm tip.

**Tech Stack:** Git branches/worktrees, PowerShell, Python unittest, NumPy raw-result verifiers.

---

### Task 1: Protect the current paired history

**Files:**
- Verify: repository refs and worktree state

- [ ] Confirm the current worktree is clean and record local/remote SHAs for `exp/universal-representation` and `ff178de`.
- [ ] Create local branch `exp/universal-representation-bu` at the current tip `1d4de40`.
- [ ] Push `exp/universal-representation-bu` and verify the remote SHA matches before modifying the old ref.
- [ ] Attach or create an isolated worktree for the new branch.

Commands:

```powershell
git status --porcelain=v1
git rev-parse exp/universal-representation
git rev-parse origin/exp/universal-representation
git cat-file -t ff178de
git branch exp/universal-representation-bu 1d4de40
git push -u origin exp/universal-representation-bu
```

Expected: clean status, commit objects resolve, and local/remote new-branch SHAs are identical.

### Task 2: Inventory and clean the B/U branch tip

**Files:**
- Retain: `universal_experiment/paired.py`
- Retain: `universal_experiment/paired_run.py`
- Retain: `universal_experiment/verify_paired.py`
- Retain: `universal_experiment/convergence.py`
- Retain: `universal_experiment/test_paired.py`
- Retain: `universal_experiment/test_paired_run.py`
- Retain: `universal_experiment/test_verify_paired.py`
- Retain: `universal_experiment/test_convergence.py`
- Create: `docs/experiments/universal-representation-bu.md`
- Create: `results/universal/paired-baselines.json`
- Remove from the new branch only when not imported by retained code: four-arm runners/verifiers/tests and four-arm result directories.

- [ ] Use `rg` imports and test references to classify every file under `universal_experiment/` as paired-required, shared-required or four-arm-only.
- [ ] Use `rg --files results/universal` and the historical ledger to classify result directories.
- [ ] Write `paired-baselines.json` with the verified Census protocol tuple, raw/checkpoint paths, AUC values and hashes copied from saved reports.
- [ ] Write the B/U ledger containing the no-read and detached-read results, convergence limits and reproduction commands.
- [ ] Remove only files proven four-arm-only from the new branch tip with explicit paths.
- [ ] Update package entry points so the new branch exposes only paired B/U commands.
- [ ] Commit the cleanup without regenerating any experiment.

Expected: no B/U/R/G experiment command or four-arm result directory remains at the new branch tip, while paired imports resolve.

### Task 3: Verify the cleaned B/U branch

**Files:**
- Test: retained paired test modules
- Verify: both paired Census result directories

- [ ] Run the complete paired suite:

```powershell
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m unittest `
  universal_experiment.test_convergence `
  universal_experiment.test_paired `
  universal_experiment.test_paired_run `
  universal_experiment.test_verify_paired -v
```

Expected: all tests pass with zero failures.

- [ ] Verify both raw result directories:

```powershell
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-no-read-frozen-seed1685480945
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-read-detached-frozen-seed1685480945
git diff --check
```

Expected: both verifiers report `passed: true`; diff check is empty.

- [ ] Push the cleaned branch and verify local/remote SHAs match.

### Task 4: Restore the historical four-arm branch

**Files:**
- No working-tree edits; update refs only.

- [ ] Fetch the remote and confirm its old-branch SHA still equals the SHA recorded in Task 1.
- [ ] Update local `exp/universal-representation` to `ff178de` from a worktree-safe context.
- [ ] Push with an explicit lease:

```powershell
git push --force-with-lease=refs/heads/exp/universal-representation:<recorded-remote-sha> origin ff178de:refs/heads/exp/universal-representation
```

Expected: push succeeds only if no external update occurred.

- [ ] Verify `origin/exp/universal-representation` resolves to `ff178de` and `origin/exp/universal-representation-bu` resolves to the cleaned B/U tip.
- [ ] Confirm both worktrees are clean and report the two branch SHAs and retained protocol scopes.

### Task 5: Final evidence audit

**Files:**
- Verify: both branch trees and manifests

- [ ] On the old branch, confirm the four-arm runner, verifier, ledger and four-arm result directories remain present.
- [ ] On the new branch, confirm the paired runner, verifiers, baseline manifest, B/U ledger and two paired Census result directories remain present.
- [ ] Confirm no experiment was rerun and no raw artifact hash changed during organization.
- [ ] Record final local/remote SHAs and clean status in the handoff.
