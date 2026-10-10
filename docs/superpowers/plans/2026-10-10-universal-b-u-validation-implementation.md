# Universal Representation B–U Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run converged, paired B–U experiments that isolate old-task U reading and third-task U freezing on CensusIncome and AliCCP.

**Architecture:** Add one unified B–U runner inside `universal_experiment` rather than another branch or direction package. It constructs matched `no_read` and `read_detached` Stage-1 models, gives both U variants the same direct-U Stage-2 residual head, supports either frozen cached U or online trainable U, and writes immutable raw artifacts. A separate verifier recomputes AUC and convergence/provenance checks from raw files.

**Tech Stack:** Python, PyTorch, NumPy, scikit-learn, unittest, Git.

---

### Task 1: Convergence policy

**Files:**
- Create: `universal_experiment/convergence.py`
- Create: `universal_experiment/test_convergence.py`

- [ ] **Step 1: Write failing tests** for strict validation improvement, patience stopping, cap detection, and budget expansion.

```python
def test_cap_hit_requires_budget_expansion(self):
    decision = convergence_decision([.50, .51, .52], best_epoch=3, budget=3, patience=5)
    self.assertFalse(decision.converged)
    self.assertEqual(decision.reason, "best_epoch_at_cap")

def test_early_stopped_curve_is_converged(self):
    curve = [.50, .52, .519, .518, .517, .516, .515]
    decision = convergence_decision(curve, best_epoch=2, budget=40, patience=5)
    self.assertTrue(decision.converged)
```

- [ ] **Step 2: Run** `python -m unittest universal_experiment.test_convergence -v` and confirm failure because `convergence.py` does not exist.
- [ ] **Step 3: Implement** `EarlyStopper`, `ConvergenceDecision`, and `next_budget`. Formal defaults are Stage-1/Stage-2 maximum 60 epochs and patience 6. A run is converged only when it stops through patience and its best epoch is strictly below the executed budget. A cap hit expands the paired B/U budget to 100, then 160; a cap hit at 160 is reported as unresolved rather than called converged.
- [ ] **Step 4: Run the test module and confirm all tests pass.**
- [ ] **Step 5: Commit** the tested convergence utility.

### Task 2: Matched Stage-1 B/U models

**Files:**
- Create: `universal_experiment/paired.py`
- Create: `universal_experiment/test_paired.py`
- Reuse: `universal_experiment/model.py`

- [ ] **Step 1: Write failing tests** for `no_read` and `read_detached` behavior.

```python
def test_read_detached_old_loss_cannot_update_u(self):
    model = make_tiny_read_model()
    old_task_loss(model(sample_features())).backward()
    self.assertTrue(all(p.grad is None for p in model.universal.parameters()))
    self.assertTrue(any(p.grad is not None for p in model.projections.parameters()))

def test_no_read_has_unchanged_old_prediction_path(self):
    base, wrapped = make_tiny_no_read_pair()
    self.assert_tensors_equal(base.predict(X), wrapped.predict(X))
```

- [ ] **Step 2: Run** `python -m unittest universal_experiment.test_paired -v` and confirm the desired APIs are missing.
- [ ] **Step 3: Implement** `ReadDetachedStage1` by porting only the validated U behavior from the historical Read-U branch: old fused representations receive `sigmoid(gate) * projection(U.detach())`; U still trains only through its existing auxiliary objective. Keep `UniversalStage1` as `no_read`.
- [ ] **Step 4: Add a paired Stage-1 trainer** that starts B and U from the same seed, uses the same data order and maximum budget, selects by the sum of the two old-task validation AUCs, records both old-task raw predictions, and invokes the convergence utility. B is the unmodified MPT-Rec model. U is the selected read policy.
- [ ] **Step 5: Run focused and existing model tests**, then commit.

### Task 3: Unified direct-U Stage-2

**Files:**
- Modify: `universal_experiment/paired.py`
- Modify: `universal_experiment/test_paired.py`

- [ ] **Step 1: Write failing tests** showing that both read policies use the same `ResidualHead`, B and U core heads have identical initialization, frozen U is absent from the optimizer, and trainable U is present while base parameters are absent.

```python
def test_trainable_u_optimizer_excludes_base(self):
    groups = stage2_parameters(head, universal=u, base=base, freeze_u=False)
    ids = {id(p) for p in groups}
    self.assertTrue(all(id(p) in ids for p in u.parameters()))
    self.assertTrue(all(id(p) not in ids for p in base.parameters()))
```

- [ ] **Step 2: Confirm the tests fail for missing Stage-2 APIs.**
- [ ] **Step 3: Implement frozen mode** using the existing cached U representation and `ResidualHead`; B uses the original `NewTask`. Remove R/G construction from this runner only, leaving historical runners untouched.
- [ ] **Step 4: Implement trainable mode** by recomputing `U(base.embedding_network(features).detach())` online and optimizing only U plus the U head. The complete base, old heads, and G/S path remain frozen. Re-evaluate old-task raw predictions after Stage-2 when the read policy is `read_detached`.
- [ ] **Step 5: Add equal-budget adaptive training** for B and U. If either arm hits the cap, rerun the pair at the next registered budget from identical initialization; never extend only the better arm.
- [ ] **Step 6: Run focused and regression tests**, then commit.

### Task 4: Formal runner, verifier, and CLI

**Files:**
- Create: `universal_experiment/verify_paired.py`
- Modify: `universal_experiment/__main__.py`
- Create: `universal_experiment/test_verify_paired.py`
- Modify: `docs/experiments/universal-representation.md`

- [ ] **Step 1: Write failing verifier tests** using a tiny synthetic result directory. Check raw rank-AUC, B/U label equality, seed/split/budget equality, clean commit, best-epoch selection, convergence status, hashes, old-task deltas, and U−B classification.
- [ ] **Step 2: Confirm verifier tests fail because the verifier does not exist.**
- [ ] **Step 3: Implement** `python -m universal_experiment <dataset> --paired --read-policy <no_read|read_detached> --u-mode <frozen|trainable> --seed <n> --out <dir>`. Reject nonempty outputs and dirty formal starts.
- [ ] **Step 4: Implement independent verification** without importing the training AUC helper. It must fail the run if either arm is marked converged without patience stopping or if B/U provenance differs.
- [ ] **Step 5: Run the full unit-test suite, a Census smoke, and an AliCCP smoke.**
- [ ] **Step 6: Commit and push the clean implementation before formal training.**

### Task 5: Experiment 1 — old-task read policy

**Files:**
- Create under: `results/universal/paired/`
- Modify: `docs/experiments/universal-representation.md`

- [ ] **Step 1: Run canonical seed 1** for CensusIncome `no_read/frozen` and `read_detached/frozen`, each as a B–U pair. If a pair hits the epoch cap, use the registered expanded budget.
- [ ] **Step 2: Run the same two pairs on AliCCP.**
- [ ] **Step 3: Run `verify_paired` and independently inspect raw prediction arrays, curve endpoints, selected epochs, hashes, sample counts, elapsed time, and old-task AUC.**
- [ ] **Step 4: Append results and select the read policy using third-task U−B first, then old-task material-decline rule and cost. Commit valid raw artifacts and push.**

### Task 6: Experiment 2 — frozen versus trainable U

**Files:**
- Create under: `results/universal/paired/`
- Modify: `docs/experiments/universal-representation.md`

- [ ] **Step 1: On the selected read policy and same seed, run `trainable` U against the same paired B protocol.**
- [ ] **Step 2: Verify convergence and raw AUC. For `read_detached`, verify both old-task AUCs after Stage-2; for `no_read`, verify bitwise-stable old-task predictions.**
- [ ] **Step 3: Select frozen or trainable U by third-task AUC, rejecting a candidate for final recommendation if either old task declines by at least 0.001 versus its paired pre-Stage-2 value.**
- [ ] **Step 4: Record, commit, and push the result.**

### Task 7: Experiment 3 — canonical-seed stability

**Files:**
- Create under: `results/universal/paired/`
- Modify: `docs/experiments/universal-representation.md`

- [ ] **Step 1: Run the selected configuration on canonical seeds 2 and 3 with fresh Stage-1 and Stage-2 training.**
- [ ] **Step 2: Expand any capped B–U pair equally until convergence or the registered 160-epoch ceiling.**
- [ ] **Step 3: Independently verify every seed, then calculate paired U−B mean, sample standard deviation, 95% t interval, positive count, worst seed, selected epochs, wall time, and parameter counts.**
- [ ] **Step 4: State the final recommendation or retain B if U lacks repeatable utility. Commit and push all valid evidence.**

### Task 8: Final verification

**Files:**
- Modify if required: `docs/experiments/universal-representation.md`

- [ ] **Step 1: Run all `universal_experiment` unit tests and both dataset verifiers.**
- [ ] **Step 2: Run `git diff --check`, confirm a clean worktree, and confirm the local/remote branch tips match.**
- [ ] **Step 3: Audit the direction ledger against every raw `report.json`; correct any mismatch before reporting completion.**

