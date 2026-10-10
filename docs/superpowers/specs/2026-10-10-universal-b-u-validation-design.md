# Universal Representation B–U validation design

## Scope

This direction covers the two original tasks and the currently added third task only. All follow-up code, tests, experiment records, and valid positive or negative results live on `exp/universal-representation`. The historical `exp/stage1-read-u` branch remains read-only evidence. No R/G arm, fourth-task claim, RP, or Null Expert is part of this experiment series.

## Question

Determine whether a self-supervised Universal Representation (U) improves the third task relative to the original MPT-Rec baseline (B), and which of two choices is most effective:

1. whether the two original task heads read U during their training; and
2. whether U is frozen or trainable while learning the third task.

Each experiment changes one factor only. B and U use the same data split, model seed, head initialization, training budget, validation selection rule, and original G/S pathway.

## Matched architecture

U is trained by its existing masked reconstruction, decorrelation, and variance objectives. Original-task supervision must not backpropagate into U. In the `no_read` arm, old task heads do not consume U. In the `read_detached` arm, each old task head consumes a projected, gated `U.detach()`, so U remains shaped only by its auxiliary objective while the old-task prediction path may learn to use it.

For both designs, the third-task U model uses the same direct residual pathway:

`original third-task fused representation + sigmoid(gate) * projection(U)`.

B uses the unmodified third-task head. This makes the Stage-2 comparison identical across the two Stage-1 designs.

## Experiment sequence

### Experiment 1: old-task read policy

Run `no_read` and `read_detached` as separate matched B–U pairs on CensusIncome and AliCCP. U is frozen during third-task training. Record third-task validation/test AUC, selected epoch, old-task validation/test AUC after Stage 1, parameter counts, and elapsed time.

This experiment answers whether allowing the original two tasks to use U creates a better base for the third task. Existing Read-U transfer results are retained as an indirect-transfer ablation but do not answer this question because their third-task head did not consume U directly.

### Experiment 2: frozen versus trainable U

Use the better Stage-1 read policy from Experiment 1. Compare two U variants against the same B result:

- `u_frozen`: cache U and train only the third-task head;
- `u_trainable`: recompute U online, optimize U and the third-task head, detach the base embedding so gradients cannot enter the frozen MPT-Rec backbone.

For `no_read`, updating U cannot change old-task predictions because old heads do not consume U; verify this from raw predictions. For `read_detached`, re-evaluate both old tasks after third-task training because their prediction paths consume U.

### Experiment 3: stability and efficiency

Only the best B–U configuration from Experiments 1–2 receives additional canonical seeds. Report paired U−B deltas, mean, sample standard deviation, 95% confidence interval, positive-seed count, worst seed, selected epochs, wall time, and trainable/inference parameter counts. No unrelated hyperparameter search is included.

## Decision rules

The third-task test AUC is primary; validation direction, old-task AUC, stability, and cost are supporting evidence. Per paired run:

- `U−B >= +0.001`: positive improvement;
- `-0.02 < U−B < +0.001`: no clear improvement;
- `U−B <= -0.02`: clear decline.

Old-task changes are reported rather than hidden. Because the scope is the current three tasks, a trainable U is acceptable only if its third-task gain is not purchased by a material decline in either current old task. A decline of at least 0.001 AUC in either old task is treated as material for ranking candidate configurations, while the raw value remains visible.

## Evidence and implementation constraints

- Add functions/configuration inside the existing `universal_experiment` package; do not create another direction branch.
- Never overwrite historical raw outputs. Every run gets a unique result directory and records branch, commit, dirty state, seed, split, budget, selected checkpoint, hashes, raw labels/predictions, and timing.
- Formal training starts only from a clean committed implementation. CensusIncome uses at most 20 epochs and AliCCP at most 30 epochs, both with patience 5. Convergence requires patience-based stopping and a best epoch earlier than the final two executed epochs. A cap hit is reported as not converged and inspected; it does not trigger an automatic large-budget run.
- Add tests first for detached old-task gradients, matched third-task initialization, frozen-base isolation, trainable-U optimizer membership, and old-task prediction preservation where mathematically required.
- Independently recompute AUC from raw predictions before adding results to the direction ledger.
- Valid negative results are retained. Confirmed code/configuration/analysis errors are excluded and regenerated from valid raw evidence.

## Expected output

The final recommendation selects one of `no_read/frozen`, `no_read/trainable`, `read_detached/frozen`, or `read_detached/trainable`, or concludes that B remains preferable. The conclusion must state third-task benefit, effects on both old tasks, stability, and computational cost.
