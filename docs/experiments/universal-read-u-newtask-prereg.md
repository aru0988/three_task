# Read-U Stage-1 → new-task transfer: preregistration

Registered 2026-10-10 on `exp/stage1-read-u` before any Read-U new-task training. This is a one-seed-per-dataset screen, not a stability or significance claim. The eight existing Stage-1 checkpoints and their `*_metrics.json`/`*_predictions.npz` are inputs, not results to regenerate.

## Question and attribution

Does letting old-task heads read a detached Universal Representation during Stage-1 improve the representation subsequently available to a new task, **without giving the new-task head U as an extra input**? B is the no-read control; U reads the learned self-supervised encoder; R reads an equally initialized but frozen random encoder; G reads the general expert again. Each Stage-1 arm trained its own base and was selected on its old-task validation sum. Thus U−B measures the complete Stage-1 design, including any changed G/S trajectory. U−R is the key content control. U−G is reported as a secondary specificity control. No result may be described as the effect of adding U to a fixed G/S backbone.

The earlier `exp/stage1-universal-representation` AliCCP result is **not** a paired comparator: that U was not read by old-task heads, and its new-task head received U directly.

## Frozen inputs and preflight

Source directories under `results/universal/`: `census-read-u-seed1685480945` and `aliccp-read-u-seed1688723512`. For each B/U/R/G arm, load its own `{arm}_model.pt` at the epoch named in `{arm}_metrics.json`; never use the historical two-/three-epoch backbone as B. Checkpoints are local/gitignored, so the runner must record each file SHA-256, refuse missing or changed files, and verify the loaded base's `backbone_sha256` against that arm's metric file. B has unprefixed MPTRec keys; U/R/G store the base under `base.`. U and R contain auxiliary encoders, but **none** is passed to the new-task head. The supervising agent already independently checked all eight base hashes and validation argmax epochs, and the two U encoder hashes. The earlier claim that Census/AliCCP checkpoints were swapped was a false alarm; do not move or retrain them.

## Paired protocol

| Item | CensusIncome | AliCCP |
| --- | --- | --- |
| New task | Education | BSI |
| Model seed | 1685480945 | 1688723512 |
| Train / val / test rows | 199523 / 49881 / 49881 | 2000000 / 500000 / 1000000 |
| Split | existing canonical 20260929 split | existing p2M-v500k-t1M prefixes |

All four arms share the same NewTask head initialization (construct with the canonical `seed_model → base → NewTask` sequence, copy the initial head state to every arm and record its hash), optimizer, learning rate, minibatch size, row order, loss, 20-epoch maximum and patience 3. Freeze the loaded Stage-1 base. Use the same original NewTask G/S+environment path in every arm; no residual U/G/R input, no extra trainable parameter, no official val/test labels in training. Choose the best head by validation AUC under a declared deterministic tie rule; evaluate the selected head once on test. Do not choose epochs or hyperparameters from test.

Primary outcome is new-task test AUC. Record paired U−B and U−R deltas, corresponding validation deltas, U−G as secondary, every arm's best epoch, test/val sample counts, head parameter count, Stage-1 and Stage-2 wall time, base/head hashes, and raw labels/predictions. Old-task AUCs are copied from the existing Stage-1 raw predictions only as contextual trade-offs; they are not new Stage-2 measurements or evidence that old tasks were unaffected.

## Frozen decision rule

For each dataset separately, call this screen a positive transfer/content signal only if **both** U−B and U−R test AUC are at least +0.001 and both matching validation deltas are positive. Otherwise record NO-GO for that dataset; still report U−G and all signed deltas. If U−B is positive but U−R is not, the gain cannot be attributed to learned U content. If only test is positive while val disagrees, treat it as unstable. A positive single-seed screen justifies an independently registered seed expansion, not a generalization claim. Classification fields follow the project rule: Δtest ≥ +0.001 positive; −0.02 < Δtest < +0.001 no clear improvement; Δtest ≤ −0.02 clear decline. Retain all valid negative outcomes.

## Implementation and verification gates

Commit the runner and an independent raw-artifact verifier before a formal run; the formal runner must reject a dirty worktree and nonempty output directory. First smoke all arms in a separate `results/universal-smoke/` directory, with smaller row prefixes and 1–2 epochs. Smoke outputs never enter formal summaries. Formal output directories must be fresh and contain `config.json`, per-arm `*_metrics.json` and `*_predictions.npz`, checkpoint/head hashes, and a report. The independent verifier must recompute AUC from raw labels/predictions without calling the training metric function; check equal labels and head-init hashes across arms, row counts, selected epoch against validation history, budget, input checkpoint hashes, frozen-base hashes, `dirty=false`, and the no-U-input contract. The supervising agent reviews that verifier and raw evidence before any result is accepted or pushed. A future experiment giving the new head U directly requires a separate preregistration and branch.
