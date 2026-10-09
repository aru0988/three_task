# Universal Representation: preregistered CensusIncome screen

Approved direction: scheme 2, task-supervised General G plus masked-feature-prediction Universal U. Branch `exp/stage1-universal-representation`, base `infra/fair-benchmark` @ `153ecfe`. All experiments in this direction stay in this branch. No novelty claim.

## Frozen protocol (before outcomes)

- CensusIncome only initially; remove Education from inputs exactly as existing benchmark. Stage 1 uses only Income/Marital labels. Model seed 1685480945, split/env seed 20260929; 199523/49881/49881 rows; 2 Stage-1 epochs, 5 Stage-2 maximum epochs, patience 2, batch 256, LR .001 and original regularization. Select Stage-1 checkpoint by old-task validation AUC sum; select each Stage-2 head by Education validation AUC. Test is evaluated after selection.
- G/S and original supervised/environment objectives stay unchanged. U reads **detached** shared embeddings: current tasks influence U through input geometry, while U cannot change the base. This is a deliberately one-way first-stage extension, not evidence for fully joint optimization.
- Fit coordinate mean/std on training embeddings at initialization, then freeze them (std floor .1; standardized coordinates clipped to [-10,10]). No validation/test statistics. Mask 30% of fields (whole categorical embedding blocks; at least one field), with private RNG. U: [input plus field mask indicators, 256, 128] with ReLU between linear layers. Decoder: linear 128 -> input dimension. Detached targets are the standardized embedding coordinates. Masked MSE gives equal weight to each field. Clean U/G cross-correlation penalty .01; clean U variance hinge (std target .5) penalty .01. Reconstruction weight 1. No grid or post-result coefficient adjustment.
- U is not used in old-task heads. Fixed U initialization seed 1685480946 and masking seed 20261009. Model initialization and normalization fitting preserve the base RNG stream. A second frozen U retains the pretraining initialization for the random-feature control.
- Stage 2 freezes the entire base and U. Four arms: B original NewTask; U same head plus projected U residual; R identical residual head with random U; G identical residual head with G as the residual input. Residual = sigmoid(scalar) * linear(extra), scalar initialized to logit(.1), linear initialized to identity. All original-head weights are identical at initialization. U/R/G have equal trainable parameter counts. Same training order, budget, and selection rule. Cache frozen representations to avoid repeated backbone computation. Include deterministic shuffled-U evaluation as a diagnostic only (not checkpoint selection).
- Reuse historical baseline only if the freshly trained G/S parameter hash equals original `a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85`; otherwise report mismatch and run a fresh base-only Stage-1 with identical protocol before interpreting changes. B is freshly trained with this runner, and its raw predictions compared to historical B when available.

## Outcomes and decision

Record old-task val/test AUC; all four Education val/test AUCs, deltas, selected epochs, trainable parameters; U per-dimension std, near-zero-variance fraction, U/G correlation; masked reconstruction versus zero predictor; U auxiliary losses; gradient isolation tests; data/checkpoint/prediction hashes, raw labels/predictions, clean code commit, configuration and elapsed time.

Delta >= .001 = positive; -.02 < delta < .001 = no clear improvement; delta <= -.02 = clear decline. Historical rule unchanged. Candidate GO requires U-B test >= .001 and val > 0, plus U-R and U-G test >= .001 with positive val directions, old-task AUC changes within 1e-6 when base hashes match, finite noncollapsed U (at least half dimensions std > .01), and masked MSE below zero predictor on a fixed train-only diagnostic batch. Gate must not be interpreted as causal contribution; shuffled-U performance is diagnostic. If utility gates fail, record valid negative result and do not claim a universal representation. One seed cannot establish robustness or unknown-task generality.

## Implementation plan

1. Add failing tests for whole-field masking, detached gradients, train-only normalization, and residual-head initialization/capacity.
2. Implement U in `universal_experiment/model.py`; wrap baseline only at its forward/auxiliary-loss boundary, preserving old-task interfaces.
3. Add `universal_experiment/run.py`: strict data/checkpoint assertions, Stage-1 training, frozen-cache four-arm training, raw outputs. Commit implementation before formal run.
4. Run unit tests and tiny synthetic smoke. Run the single preregistered full-data short experiment. Independently recompute AUC from saved raw NPZ and verify hashes. Append outcomes here and push code plus valid records (including negative results).
