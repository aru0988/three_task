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
4. Run unit tests and a small-data smoke. Run the single preregistered full-data short experiment. Independently recompute AUC from saved raw NPZ and verify hashes. Append outcomes here and push code plus valid records (including negative results).

## Initial CensusIncome result — 2026-10-09

Preregistration commit `a52303f`; training code `51a1c7d6259a0ed069cfd5838e1096b8a982f904`, `dirty=false`. Raw evidence: `results/universal/census-seed1685480945/`. Total runner wall time 303.4104 seconds, excluding Python import/startup. This was one full-data short-budget experiment, not a multi-seed confirmation. All four heads used five training epochs, with validation-only checkpoint selection.

| New-task Education arm | Validation AUC | Test AUC | Selected epoch | Trainable Stage-2 parameters |
| --- | ---: | ---: | ---: | ---: |
| B: original baseline | 0.8527881906 | 0.8500685307 | 5 | 27063 |
| U: learned self-supervised expert | 0.8525359330 | 0.8498233900 | 5 | 43448 |
| R: frozen random expert control | 0.8523018109 | 0.8495605306 | 5 | 43448 |
| G: extra General residual control | 0.8506287793 | 0.8477891230 | 4 | 43448 |

| U minus control | Validation delta | Test delta | Added threshold classification |
| --- | ---: | ---: | --- |
| B | -0.0002522576 | -0.0002451407 | No clear improvement |
| R | +0.0002341221 | +0.0002628594 | No clear improvement |
| G | +0.0019071537 | +0.0020342670 | Positive relative to G control only |

**Preregistered decision: NO-GO for this configuration.** U does not beat B or the equal-capacity random-feature control by +0.001. Beating the weaker G residual arm is not an improvement over MPT-Rec. No automatic seed expansion or coefficient search is justified by this run.

### Old tasks and mechanism

| Original task | Validation AUC | Test AUC |
| --- | ---: | ---: |
| Income | 0.9373971774 | 0.9381688584 |
| Marital | 0.9909744587 | 0.9908426142 |

Both old-task AUCs and the full G/S parameter hash exactly reproduce the historical paired baseline. Fresh B also reproduces historical Education AUC. Old-task preservation is **by one-way gradient isolation and unchanged prediction paths**, not evidence that U improves existing tasks.

- U masked reconstruction MSE: 0.1141255274 versus zero-predictor 1.7759345770 on the fixed training-only probe.
- All 128 U dimensions have standard deviation >0.01; mean standard deviation 0.5451976006. Mean squared U/G cross-correlation: 0.0464687955. These show learning/noncollapse, not mathematical independence or future-task universality.
- Shuffling U at inference reduces Education AUC to validation 0.8069508788 / test 0.8036833618. The fitted head depends on the aligned representation, but this distribution-shift diagnostic does not establish incremental useful information: the matched random control is nearly as good.
- Improvement space: possible but unproven. The current self-supervised objective predicts task-shaped embedding coordinates, which may retain nuisance information or duplicate what the existing path already supplies. Only one seed and one budget exist; there is no cross-seed or long-budget evidence. A future falsifiable test could change only the self-supervised target to original masked fields, while retaining these controls. It is not run or authorized as a claimed success in this record.

### Reproduction and evidence

```powershell
# Run from this worktree; source supplies the existing dataset and fixed split.
python -m unittest universal_experiment.test_model -v
python -m universal_experiment census --source D:\MPT-Rec-three_task\MPT-Rec --out results/universal/census-seed1685480945
python -m universal_experiment.verify results/universal/census-seed1685480945
```

The runner rejects nonempty output paths and dirty formal starts. Use the recorded training commit and a new output path to reproduce. Six model tests passed before the formal run; the 1024-row/one-epoch smoke completed all four arms and is not included in scientific results. The independent verifier uses rank-statistic AUC, NumPy reconstruction of the U encoder/decoder, raw sufficient statistics for representation diagnostics, and explicit checkpoint/data-split checks. It does not call the training metric implementation or U loss implementation. `verification.json` is the machine-readable audit outcome.

Saved evidence includes train/validation/test sizes, source-file fingerprints, exact split indices, all validation epoch scores, raw validation/test labels and predictions, old-task predictions, selected checkpoints, per-step auxiliary losses, representation sufficient statistics and the masked training probe. All small checkpoints are versioned alongside the results. The decoder is training-only; Stage-2 U is frozen. Remaining limitations include a single held-out new task, a single seed, fixed initial normalization despite evolving embeddings, and only one residual readout design.

## AliCCP transfer preregistration — before outcomes

User requested the same branch and same method. No Census-driven tuning. Use canonical AliCCP baseline code from `8133d32`, whose core model/train/dataset files match this branch. Seed 1688723512, environment seed 20261003; fixed first 2,000,000 / 500,000 / 1,000,000 rows from train/dev/test, fingerprint `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`. Remove 101 and 301 as the established loader/vocabulary do; BSI target must not enter features. Original CTR/CVR Stage 1: 3 epochs, patience 2, batch 2000, LR .0001, dropout [.1,.3], original regularization and objective coefficients. Stage 2 BSI: max 5 epochs, patience 2, same batch/LR. Best checkpoints selected by original validation rules only.

U keeps hidden width 256, masking 30%, loss coefficients and normalization unchanged; output width 64 follows AliCCP G (Census G is 128). U seed remains 1685480946 and private mask seed 20261009. Auxiliary G extraction preserves both CPU/CUDA RNG streams so dropout cannot perturb base training. Readouts B/U/R/G are exactly the same designs as Census; B uses historical CPU BCE evaluation path. Frozen caches are disk-backed to bound RAM. Fresh base must equal historical parameter hash `5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee`, otherwise stop interpretation and investigate. Fresh B is the paired primary comparator; record any historical discrepancy, never silently substitute another seed.

The same GO rule applies: U exceeds B, R and G by test +.001 with positive val directions; old CTR/CVR unchanged; no U collapse; train-only reconstruction beats zero. Record per-arm raw predictions/epochs/weights, old-task AUC, provenance and independent rank-AUC audit. Strong historical environment collapse and existing duplicate rows remain limitations of this protocol; this experiment does not repair them. A negative outcome is retained without expansion or tuning.

Implementation plan (inline, approved cross-dataset transfer):

1. Add a dropout RNG regression test and AliCCP shape test; observe failures before adapting wrapper.
2. Import unchanged benchmark support from `8133d32`; add `universal_experiment/aliccp.py` for prefix verification, Stage-1 recording, bounded caches and the four heads. Keep Census runner unchanged.
3. Run model/adapter tests and small-prefix smoke, then commit clean code and execute the canonical short run once.
4. Independently recompute AUC, hashes, sample counts, U reconstruction and statistics; append results here and push this same branch.

## Initial AliCCP result — 2026-10-09

Preregistration `a9b8498`; clean training commit `d9a2a040c39ef8953fd326ed454e50ca9b486b85`. Evidence: `results/universal/aliccp-seed1688723512/`. Runtime 446.9629 seconds excluding interpreter imports; Stage 1 selected epoch 3, all four Stage-2 heads selected epoch 5. Ten model/transfer tests passed, including CPU and CUDA random-state isolation; a corrected small-prefix smoke completed before this formal run. Unchanged AliCCP benchmark support was imported from `8133d32`, not from an experimental model branch.

| BSI arm | Validation AUC | Test AUC | Stage-2 trainable parameters |
| --- | ---: | ---: | ---: |
| B: original baseline | 0.5781533414 | 0.5988392178 | 8129 |
| U: learned self-supervised expert | 0.5819449076 | 0.5986469590 | 12226 |
| R: frozen random expert control | 0.5799961346 | 0.6004200533 | 12226 |
| G: extra General residual control | 0.5793536311 | 0.6017272594 | 12226 |

| U minus control | Validation delta | Test delta | Added threshold classification |
| --- | ---: | ---: | --- |
| B | +0.0037915661 | -0.0001922588 | No clear improvement |
| R | +0.0019487730 | -0.0017730943 | No clear improvement |
| G | +0.0025912765 | -0.0030803004 | No clear improvement |

**Preregistered NO-GO.** U improves validation relative to all controls but loses on test against all three. R and G exceed B test AUC by +0.0015808355 and +0.0028880416 respectively; these are control-arm observations on a single seed, not evidence that self-supervised U helps, nor grounds to select a new winner using test results.

| Preserved old task | Validation AUC | Test AUC |
| --- | ---: | ---: |
| CTR | 0.5493130789 | 0.5481837665 |
| CVR | 0.5132119173 | 0.5280080347 |

Old-task AUC and the entire base parameter hash exactly match history; fresh B also exactly reproduces the historical BSI metrics and validation learning curve. Final environment counts remain 566 / 1,999,434. Preservation is by design, not an old-task improvement.

U masked reconstruction 0.5178612471 versus zero predictor 0.8125898242; all 64 dimensions have std >.01, mean std 0.6400022907; mean squared U/G correlation 0.0981099295. Shuffled-U validation/test AUC are 0.5552463182 / 0.5779087260: the head uses aligned U, but use does not imply incremental utility. The independent verifier recomputed the AUCs from raw arrays, checked matched labels/counts/selection/budgets/hashes, and reconstructed the U probe using NumPy; all checks in `verification.json` passed.

**Across the two datasets:** U-B test deltas are CensusIncome -0.0002451407 and AliCCP -0.0001922588. Both are no clear improvement, not marked large regressions. Reconstruction and noncollapse are demonstrated; incremental transfer value is not. AliCCP validation/test direction reversal and stronger simple controls weaken the case for simply extending this configuration. A different target/readout remains a hypothesis, not a demonstrated opportunity; no automatic tuning, extra seeds or longer training were performed. This does not rule out all G/U decompositions, only this detached embedding-reconstruction implementation at these budgets.

Reproduce from the recorded clean code commit with a fresh output directory:

```powershell
python -m unittest universal_experiment.test_aliccp universal_experiment.test_model -v
python -m universal_experiment aliccp --source D:\MPT-Rec-three_task\MPT-Rec --out results/universal/aliccp-seed1688723512
python -m universal_experiment.verify_aliccp results/universal/aliccp-seed1688723512
```

All result JSON, raw prediction arrays, U diagnostic statistics/probe and selected checkpoints are versioned. Large regenerable representation caches under `cache/` are local and ignored. The same branch contains both datasets; neither original baseline nor master is modified.
