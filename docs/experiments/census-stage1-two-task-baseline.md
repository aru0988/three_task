# CensusIncome Stage-1 two-task baseline

This is the **original Stage-1 Income/Marital model**, not the newly added Stage-2 Education task. No training or model-code change was made for this report. On 2026-10-09, the previously saved checkpoint was evaluated once on the existing validation and held-out test partitions using `MPTRec.predict()` and `census_benchmark.metrics.auc()` from fair-benchmark commit `87afe037505a29046e1a9a8ba66b950fb30fe820`.

| Split | Samples | Income AUC | Marital AUC | Income positives | Marital positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 49,881 | 0.9373971773636002 | 0.9909744586701101 | 3,050 | 21,655 |
| Test | 49,881 | 0.9381688583685694 | 0.9908426142246791 | 3,136 | 21,488 |

## Provenance and evaluation protocol

- Stage-1 artifact: `s1-096f8f16-m1685480945-e2-cb2094b3`, originally recorded at code commit `be20ff9`; model seed `1685480945`, environment seed and split seed `20260929`.
- Same original training protocol: two epochs, patience 2, batch size 256; 199,523 training samples. The saved checkpoint is from **epoch 2**, selected by the sum of the two validation AUCs. This is a short-budget baseline, not a multi-seed or converged estimate.
- `test.gz` was split into validation and test using the saved `split_indices.npz` (49,881 each). The split fingerprint `096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c` and each partition hash matched the original Stage-1 `meta.json` before evaluation.
- The loaded model's parameter hash matched `meta.json`: `a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85`. The evaluated validation AUC pair exactly matched the original **epoch-2** record, an independent check of the checkpoint/evaluator pairing.
- `meta.json` also reports `val_auc_marital_max = 0.9912246018108644`, but that is from **epoch 1**, a different checkpoint. Do not pair it with epoch-2 Income AUC or use it as the selected-model Marital baseline.

Input/artifact SHA-256 (file bytes): `train.gz` `e2f2d57925a8020d23fcd05802284b3bc1d91fcdd451f1eaaefd683e76d1b2dc`; `test.gz` `8de690375028ac531de37fcbb117bd6527c5f3542d3bfead8975226e45100cec`; `backbone.pt` `f41aa2c06deecc0a3fe29c94c91c1726a3414d7a6d57bd8e87d7a2c4fef9add2`; `split_indices.npz` `269cda6f1c34c403a8aeb7ed8c3560500180aeaff3874e520a48943d05ea9977`.

The model and split artifacts remain in the local `artifacts/census_stage2/` directory and are ignored by Git; this GitHub report records their IDs and hashes but **does not contain their bytes**. Future structural comparisons should use the same dataset files, split, model seed, Stage-1 training budget, checkpoint-selection rule, and separate Income/Marital AUCs; a different budget or seed requires its own paired baseline.
