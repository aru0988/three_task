# AliCCP Stage-1 two-task baseline

This records the existing Stage-1 CTR/CVR model, not the newly added Stage-2 BSI task. No training or model-code change was made for this report. The values below come from the previously completed canonical Stage-1 artifact and its recorded validation/test evaluation.

| Split | Prefix samples | CTR AUC | CVR AUC |
| --- | ---: | ---: | ---: |
| Validation | 500,000 | 0.5493130789281618 | 0.5132119172500262 |
| Test | 1,000,000 | 0.5481837665250074 | 0.5280080347106456 |

## Provenance and scope

- Stage-1 artifact: `s1-5c060b9c-m1688723512-e3-3a30e2c0`; original code commit `b2e17f9`; model seed `1688723512`; environment seed `20261003`.
- Fixed prefix protocol: 2,000,000 train / 500,000 validation / 1,000,000 test examples; three epochs, patience 2, batch size 2,000. Epoch 3 was the selected checkpoint and contains both reported validation AUCs.
- Protocol fingerprint: `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`; model parameter hash recorded by the benchmark: `5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee`.
- Artifact file SHA-256: `meta.json` `5f3052514433451c132274355649afc9d2c168862225e4c17a02bb0d0b599dd3`; `backbone.pt` `a79fb892554d55abd066cc05dd90e6f7ff178e04d9414795a69907a42fc028e8`; `env_ids.pt` `7d159392a1816e7ba1b685422cd8ca92ecbbaa98065dc88cfb7f8c1b83e7589`.
- This is a single-seed, short-budget baseline. Its Stage-1 environment assignment collapsed strongly (`env_acc = 0.99977`; final environment counts 566 / 1,999,434), so it is valid as the exact historical performance comparator but not evidence that the environment-learning mechanism was healthy.

The artifact bytes remain in the local ignored `artifacts/aliccp_bench/` directory and are not committed to GitHub. Future structural comparisons should use the same prefix datasets, seeds, training budget, checkpoint-selection rule, and separate CTR/CVR AUCs. A different seed or budget requires its own paired baseline.
