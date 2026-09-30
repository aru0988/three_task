# 阶段 1 梯度冲突审计报告

> 梯度手术（PCGrad / GradNorm / CAGrad / Gradient Vaccine 等）是**既有文献**，本审计不主张新颖性；审计本身**不改变训练更新**，也不主张任何性能提升。

- 状态：**NO_ACTION**（actionable=False，smoke=False）
- 采样步数：68，按 epoch：{'1': 34, '2': 34}
- 判定组：`shared_trunk`，判定对：`income_vs_marital`
- meta：{"batch_size": 256, "commit": "87afe03", "env_coe": 0.1, "env_seed": 20260929, "epochs": 2, "group_sizes": {"env_classifier": 2, "shared_embeddings": 28, "shared_expert": 4, "shared_trunk": 32, "task_exclusive": 23, "unclassified": 0}, "loss_identity_gap": 0.0, "loss_identity_ok": true, "lr": 0.001, "max_batches": null, "meets_min_sampled_steps": true, "model_seed": 1685480945, "n_batches": 780, "n_parameters": 217548, "n_sampled_steps": 68, "n_train": 199523, "patience": 2, "reg_dnn": 3e-05, "reg_embedding": 0.006, "sampled_steps_per_epoch": 34, "smoke": false, "spaced_steps": 25, "split_fingerprint_sha256": "096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c", "split_seed": 20260929, "unclassified_parameters": [], "uni_coe": 0.9, "warmup_steps": 10}

## 门禁判据（预注册阈值）

| 编号 | 观测值 | 要求 | 通过 |
|---|---|---|---|
| G0 | 68 | >= 50 | 是 |
| G1 | 0.5 | >= 0.25 | 是 |
| G2 | 0.000804110823156787 | <= -0.05 | 否 |
| G3 | 1.0 | >= 0.95 | 是 |

- 未过原因：G2 未过：income-vs-marital 平均余弦（shared_trunk）；observed=0.000804110823156787，requirement <= -0.05
- 决策规则：G0 且 G1 且 G2 且 G3 → ACTIONABLE（另开分支试 PCGrad）；否则 NO_ACTION（主线继续）
  未过门禁即 `NO_ACTION`：不做梯度手术，主线继续。

## 结论解读

- 门禁未过（G2）→ `NO_ACTION`：按预注册**不做梯度手术**，主线继续；阈值事后不回改（要改只能新开预注册、从零重跑）。
- 关键区分：负余弦出现在 50.0% 的采样步（34/68），但平均余弦 = +0.000804（逐步幅度 min -0.624 / max +0.550）——冲突是**双向抖动、在均值上相互抵消**，而非持续的方向性对立；「半数步冲突」本身不构成 PCGrad 的依据。
- env 相关对（`income_vs_env` / `marital_vs_env`）的负余弦是 `ReverseLayerF` 的**机制本身**：只作诊断，不参与门禁，也不构成行动依据。

## income-vs-marital（shared_trunk，门禁所用）

| 统计量 | 值 |
|---|---|
| mean_cos | 0.000804 |
| neg_cos_rate | 0.500000 |
| n_negative / n_defined | 34 / 68 |
| nonzero_rate_a (income) | 1.000000 |
| nonzero_rate_b (marital) | 1.000000 |
| mean_of_ratio (‖g_a‖/‖g_b‖) | 0.636591 |
| ratio_of_means | 0.298293 |
| mean_dominance_ratio | 0.303138 |
| mean_norm_a / mean_norm_b | 0.170998 / 0.573255 |

## 各参数组（income-vs-marital）

| 组 | mean_cos | neg_cos_rate | nonzero_a | nonzero_b | dominance |
|---|---|---|---|---|---|
| `shared_embeddings` | 0.017113 | 0.411765 | 1.000000 | 1.000000 | 0.342030 |
| `shared_expert` | -0.000014 | 0.500000 | 1.000000 | 1.000000 | 0.296794 |
| `shared_trunk` | 0.000804 | 0.500000 | 1.000000 | 1.000000 | 0.303138 |
| `env_classifier` | n/a | n/a | 0.000000 | 0.000000 | n/a |
| `task_exclusive` | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.342268 |

## 环境分量（**仅诊断**，不参与门禁）

| 对 | 组 | mean_cos | neg_cos_rate |
|---|---|---|---|
| `income_vs_env` | `shared_embeddings` | 0.045553 | 0.417910 |
| `income_vs_env` | `shared_expert` | 0.019111 | 0.462687 |
| `income_vs_env` | `shared_trunk` | 0.019090 | 0.462687 |
| `income_vs_env` | `env_classifier` | n/a | n/a |
| `income_vs_env` | `task_exclusive` | n/a | n/a |
| `marital_vs_env` | `shared_embeddings` | -0.043699 | 0.582090 |
| `marital_vs_env` | `shared_expert` | -0.062314 | 0.552239 |
| `marital_vs_env` | `shared_trunk` | -0.063170 | 0.567164 |
| `marital_vs_env` | `env_classifier` | n/a | n/a |
| `marital_vs_env` | `task_exclusive` | n/a | n/a |

env 支路经 `ReverseLayerF` 反转，与任务的负余弦是**机制本身**而非新发现；因此环境冲突单独不构成任务间 PCGrad 的依据。

## 零梯度计数（精确零：embedding 行未被 batch 命中 / 不可达参数 / alpha=0）

| 分量 | 组 | 范数为零的步（by epoch） | 占比 | 零梯度张量对数 | 出现的步占比 |
|---|---|---|---|---|---|
| `income` | `shared_embeddings` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `income` | `shared_expert` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `income` | `shared_trunk` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `income` | `env_classifier` | {'1': 34, '2': 34} | 1.000000 | 136 | 1.000000 |
| `income` | `task_exclusive` | {'1': 0, '2': 0} | 0.000000 | 748 | 1.000000 |
| `marital` | `shared_embeddings` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `marital` | `shared_expert` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `marital` | `shared_trunk` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `marital` | `env_classifier` | {'1': 34, '2': 34} | 1.000000 | 136 | 1.000000 |
| `marital` | `task_exclusive` | {'1': 0, '2': 0} | 0.000000 | 748 | 1.000000 |
| `env` | `shared_embeddings` | {'1': 1, '2': 0} | 0.014706 | 28 | 0.014706 |
| `env` | `shared_expert` | {'1': 1, '2': 0} | 0.014706 | 4 | 0.014706 |
| `env` | `shared_trunk` | {'1': 1, '2': 0} | 0.014706 | 32 | 0.014706 |
| `env` | `env_classifier` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `env` | `task_exclusive` | {'1': 34, '2': 34} | 1.000000 | 1564 | 1.000000 |
| `reg` | `shared_embeddings` | {'1': 0, '2': 0} | 0.000000 | 0 | 0.000000 |
| `reg` | `shared_expert` | {'1': 0, '2': 0} | 0.000000 | 136 | 1.000000 |
| `reg` | `shared_trunk` | {'1': 0, '2': 0} | 0.000000 | 136 | 1.000000 |
| `reg` | `env_classifier` | {'1': 34, '2': 34} | 1.000000 | 136 | 1.000000 |
| `reg` | `task_exclusive` | {'1': 0, '2': 0} | 0.000000 | 884 | 1.000000 |

## 说明

- 梯度手术（PCGrad/GradNorm/CAGrad 等）是既有文献，本审计不主张任何新颖性。
- 本审计只读梯度，不改变训练更新，不主张任何性能提升。
- env 分量含 ReverseLayerF 反转：task-vs-env 的负余弦有设计上的先验，只作诊断。
- 未定义的复合量（损失空间不存在可加分解）：general_vs_specific。
- 组内余弦按该组所有参数梯度拼接后的整体向量定义，不是逐参数余弦再平均。

- 预注册：`docs/superpowers/specs/2026-09-30-census-stage1-gradient-conflict-audit-prereg.md`
