# Stage-2 Null Expert：源任务路由追加"零候选"

- **状态**：已预注册，尚未运行（代码与本说明同批交付；**未跑过任何本臂训练**）
- **日期**：2026-09-29
- **适用分支**：`exp/stage2-null-expert`（自 `infra/fair-stage2-benchmark` 独立拉出，HEAD `87afe03`）
- **协议依赖**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（本分支**只引用**该协议，不修改其文件；`census_benchmark/protocol.py` 与 `run_census_benchmark.py` 的协议语义不变）
- **唯一事实来源**：本文件。第 5 节的接受标准在**看到任何本臂结果之前**写死；与第 6.3 节阶梯验证一致（单 seed 单次短跑 → 达标才扩展）。

---

## 1. 动机与假设

阶段 2 的新任务头把 K 个旧任务（源任务）的 specific 表征按 router 权重混合成 `new_spec_rep`：

```
W = softmax(H_out · env_emb_k / T)        # (B, K)，Σ_k W_k = 1
new_spec_rep = Σ_k W_k · spec_rep_k
```

K = 2 且权重和为 1 ⇒ **新任务表征恒为两个旧任务表征的凸组合**：即使某个样本与两个旧任务都无关，router 也必须把概率质量分给它们，无法表达"这一样本不借用任何旧任务"。

**假设**：在候选集合中追加一个**值恒为零**的候选（Null Expert），并给它一个**可学习的 router key**，让 router 能对"不需要旧任务迁移"的样本把权重集中到零候选上，从而改善新任务 AUC。

**机制含义**：`new_spec_rep` 会被按 `(1 − W_null)` 缩放；`W_null → 1` 等价于关掉 specific 分支（此时 `fused_rep` 退化为 `gate_general · gen_rep`）。

---

## 2. 改动面（最小）

| 位置 | 改动 |
|---|---|
| `multitaskrec/model.py` → `NewTask.__init__` | 新增 `use_null_expert=False`；为 True 时在**末尾**创建 `null_key`（`nn.Parameter`，形状 `(rep_dim,)`，**零初始化**），是唯一新增参数 |
| `multitaskrec/model.py` → `NewTask.routing_weights(dnn_input, env_embs)` | 新增，forward 与诊断共用：`H_out = projection_network(dnn_input)`，keys = `[env_emb_0..K-1]`（True 时再拼接 `null_key`），`softmax(H_out @ keys / temperature)` → `(B, K)` 或 `(B, K+1)` |
| `multitaskrec/model.py` → `NewTask.routing_values(spec_reps)` | 新增，forward 与测试共用：`(B, rep_dim, K)`；True 时在**最后一列**追加 `zeros_like` → `(B, rep_dim, K+1)`，该列**逐位精确为零** |
| `multitaskrec/model.py` → `NewTask.forward` | 改为 `matmul(routing_values(...), routing_weights(...).unsqueeze(2))`；**其余语句（new_env_emb / gate / tower）与 master 逐字相同** |
| `census_benchmark/metrics.py` | 新增 `NullRouteStats`（M7）、预注册阈值常量与 `null_arm_verdict`；`evaluate_newtask` 在 `mechanism=True` 且检测到 `newtask.use_null_expert` 时追加 null 指标 |
| `run_census_benchmark.py` | `run_stage2(..., null_expert=False)`；`stage2` 子命令新增 `--null-expert`（默认 false）；`config.json` / `metrics.json` 记录 `null_expert`；启用时 `run_id` 追加 `-nullx` 后缀并写入 `null_arm` 判定 |

**明确不改**：Stage-1 相关类（`MPTRec` / `SharedBottom` / `MMOE` / `PLE` / `STEM` / `SparseSharing` / `SingleTask`）、`CensusIncome_*` / `AliCCP_*` 入口脚本、`multitaskrec/train.py`、`config.py`、`census_benchmark/protocol.py`（含 `SUMMARY_COLUMNS` 与 A/B 门禁阈值）、AliCCP / ByteRec 任何代码路径。

### 2.1 为什么默认臂能逐位等于 master

- `use_null_expert=False` 时**不创建** `null_key`：参数集合、`state_dict` 键集与 master 完全相同（`load_state_dict(..., strict=True)` 可过）。
- `routing_weights` / `routing_values` 在关闭时执行的就是 master 里那两行原式（`torch.mm` → `/temperature` → `F.softmax`；`torch.stack(spec_reps, dim=2)`），算子、操作数、顺序都不变 ⇒ 浮点结果逐位相同。
- `null_key` 用**零初始化**且建在 `__init__` 末尾：不消耗 RNG（两臂 shared 参数初始化逐位一致），不进入 `get_l2_reg()`（loss 形式不变）。

---

## 3. 不变量（由测试强制，不接受人工目测）

`census_benchmark/tests/test_null_expert.py`（新增）与 `test_smoke.py`（静态守卫收窄）：

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂前向/反向与 master **逐位一致** | `TestDefaultArmBitIdenticalToMaster`：从 git `master:multitaskrec/model.py` 动态加载参考实现，`torch.equal` 比对输出与每个参数的梯度 |
| I2 | 默认臂参数集合 = master 参数集合（无 `null_key`） | 同上 + `test_default_arm_has_no_null_key` |
| I3 | 开启后 router 权重为 `(B, K+1)` 且是概率 | `test_routing_weights_shape_is_k_then_k_plus_one`、`test_routing_weights_are_probabilities` |
| I4 | 零候选值**精确为零**，前 K 列逐位不变 | `test_null_value_is_exactly_zero_and_other_columns_untouched` |
| I5 | `null_key` 是唯一新增参数且可学习、有非零梯度 | `test_null_key_is_the_only_new_parameter`、`test_null_key_receives_gradient` |
| I6 | 开启后的前向等于本说明写死的公式 | `test_enabled_forward_matches_reference_formula` |
| I7 | M7 指标（`null_mean` / `null_top1_rate`）计算正确、流式累加、CPU 累加器 | `TestNullRouteStats`（含 CUDA 回归） |
| I8 | 接受标准常量与本文档一致 | `TestNullArmVerdict`（含 `test_doc_preregisters_same_numbers`） |
| I9 | 两臂共用同一 Stage-1 产物、冻结门禁 A1 仍 PASS、默认臂落盘形状不变 | `TestRunnerNullArm`（CPU 极小夹具端到端） |
| I10 | 除 `model.py` 外模型/master 文件零改动 | `test_smoke.py::test_static_guards_only_model_py_differs_from_master` |

> 测试用 CPU 极小夹具（`input_size=8`、`rep_dim=4`、64 样本、2 epoch），**不构成**任何性能证据，只验证语义与接线。

---

## 4. 运行方式

```powershell
# 0) 测试（全部不变量）
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v

# 1) 基线臂（协议既有路径，null_expert=False 等价于 master 行为；本臂已存在同 checkpoint 结果，无需重跑）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --epochs 5 --gpu 0

# 2) 处理臂（本实验唯一要跑的东西）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --epochs 5 --null-expert --gpu 0
```

- 两臂**必须**引用同一 `--stage1-dir`（`s1-096f8f16-m1685480945-e2-cb2094b3`，即基线 run 用的那份 checkpoint）。
- split seed / model seed / env seed 沿用协议默认，不另设；不搜参、不换结构、不调温度。
- 处理臂 `run_id` 形如 `YYYYMMDD-HHmm-s20260929-m1685480945-short-<commit>-nullx`，与基线 run 在 `SUMMARY.md` 中天然可区分。
- **前置条件**：开跑前先把本批改动提交，否则 `run_id` / `metrics.json` 里的 `commit` 记的是 `87afe03`（不含本实验代码），可审计性有缺口。

---

## 5. 预注册接受标准（写死；看到结果后不得修改）

| 项 | 值 |
|---|---|
| 对照基线 | **同 checkpoint、同 split、同 model seed** 的既有 run `20260929-1735-s20260929-m1685480945-short-904f8d0`（`stage1_id=s1-096f8f16-m1685480945-e2-cb2094b3`） |
| 基线值 | `AUC-Test-Education = 0.8500685307` |
| **主判据** | 处理臂 `AUC-Test-Education ≥ 0.8521` |
| **机制判据** | 处理臂 `null_top1_rate ∈ [0.05, 0.95]`（两端含等号；null 从不被选或总被选都视为机制未成立） |
| 判定 | 两条**同时**满足才允许进入多 seed 扩展；否则**停止** |
| 机械判定 | `metrics.null_arm_verdict(test_auc, null_top1_rate)`，结果落盘 `metrics.json["null_arm"]` |

**止损规则（与协议 6.3 一致）**：

1. 未达阈值 → 如实记录该次结果（`SUMMARY.md` 追加行 + `artifacts/` 留档），**不得**重跑挑好的、不得换阈值、不得跳到多 seed 或全量。
2. 机制判据不成立（null 权重被压到 0 或 1）→ 视为"该机制在本设置下不激活"，判为失败并停止，不做温度/初始化等补救性调参。
3. 阈值只允许在**看到任何本臂 run 结果之前**修改；每次修改单独 commit 并在本文件记录理由；已判定的 run 不得用新阈值重判。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。

---

## 6. 指标落盘

| 键 | 位置 | 含义 |
|---|---|---|
| `null_expert` | `config.json` / `metrics.json` 顶层 | 本 run 是否启用零候选（默认 `false`） |
| `mechanism.null_mean` | `metrics.json` | 验证集上 null 候选权重的样本均值（`W[:, -1]`） |
| `mechanism.null_top1_rate` | `metrics.json` | 验证集上 null 候选为最大权重的样本占比 |
| `null_arm` | `metrics.json` | `null_arm_verdict` 的完整判定（含 `test_auc`、`null_top1_rate`、`baseline_test_auc`、`auc_min`、`top1_range`、`checks`、`pass`） |

- null 指标只在**处理臂**出现；基线臂 `mechanism` 的键集保持 `{gate_mean, cos_gen_spec, gen_std, env_acc_stage1}` 不变（由 I9 强制）。
- 与 M3/M4 同口径：只在 **val** 上算一次（`mechanism=True` 的那一次），**test 不参与机制诊断**，更不参与任何选择。
- `null_arm` 是**臂级**判定，不并入 `judge()` 的 `overall_pass`（A/B 门禁语义不变，`protocol.py` 未动）。

---

## 7. 报告与分支处置

- `SUMMARY.md` 只追加、不重写：处理臂即使失败也必须有行（协议 7.3「不得只报告成功的 run」）。
- 分支处置（用户硬规则）：**无论成功失败，只保留 `exp/stage2-null-expert` 本分支，不合并回 `master`**；失败同样记录在案（`SUMMARY.md` + 本文件补记 `结果` 小节）。
- 与本实验无关的既有探索（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 / attn-env-prior）一律不引用、不重建。

---

## 8. 结果（空：运行后追加，不得回填预期值）

| run_id | AUC-Test-Education | null_mean | null_top1_rate | 判定 |
|---|---|---|---|---|
| （待运行） | | | | |
