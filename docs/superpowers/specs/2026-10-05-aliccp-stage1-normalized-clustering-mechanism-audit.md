# AliCCP Stage-1 归一化聚类：seed1/seed2 既有证据的独立机制审计（seed3 实验的前置审计）

- **状态**：审计完成（本文件在 seed3 分支任何新 run 之前提交；结论只引用既有产物与已提交记录）
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage1-normalized-clustering-seed3`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出）
- **被审计对象（两个既有分支，均为只读引用）**：
  - `exp/aliccp-stage1-normalized-env-clustering`（tip `2058de8`，已 push）：seed1 = `1688723512` 的归一化聚类修复；结论 `REPAIRED` + `NO_MATERIAL_DEGRADATION`（预注册 `4b83863`、实现 `13d2100`、结果 `529090e`/`2058de8`）
  - `exp/aliccp-stage1-normalized-clustering-seed-replication`（tip `7ab445c`，已 push）：seed2 = `1688723740` 的独立复现；结论 `NOT_SUPPORTED`（机制 10/10 复现，效用方向未复现；预注册 `8ef0c9d`、移植 `a39c170`、结果 `7b2c26a`/`7ab445c`）
- **审计工具（入库）**：`aliccp_benchmark/audit_prior_seeds.py`（只读；无训练；重新落盘的捕获不做任何写操作）。输出：`artifacts/aliccp_bench/audit/prior-seeds/audit_prior_seeds.json`（gitignore）。
- **证据物理位置**：seed1 raw 产物与 run 在主 worktree `D:\MPT-Rec-three_task\MPT-Rec\artifacts\aliccp_bench\`；seed1 norm、seed2 raw、seed2 norm 产物与 run 在 `D:\MPT-Rec-three_task\MPT-Rec-exp-aliccp-atten-seed2\artifacts\aliccp_bench\`。审计**只读打开**，未复制、未修改任何既有产物。

---

## 1. 审计方法

对四个（seed × 臂）配对做**从原始产物出发**的机械复核（不依赖文档转述）：

1. **定位与哈希**：按钉死 `stage1_id` / `run_id` 定位产物；重算 `env_ids.pt` 与 `backbone.pt` 的 sha256 == meta 记录；重建 `MPTRec` 载入 backbone，复算 `protocol.backbone_sha256` == meta 记录；
2. **标签比例（交叉表）**：独立扫描训练集前缀 2,000,000 行原始标签（click/purchase），与产物 `env_ids.pt` 逐位对齐，复算 env × 标签交叉表（含"env_0 ≡ purchase=1 集合"逐位判定）；
3. **聚类可预测性/退化诊断**：重建模型 + `AliCCPDataset`，复算 env_pred 探针（前 200 个 batch = 400k 行，整数计数口径；acc / balanced_acc / 各环境 recall）；
4. **下游 Stage-2**：读取 run 的 `metrics.json` / `gate_report.json`，逐项断言（best val BSI、test BSI、逐 epoch、gate_mean、A/B 门禁逐项、hard_pass、commit、dirty）；
5. **模式导出**：由上述复算值机械导出 B4 修复与效用方向模式（`pattern` 块）。

**审计结果：四个臂 ×（stage1 13 项 + 交叉表 4 项 + 探针 3–4 项 + run 9 项）全部 pin 通过（`all_pins_pass = true`，wall 79.3 s，审计 commit 记录于 JSON）。** 其中探针值为本次**独立复算**，与既有记录逐位一致（seed1 raw 的 `env_acc` 为旧 float32 均值口径 0.99977，复算整数计数口径 0.9997699856758118，差异 ≤1e-6，系 `ebefd19` 的口径修正，属已知记录差异）。

---

## 2. 机制证据（Stage-1；复算值）

### 2.1 四臂聚类事件与环境分配

同一前缀 `p2M-v500k-t1M`（指纹 `5c060b9c…`）、同一 env seed `20261003`、每臂恰 1 次聚类事件（epoch 2 末）：

| 臂 | stage1_id | 事件（diff_num / env_0 / env_1） | 占比 | B4（协议需 ≥5%/≥5%） |
|---|---|---|---|---|
| seed1 raw | `s1-5c060b9c-m1688723512-e3-3a30e2c0` | 1000466 / **566** / 1999434 | **0.0283%** / 99.9717% | **FAIL** |
| seed1 norm | `s1-5c060b9c-m1688723512-e3-ad3b353f` | 999966 / 959244 / 1040756 | 47.9622% / 52.0378% | **PASS** |
| seed2 raw | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | 1000418 / **1532** / 1998468 | **0.0766%** / 99.9234% | **FAIL** |
| seed2 norm | `s1-5c060b9c-m1688723740-e3-820697f6` | 998878 / 1106286 / 893714 | 55.3143% / 44.6857% | **PASS** |

**结论 A1（环境均衡修复，跨 seed 稳定）**：raw 臂两 seed 均坍缩到 ~10⁻⁴–10⁻³ 量级小组（B4 FAIL）；归一化臂两 seed 均 >44%（B4 PASS）。修复由 seed1 的认证捕获（`--reproduce` 10/10 逐位复现）归因于跨任务损失尺度不可比（平均 65×、负例中位 383×；raw argmin 退化为 CVR 标签函数），归一化（逐任务 rank/quantile）对单调重标定免疫。

### 2.2 标签比例（env × 标签交叉表；本次独立复算）

| 臂 | env_0 ∩ purchase=1 | env_0 ∩ click=1 | env_0 ≡ purchase=1 集合（逐位） | env_0 ⊆ purchase=1 |
|---|---|---|---|---|
| seed1 raw | 566 / 566（100%） | 566 | **True**（逐位相等） | True |
| seed1 norm | **564**（0.0588%·|env_0|） | 4807 | False | False |
| seed2 raw | 564 | 564 | False（env_0 含 968 非 purchase 行；env_1 含 2 purchase 行） | False |
| seed2 norm | **565**（0.0511%·|env_0|） | 4803 | False | False |

**结论 A2（非标签恢复，跨 seed 稳定）**：raw 臂分配由 CVR 标签主导（seed1 逐位等于标签集合；seed2 同型但非严格）；归一化臂 env_0 中的 purchase 行占比 ≤0.06%，分配不再是标签函数。

**披露**：三臂 env_0 ∩ purchase=1 恰为 564/564/565（train 前缀 purchase 总数 = 566）——归一化后 env_0 中仍几乎不含 purchase 正例；即归一化**不把** CVR 正例并入 env_0，而是把损失序结构（非标签）作为划分依据（env_0∩click1 = 4807/564 与 raw 的 566 不同）。这是分配语义的记录，不作优劣判断。

### 2.3 聚类可预测性/退化诊断（env_pred 探针；本次独立复算）

| 臂 | acc（整数计数） | balanced_acc | recall(env_0) | recall(env_1) | 退化签名 |
|---|---|---|---|---|---|
| seed1 raw | 0.999770 | **0.500000** | **0.000000** | 1.000000 | 多数类失真：acc≈1 但从不命中 env_0 |
| seed1 norm | 0.183750 | 0.184482 | 0.203993 | 0.164972 | 两类同量级，无多数类失真 |
| seed2 raw | 0.999252 | **0.500000** | **0.000000** | 1.000000 | 同型退化 |
| seed2 norm | 0.668745 | 0.647417 | 0.880935 | 0.413898 | 两类均被命中 |

**结论 A3（退化修复，跨 seed 稳定）**：raw 臂 balanced_acc = 0.5 且 recall(env_0) = 0（acc 完全由多数类贡献）；归一化臂 balanced_acc 与 acc 同量级、两环境 recall 均 > 0。注：归一化臂 env 头的绝对精度不高（seed1 仅 0.18）——聚类结果与 env 头拟合是两件事，探针只证明塌缩签名消失，不主张 env 头质量。

---

## 3. 下游 Stage-2 证据（配对 Δ；复算自 run 记录）

配对口径：同 seed，norm 臂 run − raw 臂 run（未改动 NewTask 头；协议默认 5 epoch/patience 2/batch 2000；同前缀、同 stage-2 过程）。

| 项 | seed1 raw（`…-short-b2e17f9`） | seed1 norm（`…-norm-529090e`） | seed2 raw（`…-short-79b5e07`） | seed2 norm（`…-norm-7b2c26a`） |
|---|---|---|---|---|
| best val BSI | 0.5781533414372665 | 0.5882126133134417 | 0.5809347091990792 | 0.5794515447344156 |
| test BSI | 0.5988392178311113 | 0.6070820576951792 | 0.5974422649550507 | 0.5979010844832302 |
| **Δtest（norm−raw）** | — | **+0.0082428399** | — | **+0.0004588195** |
| **Δval（norm−raw）** | — | **+0.0100592719** | — | **−0.0014831645** |
| 逐 epoch val BSI | 0.4937/0.5037/0.5178/0.5434/0.5782 | 0.4991/0.5086/0.5289/0.5584/0.5882 | 0.4646/0.4856/0.5142/0.5509/0.5809 | 0.4667/0.4870/0.5142/0.5492/0.5795 |
| gate_mean（val，routing 诊断） | [0.851149, 0.148851] | [0.551520, 0.448480] | [0.789178, 0.210822] | [0.808680, 0.191320] |
| B 类门禁 | B1 FAIL / B4 FAIL | B1 FAIL（共享边界）/ B4 PASS | B4 FAIL | 全 PASS |
| hard_pass | false | false | false | true |
| commit / dirty | b2e17f9 / false | 529090e / false | 79b5e07 / false | 7b2c26a / false |

- **结论 A4（效用方向，跨 seed 不稳定）**：seed1 双向为正（Δtest +0.00824、Δval +0.01006）；seed2 Δtest 边际为正（+0.00046，比 seed1 小 ~18×）而 Δval 反向（−0.00148）。两 seed 的 val 方向不一致。
- **披露（seed2 raw 产物系谱）**：seed2 raw 配对基线（stage1 `79b5e07`、run `79b5e07`）产生于 `exp/aliccp-stage2-attenuation-seed-replication` 系谱，其 bench 含 `spec_attenuation` 扩展但默认 1.0（对 stage-2 无行为差异）；seed2 分支已做 35/35 只读参考核验（含 head 严格载入 + val/test BSI 与 gate_mean 逐位重评测，`verify_reference_seed2.py`），本次审计又独立复算了其 meta/run 哈希与探针，全部通过。
- **披露（seed1 norm 的 B1）**：seed1 norm run 的 B1 FAIL 为**共享既有边界**（其 CTR-val 0.5475 与 raw 基线 0.5493 同为 <0.55 的协议活动下限，Δ 在容差内）——该 run 的 `hard_pass=false` 因此记录，非本审计的效用判据。seed2 两臂 B1 均 PASS。
- **披露**：全部 8 个 run（4 结果 run + 审计复现等）`git.dirty=false`；A 类（A1/A2/A4/A5/A6）在全部 4 个 stage2 run 中 PASS（A3 SKIP，协议口径）。

---

## 4. 模式判定（本次审计的结论）

从复算值机械导出（`pattern` 块，`all_pins_pass=true`）：

| 判定 | 结果 |
|---|---|
| 环境均衡/退化修复在 seed1 复现 | ✅（raw 0.0283% FAIL → norm 47.96% PASS；bal 0.5→0.184 两 recall>0） |
| 环境均衡/退化修复在 seed2 复现 | ✅（raw 0.0766% FAIL → norm 55.31% PASS；bal 0.5→0.647 两 recall>0） |
| 效用 seed1 为正 | ✅（Δtest +0.00824 ∧ Δval +0.01006） |
| 效用 seed2 近零 + val 反向 | ✅（Δtest +0.00046 < +0.001 ∧ Δval −0.00148） |

**与用户陈述的先验模式一致：归一化聚类在两个 seed 上均修复环境均衡与退化（机制），但效用 seed1 为正、seed2 近零且验证方向反向（跨 seed 不稳定）。**

对 seed3（= `1688738016`，canonical 列表位置 3，同为本仓库 `AliCCP_NewTask.py` 默认 seed）的判定问题因此为：**第三 canonical seed 上，效用落在哪一侧、以及模式是否稳定** —— 判定规则、配对定义、门禁与预测已在同批提交的预注册文档 `2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md` 中写死（先于任何新 run）。

---

## 5. 局限（预声明）

1. 两 seed 均为**单 run 配对**（协议 A3 SKIP，无 run-to-run 噪声估计）；Δ 的绝对值不作显著性主张。
2. seed2 臂无 bit 级事件预测（无审计 reproduce 捕获）；机制判定依赖 M 组判据与本次复算，而非 bit 等式。seed1 有认证捕获（seed1 raw `--reproduce` 10/10）。
3. 交叉表与探针为本次复算（只读）；探针只证明退化签名消失/存在，不度量 env 头质量；环境分配的语义优劣不在审计范围。
4. CVR 侧统计功效极低（test 正例 346），CVR 数值仅作记录。
5. 审计覆盖的范围 = 上述四个（seed × 臂）配对；两个分支的其余运行（attenuation、rpg 等系谱产物）不在本审计范围。
