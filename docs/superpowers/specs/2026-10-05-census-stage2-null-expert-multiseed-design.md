# CensusIncome Stage-2 Null Expert：多 seed 配对验证（三 seed 检查点）

- **状态**：三 seed 检查点（§10）与 5-seed 扩展（§12）均**已完成**。判据 §3、§6 在**结果产生之前** commit（`be82ee9`），运行后未修改、未回填；§10 为 append-only 冻结记录（5-seed 扩展未改动 §10 任何一行）；新增诊断仪器（§5）在结果前 commit（`b4933e2`，测试全绿）。
- **日期**：2026-10-05
- **适用分支**：`exp/census-stage2-null-expert-multiseed`（自 `infra/fair-stage2-benchmark` @ `87afe03` 拉出；已 cherry-pick `1eadd05` + `87e2b8d`，当前树与 `exp/stage2-null-expert` tip **逐字节相同**，`git diff exp/stage2-null-expert HEAD` 为空）
- **协议依赖**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（公平评测协议，唯一事实来源；本分支**只引用**，不修改其文件、不改 `census_benchmark/protocol.py` 语义）
- **历史前置**：`docs/superpowers/specs/2026-09-29-stage2-null-expert-design.md`（单 seed 短跑）。该文件第 5 节判据与第 8 节结论（FAILED → stop）**逐字保留、不改写**；本文件只**新增**二级分类字段，不修改历史判定。
- **本轮唯一事实来源**：本文件（多 seed 验证部分）。

---

## 0. 背景、授权与边界

| 项 | 内容 |
|---|---|
| 历史 seed-1 结果 | `Δtest = +0.001296296526501206`（0.8513648272440768 − 0.8500685307175756），机制激活（`null_top1_rate = 0.37256670876686515`），未过旧阈值 0.8521 → 历史判定 **FAILED（utility under bar; mechanism active）→ stop**（逐字保留） |
| 姊妹线背景（仅参考） | AliCCP null expert：utility 过 PI 阈值、机制门失败（top1 = 0）→ 后验归因于近似常量衰减 → 常量控制复现 91.8% → 长预算 NOT_PERSIST。该线结论**不迁移**到本线 |
| PI 指示 | 对 Census null expert 做 **三 seed 配对验证**；新增二级分类（本文件 §6.1）；最小 3 个 canonical seed；三 seed 检查点记录后才允许考虑扩展 5 seed |
| 明确非目标 | ①不改写历史 verdict；②不新增/修改机制（不做温度、初始化、损失等任何补救性调参）；③不写论文、不做 novelty 声明、不做图表；④不重开 CGR / affinity gate / gradient surgery / attenuation；⑤本文件不预设任何性能结论 |
| 与筛查合成的边界 | PI 指示优先于 `docs/mptrec-screening-synthesis` 的「不再做自动化筛查」总建议；本分支为 PI 指定的验证性实验（非新方法筛查），结果只按本文件 §6 分类记录 |

---

## 1. Canonical seed 列表（来自已提交证据，不发明）

**五 seed 全表**（项目多处一致，逐字）：

`[1685480945, 1685463909, 1685477428, 1685459668, 1685496394]`

已提交出处（HEAD 树，行号为当前分支）：`CensusIncome_MPTRec.py:87`（注释）、`baseline/csrec/CensusIncome_train_single.py:180`、`baseline/csrec/CensusIncome_train_mtl.py:80`、`baseline/sparsesharing/CensusIncome_train_mtl.py:69`、`baseline/ple/CensusIncome_NewTask.py:80`。

**三 seed 前缀**（多数 baseline 脚本使用）：

`[1685480945, 1685463909, 1685477428]`

出处：`baseline/sharedbottom/CensusIncome_SharedBottom.py:79`、`baseline/mmoe/CensusIncome_MMOE.py:80`、`baseline/ple/CensusIncome_PLE.py:82`、`baseline/stem/CensusIncome_STEM.py:82`、`baseline/singletask/CensusIncome_SingleTask.py:73`、`CensusIncome_NewTask.py:152`（注释）。

**本轮检查点使用**：三 seed 前缀（最小的 3 个 canonical seed），seed 1 = `1685480945`（已有历史证据，见 §3）。
**扩展（仅在检查点记录之后）**：第 4、5 位 `1685459668`、`1685496394` —— 本文件不启动，需要时按同判据补充。

---

## 2. 协议冻结（全部沿用公平评测协议，结果后不得再改）

| 项 | 取值（逐字沿用 `census_benchmark/protocol.py`） |
|---|---|
| 数据 | `dataset/Census-income/train.gz` 全量训练；`test.gz` 全量按 split seed 50/50 切 val/test |
| split seed | `20260929`（唯一决定划分；跨 model seed 恒定；A2 指纹必须与既有基准一致） |
| env seed | `20260929`（只决定初始 env_ids） |
| model seed | 每个 run 取 §1 列表中的值（**唯一变量**） |
| 阶段 1 | `epochs=2`、`patience=2`、`batch_size=256`、`lr=1e-3`、`uni_coe=0.9`、`env_coe=0.1`、`reg_embedding=0.006`、`reg_dnn=3e-5` |
| 阶段 2 | `epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`、`tag=short` |
| 架构 | `input_size=123`、`embedding_size=4`、`expert_hidden=(256,128)`、`tower_hidden=(64,32)`、`num_tasks=2`、`num_envs=2`、`temperature=150` |
| 评测逻辑 | backbone 真冻结三件套；val 仅用于 early stop 选点；**test 只在训练结束后评一次**；AUC 取 `fused_preds`；A/B 门禁逐项落盘（A1/A2/A4/A5/B1/B2/B3/B4） |

**硬约束**：除 model seed 外，任何超参、结构、优化器、epoch 预算、评测逻辑**不得**因观察结果而改变；已完成 run 不得为取更好数值重跑（§4.3）。

---

## 3. 已有证据审计与复用裁定（在写死判据之前完成）

### 3.1 产物完整性（本 worktree 实测）

| 项 | 值 |
|---|---|
| stage1_id | `s1-096f8f16-m1685480945-e2-cb2094b3`（与预测式重算一致） |
| backbone.pt sha256 | `f41aa2c06deecc0a3fe29c94c91c1726a3414d7a6d57bd8e87d7a2c4fef9add2` |
| env_ids.pt sha256 | `3977f76d0c8a541e40257ab5538b9e0fc2ba12fd53bc50e20aa176b09b724154` |
| meta.json sha256 | `c67f1a2aa46b2213fea33f3c5e70508966a766f56aef49b1ac4610bcd6ed6ebc` |
| backbone_sha256（按参数） | `a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85` |
| env_ids_sha256 | `9605cef86202e4f6fec6bd27d34f57245123d8a7e143aaeac6d7359a24bf8395`（重算一致） |
| split 指纹 | `096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c`（由 `split_indices.npz` 重算 == 落盘基准 == stage1 meta） |
| 训练/验证/测试行数 | 199523 / 49881 / 49881；disjoint=true；union_complete=true |

### 3.2 历史 run（seed 1，已提交证据）

| 项 | 基线臂 | 处理臂 |
|---|---|---|
| run_id | `20260929-1735-s20260929-m1685480945-short-904f8d0` | `20260929-1802-s20260929-m1685480945-short-1eadd05-nullx` |
| commit | `904f8d0` | `1eadd05` |
| AUC-Test-Education | 0.8500685307175756 | 0.8513648272440768 |
| best val AUC（best epoch） | 0.8527881905614896（ep5） | 0.853020431042571（ep5） |
| Δtest / Δval | — | **+0.001296296526501206** / +0.00023224048108139161 |
| null_mean / null_top1_rate | — | 0.34177784155227137 / 0.37256670876686515 |
| 臂级旧判定（逐字） | — | `null_arm.pass = false`，`checks = {auc: false, top1: true}`，`auc_min = 0.8521` |
| 预算状态 | ep5 = 最后 epoch（**右删失**，无早停） | 同左（**右删失**） |

### 3.3 机制审计（不盲信结论，重查实现）

1. 机制 = 在 K 个源任务候选**末尾追加一个值恒为零的候选** + 一个 `nn.Parameter` `null_key`（形状 `(rep_dim,)`，**零初始化**，不消耗 RNG，不入 `get_l2_reg()`）；router 权重变为 `softmax(H_out @ [env_emb_0..K-1, null_key] / T)`，`(B, K+1)`。
2. 默认臂（`use_null_expert=False`）与 `master` **逐位一致**：由 `test_null_expert.py::TestDefaultArmBitIdenticalToMaster` 从 git `master:multitaskrec/model.py` 动态加载参考实现，逐位比对前向输出与**每个参数的梯度**（I1），且参数集合/state_dict 键集相同（I2）。
3. 模型侧唯一改动 = `multitaskrec/model.py` 的 NewTask 分支（静态守卫 I10 强制），`protocol.py` / `census_benchmark/metrics.py` 的既有门禁语义未动。

### 3.4 复用裁定

- 历史 seed-1 证据**协议兼容**（同 split/env/model seed、同数据、同架构、同评测；artifact 哈希如上）→ 语义上可作为 seed 1 的证据。
- **但本轮仍对全部 3 个 seed 的两臂重跑**，理由：(a) 本轮新增 C 类诊断仪器（§5），重跑使三个 seed 的记录口径完全一致；(b) 顺带获得 Census 的 run-to-run 复现数据（合成文档标注该维度未测）；(c) 历史对子降级为**预注册复现锚**（§6.5）。
- **复现锚规则（预注册）**：seed 1 新跑与历史值的 `AUC-Test-Education` 差 ≤ `1e-9` 视为复现成功（A3 口径）；若超出，**两者全部记录**并在检查点显著标注为噪声发现的证据，检查点的主对子仍取**新跑**（同一冻结代码、同一批次）。

---

## 4. 本轮运行矩阵

### 4.1 运行清单（每 seed 三件套）

| # | seed | 阶段 1 | 基线臂（阶段 2） | 处理臂（阶段 2 + `--null-expert`） |
|---|---|---|---|---|
| 1 | 1685480945 | **选取**既有产物 `s1-096f8f16-m1685480945-e2-cb2094b3`（只读、哈希已验） | 新跑 | 新跑 |
| 2 | 1685463909 | 新训（预测 id 见 4.2） | 新跑 | 新跑 |
| 3 | 1685477428 | 新训（预测 id 见 4.2） | 新跑 | 新跑 |

共 8 个新 run（2×stage1 + 6×stage2）。两臂必须引用**同一** `--stage1-dir`；`run_stage2` 从 artifact meta 读取 model seed，结构上排除"跨 seed 对比"。

### 4.2 预注册预测（运行后核对）

| seed | 预测 stage1_id |
|---|---|
| 1685480945 | `s1-096f8f16-m1685480945-e2-cb2094b3`（已存在，重算命中） |
| 1685463909 | `s1-096f8f16-m1685463909-e2-8b53a3bf` |
| 1685477428 | `s1-096f8f16-m1685477428-e2-15eabcb4` |

（由 split 指纹 + 各 seed 的 config 哈希确定性推得；若实际产物 id 不符 → 视为实现缺陷，按 4.3 处理。）

### 4.3 运行纪律

1. 逐条前台执行（单卡 RTX 3060 6GB，串行）；每个阶段 2 run 完成即把 SUMMARY 行提交（append-only），保持下一 run 从干净树启动。
2. **已完成 run 不得重跑**；崩溃/中断的 run 保留产物并如实记录后允许重执行（失败证据不删）。
3. 运行期间不搜参、不改超参、不改结构、不改评测；`commit` 字段必须包含本轮代码。
4. 全部命令记录于 §8，运行环境 = 本 worktree + `D:\MPT-Rec-three_task\MPT-Rec\.venv`。

---

## 5. 仪器化新增（additive；不改训练/评测语义）

从姊妹线 `exp/aliccp-stage2-null-expert`（`6224c0f`，`aliccp_benchmark/metrics.py`）**移植** C 类诊断：

| 新增 | 内容 |
|---|---|
| `NullRouteDiagnostics` | null 候选质量分布（`null_std`、`null_var`、`null_q10/q25/q50/q75/q90`）、router 熵（`route_entropy_mean/std`）、逐样本路由方差（`route_var_mean`） |
| `null_supervision_stats` | 标签分层 null 均值（`null_mean_edu_pos/neg`、`null_label_gap`）、`corr_null_pred`、`corr_null_abs_err`、`pred_std`（零方差 → 相关返回 None） |
| runner 探针 | `newtask_null_probe(...)`：**仅处理臂**在训练结束后对 val 做一次额外 pass，`mechanism` 追加上述键（含 M7 的 `null_mean/null_top1_rate`，取自同一 pass；以测试断言与 `evaluate_newtask(mechanism=True)` 的值**逐位一致**） |

不变式：①基线臂 `mechanism` 键集逐字不变 `{gate_mean, cos_gen_spec, gen_std, env_acc_stage1}`（测试强制）；②`metrics.null_arm_verdict` 与常量 `NULL_ARM_AUC_MIN = 0.8521` 等**逐字保留**（历史阈值字段，不删不改）；③诊断只在 **val** 上算，test 不参与任何机制量；④新增代码不进入基线路径。

**注**：`null_arm` 的绝对阈值 `0.8521` 是 **seed-1 专属**历史判据；对 seed 2/3 的 run，该字段按原样落盘（保持字段逐字），但它**不是**本轮的 per-seed 判定量 —— per-seed 判定量是 §6.1 的 Δtest 二级分类（配对、同 seed、同 artifact）。

---

## 6. 预注册判据（写死；看到结果后不得修改）

### 6.1 二级分类（新增；不改历史 verdict）

对每个 seed：`Δtest = AUC-Test-Education(处理臂) − AUC-Test-Education(同 seed 基线臂)`（配对、同 artifact）：

| 分类 | 条件（边界含等号，逐字） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest >= +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `-0.02 < Δtest < +0.001` |
| `CLEAR_DEGRADATION` | `Δtest <= -0.02` |

历史阈值/verdict 字段（`null_arm`、`auc_min`、`checks`、`pass`，以及旧文档 §5/§8 的全部数值与结论）**逐字保留**；新分类作为**独立字段**追加，不改写任何历史字段。

### 6.2 `NO_CLEAR_IMPROVEMENT` 的 headroom 记录（预注册机械规则）

逐 seed 记录四个维度（原始值 + 布尔），并给出：

```
headroom_remains = mechanism_active AND val_direction_positive
                   AND (budget_right_censored AND budget_last_delta_positive)
                   AND cross_seed_consistent
```

| 维度 | 机械定义 |
|---|---|
| `mechanism_active` | `null_top1_rate ∈ [0.05, 0.95]`（历史机制判据逐字沿用） |
| `val_direction_positive` | `Δval > 0`（处理臂 best val − 基线臂 best val） |
| `budget_right_censored` | 处理臂 best epoch == 该臂最后一个训练 epoch（预算终点仍在改善、无早停） |
| `budget_last_delta_positive` | 最后一个共同 epoch 的 `Δval(epoch) > 0` |
| `cross_seed_consistent` | 三个 seed 中 `Δtest > 0` 占多数 **且** 无 seed `Δtest <= -0.02` |

该记录仅适用于 `NO_CLEAR_IMPROVEMENT`（其它分类下字段照记、不作解释），且**只是 headroom 描述，不构成有效性主张**；分类结论与 headroom 记录相互独立。

### 6.3 三 seed 检查点统计（方法写死）

- 逐 seed `Δtest`；`mean`、`std`（**样本标准差，ddof=1**）；`positive_count = #{Δtest >= +0.001}`；`worst_seed = argmin(Δtest)`（并列取 `CHECKPOINT_SEEDS` 列表序靠前者；本轮列表序 = `[1685480945, 1685463909, 1685477428]`）。
- **95% 置信区间**：对 3 个配对 Δtest 的 **Student t 区间**：`mean ± t(0.975, df=2) · s/√3`，`t = 4.303`。方法原文写入记录；并注明 n=3 下该区间为**描述性**（不构成确证性推断）。

### 6.4 门禁与塌缩检查（逐 seed 记录，不新增阈值）

- A/B 门禁逐项（A1/A2/A4/A5/B1/B2/B3/B4）按协议判定，结果原样记录（B3 历史两臂均 FAIL，属继承，不因本轮改变）。
- 塌缩检查（记录值）：gate 权重（B3 量）、`gen_std`（常量表征）、`null_top1_rate` 端点、`null_std`、`route_entropy_mean`。
- 机制判据（记录值，不并入 overall_pass）：`null_top1_rate ∈ [0.05, 0.95]`。

### 6.5 复现锚

seed 1 新旧值比对（§3.4 规则）。记录：新旧 run_id、两值、差；≤ `1e-9` → `REPRODUCED`；否则 `DIVERGED`（显著标注）。

### 6.6 阈值纪律

本节全部数值（+0.001 / −0.02 / 0.05 / 0.95 / 1e-9 / t=4.303）只允许在**看到任何新 run 结果之前**修改；每次修改单独 commit 并记录理由。已判定的 run 不得用新阈值重判。

---

## 7. 每 seed 必录字段（检查点产物）

1. run id、commit、（若有）树状态；stage1_id + `backbone.pt` / `env_ids.pt` / `meta.json` 文件 sha256。
2. config（落盘 `config.json`）+ 数据指纹（split fingerprint / env_ids sha256）。
3. 基线臂与处理臂的 `AUC-Test-Education`、best val AUC、best epoch、逐 epoch val 轨迹。
4. 配对 Δtest / Δval；§6.1 分类；§6.2 headroom 记录（四维度原始值 + 布尔）。
5. 处理臂机制量：`null_mean`、`null_top1_rate`、`null_std`、分位数、`route_entropy_mean/std`、`route_var_mean`、监督相关（§5）；基线臂 M3/M4。
6. 门禁逐项 + 塌缩检查（§6.4）；stage-1 epoch 记录（`env_acc`、cluster 记录）。
7. 右删失/早停状态；与历史锚的比对（§6.5）。

检查点汇总另录：mean/std、positive_count、worst_seed、95% CI（含方法原句）、以及各 seed 分类表。

---

## 8. 运行命令（预注册；执行时逐字使用）

```powershell
# 0) 测试（先红后绿；不变量见 §5）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests

# 1) seed 1685463909：阶段 1（seed 1 选取既有产物，不重训）
... python.exe run_census_benchmark.py stage1 --model-seed 1685463909 --tag short --gpu 0
... python.exe run_census_benchmark.py stage1 --model-seed 1685477428 --tag short --gpu 0

# 2) 每 seed：基线臂 → （提交 SUMMARY 行）→ 处理臂 → （提交 SUMMARY 行）
... python.exe run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/<sid> --tag short --epochs 5 --gpu 0
... python.exe run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/<sid> --tag short --epochs 5 --null-expert --gpu 0

# 3) 检查点
... python.exe -m census_benchmark.multiseed --root artifacts/census_stage2
```

---

## 9. 报告纪律

- `SUMMARY.md` 只追加、不重写（协议 7.3）；处理臂即使失败也必须有行。
- 结果追加到本文件 §10（运行后），**不回填预期值**、不删除失败 run、不挑 seed。
- 不推送远端（由 PI 指示本轮结束统一 push）；**永不 merge master**。
- 结果后不得再改 §6 判据；若发现实现缺陷，保留失败产物 + 说明 + 修复后重跑（§4.3.2）。

---

## 10. 结果（2026-10-05 实测；本节为运行后追加，判据未回填、未修改）

**状态**：三 seed 检查点完成。判据 §6 在结果产生前已 commit（`be82ee9`）；本轮共 8 个新 run（2×stage1 + 6×stage2），全部前台单跑、无重跑、无调参。

### 10.1 运行清单与产物哈希

| seed | stage1_id | 基线臂 run_id（commit） | 处理臂 run_id（commit） |
|---|---|---|---|
| 1685480945 | `s1-096f8f16-m1685480945-e2-cb2094b3`（**选取**既有产物） | `20261005-0225-s20260929-m1685480945-short-b4933e2`（b4933e2） | `20261005-0227-s20260929-m1685480945-short-9898f9e-nullx`（9898f9e） |
| 1685463909 | `s1-096f8f16-m1685463909-e2-8b53a3bf`（新训；id **预测命中**） | `20261005-0231-s20260929-m1685463909-short-9c6cc38`（9c6cc38） | `20261005-0233-s20260929-m1685463909-short-5b2ca79-nullx`（5b2ca79） |
| 1685477428 | `s1-096f8f16-m1685477428-e2-15eabcb4`（新训；id **预测命中**） | `20261005-0236-s20260929-m1685477428-short-1928ee9`（1928ee9） | `20261005-0238-s20260929-m1685477428-short-ecc2148-nullx`（ecc2148） |

Stage-1 产物文件哈希（sha256，逐字节）：

| seed | backbone.pt | env_ids.pt |
|---|---|---|
| 1685480945 | `f41aa2c06deecc0a3fe29c94c91c1726a3414d7a6d57bd8e87d7a2c4fef9add2` | `3977f76d0c8a541e40257ab5538b9e0fc2ba12fd53bc50e20aa176b09b724154` |
| 1685463909 | `41d2827a8bed9a119d8086b0db4bc5b9a1ed239f34f79703f0a87b5ccc5119fc` | `9394ef91552714ef43fb8efc11c099d87c15bbc2d69967b60699de22fd29917f` |
| 1685477428 | `d06f18c668e8ba2787cb7d98fbf635d81084e83b015499c522dc92516a5ad802` | `706ae0a93eebd5d99ffd5feddbc2d843297dacfa59bd5d39657a8ddedcd0617d` |

meta.json 及每个 run 目录 7 个文件的完整 sha256 见检查点 JSON（§10.9）。每 seed 两臂 `split_sha256.fingerprint = 096f8f16…`、`env_ids_sha256` 与对应 stage1 meta 一致（`find_runs` 硬校验，不一致即拒绝配对）。

**复现锚（§6.5）**：`REPRODUCED`。seed 1 新跑与历史对子 `AUC-Test-Education` **逐位相等**（|Δ| = 0.0 ≤ 1e-9；基线 0.8500685307175756、处理 0.8513648272440768；逐 epoch 记录与 M7 值均逐位一致）；历史目录值与常量一致（`historical_dirs_match_constants = true`）。→ **Census 管线在同机、同配置、同 artifact 下 run-to-run 逐位可复现**（该维度此前在合成文档中被标注"未测"，本轮在本配置上实测为 0 噪声；不外推到其它配置/机器）。

### 10.2 配对结果（主表）

| seed | AUC-test 基线 | AUC-test 处理 | **Δtest** | best-val 基线 / 处理 | **Δval** | 分类（§6.1） |
|---|---|---|---|---|---|---|
| 1685480945 | 0.8500685307175756 | 0.8513648272440768 | **+0.001296296526501206** | 0.8527881905614896 / 0.853020431042571 | +0.00023224048108139161 | **POSITIVE_IMPROVEMENT** |
| 1685463909 | 0.845256873871603 | 0.8447283846542692 | **−0.0005284892173337274** | 0.8476002975447581 / 0.8473110085312799 | −0.0002892890134781334 | **NO_CLEAR_IMPROVEMENT** |
| 1685477428 | 0.8427796966516823 | 0.8456572919817906 | **+0.00287759533010834** | 0.8436747451693688 / 0.8476304556509646 | +0.00395571048159582 | **POSITIVE_IMPROVEMENT** |

### 10.3 检查点统计（§6.3）

| 统计量 | 值 |
|---|---|
| n | 3 |
| mean(Δtest) | +0.0012151342130919396 |
| std(Δtest)（样本标准差，ddof=1） | 0.0017044921463354903 |
| positive_count（Δtest ≥ +0.001） | 2 |
| worst_seed | 1685463909（Δ = −0.0005284892173337274） |
| 95% CI | [−0.0030194007518357284, +0.005449669178019608] |
| CI 方法（原句） | `Student t 95% CI on paired deltas, df=n-1, t=4.303 (n=3)` |

n=3 下该区间为**描述性**且**含 0**——不构成"正效应"的确证性结论；本节不含任何超出 §6.1 分类的主张。

### 10.4 headroom 记录（§6.2）

| seed | 分类 | mechanism_active | val_direction_positive | budget_right_censored | budget_last_delta_positive | cross_seed_consistent | headroom_remains |
|---|---|---|---|---|---|---|---|
| 1685480945 | POSITIVE | true | true | true | true | true | true（context；分类已为 POSITIVE） |
| 1685463909 | NO_CLEAR | true | **false** | true | **false** | true | **false** |
| 1685477428 | POSITIVE | true | true | true | true | true | true（context） |

seed 1685463909（唯一 NO_CLEAR_IMPROVEMENT）：机制激活、跨 seed 一致性成立、预算右删失，但 **val 方向为负、末 epoch Δval 为负** → 按预注册规则 `headroom_remains = false`（该 seed 上"仍有改进空间"的证据不成立）。headroom 记录仅为描述，不改变分类。

### 10.5 机制量与塌缩检查（处理臂，val；§5 新仪器）

| seed | null_top1_rate | null_mean | null_std | q10 / q50 / q90 | route_entropy_mean (max ln3=1.0986) | route_var_mean | 分层差(edu 正−负) | corr(null,pred) | corr(null,\|err\|) | pred_std |
|---|---|---|---|---|---|---|---|---|---|---|
| 1685480945 | 0.37256670876686515 | 0.34177784155227137 | 0.17065749985572967 | 0.1245 / 0.3394 / 0.5791 | 0.9130 | 0.04254 | +0.1014 | +0.4201 | +0.3790 | 0.13307 |
| 1685463909 | 0.40646739239389745 | 0.35102747942742163 | 0.16234055106992418 | 0.1431 / 0.3597 / 0.5673 | 0.9233 | 0.04005 | +0.0631 | +0.2584 | +0.2717 | 0.13250 |
| 1685477428 | 0.4249112888675047 | 0.4071581977192909 | 0.251825084404169 | 0.1289 / 0.3312 / 0.7478 | 0.8889 | 0.04761 | **−0.2160** | **−0.5706** | −0.4643 | 0.14055 |

- 机制判据（`null_top1_rate ∈ [0.05, 0.95]`，历史判据逐字沿用）：**3/3 成立** → `mechanism_active = true`。
- 塌缩检查：null 权重分布宽（q10–q90 跨度 0.44–0.62），`null_std` 0.16–0.25 —— 逐样本变化，**不是** AliCCP 线的近似常量衰减（该线 std 0.0085）；两线机制形态不同，如实并列记录、不解释。`gen_std` 3.477 / 2.278 / 1.110（无表征常量塌缩）。
- 事实性记录（不解读）：seed 3 的 `null_label_gap`（−0.2160）与 `corr_null_pred`（−0.5706）符号与 seed 1/2 相反——跨 seed 的监督相关性符号不一致。

### 10.6 门禁（逐 seed 逐臂，原样记录）

| seed | 基线臂 | 处理臂 |
|---|---|---|
| 1685480945 | A1✓ A2✓ A4✓ A5✓ B1✓ B2✓ **B3✗** B4✓ | A1✓ A2✓ A4✓ A5✓ B1✓ B2✓ **B3✗** B4✓ |
| 1685463909 | 全部 PASS | 全部 PASS |
| 1685477428 | 全部 PASS | 全部 PASS |

B3 FAIL 仅出现在 seed 1 两臂（`gate_mean[0] = 0.9524 > 0.95` 上界；seed 2/3 分别为 0.9291/0.9301），与历史记录一致（同 run 逐位复现）；属**继承**的协议门禁行为，本轮不重判、不调阈值。

### 10.7 预算状态（右删失与逐 epoch Δval）

6 个 run（3 seed × 2 臂）**全部右删失**：best epoch = 5 = 最后训练 epoch（patience=2 未触发早停）。逐 epoch Δval（处理 − 基线）：

| seed | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| 1685480945 | +0.001784488962795261 | +0.0011923104451476707 | +0.0015461903719412584 | +0.000865779926852217 | +0.00023224048108139161 |
| 1685463909 | +0.00045756882560732404 | −0.00017982287466244973 | +0.00064170978755973 | +0.0007852707763615907 | −0.0002892890134781334 |
| 1685477428 | +0.00033184192550639313 | +0.0016075501411455528 | +0.003375947321216599 | +0.004876488363168119 | +0.00395571048159582 |

### 10.8 历史阈值 / verdict 字段（逐字保留）

三个处理臂 run 的 `metrics.json["null_arm"]` 均为历史逐字判据（seed-1 专属绝对阈值）：`auc_min = 0.8521`、`baseline_test_auc = 0.8500685307`、`top1_range = [0.05, 0.95]`、`checks = {auc: false, top1: true}`、`pass = false`。历史文档 `2026-09-29-stage2-null-expert-design.md` §5/§8 **未做任何修改**；二级分类（§6.1）是**独立新增字段**，不改写历史结论。**注意**：对 seed 2/3，`null_arm` 的绝对阈值不是本轮判定量（其对照常量是 seed-1 基线）；per-seed 判定量一律为 §10.2 的配对 Δtest。

### 10.9 记录与可审计性

- `SUMMARY.md`（本分支）：8 行（历史 2 行 + 本轮 6 行），append-only，未重写。
- 检查点 JSON（机器可读全量记录，含逐文件 sha256 / 逐 epoch 记录 / 门禁 / 机制量表 / 复现锚）：
  `docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint.json`。
- run 目录（本地保留，gitignore）：`artifacts/census_stage2/runs/` 6 个新 run + 2 个历史锚 run；`stage1/` 3 个产物。
- 代码：`b4933e2`（仪器 + 检查点模块 + 85/85 测试）；各 run 的 `commit` 见 §10.1。

### 10.10 结论（按预注册判据逐字陈述，不越界）

1. **分类结果**：3 个 canonical seed 中 **2 个 POSITIVE_IMPROVEMENT**（1685480945：+0.001296296526501206；1685477428：+0.00287759533010834）、**1 个 NO_CLEAR_IMPROVEMENT**（1685463909：−0.0005284892173337274，headroom_remains=false）、**0 个 CLEAR_DEGRADATION**。
2. **统计**：mean Δtest = +0.0012151342130919396；std(ddof=1) = 0.0017044921463354903；95% CI（Student t，t=4.303，n=3，描述性）= [−0.0030194007518357284, +0.005449669178019608]（**含 0**）。
3. **复现锚**：REPRODUCED（逐位相等）；Census 同配置 run-to-run 噪声在本机本配置下于 1e-9 精度内为 0。
4. **明确不主张**：本文不构成有效性/通用性结论，不改写历史 verdict，无任何检验后调参；与 AliCCP 线的机制形态差异仅为事实记录。
5. **扩展条件**：三 seed 检查点已记录 → 按 PI 指示，允许（但不自动启动）扩展至 5 seed（§11.1）。

---

## 11. 三 seed 检查点之后的候选（本轮不启动）

1. 扩展至 5 seed：追加 `1685459668`、`1685496394`（同判据、同协议；仅在检查点记录后）。
2. 预算延长（如 10 epoch / patience 3）—— 属独立实验，需另行预注册；本轮不做。

> 注：§11.1 已在三 seed 检查点记录后由 PI 指示启动，结果见 §12（本节文字保持预注册时原样）。

---

## 12. 5-seed 扩展结果（2026-10-05 实测；执行 §11.1；§10 未改动，本节 append-only）

**授权**：PI 在三 seed 检查点记录后指示启动 §11.1（追加 canonical seed 4 = `1685459668`、seed 5 = `1685496394`）。判据、协议、仪器、运行纪律与 §2/§4.3/§6 完全一致：**无任何调参、无新机制、seed 是唯一变量**。本轮新增 6 个 run（2×stage1 + 4×stage2），全部前台单跑、无重跑。

### 12.1 运行清单与产物哈希（新增部分）

| seed | stage1_id | 基线臂 run_id（commit） | 处理臂 run_id（commit） |
|---|---|---|---|
| 1685459668 | `s1-096f8f16-m1685459668-e2-8611b794`（新训；id **预测命中**） | `20261005-0409-s20260929-m1685459668-short-caad751`（caad751） | `20261005-0411-s20260929-m1685459668-short-0e1c7d6-nullx`（0e1c7d6） |
| 1685496394 | `s1-096f8f16-m1685496394-e2-2231eae1`（新训；id **预测命中**） | `20261005-0415-s20260929-m1685496394-short-d86a4bc`（d86a4bc） | `20261005-0416-s20260929-m1685496394-short-3396e2e-nullx`（3396e2e） |

（stage1_id 预测在运行前按 §4.2 同一确定性公式计算，两个全部命中。）

Stage-1 产物文件哈希（sha256）：seed 4 `backbone.pt = 746681c0111413a048b3b8ca6de260273f115f451000388cdf8f000cd25f6a57`；seed 5 `backbone.pt = 5b23036ba6659d2f944676d69af3d5f7a1f07deda9245842b0a34133d20b4566`（env_ids.pt / meta.json 及逐 run 文件哈希见 §12.7 的 5-seed 检查点 JSON）。

### 12.2 配对结果（全 5 seed 主表）

| seed | AUC-test 基线 | AUC-test 处理 | **Δtest** | best-val 基线 / 处理 | **Δval** | 分类（§6.1） |
|---|---|---|---|---|---|---|
| 1685480945 | 0.8500685307175756 | 0.8513648272440768 | **+0.001296296526501206** | 0.8527881905614896 / 0.853020431042571 | +0.00023224048108139161 | POSITIVE_IMPROVEMENT |
| 1685463909 | 0.845256873871603 | 0.8447283846542692 | **−0.0005284892173337274** | 0.8476002975447581 / 0.8473110085312799 | −0.0002892890134781334 | NO_CLEAR_IMPROVEMENT |
| 1685477428 | 0.8427796966516823 | 0.8456572919817906 | **+0.00287759533010834** | 0.8436747451693688 / 0.8476304556509646 | +0.00395571048159582 | POSITIVE_IMPROVEMENT |
| 1685459668 | 0.8432785266659345 | 0.8440976853967682 | **+0.0008191587308337134** | 0.8460871904349292 / 0.8453586690532408 | −0.0007285213816884406 | NO_CLEAR_IMPROVEMENT |
| 1685496394 | 0.8474462695821553 | 0.8476911350393439 | **+0.0002448654571886033** | 0.84798146522571 / 0.8473791525613715 | −0.0006023126643385224 | NO_CLEAR_IMPROVEMENT |

### 12.3 5-seed 检查点统计（§6.3；t=2.776 由 df=4 查表）

| 统计量 | 值 |
|---|---|
| n | 5 |
| mean(Δtest) | +0.0009418853654596271 |
| std(Δtest)（样本标准差，ddof=1） | 0.00127822927632415 |
| positive_count（Δtest ≥ +0.001） | 2 |
| worst_seed | 1685463909（Δ = −0.0005284892173337274） |
| 95% CI | [−0.0006449914677945057, +0.0025287621987137602] |
| CI 方法（原句） | `Student t 95% CI on paired deltas, df=n-1, t=2.776 (n=5)` |

5-seed CI 较 3-seed **收窄**（[−0.000645, +0.002529] vs [−0.003019, +0.005450]）且**仍含 0**；mean 由 +0.001215 降至 +0.000942（新增两个 seed 均为 NO_CLEAR 且 Δval 为负，拉低均值）。**分类计数**：POSITIVE 2 / NO_CLEAR 3 / **CLEAR_DEGRADATION 0**。仍为描述性统计（n=5），不构成确证性结论。

### 12.4 验证方向、headroom 与跨 seed 一致性（§6.2）

**跨 seed 一致性**：`cross_seed_consistent = true`（5 个 Δtest 中 4 个 > 0，严格过半；且无任一 Δtest ≤ −0.02）。

| seed | 分类 | mechanism_active | val_direction_positive | budget_right_censored | budget_last_delta_positive | headroom_remains |
|---|---|---|---|---|---|---|
| 1685480945 | POSITIVE | true | true | true | true | true（context） |
| 1685463909 | NO_CLEAR | true | **false** | true | **false** | **false** |
| 1685477428 | POSITIVE | true | true | true | true | true（context） |
| 1685459668 | NO_CLEAR | true | **false** | true | **false** | **false** |
| 1685496394 | NO_CLEAR | true | **false** | true | **false** | **false** |

**验证方向**：5 个 seed 中 **2 正 3 负**（Δval：+0.000232 / −0.000289 / +0.003956 / −0.000729 / −0.000602）。3 个 NO_CLEAR seed 的 headroom 全部为 **false**（均为 val 方向负 + 末 epoch Δval 负）。

### 12.5 机制量与塌缩检查（新增两 seed；全 5 seed 汇总）

| seed | null_top1_rate | null_mean | null_std | q10 / q50 / q90 | entropy (max 1.0986) | 分层差(edu) | corr(null,pred) | corr(null,\|err\|) | gen_std | gate_mean[0] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1685480945 | 0.3726 | 0.3418 | 0.1707 | 0.1245 / 0.3394 / 0.5791 | 0.9130 | +0.1014 | +0.4201 | +0.3790 | 3.477 | 0.9524 |
| 1685463909 | 0.4065 | 0.3510 | 0.1623 | 0.1431 / 0.3597 / 0.5673 | 0.9233 | +0.0631 | +0.2584 | +0.2717 | 2.278 | 0.9291 |
| 1685477428 | 0.4249 | 0.4072 | 0.2518 | 0.1289 / 0.3312 / 0.7478 | 0.8889 | −0.2160 | −0.5706 | −0.4643 | 1.110 | 0.9301 |
| 1685459668 | 0.4524 | 0.3484 | 0.1172 | 0.1939 / 0.3784 / 0.4805 | 0.9738 | +0.0441 | +0.2931 | +0.3400 | 0.851 | 0.8176 |
| 1685496394 | 0.3012 | 0.3316 | 0.1192 | 0.1922 / 0.3125 / 0.5087 | 0.9497 | +0.0531 | +0.3831 | +0.3417 | 2.816 | 0.9898 |

- 机制判据：**5/5 成立**（top1 ∈ [0.31, 0.45] ⊂ [0.05, 0.95]）。
- 塌缩检查：`null_std` 0.117–0.252、分布宽 → 逐样本自适应（非常量衰减）；`gen_std` 0.851–3.477（seed 4 为 5 者最低，但仍非零；如实记录，不解读）。
- 事实性记录（不解读）：`null_label_gap` 1 负 4 正；`corr_null_pred` 1 负 4 正（负者均为 seed 1685477428）；`route_entropy_mean` 0.889–0.974 均低于 max ln3。

### 12.6 门禁与预算（新增部分）

| seed | 基线臂 | 处理臂 |
|---|---|---|
| 1685459668 | 全部 PASS | 全部 PASS |
| 1685496394 | A1✓A2✓A4✓A5✓B1✓B2✓ **B3✗** B4✓ | A1✓A2✓A4✓A5✓B1✓B2✓ **B3✗** B4✓ |

B3 FAIL 现出现于 seed 1 与 seed 5 两臂（`gate_mean[0] = 0.9524 / 0.9898 > 0.95`）；属**继承**的协议门禁行为，原样记录、不重判。

预算：新增 4 个 run 同样**全部右删失**（best epoch = 5）。逐 epoch Δval：

| seed | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| 1685459668 | +0.001541534327529015 | +0.0009166401903142329 | +0.00048077825935344265 | −0.0009891548280591689 | −0.0007285213816884406 |
| 1685496394 | −0.00008202767404841271 | −0.00035797194817044353 | −0.00028454162930269256 | −0.00020289986575250385 | −0.0006023126643385224 |

### 12.7 记录（§10 保留 + 新增）

- `SUMMARY.md`：12 行（历史 2 + 三 seed 6 + 扩展 4），append-only。
- 5-seed 检查点 JSON（机器可读全量记录）：`docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json`；三 seed 版 `…-checkpoint.json` **原样保留**。
- §10（三 seed 记录）**未改动一行**（append-only 可经 git diff 验证：本节只新增 §12 与状态行）。

### 12.8 结论（按预注册判据逐字陈述，不越界）

1. **分类**：5 个 canonical seed 中 **2 POSITIVE_IMPROVEMENT**（1685480945、1685477428）、**3 NO_CLEAR_IMPROVEMENT**（1685463909、1685459668、1685496394，headroom 全为 false）、**0 CLEAR_DEGRADATION**。
2. **统计**：mean Δtest = +0.0009418853654596271；std(ddof=1) = 0.00127822927632415；95% CI（Student t，t=2.776，df=4，描述性）= [−0.0006449914677945057, +0.0025287621987137602]（**含 0**）；cross_seed_consistent = true。
3. **复现锚**：仍为 REPRODUCED（同 §10.1；本轮未重跑锚）。
4. **明确不主张**：扩展**不改变**任何既有结论的性质——短预算下本方法的 Δtest 在 5 seed 上为正均值、但 CI 含 0、按预注册仅 2/5 达 POSITIVE；无有效性/通用性主张；无检验后调参。
5. **候选下一步（§11.2，需另行预注册的独立分支）**：见 §13。

---

## 13. 5-seed 检查点之后的建议（本轮不启动）

**建议的下一独立分支（推荐）**：`exp/census-stage2-null-expert-longer-budget` —— 预注册的**长预算持续性检验**（stage2 `epochs 5→10`、`patience 2→3`、`tag long`，其余全部冻结；同 5 canonical seed × 双方案配，共 10 run）。理由：①5-seed 逐 epoch Δval 轨迹已给出可预注册的方向性预测——seed 1 单调收窄（+0.00178→+0.00023）、seed 3 于 ep4 达峰后回落、seed 4/5 在 ep4–5 转负，全部 10 个 run 右删失；②该检验直接回答"短预算下的小幅正 Δ 是否在更长预算下保持"，与 AliCCP 线的 longer-budget 先例（`47ecb0c`/`bbd8a61`）同构，机器与门禁全部复用；③判定应为 PERSISTS / NOT_PERSIST 的预注册规则（对照现 5-epoch 对子作上下文，仅作动机记录）。

**备选（仅在持续性通过后再启动，PI 已明确本轮不做）**：归因/特异性对照分支（如固定随机 key 对照，检验增益来自 null 候选语义还是额外路由容量）——Census 机制非 AliCCP 式常量衰减（`null_std` 0.12–0.25），故 AliCCP 的常量控制不能直接照搬，需新设计、新预注册。
