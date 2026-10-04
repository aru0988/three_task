# CensusIncome Stage-2 Null Expert 增量在更长优化预算下的持久性检验（robustness / alternative-explanation，非新方法）

- **状态**：预注册已写死（本文件在本分支任何新 run 之前**单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 0–9 节的判据与数字不得在看到结果后改动。
- **日期**：2026-10-05
- **适用分支**：`exp/census-stage2-null-expert-longer-budget`（自 `exp/stage2-null-expert` @ `87e2b8d` 拉出；不从任何其它 worktree 运行、不修改其内容；**永不 merge master**；`census_benchmark/protocol.py` 零修改）。
- **上游协议（只引用、不修改）**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（CensusIncome 公平评测协议，唯一事实来源）+ `census_benchmark/protocol.py`。
- **历史前置（只读引用；逐字保留，不改写）**：
  1. `docs/superpowers/specs/2026-09-29-stage2-null-expert-design.md`（单 seed 短跑；§5 判据与 §8 的 FAILED → stop 逐字保留）；
  2. `exp/census-stage2-null-expert-multiseed` @ `1196acd`（5-seed 检查点；判据 §6 + 结果 §10/§12；本实验的动机与短预算 context 来源）。该分支的设计文档**不复制**到本分支，一律按 `1196acd` 引用。
- **定位声明**：本实验是**稳健性 / 替代解释检验**（短预算下的小幅正 Δ 是持久收益还是仅加速收敛），不是新方法，不主张任何新颖性；不改机制、不调参、不换结构、不改损失、不放宽任何门限。与短预算协议相比**唯一变化 = Stage2 max epochs 5→10 与 patience 2→3（tag `long`）**（§3）。结论口径：5-seed 短预算观察到的（小幅、CI 含 0 的）正 Δ 是否在更长 Stage2 预算下按 §5 预注册判据**仍然成立**（`PERSISTS` / `NOT_PERSIST`）。

**提交时序（预注册纪律）**：本文件 + 预注册前核验脚本（`verify_census_stage1_artifacts.py`）先于实现与运行**单独提交**（C1）→ 实现 + 测试（TDD；10 条 run 的 `commit` 字段即后续提交）→ **恰好一次**/臂/seed 的 10 条 run（前台、串行、run 间提交 SUMMARY 行，§4）→ 分析恰一次 → 结果只在第 10 节与 `SUMMARY.md` 以**新增小节 / 新行**回填。

---

## 0. 背景、授权与判定问题

5-seed 短预算（Stage2 epochs=5 / patience=2）检查点已完成并提交（`exp/census-stage2-null-expert-multiseed` @ `1196acd`；判据在结果前 commit）。五个 canonical seed 的配对 Δtest（处理臂 − 同 seed、同 stage1_id 基线臂）为

`+0.001296296526501206` / `−0.0005284892173337274` / `+0.00287759533010834` / `+0.0008191587308337134` / `+0.0002448654571886033`

（mean `+0.0009418853654596271`；4/5 为正；2/5 ≥ +0.001；Student t 95% CI `[−0.0006449914677945057, +0.0025287621987137602]` **含 0**）。**全部十条短预算 Stage-2 run（5 seed × 2 臂）均为右删失**（best epoch = 5 = 最后训练 epoch；patience=2 从未被行使）——判定点截断了仍在演化的轨迹，无法区分"持久收益"与"仅加速收敛"。该事实与全部数值见 §1（已提交记录逐项复核）。

**判定问题 Q**：在同一五份 Stage-1 产物上（同 split/env seed、同 5 个 model seed、同数据、同架构、同优化器/批量/评测/Null 构造/仪器；**唯一变化 = Stage2 10 epochs / patience 3 / tag `long`**），Null Expert 处理臂相对**本实验新跑的配对 10-epoch 基线臂**的正 Δ 是否按 §5 预注册判据成立？
- 全部条件成立 → **`PERSISTS`**（增量在更长预算下持续；非纯加速解释）
- 任一判定条件不成立 → **`NOT_PERSIST`**（增量被更长预算抹平或反向；与"仅加速收敛"解释一致）
- identity/记录/构造一致性校验不通过 → **`INVALID`**（完备性分支：出现即表示运行偏离预注册，按实记录并停止解读）

**纪律声明（预注册）**：短预算 5-seed 结果（§1）**只作本实验的动机与事后对照（context），不构成本实验判定结果的任何部分**；本实验的判定只用 §4 新跑的 10 条成对 run（§5 判据），不使用任何旧 run 数值入判。seed-1 历史单跑与 AliCCP 姊妹线（含其 longer-budget 先例 `47ecb0c`/`bbd8a61` 与 NOT_PERSIST 结果）同样**只作背景**，其结论**不迁移**到本线。

---

## 1. 审计（先于任何新 run；只读，来自已提交记录与本机磁盘复核）

### 1.1 已提交 5-seed 检查点逐项复核（`1196acd` 树内）

来源：`docs/superpowers/specs/2026-10-05-census-stage2-null-expert-multiseed-checkpoint-5seed.json`（字节级复制入本分支，content sha256 钉死 `605196e56b7e92a78a5953205c78b447e4691216593eb1436b1c8ab26b83e2ab`；git blob `074d893a801e118598c416accfc6e94a05b550eb`）与 `artifacts/census_stage2/SUMMARY.md`（12 行）。逐 seed 配对（两臂同 `stage1_id`、同划指纹、单次 test 评测）：

| seed | stage1_id | 基线 test / val | 处理 test / val | **Δtest** | **Δval** | 分类 | 臂 best/last epoch |
|---|---|---|---|---|---|---|---|
| 1685480945 | `s1-096f8f16-m1685480945-e2-cb2094b3` | 0.8500685307175756 / 0.8527881905614896 | 0.8513648272440768 / 0.853020431042571 | **+0.001296296526501206** | +0.00023224048108139161 | POSITIVE_IMPROVEMENT | 5 / 5（右删失） |
| 1685463909 | `s1-096f8f16-m1685463909-e2-8b53a3bf` | 0.845256873871603 / 0.8476002975447581 | 0.8447283846542692 / 0.8473110085312799 | **−0.0005284892173337274** | −0.0002892890134781334 | NO_CLEAR_IMPROVEMENT | 5 / 5（右删失） |
| 1685477428 | `s1-096f8f16-m1685477428-e2-15eabcb4` | 0.8427796966516823 / 0.8436747451693688 | 0.8456572919817906 / 0.8476304556509646 | **+0.00287759533010834** | +0.00395571048159582 | POSITIVE_IMPROVEMENT | 5 / 5（右删失） |
| 1685459668 | `s1-096f8f16-m1685459668-e2-8611b794` | 0.8432785266659345 / 0.8460871904349292 | 0.8440976853967682 / 0.8453586690532408 | **+0.0008191587308337134** | −0.0007285213816884406 | NO_CLEAR_IMPROVEMENT | 5 / 5（右删失） |
| 1685496394 | `s1-096f8f16-m1685496394-e2-2231eae1` | 0.8474462695821553 / 0.84798146522571 | 0.8476911350393439 / 0.8473791525613715 | **+0.0002448654571886033** | −0.0006023126643385224 | NO_CLEAR_IMPROVEMENT | 5 / 5（右删失） |

统计（检查点记录，§6.3 方法）：n=5；mean `+0.0009418853654596271`；std(ddof=1) `0.00127822927632415`；positive_count（≥ +0.001）**2**；worst_seed `1685463909`；Student t 95% CI（t=2.776）`[−0.0006449914677945057, +0.0025287621987137602]`。机制（处理臂，val）：`null_top1_rate` ∈ [0.3012, 0.4524]（5/5 ∈ [0.05, 0.95]）；`null_std` 0.117–0.252（分布宽，非 AliCCP 线式常量衰减）。复现锚 `REPRODUCED`（seed-1 新旧对子逐位相等，1e-9 容差内噪声为 0）。

**逐 epoch Δval（处理 − 基线，短预算；§1.2 轨迹审计的原始记录）**：

| seed | ep1 | ep2 | ep3 | ep4 | ep5 |
|---|---|---|---|---|---|
| 1685480945 | +0.001784489 | +0.001192310 | +0.001546190 | +0.000865780 | +0.000232240 |
| 1685463909 | +0.000457569 | −0.000179823 | +0.000641710 | +0.000785271 | −0.000289289 |
| 1685477428 | +0.000331842 | +0.001607550 | +0.003375947 | +0.004876488 | +0.003955710 |
| 1685459668 | +0.001541534 | +0.000916640 | +0.000480778 | −0.000989155 | −0.000728521 |
| 1685496394 | −0.000082028 | −0.000357972 | −0.000284542 | −0.000202900 | −0.000602313 |

### 1.2 轨迹审计结论（决定预算选择的事实，与结果无关）

1. 十条 run **全部右删失**（best=ep5），patience=2 在 5 epoch 内**从未被行使**——延长预算后 patience 必须先于 run 重新钉死（§2）。
2. Δval 轨迹形态跨 seed 分化：seed 1 单调收窄（+0.00178 → +0.00023）；seed 3 于 ep4 达峰后回落（+0.00488 → +0.00396）；seed 4/5 在 ep4–5 转负；seed 2 波动。**增量是否在更长预算下继续收窄/翻转，必须实测。**
3. 同机同配置同 artifact 的 run-to-run 噪声在本项目已被实测为 0（锚 `REPRODUCED`，逐位相等）——因此长短预算的 Δ 差异在本机上不会被 run 噪声混淆（但仍受 seed 数 n=5 限制，§9）。

### 1.3 Stage-1 产物复用裁定（本 worktree 实测；预注册前核验脚本 `verify_census_stage1_artifacts.py`）

五份 Stage-1 产物**字节级复制**自已完成检查点的 worktree（`D:\MPT-Rec-three_task\MPT-Rec-exp-census-null-multiseed\artifacts\census_stage2\`），复制后在本 worktree **重跑同一核验：75/75 ALL PASS**（报告 `artifacts/census_stage2/audit/prereq_report.json`，gitignore）。核验内容（只读）：逐文件 sha256 == 已提交检查点记录；`meta.json` 全 config 重算（`config_hash`、`stage1_id` 内容寻址重算命中）；参数级 `backbone_sha256` / `env_ids_sha256` 重算 == meta；split 指纹重算 == `096f8f16…`（train/val/test = 199523/49881/49881，disjoint & union_complete）。

| seed | stage1_id | backbone.pt sha256 | env_ids.pt sha256 | meta.json sha256 |
|---|---|---|---|---|
| 1685480945 | `s1-096f8f16-m1685480945-e2-cb2094b3` | `f41aa2c0…9add2` | `3977f76d…24154` | `c67f1a2a…d6ebc` |
| 1685463909 | `s1-096f8f16-m1685463909-e2-8b53a3bf` | `41d2827a…119fc` | `9394ef91…9917f` | `6b22b051…1b6e2` |
| 1685477428 | `s1-096f8f16-m1685477428-e2-15eabcb4` | `d06f18c6…5ad802` | `706ae0a9…d0617d` | `66305ecb…1a44fac` |
| 1685459668 | `s1-096f8f16-m1685459668-e2-8611b794` | `746681c0…5f6a57` | `89926aac…ea9b77` | `2ce2cd89…fc04061` |
| 1685496394 | `s1-096f8f16-m1685496394-e2-2231eae1` | `5b23036b…0b4566` | `07a2f5c6…16e6dca` | `ebd84a2c…1790346` |

（完整 64 位十六进制值与全部 75 项核验明细见报告 JSON 与已提交检查点；上表为摘要截断。）数据文件：`dataset/Census-income/train.gz` sha256 `e2f2d579…d1b2dc`、`test.gz` sha256 `8de69037…100cec`（与检查点运行所用文件字节一致；`dataset/` 为 gitignore，复制自 main tree 只读源）。

**裁定**：`REUSE`（内容寻址 id + 全部哈希 + config + 划分指纹逐位验证通过）。**失败处置**：任一核验不一致 ⇒ **停止**（不重训、不替换、不静默换产物），保留现场并如实报告——本实验**不重训任何 Stage-1**。

---

## 2. 预算选择（基于 §1 轨迹，不基于任何结果；先于 run 写死）

| 项 | 取值 | 依据 |
|---|---|---|
| Stage2 **max epochs** | **10** | 5-epoch 为右删失点（§1.2 结论 1）；10 = canonical 5 的 2×，与 AliCCP 线已建立的 longer-budget 协议（`47ecb0c` §2 / `bbd8a61`）**完全相同**的固定预算 |
| Stage2 **patience** | **3** | 与 AliCCP 线 longer-budget 协议相同：canonical 值 2 在 5 epoch 内从未被行使（无任何非改进 epoch）；延长预算后允许**至多 2 个连续非改进 epoch** 后停止，既容忍暂时波动又保证终止；两臂**同一规则**，先于 run 写死 |
| 其余全部 | 不变 | §3 |

- 若某臂在 10 epoch 内触发 early stop：属预注册行为，**照实记录**（停止 epoch、best epoch 仍取 val 最优并重载），不作为失败、不重跑。
- **禁止**：看到轨迹后改预算/patience；事后挑选 epoch 报数（判定只用各臂 val 选出的 best checkpoint 与单次 test 评估，§5）。
- 墙钟估计：5-epoch 单臂 ≈ 1–2 min（按已完成 run 的 run_id 实测间隔）→ 10-epoch 单臂 ≈ 3–5 min；10 条 run 合计 ≈ 40 min，前台串行可控。

---

## 3. 控制变量（哪些变、哪些钉死）

| 项 | 短预算（已提交检查点，仅 context） | 本实验（10-epoch） | 说明 |
|---|---|---|---|
| **Stage2 epochs** | 5 | **10** | 唯一变化之一（§2） |
| **Stage2 patience** | 2 | **3** | 唯一变化之二（§2） |
| tag / run_id | `short` | **`long`**（新 tag 词汇，§11 披露；`smoke`/`short` 语义不变） | run_id 身份标识；处理臂 run_id 由 runner 追加 `-nullx` 后缀 |
| Stage-1 产物 | 5 份（§1.1 表） | **同 5 份钉死（不重训）** | 内容寻址 id + 文件 sha256 + 参数级哈希 + 划分指纹全部钉死（§1.3，75/75） |
| model seed | 5 canonical：`1685480945 / 1685463909 / 1685477428 / 1685459668 / 1685496394` | **同（不变）** | 每 seed 两臂同进程重播种 → 头初始化与样本顺序构造性相同 |
| split seed / env seed | `20260929` / `20260929` | **同（不变）** | 划分指纹 `096f8f16…`；`env_ids` 来自产物（不重抽） |
| 数据 | `dataset/Census-income/{train,test}.gz`（sha256 钉死，§1.3） | **同（不变）** | train 全量训练；test.gz 按 split seed 50/50 切 val/test |
| 架构 / 超参 | `input_size=123`、`embedding_size=4`、`expert_hidden=(256,128)`、`tower_hidden=(64,32)`、`num_tasks=2`、`num_envs=2`、`temperature=150`、batch 256、lr 1e-3、`reg_embedding=0.006`、`reg_dnn=3e-5`、`uni_coe=0.9`、`env_coe=0.1` | **同（不变）** | `protocol.py` 零修改 |
| Null 构造 | 处理臂 = `NewTask(use_null_expert=True)`（零候选 + 零初始化 `null_key`）；基线臂 = 默认（与 master 逐位一致） | **同（不变；`multitaskrec/model.py` 与 `87e2b8d`/多 seed 分支 tip 字节相同）** | 机制无任何改动 |
| 仪器化 | C 类诊断（`NullRouteDiagnostics` + `null_supervision_stats` + runner 探针 `newtask_null_probe`；val-only） | **同（逐字移植，§6）** | 处理臂新增键；基线臂 `mechanism` 键集逐字不变 |
| 评测口径 | val 选点 + best 重载 + **test 单次评估**；A/B 门禁逐项落盘 | **同（不变）** | 判定只用 run 记录值；无重新评测、无新增 test 遍历 |

---

## 4. 运行计划（恰好一次）与清洁树纪律

**运行矩阵**：5 个 canonical seed × 2 臂 = **10 条 Stage-2 run**（tag `long`、epochs 10、patience 3；每 seed 两臂引用**同一** `--stage1-dir`）。不重训 Stage-1；已完成的有效 run 一律不重跑；不调参、不改结构、不改评测。

**清洁树纪律（run 间提交；每一 run 启动时 `git status --short` 必须为空）**：
1. 提交预注册 + 核验脚本（C1）→ 提交实现与测试（C2）→ 树干净；
2. 逐 seed、按列表序 `[1685480945, 1685463909, 1685477428, 1685459668, 1685496394]`：跑**基线臂**（run 记录 commit = 当时 HEAD）→ **提交其 SUMMARY 追加行**（只允许 `artifacts/census_stage2/SUMMARY.md` 一个文件变化）→ 跑**处理臂** → **提交其 SUMMARY 追加行**；
3. 运行后核验代码同一性：所有 run 的代码路径逐字相同（§7 I10 的 diff 守卫；run_id 记不同 commit 仅为 SUMMARY 追加行的归属）；
4. 分析（纯分析：只读 run 记录，**不重新评测、不新增 test 遍历**）恰一次（§8）。

**无效执行处置（预注册）**：仅当 run 因**工具性原因**（进程崩溃 / 中断 / 环境故障，run 未完整落盘）无效时，保留全部现场（含失败日志与半成品目录，不删除），记录原因后可重跑一次；因**结果原因**（判据未过 / 门禁失败）不构成无效，不重跑，如实记录（失败证据保留）。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

### 5.1 每 seed 量（配对，同 seed 同 stage1_id）

- `Δtest_seed` = `test_auc(处理臂)` − `test_auc(基线臂)`；`Δval_seed` = `best_val_auc(处理臂)` − `best_val_auc(基线臂)`。
- **二级分类字段（新增；与短预算线同一阈值，逐字）**：

| 分类 | 条件（边界含等号） |
|---|---|
| `POSITIVE_IMPROVEMENT` | `Δtest >= +0.001` |
| `NO_CLEAR_IMPROVEMENT` | `-0.02 < Δtest < +0.001` |
| `CLEAR_DEGRADATION` | `Δtest <= -0.02` |

- 历史阈值/verdict 字段（`null_arm`、`auc_min`、`checks`、`pass` 等）**逐字保留**；新分类只作为**独立字段**追加，不改写任何历史结论。

### 5.2 `NO_CLEAR_IMPROVEMENT` 的 headroom 记录（复用短预算线冻结的机械规则，逐字同语义）

逐 seed 记录四个维度（原始值 + 布尔）：`mechanism_active`（`null_top1_rate ∈ [0.05, 0.95]`）、`val_direction_positive`（`Δval_seed > 0`）、`budget_right_censored`（**处理臂** best epoch == 该臂最后训练 epoch）与 `budget_last_delta_positive`（最后一个共同 epoch 的 `Δval(epoch) > 0`）、`cross_seed_consistent`（见下），并给出

```
headroom_remains = mechanism_active AND val_direction_positive
                   AND (budget_right_censored AND budget_last_delta_positive)
                   AND cross_seed_consistent
cross_seed_consistent = (#{Δtest_seed > 0} 严格过半) AND 无 seed Δtest <= -0.02
```

该记录仅适用于 `NO_CLEAR_IMPROVEMENT`（其它分类下字段照记、不作解释），且**只是 headroom 描述，不构成有效性主张**。

### 5.3 持久性判定（对照 = **本实验新跑的 10-epoch 基线臂**；不得用短预算 run 做对照）

设 `δ_i` = 第 i 个 seed 的 `Δtest_seed`（长预算），`v_i` = `Δval_seed`，`e_i` = 最后一个共同 epoch 的 Δval。**`PERSISTS` ⇔ 以下全部成立**：

| 编号 | 维度 | 条件（边界逐字） |
|---|---|---|
| V1 | aggregate | `mean(δ) >= +0.001`（五 seed 均值达到本线冻结的 POSITIVE 下限） |
| V2 | per-seed | `#{δ_i > 0} >= 3` **且** `min(δ_i) > -0.02`（严格过半为正；无 `CLEAR_DEGRADATION`） |
| V3 | validation direction | `#{v_i > 0} >= 3`（best-val 方向严格过半为正） |
| V4 | budget trajectory | `mean(e_i) > 0`（更长预算末端、最后一个共同 epoch 上，臂在汇总上仍领先；严格 > 0） |
| V5 | mechanism | 5/5 处理臂 `null_top1_rate ∈ [0.05, 0.95]`（历史机制判据逐字沿用） |

- `PERSISTS` ⇔ V1 ∧ V2 ∧ V3 ∧ V4 ∧ V5 且 identity/记录校验（下）全部通过；否则 **`NOT_PERSIST`**（失败条件集合逐项落盘，照实披露）。
- **identity/记录/构造校验（INVALID 分支；任一项 false ⇒ `INVALID`，按实记录并停止解读）**：每对两臂 `stage1_id` 相同**且** == §1.3 钉死值（逐 seed）；两臂 split 指纹相同 == `096f8f16…`；两臂 `env_ids_sha256` 相同；10 条 run 记录的 `tag == "long"`、`epochs == 10`、`patience == 3`；run_id 与 config 内部一致；**A 类门禁（A1/A2/A4/A5，构造/冻结完整性）在 10 条 run 上全部 PASS**。B 类门禁（B1–B4）照常判定并披露，**不参与判定**；B3 在部分 seed 两臂上 FAIL 属继承的协议门禁行为（两臂共享同一 Stage-1 产物；短预算线口径逐字沿用），如实记录、不重判。
- 阈值来源（全部沿用既有冻结数字，不新发明）：`+0.001` 与 `-0.02` = 本线短预算预注册的二级分类阈值（`1196acd` §6.1）；`0.05 / 0.95` = 历史机制判据（§6.4）；majority = 严格过半（5 seed ⇒ ≥3）；Student t `t = 2.776`（df=4）。

### 5.4 汇总统计（方法写死；`1196acd` §6.3 逐字同法）

逐 seed `δ_i`；`mean`、`std`（**样本标准差，ddof=1**）、`positive_count = #{δ_i >= +0.001}`、`worst_seed = argmin(δ_i)`（并列取 seed 列表序靠前者）；**Student t 95% CI**：`mean ± t(0.975, df=4) · s/√5`，`t = 2.776`——方法原句落盘，n=5 下为**描述性**（不构成确证性推断）。同时记录：与短预算检查点的逐 seed 对照（`Δtest_long − Δtest_short`、`Δval` 对照）与两套 CI 的并列——**仅作 context**。

### 5.5 阈值纪律与判据回放（retrodiction）

本节全部数值只允许在**看到任何新 run 结果之前**修改；每次修改单独 commit 并记录理由。已判定的 run 不得用新阈值重判。

**回放（机械性质，防"规则只为本轮结果而设"的质疑；属已提交数据的性质，不构成本轮结果）**：把 §1.1 的已提交短预算数值代入 §5.3 判据函数：V1 = false（mean `+0.000942 < +0.001`）；V2 = true（4/5 > 0；min `−0.000528 > −0.02`）；V3 = false（2/5 Δval > 0）；V4 = true（mean(e) = `+0.000513565580634423 > 0`）；V5 = true。⇒ 短预算数据代入本判据 = **`NOT_PERSIST`**。判据对动机数据不"放水"。

---

## 6. 实现面（最小；TDD；默认臂逐位一致）

| 文件 | 改动 | 来源 / 守卫 |
|---|---|---|
| `census_benchmark/metrics.py` | 追加 C 类：`NullRouteDiagnostics`（null 质量分布/分位数、router 熵、逐样本路由方差）与 `null_supervision_stats`（标签分层、相关、`pred_std`） | **逐字移植** `exp/census-stage2-null-expert-multiseed` @ `1196acd:census_benchmark/metrics.py`（其自身逐字移植自 `exp/aliccp-stage2-null-expert` @ `6224c0f`）；历史 `NULL_ARM_*` 常量与 `null_arm_verdict` **零改动** |
| `run_census_benchmark.py` | ① `newtask_null_probe`（val-only、no_grad、仅处理臂）逐字移植并接线（`mechanism.update(null_probe)`）；② Stage2 新增 `--patience`（默认 `P.PATIENCE`，逐位不变）并线程到 early stop 与 `config.json` 记录；③ `stage2` 记录追加 `epochs`/`patience` 键（仅 JSON 记录，训练/评测数值不变）；④ `--tag` choices 增加 `"long"`（默认仍 `"short"`） | ①逐字移植（源 blob 守卫）；②③④为本实验所需最小 CLI/记录支持（§11 披露）；默认路径行为逐位不变（I1/I3/I4） |
| `census_benchmark/multiseed.py` | **逐字移植** `1196acd:census_benchmark/multiseed.py` + 唯一适配：`find_runs`/`seed_record` 增加 `tag` 参数（默认 `"short"`，缺省行为逐位不变） | 复用冻结的二级分类 / headroom / 统计 / 配对硬约束语义（测试同源移植） |
| `census_benchmark/longer_budget.py`（新） | 钉死常量（预算 10/3、tag `long`、5 seed、5 个 stage1_id、判据阈值、短预算 context 常量）+ `persistence_verdict`（V1–V5 纯函数）+ `analyze`（只读 run 记录 → 判定 + 描述量 + identity checks + context；可选 JSON 落盘）+ `python -m` CLI | 本分支新写（判定规则 = §5）；统计/分类/headroom 复用 `multiseed` |
| `census_benchmark/tests/test_null_diagnostics.py`（新） | C 类诊断不变量（移植 + 与 `evaluate_newtask` M7 逐位一致 + 基线臂键集不变） | 同源移植 `1196acd` |
| `census_benchmark/tests/test_multiseed.py`（新） | 移植 + 新增 `tag="long"` 选择测试 | 同源移植 `1196acd` + 适配断言 |
| `census_benchmark/tests/test_longer_budget.py`（新） | §7 I4–I9 不变量（V1–V5 边界、INVALID、CLI、纯度、文档 token、静态守卫） | 本分支新写 |
| `verify_census_stage1_artifacts.py`（新，本提交） | 预注册前只读完整性核验（§1.3；75/75） | 本分支新写 |

**明确不改**：`census_benchmark/protocol.py`（零 diff）、`multitaskrec/model.py`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`config.py`、`CensusIncome_*.py`、`baseline/*`、`master`。SUMMARY 只追加行、**永不重写**。

---

## 7. 不变量（由测试强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂（`use_null_expert=False`）前向/反向与 `master` 逐位一致；参数集合/state_dict 相同；null 构造不消耗 RNG | 既有 `test_null_expert.py` 守卫保持全绿（动态加载 `master:multitaskrec/model.py`） |
| I2 | C 类诊断与探针为**逐字移植**：`metrics.py` 移植段与 `run_census_benchmark.py::newtask_null_probe` 的 AST/字节段 == `1196acd` 源 blob 对应段 | `git show` + 段级字节比较 |
| I3 | 基线臂 `mechanism` 键集逐字不变（`{gate_mean, cos_gen_spec, gen_std, env_acc_stage1}`）；处理臂新增键含 M7（`null_mean`、`null_top1_rate`）与 C 类全部键；探针值与 `evaluate_newtask(mechanism=True)` 的 M7 值**逐位一致** | 夹具端到端 + `torch.equal` |
| I4 | CLI/override：`--epochs`/`--patience` 默认值 == 协议常量（逐位不变）；`--tag long` 可解析且默认仍 `short`；`--epochs 10 --patience 3` 经 `main()` 正确线程化到 `run_stage2` 调用参数（monkeypatch 捕获 kwargs）；run 记录 `epochs=10 / patience=3` | parser 断言 + `main()` 捕获式测试 |
| I5 | 判定边界：V1 `+0.001` 过（闭）、`+0.000999…` 不过；V2 3/5 正值过、2/5 不过、`max` 恰 `−0.02` 不过；V3 恰 0 不算正；V4 恰 0 不过（严格）、`+1e-12` 过；V5 端点 `0.05/0.95` 含（过）、越界不过；identity 任一 false ⇒ `INVALID`（即使 V1–V5 全过） | 真值表/边界测试 |
| I6 | 分析器只读（不改 run 产物；运行前后文件哈希不变）；输出键齐全（判定/逐 seed 记录/统计/CI/context/identity checks）；context 常量 == 已提交检查点 JSON 的钉死值；夹具端到端 | CPU 极小夹具 + 篡改检测 |
| I7 | 短预算 context JSON 字节一致：content sha256 == `605196e5…`（分析器加载时校验；不一致 ⇒ 抛错） | 常量钉死 + 校验测试 |
| I8 | 预注册文档 token 钉死：预算 10/3、tag `long`、5 seed 列表、5 个 stage1_id、阈值 `+0.001 / −0.02 / 0.05 / 0.95 / 2.776`、`PERSISTS`/`NOT_PERSIST`/`INVALID`、短预算 context 数值、"仅动机"表述 | 文档文本断言 |
| I9 | 静态守卫：相对 `87e2b8d` 的全部跟踪改动 ⊆ §6 白名单；`protocol.py` / `multitaskrec/*` 零 diff | `git diff --name-only` |
| I10 | 全测试套件绿（既有 46 + 新增）；测试先于实现（红 → 绿） | unittest discover |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 8. 运行方式（恰好一次；命令留档）

```powershell
# 0) 预注册前核验（只读；75/75 通过方能继续）——已在 C1 前执行
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe verify_census_stage1_artifacts.py --out artifacts/census_stage2/audit/prereq_report.json

# 1) 测试（全部不变量；先写测试后写实现）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests

# 2) 每 seed（列表序）：基线臂 → （提交 SUMMARY 行）→ 处理臂 → （提交 SUMMARY 行）
#    seed 1685480945（其余 seed 仅 stage1-dir 与 run 记录不同）
python run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 --tag long --epochs 10 --patience 3 --gpu 0
python run_census_benchmark.py stage2 --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 --tag long --epochs 10 --patience 3 --null-expert --gpu 0
# 其余四个 stage1-dir：…-m1685463909-e2-8b53a3bf / …-m1685477428-e2-15eabcb4 / …-m1685459668-e2-8611b794 / …-m1685496394-e2-2231eae1

# 3) 持久性分析（纯分析，恰一次；只读 run 记录）
python -m census_benchmark.longer_budget --root artifacts/census_stage2 --out docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-checkpoint.json
```

---

## 9. 局限（预声明）

1. **seed 数 n=5**：判定依赖 5 个配对点的汇总统计；CI 为描述性，不构成确证性推断。
2. **预算固定为 10 epochs**：更长（20+）预算下的行为不外推；本轮不外延预算。
3. **无独立重评测 pass**：判定直接取 run 记录值（单次 test 评估纪律）；记录值正确性依赖 runner 代码路径与短预算协议逐字一致（§7 I2/I9）+ 短预算线已实测的同配置 run-to-run 确定性（噪声 0 @1e-9）。
4. **B3 继承 FAIL**（部分 seed 两臂）：由共享 Stage-1 产物决定，属继承缺陷，不归因于处理臂、不据此否定任何臂。
5. **单数据集（CensusIncome）**：结论不推及 AliCCP / ByteRec。
6. **归因未做**：若 `PERSISTS`，增益来源（null 候选语义 vs 额外路由容量）需另立独立预注册分支（§10 结论后给出精确建议）；若 `NOT_PERSIST`，本线关闭，不启动归因。

---

## 10. 结果（2026-10-05 实测；运行后追加，判据未回填、未修改）

**状态**：10 条 run（5 canonical seed × 2 臂）全部完成；全部前台、串行、单次执行（无重跑、无调参、无崩溃）；每条 run 启动时树干净、run 间提交 SUMMARY 行（§4 纪律）。判定 = §5.3 机械计算（`census_benchmark/longer_budget.py`，恰一次）；独立复核 58/58 ALL PASS（§10.10）。**§0–§9 未改动一行**（判据、阈值、预算与全部分析方法均保持预注册原样）。

### 10.1 运行清单与产物哈希

| seed | stage1_id（钉死复用） | 基线臂 run_id（记录 commit） | 处理臂 run_id（记录 commit） |
|---|---|---|---|
| 1685480945 | `s1-096f8f16-m1685480945-e2-cb2094b3` | `20261005-0519-s20260929-m1685480945-long-618baaf`（618baaf） | `20261005-0522-s20260929-m1685480945-long-5510517-nullx`（5510517） |
| 1685463909 | `s1-096f8f16-m1685463909-e2-8b53a3bf` | `20261005-0525-s20260929-m1685463909-long-3e5c6a5`（3e5c6a5） | `20261005-0528-s20260929-m1685463909-long-6f38b3c-nullx`（6f38b3c） |
| 1685477428 | `s1-096f8f16-m1685477428-e2-15eabcb4` | `20261005-0531-s20260929-m1685477428-long-437aec2`（437aec2） | `20261005-0534-s20260929-m1685477428-long-e274588-nullx`（e274588） |
| 1685459668 | `s1-096f8f16-m1685459668-e2-8611b794` | `20261005-0537-s20260929-m1685459668-long-4774af0`（4774af0） | `20261005-0541-s20260929-m1685459668-long-e7cf912-nullx`（e7cf912） |
| 1685496394 | `s1-096f8f16-m1685496394-e2-2231eae1` | `20261005-0544-s20260929-m1685496394-long-cac5e33`（cac5e33） | `20261005-0547-s20260929-m1685496394-long-b7ae211-nullx`（b7ae211） |

全部 run 的逐文件 sha256（run 目录 7 文件 + stage1 目录 4 文件）、逐 epoch 记录、门禁、机制量见检查点 JSON（`docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-checkpoint.json`，`longer_budget` CLI 恰一次生成）。运行间代码同一性：全部 10 条 run 的代码路径与 §5.3 实现同一（I9 白名单 + §10.10 复核）。

### 10.2 配对结果（主表；同 seed、同 stage1_id、同划分；test 单次评估）

| seed | AUC-test 基线 | AUC-test 处理 | **Δtest** | best-val 基线 / 处理 | **Δval** | 分类（§5.1） | best epoch 基线/处理 | last epoch 基线/处理 |
|---|---|---|---|---|---|---|---|---|
| 1685480945 | 0.8500685307175756 | 0.8513648272440768 | **+0.001296296526501206** | 0.8527881905614896 / 0.853020431042571 | +0.00023224048108139161 | **POSITIVE_IMPROVEMENT** | 5 / 5 | 8 / 8（两臂 ep8 早停） |
| 1685463909 | 0.8449392888699385 | 0.8467710426729359 | **+0.0018317538029973823** | 0.8480409456792931 / 0.8490443243919258 | +0.0010033787126326743 | **POSITIVE_IMPROVEMENT** | 8 / 8 | 10 / 10 |
| 1685477428 | 0.8463878771131822 | 0.8478916659703787 | **+0.0015037888571964864** | 0.8483220346655591 / 0.8499875718551647 | +0.0016655371896056037 | **POSITIVE_IMPROVEMENT** | 9 / 8 | 10 / 10 |
| 1685459668 | 0.8468816103883973 | 0.8459363628622468 | **−0.0009452475261504389** | 0.8491575377445129 / 0.84771385570495 | −0.0014436820395629102 | **NO_CLEAR_IMPROVEMENT** | 8 / 8 | 10 / 10 |
| 1685496394 | 0.8493675426339112 | 0.8490419395915664 | **−0.0003256030423448575** | 0.8490294382940713 / 0.8488915769064728 | −0.00013786138759852573 | **NO_CLEAR_IMPROVEMENT** | 10 / 7 | 10 / 10 |

### 10.3 持久性判定（§5.3 预注册判据，机械计算）

| 编号 | 维度 | 结果 | 值 |
|---|---|---|---|
| V1 | aggregate | **FAIL** | mean(Δtest) = `+0.0006721977236399557` **< +0.001** |
| V2 | per-seed | PASS | 3/5 为正值（>0）；min = `−0.0009452475261504389 > −0.02`（无 CLEAR_DEGRADATION） |
| V3 | validation direction | PASS | 3/5 Δval > 0 |
| V4 | budget trajectory | PASS | mean(last-common-epoch Δval) = `+0.00040311434247657373 > 0` |
| V5 | mechanism | PASS | 5/5 处理臂 `null_top1_rate ∈ [0.3244, 0.4810] ⊂ [0.05, 0.95]` |

**identity/记录/构造校验（INVALID 分支）：5/5 seed × 9 项全过 → 无 INVALID**（两臂 stage1_id 相同且 == 钉死值；split 指纹相同 == `096f8f16…`；env_ids 相同；tag=long / epochs=10 / patience=3 落盘一致；A 类门禁 10/10 run PASS）。

**判定：`NOT_PERSIST`**（失败条件集合 = {V1}；V2–V5 全过）。短预算下的小幅正增量在 10-epoch / patience-3 预算下**未达到预注册的持续性判据**（聚合均值回落至 +0.000672，低于 POSITIVE 下限 +0.001）。

### 10.4 汇总统计（§5.4 方法）

| 统计量 | 值 |
|---|---|
| n | 5 |
| mean(Δtest) | `+0.0006721977236399557` |
| std(Δtest)（样本标准差，ddof=1） | `0.0012285508673826697` |
| positive_count（Δtest ≥ +0.001） | **3** |
| worst_seed | `1685459668`（Δ = −0.0009452475261504389） |
| 95% CI | `[−0.000853005106583309, +0.0021974005538632204]`（**仍含 0**） |
| CI 方法（原句） | `Student t 95% CI on paired deltas, df=n-1, t=2.776 (n=5)` |

### 10.5 二级分类与 headroom（新增字段；历史 verdict 未改写）

**分类计数**：POSITIVE_IMPROVEMENT **3** / NO_CLEAR_IMPROVEMENT **2** / CLEAR_DEGRADATION **0**（短预算为 2 / 3 / 0）。

| seed | 分类 | mechanism_active | val_direction_positive | budget_right_censored | budget_last_delta_positive | cross_seed_consistent | headroom_remains |
|---|---|---|---|---|---|---|---|
| 1685480945 | POSITIVE | true | true | false（臂 best=ep5 < last=ep8） | true | true | false（context） |
| 1685463909 | POSITIVE | true | true | false（best=8 < last=10） | true | true | false（context） |
| 1685477428 | POSITIVE | true | true | false（best=8 < last=10） | true | true | false（context） |
| 1685459668 | NO_CLEAR | true | **false** | false（best=8 < last=10） | **false** | true | **false** |
| 1685496394 | NO_CLEAR | true | **false** | false（best=7 < last=10） | **false** | true | **false** |

两个 NO_CLEAR seed 的 headroom 记录均为 **false**（val 方向负 + 末端 Δval 负 + 预算非删失）；按预注册该记录仅为描述，不改变分类。

### 10.6 预算状态与轨迹（右删失 / 早停 / 逐 epoch Δval）

- **早停（预注册行为，照实记录）**：seed 1685480945 两臂均于 **ep8** 触发 early stop（best=ep5；patience=3 在长预算下首次被行使）；seed 1685496394 处理臂于 ep10 达到 stale==patience，与循环终点重合（机械标志 `stopped_early=false`，如实并列记录）；其余 8 条 run 走满 10 epoch 无早停。**无任何因结果原因的重跑**。
- **右删失**：基线臂仅 seed 1685496394 为右删失（best=ep10=last）；其余 9 条 run 的 best epoch < 最后 epoch（长预算被实际使用：8/10 条 run 的 best epoch ≥ 7）。
- **逐 epoch Δval（处理 − 基线；long）**：

| seed | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 | ep9 | ep10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1685480945 | +0.001784489 | +0.001192310 | +0.001546190 | +0.000865780 | +0.000232240 | −0.000311916 | +0.000255811 | +0.000058318 | —（早停） | — |
| 1685463909 | +0.000457569 | −0.000179823 | +0.000641710 | +0.000785271 | −0.000289289 | +0.000361089 | +0.000692305 | +0.001003379 | +0.000958453 | +0.001134301 |
| 1685477428 | +0.000331842 | +0.001607550 | +0.003375947 | +0.004876488 | +0.003955710 | +0.005179287 | +0.003584266 | +0.005332637 | +0.001385503 | +0.003703880 |
| 1685459668 | +0.001541534 | +0.000916640 | +0.000480778 | −0.000989155 | −0.000728521 | −0.000890672 | −0.001820162 | −0.001443682 | −0.002187528 | −0.002391914 |
| 1685496394 | −0.000082028 | −0.000357972 | −0.000284542 | −0.000202900 | −0.000602313 | +0.000436969 | +0.000233690 | +0.000077070 | −0.000334991 | −0.000489013 |

（seed 1685480945 于 ep8 早停 → 无 ep9/ep10 记录，如实留空。）轨迹形态（事实记录，不解读）：seed 2 的正 Δ 在 ep8–10 扩大并稳定；seed 3 全程为正但幅度波动收窄；seed 4 的负 Δ 在 ep6–10 持续加深；seed 5 在零附近震荡；seed 1 与短预算完全一致（见 §10.10 确定性）。

### 10.7 机制量与塌缩检查（处理臂，val；§7 I2/I3 仪器）

| seed | null_top1_rate | null_mean | null_std | q10 / q50 / q90 | route_entropy_mean/std（max ln3=1.0986） | route_var_mean | 分层差(edu) | corr(null,pred) | corr(null,\|err\|) | pred_std | gen_std |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1685480945 | 0.3726 | 0.3418 | 0.1707 | 0.1245 / 0.3394 / 0.5791 | 0.9130 / 0.1080 | 0.04254 | +0.1014 | +0.4201 | +0.3790 | 0.13307 | 3.4770 |
| 1685463909 | 0.4578 | 0.3813 | 0.1828 | 0.1461 / 0.3969 / 0.6198 | 0.9017 / 0.0935 | 0.04472 | +0.0837 | +0.2967 | +0.3090 | 0.13929 | 2.2782 |
| 1685477428 | 0.4123 | 0.4163 | 0.3130 | 0.0945 / 0.2846 / 0.8463 | 0.7984 / 0.1696 | 0.06766 | **−0.2632** | **−0.5400** | −0.4440 | 0.13662 | 1.1099 |
| 1685459668 | 0.4810 | 0.3653 | 0.1410 | 0.1756 / 0.4024 / 0.5274 | 0.9487 / 0.0975 | 0.03416 | +0.0509 | +0.2841 | +0.3277 | 0.15591 | 0.8513 |
| 1685496394 | 0.3244 | 0.3484 | 0.1262 | 0.1989 / 0.3218 / 0.5371 | 0.9428 / 0.0750 | 0.03458 | +0.0586 | +0.3918 | +0.3410 | 0.14116 | 2.8158 |

- 机制判据（`top1 ∈ [0.05, 0.95]`）：**5/5 成立** → V5 PASS。
- 塌缩检查：`null_std` 0.126–0.313（分布宽、逐样本变化；非 AliCCP 线式常量衰减）；`route_entropy_mean` 0.798–0.949（< ln3）；`gen_std` 0.851–3.477（无表征常量塌缩）。task-wise 候选份额：本协议仪器记录的是 null 候选的逐样本分布（均值/分位数/std）+ router 熵 + 逐样本路由方差 + gate 均值（§10.8）——完整逐任务候选份额未单独落盘（如实记为 N/A，不补算）。
- 事实性记录（不解读）：seed 3 的 `null_label_gap`/`corr_null_pred` 与短预算一致为负（−0.2632 / −0.5400）；其余 4 个 seed 为正。

### 10.8 门禁（逐 seed 逐臂，原样记录）

| seed | 基线臂 | 处理臂 |
|---|---|---|
| 1685480945 | A1✓ A2✓ A4✓ A5✓ B1✓ B2✓ **B3✗** B4✓ | 同左（**B3✗**） |
| 1685463909 | 全部 PASS | 全部 PASS |
| 1685477428 | 全部 PASS | 全部 PASS |
| 1685459668 | 全部 PASS | 全部 PASS |
| 1685496394 | A1✓ A2✓ A4✓ A5✓ B1✓ B2✓ **B3✗** B4✓ | 同左（**B3✗**） |

B3 FAIL 出现在 seed 1（`gate_mean[0] = 0.9524`）与 seed 5（`0.9898`）两臂，与短预算记录一致且两臂共享同一 Stage-1 产物——属**继承**的协议门禁行为，原样记录、不重判（§5.3：B 类不参与判定）。历史 `null_arm` 字段（`auc_min=0.8521` 等 seed-1 专属阈值）在 5 个处理臂 run 中逐字落盘（`pass=false`），**未改写**。

### 10.9 与冻结的 5-epoch 配对结果对照（context only，不入判定）

| seed | Δtest 短 → 长 | Δtest 变化 | Δval 短 → 长 | Δval 变化 |
|---|---|---|---|---|
| 1685480945 | +0.001296296526501206 → +0.001296296526501206 | **0.0**（逐位相同） | +0.00023224048108139161 → +0.00023224048108139161 | **0.0** |
| 1685463909 | −0.0005284892173337274 → +0.0018317538029973823 | **+0.0023602430203311098**（由负转正） | −0.0002892890134781334 → +0.0010033787126326743 | +0.0012926677261108077 |
| 1685477428 | +0.00287759533010834 → +0.0015037888571964864 | −0.0013738064729118538（保持正、幅度收窄） | +0.00395571048159582 → +0.0016655371896056037 | −0.0022901732919902162 |
| 1685459668 | +0.0008191587308337134 → −0.0009452475261504389 | −0.0017644062569841523（由正转负） | −0.0007285213816884406 → −0.0014436820395629102 | −0.0007151606578744696 |
| 1685496394 | +0.0002448654571886033 → −0.0003256030423448575 | −0.0005704684995334608（由正转负） | −0.0006023126643385224 → −0.00013786138759852573 | +0.00046445127673999664 |

**汇总对照（context）**：mean Δtest 由 `+0.0009419`（短）→ `+0.0006722`（长）；std 0.0012782 → 0.0012286；positive_count（≥+0.001）2 → 3；worst_seed 1685463909（−0.000528）→ 1685459668（−0.000945）；两套 CI 均含 0。逐 seed 变化方向不一致（1 个 0 变化、1 个转正、1 个收窄、2 个转负）——**这是对照记录，不构成任何新主张**。

### 10.10 确定性核对与独立复核（§7 I6/I7；`verify_census_longer_budget.py`，恰一次）

**58/58 ALL PASS**（报告：入库副本 `docs/superpowers/specs/2026-10-05-census-stage2-null-expert-longer-budget-verify.json`；生成路径 `artifacts/census_stage2/audit/longer_budget_verify_report.json`，gitignore）：① 从磁盘 run 记录**独立重推**全部 Δtest/Δval/分类/统计/判定 == 检查点 JSON（逐位）；② 10 个 run 目录 + 5 个 stage1 目录的全部文件 sha256 == 检查点记录；③ **确定性**：10 条 long run 的**前 5 epoch 轨迹（val AUC + loss）与对应短预算 run 逐位一致**（两臂皆然）——长预算的 Δ 差异完全来自 epochs 6–10 与 best-epoch 选择，非同配置噪声；seed 1 两臂 best=ep5 → test AUC 亦与短预算**逐位相同**；④ 短预算 context 源 sha256 钉死一致。

该核对同时构成对短预算线的**跨分支确定性证据**（同 stage1、同 seed、跨 commit 的前缀轨迹逐位复现），如实并列记录、不外推。

### 10.11 结论（按预注册判据逐字陈述，不越界）

1. **判定：`NOT_PERSIST`**（V1 aggregate FAIL：mean Δtest = `+0.0006721977236399557 < +0.001`；V2–V5 PASS；identity 5/5 全过，无 INVALID）。预注册判据未被修改、未被回填；§5.5 的判据回放（短预算数据代入 → NOT_PERSIST）与本轮判定一致，说明判据未向"通过"倾斜。
2. **二级分类（独立新增字段）**：3 POSITIVE_IMPROVEMENT / 2 NO_CLEAR_IMPROVEMENT / **0 CLEAR_DEGRADATION**；两个 NO_CLEAR seed 的 headroom_remains = false。历史 verdict 字段与文档（`2026-09-29-stage2-null-expert-design.md`、`exp/census-stage2-null-expert-multiseed` @ `1196acd` 的 §6/§10/§12）**逐字保留、未改写**。
3. **本轮无失败 run**（无崩溃、无重跑）；失败证据条款不适用，如实记录。
4. **Census Null Expert 线按预注册处置：关闭**。短预算的小幅正 Δ（mean +0.00094、CI 含 0、2/5 达 POSITIVE）在更长预算下未通过预注册持续性判据（mean 回落 +0.00067，3/5 POSITIVE 但聚合未达 +0.001，2 个 seed 转负）——**不再在本线启动任何归因/特异性对照实验，不写论文、不做图表、不做 novelty 声明**（预注册 §9.6；本分支不做任何后续候选）。
5. **明确不主张**：本文不构成有效性/通用性/否定性结论；与 AliCCP 线的机制形态差异、与短预算的方向差异均为事实记录，不解释、不迁移。

---

## 11. 偏离披露

- **分支基点**：自 `exp/stage2-null-expert` @ `87e2b8d` 拉出（未合并入 master），沿用本项目既有工作流；协议文件零修改、SUMMARY 只追加。
- **tag `"long"` 词汇扩展**：协议 §9.1 的 run_id 语法为 `<smoke|short|full>`；本实验引入 `tag="long"`（只改 `run_census_benchmark.py` 的 choices 与 run 记录，`protocol.py` 零修改），使 10-epoch run 的身份不冒用 canonical `short`（5 epoch）语义。属最小 CLI 支持（§6），先于 run 提交。
- **run 间提交**（§4）：每条 run 的 SUMMARY 追加行在下一 run 前提交，保证 10 条 run 全部在干净树下运行。
- **产物复用 = 字节级复制**（§1.3）：Stage-1 产物自另一 worktree 复制（复制后重跑核验 75/75），不重训、不改写源 worktree。
- **分析模块复用**：`census_benchmark/multiseed.py` 及其测试自 `1196acd` 移植（唯一适配 = `tag` 参数），保证二级分类/headroom/统计语义与短预算线**逐字同源**；`SPEC_PATH` 常量保留指向多 seed 预注册文档（历史引用，不复制该文档）。
- **不做第二次 test 遍历**（§9.3）：与 seed-2 复现分支的 `replicate` 步不同，本实验判定只用 run 内单次 test 评估值。
