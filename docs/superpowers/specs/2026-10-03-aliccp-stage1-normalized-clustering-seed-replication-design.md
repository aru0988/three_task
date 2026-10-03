# AliCCP Stage-1 归一化聚类修复的独立 seed 复现（seed 2 = 1688723740；机制稳定性 + Stage-2 方向）

- **状态**：预注册已写死（本文件在本分支任何新 run 之前提交）；实现与结果见 §9（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage1-normalized-clustering-seed-replication`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；不从修复分支拉出；`aliccp_benchmark/protocol.py` 零修改；`multitaskrec/*` 零修改）
- **被复现对象（只读引用）**：AliCCP Stage-1 归一化聚类修复 @ `exp/aliccp-stage1-normalized-env-clustering` 的 `2058de8`（预注册 `4b83863`、实现 `13d2100`、结果 `529090e`/`2058de8`；已 push）。其结论：seed 1688723512 上 `REPAIRED` + `NO_MATERIAL_DEGRADATION`（未出现 `NOT_REPAIRED`；B4 由 `env_0 = 566/2M` 修复为事件 `diff_num = 999966`、`env_0 = 959244`、`env_1 = 1040756` = 47.96%/52.04%，最终 `env_ids_sha256 = 4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b`；其产物 `s1-5c060b9c-m1688723512-e3-ad3b353f`，treatment config hash `ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e`，基线 config hash `3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4`；Stage-2 相对其同 seed raw 基线 `20261003-0130-…-b2e17f9`（test BSI `0.5988392178311113`、val `0.5781533414372665`；raw stage1 test CTR `0.5481837665250074`）得 `Δtest = +0.0082428399`、`Δval = +0.0100592719`）
- **参考基线（本 worktree 既有本地产物，只读）**：raw 聚类 seed 2 配对基线 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`（stage1，commit `79b5e07`）+ run `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（stage2）；既有只读分析 `artifacts/aliccp_bench/audit/s1-5c060b9c-m1688723740-e3-4e1b5c6f-analysis/audit.json`；已记录于 `exp/aliccp-stage2-attenuation-seed-replication` 的 `f589e61:artifacts/aliccp_bench/SUMMARY.md`
- **定位声明**：这是对一项**工程修复（scale-correction ablation）**的**独立 seed 复现**（第二 canonical seed = 列表位置 2），检验 (i) 机制是否跨 seed 稳定、(ii) Stage-2 方向是否在同 seed 内为正。**无新颖性主张、无改进主张**：复现的成功定义 = 机制门禁全过 + within-seed 效用方向为正（§4/§5），不是性能提升；`+0.0055` 只报告不要求（原修复预注册 §4.2「不设提升门槛」）

---

## 0. 判定问题

- **Q1（机制复现）**：在 seed 2（`1688723740`，与参考基线同 seed）上，逐字移植的 rank/quantile 归一化聚类是否复现修复形态：单一聚类事件下 B4 双过（两环境各 ≥ 5%）、分配不再是 CVR 标签恢复、诊断有限、默认关恒等、A 类完整性通过、raw 诊断与既有记录一致？
- **Q2（效用方向复现，within-seed2）**：新 Stage-1 相对 raw seed2 基线无实质 CTR 退化（同 seed1 容差 `0.005`）？其上的未改动 Stage-2 头相对 raw seed2 配对基线 `Δtest > 0` 且 `Δval > 0`？
- Q1/Q2 独立判定（§4 M/U 组），组合分类见 §5。

---

## 1. 机制实现审计与逐字移植（pin @ `2058de8`）

### 1.1 移植范围与字节钉死

机制本体与其测试基础设施按 `2058de8` **逐字节**移植（`git checkout 2058de8 -- <path>`，不手改）。sha256（在 `2058de8` 上计算，写入静态守卫）：

| 文件 | sha256 @ `2058de8` | 角色 |
|---|---|---|
| `aliccp_benchmark/normalized_clustering.py` | `e87740a0b9cbd082b99e2538b6bd72b3103923055b76f78c877a37a8a36e7f95` | 机制本体（`per_task_rank01` + `rank_normalized_cluster_2` + manager 仅覆写 `cluster_2`） |
| `aliccp_benchmark/bench.py` | `ee28543255dd8d6b6c629adefd5ae94522bf753f0ba9b932f925b5e180841710` | `clustering_arm=None` 默认路径逐位不变；arm 注入 + cfg/meta 标记 + balanced-accuracy 探针 |
| `aliccp_benchmark/metrics.py` | `f29eb951fb0ee2ba1b768d13c740eda5b6a2d0de11e370cab4c0b9878ca0bf8e` | `env_balanced_accuracy`（macro recall） |
| `run_aliccp_benchmark.py` | `534be3ca65191c2ab55898aa383e76c820691d3409e9072646856c11bbf8bb4e` | `--clustering {raw,rank_normalized}`（默认 raw）、`--tag norm` |
| `aliccp_benchmark/audit_cluster_losses.py` | `335383aa3da1ffb88fd4d50bcb7013dc5e95dce18bfb403e134db0fdc17e03f3` | 只读审计工具（`ref_rank01` 被测试逐位对照引用；§2 重分析用） |

- **归一化规则与阈值零改动**：`average_rank → q=(rank−1)/(N−1)`、全并列 0.5、N=1 → 0、argmin 取先索引、有限性断言；容差 `0.005`/`0.01`/环境占比 `5%` 全部沿用（§4）。
- **`test_normalized_clustering.py`（@ `2058de8` = `652fed5a70c66c548cb886e6bc9b17fa8089234acd8a5f3e9377335ef9a76817`）**：随机制一并移植，**仅 2 处**分支适配（§8.2 由 AST 守卫逐项钉死为恰此 2 处）：(a) 模块级 `DOC_PATH` → 本文件；(b) `TestStaticGuards.test_tracked_changes_subset_of_whitelist` 的白名单 += 本文件与本分支新增守卫测试文件。其余（含 seed1 的默认协议身份钉死 `BASELINE_CFG_HASH`/`TREATMENT_CFG_HASH`/`PREDICTED_STAGE1_ID`/文档 token 表）逐字保留：seed1 预测值在本分支文档中作为被复现对象引用（§0/§5），token 断言对本文档依然成立。
- **禁止**：`multitaskrec/*`、`config.py`、`AliCCP_*.py`、`baseline/*`、`protocol.py` 相对 `8133d32` 零 diff（静态守卫，§8）。

### 1.2 seed 与配置（固定，先于 run）

| 项 | 值 |
|---|---|
| MODEL_SEED | `1688723740`（canonical 列表位置 2；与参考基线同 seed） |
| ENV_SEED | `20261003`（与两 seed 全部既有 run 相同） |
| 预算/前缀 | `p2M-v500k-t1M`（train 2,000,000 / val 500,000 / test 1,000,000；前缀指纹 `5c060b9c…` = 与参考基线同） |
| Stage-1 | 3 epoch / patience 2 / batch 2000 / lr 1e-4 / uni 0.9 / env 0.1（协议默认，不变） |
| Stage-2 | 5 epoch / patience 2（协议默认，不变；未改动头） |

---

## 2. 参考基线完整性核验（先于本文档提交；`verify_reference_seed2.py`）

对 raw seed2 配对基线做只读核验（35 项，**35/35 通过**，核对项含：config 由协议默认重建一致、`config_hash` 重算 == 记录 == 钉死值、`stage1_id` 三处一致、`env_ids`/`backbone` 张量 sha 重算 == 记录、前缀指纹 A2 级复算、run 产物交叉引用与门禁值、已记录 SUMMARY 行互证、head `strict=True` 载入本分支 `NewTask` 并在参考 backbone 上重评测——**val/test BSI 与 gate_mean 与 run 记录逐位相等**，同时证明本分支 Stage-2 评测路径 == 参考实现路径）。

钉死参考常量（后续判定全部引用）：

| 常量 | 值 |
|---|---|
| raw stage1_id / config_hash | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` / `4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981` |
| raw env_ids_sha256 / backbone_sha256 | `5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0` / `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c` |
| raw 聚类事件（epoch 2） | `diff_num=1000418`、`env_0=1532`（0.0766%）、`env_1=1998468` → B4 FAIL（<5%） |
| raw stage1 test CTR / CVR | `0.5534062000766764` / `0.5420522697385088`（CVR 记录不判定） |
| raw run（short, `79b5e07`）val / test BSI | `0.5809347091990792` / `0.5974422649550507`（head 无衰减：`spec_attenuation=1.0`，`8133d32→79b5e07` 的 model.py/bench.py 改动对 stage2 仅 `!=1.0` 时生效；重评测逐位一致已证） |
| raw run A 类 / B 类 | A1–A6 全 PASS（A3 SKIP）；B1/B2/B3 PASS、B4 FAIL、`hard_pass=false` |

- **raw-scale 诊断（既有只读分析，`-analysis/audit.json` @ `e377a4e`）**：尺度比 64.6×（seed1 65.1×）；env_pred `acc=0.9992525`（与 run 记录逐位一致）而 `balanced_acc=0.50`、`recall_env_0=0`——退化形态与 seed1 同型（§4 M6 将用移植后的审计工具做只读重分析复算）。
- 分支本地 `SUMMARY.md`（@ `8133d32`）尚无 seed2 行——raw seed2 行记录在 `f589e61`（已核验逐值一致）；本分支将被 run machinery 追加本实验自己的行（§11 披露分支系谱差异）。

---

## 3. 冻结预测（看到结果前写死；只预测 ex ante 可证项）

| 项 | 预测值 | 依据 |
|---|---|---|
| 新 `stage1_id` | **`s1-5c060b9c-m1688723740-e3-820697f6`**（config hash `820697f6b29dec50af3335a6b8fe35d5418e822401699441cec1388d50cc266b` = 协议默认 cfg + `{"clustering": "rank_normalized"}`，`model_seed=1688723740`） | 确定性哈希（§2 已用重建 cfg 复算）；指纹 `5c060b9c…` 与参考基线同 |
| epoch 1–2 逐 epoch 记录（7 值 × 2） | 与 raw seed2 meta **逐位相等**（epoch1：CTR `0.5375757994150185`、CVR `0.49566545141616464`、uni `0.258230060338974`/`0.10175464302301407`、fuse `0.24986110627651215`/`0.10588817298412323`、env `2.2546486854553223`；epoch2：CTR `0.5507249370912827`、CVR `0.5009371862723444`、uni `0.20124821364879608`/`0.00564573472365737`、fuse `0.19115711748600006`/`0.004963990300893784`、env `7.149273872375488`） | 首个 `cluster_2` 调用发生在 **epoch 2 训练之后、其 val 评估之前**（`train.py` `epoch % 2 == 0` → `RecordingMPTRecTrainManager.cluster_2` 覆写记录 epoch=len(val_epoch_aucs)+1=2）；此前两臂代码路径完全一致（arm manager 仅覆写 `cluster_2`，不消耗 RNG、不改权重） |
| 聚类事件结构 | 恰 1 次，记录 epoch = 2（B4 判定对象） | 同上（3 epoch ⇒ 仅 epoch 2 触发）；环境逐位可复现性由 seed1 审计 10/10 复现与本 worktree 多次 bit-equal 证据支撑 |
| **明确不预测**（ex ante 未知，不作等式断言） | 事件三元组 `diff_num/env_0/env_1`、`env_ids_sha256`、环境占比、epoch 3 轨迹、test AUC、全部 Stage-2 量 | seed2 无审计 `--reproduce` 捕获（不做；§6）。判定依据为 §4 的机制判据，而非 bit 预测 |

---

## 4. 预注册门禁（看到结果前写死；看到结果后不得修改）

### 4.1 机制复现（M 组）

| 编号 | 判据 |
|---|---|
| M1 | **B4 复现修复**：观测聚类事件（恰 1 次）`env_0 ≥ 5%·N ∧ env_1 ≥ 5%·N`（协议 `ENV_SHARE_MIN=0.05`，不改）；stage2 `gate_report` 的 B4 以 `[(env_0, env_1)]` 同口径 PASS |
| M2 | **身份与轨迹预测命中**：(a) `stage1_id` == §3 预测；(b) epoch 1–2 全部 14 个记录值与 raw seed2 meta 逐位相等 |
| M3 | **有限性**：run 未抛有限性断言；`cluster_diagnostics` 全部 raw/normalized 分位数为有限值（无 NaN/Inf） |
| M4 | **非标签恢复**：`env_0 ∩ purchase=1 ≤ 5%·|env_0|` ∧ `env_0 ≠ purchase=1 集合`（逐位不等）∧ `env_0 ⊄ purchase=1`（存在非 purchase 行）（口径同 seed1 §9.2 的只读重分析；raw 对照 = 564/1532=36.8% 且 env_0∩click1==env_0∩purchase1） |
| M5 | **有效变化**：事件 `diff_num ≥ 5%·N`（vs 初始 env_ids）；norm vs raw env_ids 逐位差 ≥ 5%·N |
| M6 | **raw 诊断一致性**：移植后审计工具对 raw 产物只读重分析（artifact_only；无训练）复算：`env_0=1532`、`env_acc=0.9992525`、`balanced_acc=0.50`、`recall_env_0=0`、尺度比与候选对照表与既有记录一致（数值逐位，env_acc 以整数计数口径——ebefd19 修正口径） |
| M7 | **A 类完整性**：stage2 run A1/A2/A4/A5/A6 全 PASS（A3 SKIP，协议口径） |
| M8 | **默认关恒等与静态守卫**：(a) 移植测试全绿（含字节钉死 §8.1、AST 守卫 §8.2、seed2 身份钉死 §8.3）；(b) smoke 规模（`p20000-v5000-t10000`，1 epoch，默认 raw 路径，**temp root、非结果 run**）meta 的科学字段与既有记录产物 `s1-550e4d92-m1688723512-e1-d069eadf` 逐位一致（仅 commit/git/versions/wall/device 允许差异） |

- `MECHANISM_REPLICATED ⇔ M1∧M2∧M3∧M4∧M5∧M6∧M7∧M8`；否则逐项如实报告。

### 4.2 效用方向复现（U 组；within-seed2；容差沿用 seed1 修复，先于 run 写死）

| 编号 | 判据 | 容差依据 |
|---|---|---|
| U1 | Stage-1 CTR 无实质退化：新产物 `test_auc_ctr ≥ 0.5534062000766764 − 0.005`（CVR 记录不判定，同 seed1 口径） | 与 seed1 G6 相同容差 `0.005`（1M 前缀 47k 正例，SE≈0.0024） |
| U2 | Stage-2 **test BSI 方向为正**：`Δtest = test_bsi_new − 0.5974422649550507 > 0` | 用户口径（within-seed 方向复现）；同时报告 seed1 型底线 `Δtest ≥ −0.005`（由 U2 蕴含） |
| U3 | Stage-2 **val BSI 方向为正**：`Δval = val_bsi_new − 0.5809347091990792 > 0` | 同上；同时报告底线 `Δval ≥ −0.01`（由 U3 蕴含） |
| 报告项（**不判定**） | `Δtest ≥ +0.0055` 是否清除；B1/B2/B3 原样披露（seed1 §4.2.1 的"共享既有边界"规则仅用于对照说明，本实验判定不依赖 B1） | 原修复预注册 §4.2「不设提升门槛」；`+0.0055` 为 attenuation 实验线阈值，仅报告 |

- `UTILITY_DIRECTION_REPLICATED ⇔ U1∧U2∧U3`；否则如实报告各分量。

### 4.3 无效执行处置

仅当 run 因工具性原因（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；**因结果原因不构成无效，不重跑**。无调参、无 epoch 挑选、test 不参与选择。

---

## 5. 复现判定与 pooled 报告（先于 run 写死）

- **分类**：`REPLICATION_SUPPORTED ⇔ MECHANISM_REPLICATED ∧ UTILITY_DIRECTION_REPLICATED`；否则 `NOT_SUPPORTED`，并按 M/U 分量披露（含逐项失败）。
- **pooled 机制平衡**（两 seed 事件并列报告，不混合成单一统计量）：seed1 `959244/1040756`（47.9622%/52.0378%）+ seed2 实测；两 seed 均须 B4 双过方构成"机制跨 seed 稳定"。
- **within-seed Δ 表**（每 seed 与其**同 seed raw 基线**配对，跨 seed 不混）：seed1 `Δtest=+0.0082428399`、`Δval=+0.0100592719`（已记录，@ `529090e`/`2058de8`）；seed2 实测。
- 标签集固定：`REPLICATION_SUPPORTED` / `NOT_SUPPORTED`（不引入新标签；分量以 M/U 布尔向量披露）。

---

## 6. 运行计划（结果 run 恰好 1+1）与非结果核验

**结果 run（各恰 1 次，不重跑）**：
1. Stage-1：`python run_aliccp_benchmark.py stage1 --tag norm --clustering rank_normalized --model-seed 1688723740`（其余=协议默认：env seed `20261003`、3 epoch / patience 2 / 2M-500k-1M）
2. Stage-2：`python run_aliccp_benchmark.py stage2 --tag norm --stage1-id s1-5c060b9c-m1688723740-e3-820697f6 --model-seed 1688723740`（**未改动头**，协议默认 5 epoch / patience 2）

**非结果核验（不产生 canonical 产物、不被计为结果 run）**：
- §2 参考基线核验（已完成，35/35；只读）；
- §4 M6 审计重分析（只读、无训练、输出入 `artifacts/aliccp_bench/audit/…-reanalysis/`）；
- §4 M8(b) smoke 规模默认路径恒等核验（temp root）；
- 跑后只读分析：norm 产物标签交叉表/平衡准确率（M4）、norm-vs-raw env_ids 逐位差（M5）。
- **不做**：seed2 审计 `--reproduce` 复现 run（无 bit 级事件预测需求；§3 已声明）；不重跑任何已完成的有效 run。

**清洁树纪律（run 间提交）**：C1 本文件（预注册）→ 树干净；C2 移植 + 测试（TDD：守卫先红后绿）→ 树干净；跑 Stage-1（`git.dirty=false`，meta commit=C2；核验 M2/M3/M5 + 记录 §9.1）→ C3（仅文档）；跑 Stage-2（`git.dirty=false`，meta commit=C3，tag `norm`；核验 M1/M4/M7 + U 组）→ C4（§9.2/§9.3 + SUMMARY 追加行）。Stage-1 不写 SUMMARY（协议现状）。

---

## 7. 实现面（TDD；默认关逐位一致）

| 文件 | 改动 | 说明 |
|---|---|---|
| `aliccp_benchmark/normalized_clustering.py`（新） | 逐字节移植 @ `2058de8` | 机制本体；无超参、不消耗 RNG |
| `aliccp_benchmark/bench.py` / `metrics.py` / `run_aliccp_benchmark.py` | 逐字节移植 @ `2058de8`（sha 见 §1.1） | 默认路径逐位不变；arm 注入 + 诊断 + CLI |
| `aliccp_benchmark/audit_cluster_losses.py` | 逐字节移植 @ `2058de8` | 只读审计（M6 用；测试引用 `ref_rank01`） |
| `aliccp_benchmark/tests/test_normalized_clustering.py`（新） | 移植 @ `2058de8` + 恰 2 处分支适配（§1.1） | 原机制不变量测试全部保留 |
| `aliccp_benchmark/tests/test_normalized_clustering_replication.py`（新） | 本分支守卫：字节钉死、AST 守卫、seed2 身份/文档 token 钉死 | §8 |
| `.gitignore` | 增 `artifacts/aliccp_bench/audit/`（与 seed1 修复分支同型） | 审计输出不入库（§11） |
| `artifacts/aliccp_bench/SUMMARY.md` | Stage-2 run 后追加 1 行 | append-only |
| `docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-seed-replication-design.md` | 本文件 | 预注册 + 结果 |

**明确不改**：`multitaskrec/*`（零 diff）、`config.py`、`AliCCP_*.py`、`baseline/*`、`aliccp_benchmark/protocol.py`（零 diff）、协议文档与 seed1 设计文档（只读引用）。禁止 `git add -A`；提交只允许显式路径。

---

## 8. 测试与不变量（TDD；CPU 极小夹具，不构成性能证据）

### 8.1 字节钉死（本分支新增守卫测试）

对 §1.1 的 5 个移植文件逐一断言 `sha256(工作树文件) == 2058de8 钉死值`（读字节计算；不依赖被钉 commit 存在）。**任一机制文件与 seed1 修复不等价即红。**

### 8.2 AST 守卫（移植测试文件 == seed1 版 + 恰 2 处适配）

对比 `test_normalized_clustering.py`（工作树）与 `2058de8:aliccp_benchmark/tests/test_normalized_clustering.py`：逐类名、逐方法名集合相等；对每个非豁免方法断言 `ast.dump(..., include_attributes=False)` 相等；模块级语句逐一相等（仅豁免 `DOC_PATH` 赋值）；**差异集合必须恰等于** `{("TestStaticGuards","test_tracked_changes_subset_of_whitelist"), 模块级 DOC_PATH}`，并断言适配后的白名单包含本分支两个新文件。被钉 commit 不可得时该测试**响亮失败**（审计分支的系谱依赖，如实处理，不静默跳过）。

### 8.3 seed2 身份与文档钉死

`_stage1_cfg`（协议默认，`model_seed=1688723740`）+ `{"clustering":"rank_normalized"}` 的 `config_hash` == `820697f6…` ⇒ `stage1_id` == `s1-5c060b9c-m1688723740-e3-820697f6`；本文档必须包含 token：预测 id/哈希、raw 基线四值（`0.5534062000766764`/`0.5420522697385088`/`0.5974422649550507`/`0.5809347091990792`）、事件 raw 值（`1532`/`1998468`）、容差（`0.005`/`0.01`）、报告项（`0.0055`）、分类标签（`REPLICATION_SUPPORTED`/`NOT_SUPPORTED`）、seed（`1688723740`）、seed1 引用值（§5 各值）。

### 8.4 继承不变量（移植文件内，全部保留）

I1–I9（rank01 语义、实现==审计参考逐位、manager 仅覆写 `cluster_2`、默认关恒等与 cfg 哈希、CLI 接线、静态守卫、seed1 文档 token）——其中 seed1 文档 token 表对本分支文档继续成立（§1.1 说明）。

---

## 9. 结果（实测；运行后追加，不得回填预期值）

### 9.0 非结果核验（运行后填写）

### 9.1 Stage-1（运行后填写）

### 9.2 Stage-2（运行后填写）

### 9.3 pooled 与分类（运行后填写）

---

## 10. 局限（预声明）

1. 复现对象为**单一 seed 对**（seed1 已跑，seed2 本次）；每配置仍 1 run，无 run-to-run 噪声估计（A3 SKIP）。
2. seed2 事件无 bit 级预测（无 reproduce 捕获，§3 声明）；机制判定依赖 §4 M 组判据而非 bit 等式。
3. within-seed Δ 为**同 seed、不同 backbone 的单 run 配对**；方向为正不等于提升主张（无提升门槛，§4.2 报告项）。
4. CVR 指标统计功效极低（test 正例 346），只作记录。
5. 仅 AliCCP 前缀 `p2M-v500k-t1M`；不外推其他预算/数据集。
6. 环境逐位可复现性依赖（M2b）：由 seed1 审计 10/10 与本 worktree 既有 bit-equal 证据支撑；若 M2b 失败按 §4 如实报告（解释规则沿用 seed1 §4.3）。

---

## 11. 偏离披露（预声明 + 运行后补）

- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（不从修复分支 `2058de8` 拉出；机制经 §8 字节/AST 守卫钉死等价）。协议未合并入 master，与同批 exp 分支先例一致。
- **分支本地 SUMMARY 系谱**：`8133d32` 的 `SUMMARY.md` 无 seed2 行（其记录在 `f589e61`）；本分支只追加本实验自己的行，不复制其他分支行。
- **审计产物忽略规则**：`.gitignore` 增 `artifacts/aliccp_bench/audit/`（与修复分支同型；审计输出不入库）。
- **非结果核验**：默认关 smoke 恒等核验与 raw 重分析均不入 canonical 产物、不计为结果 run（与 seed1 §9.3 同型）。
- **不做 seed2 审计 reproduce**：只读重分析（artifact_only）替代（§3/§6）。
