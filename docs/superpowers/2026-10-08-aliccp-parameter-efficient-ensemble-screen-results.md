# AliCCP 参数高效集成：canonical 种子首轮三臂筛查结果

- 预注册：`docs/superpowers/specs/2026-10-08-aliccp-parameter-efficient-ensemble-design.md`（先于代码与运行提交）
- 分支：`exp/aliccp-parameter-efficient-ensemble`，代码 commit `c12d0b7`（`git.dirty=false`，工作树干净）
- 状态：三臂正式短实验已完成并落盘 raw；本结果与 SUMMARY 行已在本分支准备，待提交、推送。
- 边界：全程未改代码、未重跑、未推送；结论仅依据 `artifacts/aliccp_bench/runs/` 下 raw（config.json / metrics.json / gate_report.json / predictions.pt / newtask.pt）与 `artifacts/aliccp_bench/logs/` 日志

## 1. 配对真实性核验（全部通过）

三臂共用同一 Stage-1 与同一数据前缀，身份字段在 config/metrics/predictions 三份产物间逐项一致：

| 项目 | 值（三臂相同） | 来源 |
|---|---|---|
| stage1_id | `s1-5c060b9c-m1688723512-e3-3a30e2c0` | config.json / metrics.json / predictions |
| backbone SHA256 | `5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee`（load/before/after 三处相等） | metrics.json |
| 前缀指纹 | `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8` | config.json 等三份 |
| model_seed | 1688723512 | config.json 等三份 |
| 预算 | train 2,000,000 / val 500,000 / test 1,000,000 | config.json 等三份 |
| epochs / patience | 5 / 2（三臂均跑满 5 epoch，best_epoch=5，未触发早停） | metrics.json |
| commit / dirty | c12d0b7 / false | config.json 等三份 |

- val/test 标签在 `predictions.pt` 内三臂逐项相同，已由独立检查验证。
- A 类门禁 A1/A2/A4/A5/A6 全 PASS、A3 SKIP（按 spec 10 首轮按需复跑，不阻塞）；三臂 `hard_pass=true`。
- raw 审计：B 首轮短实验与历史 `b2e17f9` 行（`20261003-0130-...-short-b2e17f9`）同 stage1、同 seed，val/test AUC 逐位一致（0.5781533414372665 / 0.5988392178311113），佐证 c12d0b7 的 B 臂为同一配对对照的干净重录。

## 2. 主结果（test BSI AUC 为唯一判定指标）

| 臂 | 头结构 | 可训练参数 | best val BSI AUC | test BSI AUC | wall (s) | 来源 |
|---|---|---|---|---|---|---|
| B | 原始 NewTask | 8,129（残差 0） | 0.5781533414372665 | 0.5988392178311113 | 140.0 | metrics.json |
| C | NewTask + 1×rank-16 残差头 | 9,169（残差 1,040） | 0.5849711199970888 | 0.6080080503402661 | 142.7 | metrics.json |
| E | NewTask + 2×rank-8 残差头（概率均值） | 9,169（残差 1,040） | 0.605159910758228 | 0.6240604138409352 | 147.4 | metrics.json |

配对差值（test / val，均取自 raw 重算口径）：

| 配对 | ΔAUC-Test-BSI | ΔAUC-Val-BSI |
|---|---|---|
| E − B | **+0.025221196009823976** | +0.0270065693209615 |
| E − C | **+0.016052363500669187** | +0.0201887907611392 |
| C − B | +0.00916883250915479 | +0.0068177785598223 |

val 方向一致性：三臂逐 epoch val AUC 单调上升，E 在每一个 epoch 均同时高于 B 与 C（E−B、E−C 的 val 差值为正，与 test 同向）。

- 按预注册阈值（test 差值 ≥ +0.001 为「正提升」）：**E−B 与 E−C 均达标** → 分类 `POSITIVE_IMPROVEMENT`。
- 预算门禁：C/E 可训练参数相差 0% ≤ 5%；E（9,169）< 2×NewTask（16,258），符合 spec 公平配对（C/E 的 budget_report 一致；B 不适用）。E 相对 B 墙钟 +5.3%（147.4s vs 140.0s），不宣称等算力。
- 备注：本方向 +0.001 阈值是预注册筛查门槛，与 2026-10-03 公平基准设计的 provisional 主判据（+0.005 且 val 同向，§10.1）是两套阈值体系，本文档不做跨口径混用。

## 3. 机制门禁（E 两残差头）

E 的两个 rank-8 残差头在 test（n=1,000,000）上的分歧统计（`predictions.pt` 成员概率，metrics.json `mechanism` 同值）：

| 统计 | 值 |
|---|---|
| 平均绝对差 | 3.308293851017952e-4 |
| 最大绝对差 | 3.0145645141601562e-3 |
| q50 / q90 / q99 | 2.696514129638672e-4 / 6.827116012573242e-4 / 1.2452012300491339e-3 |
| 相关系数 | 0.993525492812772 |
| 完全相同 | false |

- 平均绝对差 3.31e-4 高于塌缩阈值 1e-4，且非逐位相同 → **机制未塌缩**（按 spec 不触发塌缩判负）。
- 含义边界：未塌缩只说明「两头并非同一函数」，不构成「提升来自集成」的正向归因证据（见 §4）。

## 4. 反证与归因限制（重要）

E 臂输出为两成员概率均值；独立 sklearn 从 raw `predictions.pt` 重算得到：

- 两成员单独 test AUC 为 **0.6252926268772121 / 0.622372262411566**；相应的 val AUC 为 0.6075698302421041 / 0.6023643305123303。
- 按 val 选出的较强成员是第一个，其 test AUC 0.6252926268772121；**均值集成 test AUC 0.6240604138409352 更低**，相差 −0.0012322130362769。
- 这只是单种子观察，不构成总体显著性判断；但方向上不支持把收益归因于「推理时平均」。

因此现阶段**不能把 E 的提升归因于平均集成**；更保守的解释是：提升主要来自「两个残差头的联合训练 / 残差参数化本身」（共同训练两个低秩残差作用于共享主头，其收益超过了均值集成的损失）。该反证与预注册字面规则（E−B、E−C ≥ +0.001、val 同向、机制未塌缩、真实性门禁通过 → 可扩 seed）并不冲突，但**必须在任何后续扩展或结论表述中显式保留**。

## 5. 门禁全表与继承失败项（不得表述为「全部门禁通过」）

| 门禁 | B | C | E | 说明 |
|---|---|---|---|---|
| A1 冻结完整性 | PASS | PASS | PASS | sha_before==after 且 backbone 无梯度 |
| A2 前缀/指纹 | PASS | PASS | PASS | |
| A3 按需复跑 | SKIP | SKIP | SKIP | spec 10：不阻塞首轮 |
| A4 标签计数 | PASS | PASS | PASS | |
| A5 env_ids | PASS | PASS | PASS | |
| A6 Stage-1 匹配 | PASS | PASS | PASS | |
| B1 活性下限 | **FAIL** | **FAIL** | **FAIL** | CTR 0.5493 < 0.55（CVR 0.5132、BSI 达标）。CTR/CVR 取自锁定的 Stage-1 meta（`best_val_auc_ctr/cvr`），三臂同值；非本三臂引入 |
| B2 val-test 差距 | PASS | PASS | PASS | 0.0207 / 0.0230 / 0.0189 ≤ 0.05 |
| B3 gate 活性 | PASS | PASS | PASS | gate_mean 各维 ∈ [0.05, 0.95] |
| B4 聚类占比 | **FAIL** | **FAIL** | **FAIL** | 继承自 Stage-1 meta 的 `cluster_events`（events=[(566, 1999434)]，env_1 占比 99.97% ≥ 5%，env_0 0.03% < 5%）；三臂同值 |

- `hard_pass=true` 的口径：`enforce_b=false` 下只要求 A 类全 PASS/SKIP（bench 协议对 B 类「只记录不判定」），B1/B4 的 FAIL 与 hard_pass 不矛盾。
- **B1/B4 为继承性失败**（Stage-1 锁定属性：CTR 基线弱于 0.55 下限、环境聚类高度不均衡），三臂完全同值，不构成臂间差异，也不属于本方向的真实性门禁；但汇报与后续文档中必须保留 FAIL 原样，不得描述为「全部门禁通过」。

## 6. 判定与后续

按 2026-10-08 预注册规则逐条对照：

| 条件 | 结果 |
|---|---|
| E−B ≥ +0.001 | ✅ +0.025221 |
| E−C ≥ +0.001 | ✅ +0.016052 |
| val 方向一致（E−B、E−C 的 val 差为正） | ✅ |
| 机制未塌缩（平均绝对差 3.31e-4 > 1e-4） | ✅ |
| 配对真实性门禁通过（身份三方互证、标签一致、A 类全过、hard_pass=true） | ✅ |

→ `POSITIVE_IMPROVEMENT`，**满足预注册的扩 seed 条件**。预注册同时规定：扩 seed 必须**另建独立分支**执行，不在本分支进行。

**扩 seed 分支的设计输入（来自 §4 反证，建议纳入预注册再跑）**：现行 E 的定义混淆了「双残差头联合训练」与「推理均值集成」两个因素。若扩 seed 只复制三臂，无法分离该归因。建议扩 seed 前补充一个**单 rank-8 残差头臂**（与 E 同预算减半的可训练参数、同训练轮数）作为联合训练效应的对照，或在预注册中明确接受「归因不分离、结论限定为组合方案有效」。

## 7. 独立复核状态

- 已核对（本会话，纯读取）：三臂 config/metrics/gate_report 全部数值、A/B 门禁明细、日志与 raw 一致性、跨运行（b2e17f9）一致性、B1/B4 继承来源（Stage-1 meta）。
- 本线程用独立 sklearn 对三臂 raw `predictions.pt` 重算：test AUC 分别为 0.5988392178311113、0.6080080503402661、0.6240604138409352；val 第 5 epoch AUC 分别为 0.5781533414372665、0.5849711199970888、0.605159910758228，均与 metrics.json 一致。test/val 标签逐项相同。
- E 的 `newtask.pt` 中两个 up 权重范数分别为 0.0953974、0.0899001，down 权重范数分别为 1.9337124、1.8843540；两组权重均非零且互不相等。训练时参数不共享另由单元测试验证。
- 复核脚本置于 gitignore 覆盖的 `artifacts/aliccp_bench/logs/`，可再次执行：
  `artifacts/aliccp_bench/logs/_recompute_ens_screen.py`。运行方式：
  `D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe artifacts\aliccp_bench\logs\_recompute_ens_screen.py artifacts\aliccp_bench\runs`
  本线程已直接从 raw 完成上述独立重算；该脚本供复现，不把 CC 的口头结论当证据。

## 8. 产物清单（raw 路径）

```
artifacts/aliccp_bench/runs/20261008-0210-p2M-v500k-t1M-m1688723512-short-c12d0b7-ens-B/{config,metrics,gate_report}.json predictions.pt newtask.pt
artifacts/aliccp_bench/runs/20261008-0212-p2M-v500k-t1M-m1688723512-short-c12d0b7-ens-C/{同上}
artifacts/aliccp_bench/runs/20261008-0215-p2M-v500k-t1M-m1688723512-short-c12d0b7-ens-E/{同上}
artifacts/aliccp_bench/logs/20261008-021{015,251,528}-stage2-short-ens-{B,C,E}.log
（smoke: runs/20261008-0209-p20000-v5000-t10000-m1688723512-smoke-c12d0b7-ens-E/ 仅可运行性，不用于结论）
```

所有 §2/§3/§5 数值均有上述路径对应。`stage1/`、`splits/` 在本工作树通过目录联接只读引用原公平基线产物；已独立验证正式指纹自哈希、Stage-1 meta 与 backbone SHA，三臂内置 raw 核验的锚定检查均为 true。原基线文件没有被改写。
