# CensusIncome 阶段 2 低秩残差适配（LoRA-style adapter）预注册

- **状态**：预注册已冻结（阈值与设计在看到任何结果之前写定，**本次登记未改动 §4 任何阈值**）；**结果已登记（§10，2026-09-30）**：机制门禁 **R1 FAIL** + 主指标 **S1 FAIL**，按 §6.2 止损，未扩展多 seed、未调参
- **日期**：2026-09-30
- **适用分支**：`exp/stage2-lora-adapter`（起点：`infra/fair-stage2-benchmark` @ `87afe03`）
- **父协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（本文件**只引用、不修改**父协议；冲突以父协议为准）
- **定位**：**工程消融（practical ablation）**，非新颖性主张。低秩 / 残差适配是已发表方法（§7），本文档只定义"在本协议内怎么测才算公平、可复现、可证伪"。

---

## 1. 目标与非目标

### 1.1 目标

| 编号 | 目标 |
|---|---|
| E1 | 检验一个**极小的可训练低秩残差**（rank=4）加在**冻结的 Stage-1 表征**上，能否改善新任务 Education 的 AUC。 |
| E2 | 在**完全不改动 backbone**（参数值、参数集、梯度）的前提下完成 E1：`backbone_sha256` 前后一致、backbone 梯度全为 `None`。 |
| E3 | 给出**机制证据**（不是只看 AUC）：初始逐元素等同基线、适配器梯度确实流动、适配器输出占比落在合格带内、新增参数量的精确核算。 |
| E4 | 结果**可证伪**：门禁在跑之前写定，不达标即止损并如实记录，不扩展多 seed。 |

### 1.2 非目标（明确不做）

| 编号 | 非目标 |
|---|---|
| N1 | **不主张方法新颖性**：不在任何写作中把"低秩适配 / 残差适配"当作本文贡献（§7）。 |
| N2 | **不改协议**：`census_benchmark/protocol.py`、`census_benchmark/metrics.py`、`run_census_benchmark.py`、`multitaskrec/model.py`、`config.py` 一律零改动（由测试静态守卫）。 |
| N3 | **不动 Stage-1**：不重训、不微调、不重新初始化 backbone；不原地改写 backbone 的任何 Linear 权重。 |
| N4 | **不做单 seed 以外的扩展**：首轮只跑 model seed `1685480945` 一次；达标才谈多 seed（父协议 §6.3）。 |
| N5 | **不调参**：rank / alpha / 注入点 / 共享方式 / 优化器 / lr 全部预注册，不做搜索。 |
| N6 | **不跑 AliCCP / ByteRec**：与父协议 §2.3 一致。 |

---

## 2. 注入点审计（先审计，再动手）

### 2.1 候选与裁决

| 候选 | 做法 | 裁决 |
|---|---|---|
| A. backbone 内 LoRA | 给 `MPTRec` 的 expert/tower Linear 打 `W + BA` 补丁或替换模块 | **否决**。会改变 backbone 的参数集或状态字典 → `backbone_sha256` 必变（父协议 A1 直接失败）；且"冻结 backbone"是父协议存在的理由（§4.3、§4.4）。 |
| B. 与 backbone 并行的旁路网络 | 另训一个吃 `dnn_input` 的网络与冻结分支相加 | **否决**。那是新增一个完整塔（参数量远超 rank=4 的题意），且会与"低秩残差"这一待检验机制混淆：分不清增益来自低秩约束还是来自额外全秩容量。 |
| C. NewTask 内部已有层上加 LoRA | 给 `projection_network` / `gate_network` / `tower_network` 打低秩残差 | **否决**。这三层在阶段 2 **本来就是全秩从零训练**的自由参数；在其上加 rank=4 残差是被已可训练空间严格支配的子空间，既不可能提供新容量，也让结论不可解释（无法区分增益来自适配器还是来自原本就在训练的头）。 |
| D. **冻结表征上的低秩残差**（选中） | 在 `gen_rep` 与各 `spec_rep` 进入既有融合**之前**：`h' = h + scale·B(A(h))` | **选中**。这是唯一"信息被冻结、但通路可微调"的位置；改动最小（一个 1024 参数的模块）、可测（有 `h` 与 `delta` 可逐流对比）、初始恒等、且不触碰 backbone 的任何参数。 |

### 2.2 审计到的事实（来自阅读 `multitaskrec/model.py` 与 `run_census_benchmark.py`）

1. `NewTask.forward(dnn_input, gen_rep, spec_reps, env_embs)` 的四个输入**全部来自冻结 backbone**；阶段 2 里 `get_infos()` 在 `torch.no_grad()` 下调用，头部的其余部分自由训练。
2. 融合顺序为：`spec_reps` 经投影注意力加权 → `new_spec_rep` → 乘新环境嵌入 → 与 `gen_rep` 一起过 `gate_network` 的 2 路 softmax → `fused_rep` → `tower_network`。
3. 因此"融合之前的注入点"= 对 `gen_rep` 与每个 `spec_rep` 做逐元素等宽的残差修正，全部发生在 `NewTask.forward` 的第一行之前。
4. `metrics.evaluate_newtask` 只通过 `newtask(dnn_input, gen_rep, spec_reps, env_embs)` 调用头部 → **只要子类保持同签名，AUC 就由与基线完全相同的代码算出**（无需改 `metrics.py`）。
5. `protocol.backbone_sha256` 遍历 `backbone.named_parameters()` → 适配器只要不是 backbone 的子模块，哈希就不受影响。
6. 阶段 2 训练循环中 `optimizer` 只接收 `NewTask` 的参数 → 适配器作为 `LoRANewTask` 的子模块，自动进入优化器，且不会让 backbone 参数进入。
7. `get_infos()` 的**源输出逐元素不变**：适配器在 `get_infos()` 返回之后、`NewTask` 内部才作用 → backbone 的输出张量与基线完全相同。这是 A1（backbone 参数逐元素不变）加确定性前向的直接推论，不需要额外验证；训练循环里唯一的变化发生在 head 一侧。

### 2.3 子决策：共享权重 vs 逐流独立

**预注册选择：gen / source 三路共享同一份适配器权重**（`rep_adapter` 单实例，在 `gen_rep` 与每个 `spec_rep` 上复用）。

| 维度 | 共享（选中） | 逐流独立（否决） |
|---|---|---|
| 参数量 | `2·r·d = 1024` | `3·2·r·d = 3072`（3 倍） |
| 机制可解释性 | 单一假设："冻结表征空间存在一个 rank-4 的有用修正子空间" | 三个机制混在一起，单 seed 5 epoch 下无法归因 |
| 空间语义 | 三路同维（128）、同源（同一 `expert_dnn_hidden_units[-1]`、同构 MLP）→ 共享子空间修正是有意义的 | 需要为"为什么每路需要不同的子空间"额外辩护 |
| 诊断能力 | 仍可逐流报告占比/余弦（见 §5），**不失解释力** | 同样的诊断信息 |

结论：共享以 **1/3 的参数**获得同等的诊断能力与更强的可解释性。

---

## 3. 预注册设计（实现即 `census_benchmark/adapter.py`）

### 3.1 机制（只此一种）

```
h' = h + scale · up(down(h)),      scale = alpha / rank = 4 / 4 = 1.0
```

- `down`：`Linear(d, r, bias=False)`，标准 Linear 初始化；`up`：`Linear(r, d, bias=False)`，**权重零初始化**。
- 作用于 `gen_rep` 与每个 `spec_rep`（同一实例），发生在 `NewTask` 融合之前。
- 实现为 `LowRankResidualAdapter` + `LoRANewTask(NewTask)`；`forward` 签名与 `NewTask` 完全一致。

### 3.2 为什么 B 零初始化

`up` 零初始化使 `delta ≡ 0`，因此**初始前向与基线逐元素相等**（`torch.equal`）。这是可证伪性的地基：任何初始差异都说明实现有误，而不是"训练出了效果"。这也是 LoRA 的既定做法（Hu et al. 2022），不是本文的发明。

### 3.3 rank / alpha 的选择

| 项 | 取值 | 理由 |
|---|---|---|
| `rank r` | 4 | 题设给定；相对 `d = 128` 是真低秩（≤ 3.1% 的秩占比）。 |
| `alpha` | 4.0 | LoRA 惯例 `scale = alpha / r`；取 `alpha = r` 使 `scale = 1.0`，即**不引入任何人为增益**（最不任意的一档，优于 `alpha=2r` 之类的约定）。 |
| `scale` | 1.0 | `alpha / r`。 |

**参数量核算（真实维度 `d = 128`，按构造算得）**：适配器 `2·r·d = 1024`；NewTask 头（不含适配器）`27063`；合计 `28087` → 适配器占**总参数量 3.65%**（占头 3.78%）。该组数字由单元测试钉住（`test_param_report_real_dimensions_match_preregistration`），运行期实测落在 `adapter_report.json: params`。

### 3.4 优化器与正则（不引入新超参）

| 项 | 取值 | 说明 |
|---|---|---|
| 优化器 | `Adam(newtask.parameters(), lr=1e-3)` | 与基线同一优化器、同一 lr；适配器参数自动包含在内。**不**给适配器单独的 lr / weight decay。 |
| 损失 | `BCELoss(pred, y) + newtask.get_l2_reg()` | 与基线逐字相同。 |
| 适配器的 L2 | **无** | `get_l2_reg()` 只覆盖 `tower_network`（继承自 `NewTask`，未改动）→ 适配器与基线中的 `projection/gate/env_embedding` 一样不受 L2 约束。不新增正则项 = 不夸大对比优势。 |

### 3.5 随机性（RNG）核算

阶段 2 的构造顺序（`run_stage2_adapter`）：

```
P.seed_model(model_seed)            # 与基线同一起点
backbone = build_mptrec(device)     # 与基线完全相同的抽样序列
backbone.load_state_dict(...)       # 不消耗 RNG
rng_before = torch.get_rng_state()  # 记下构造前的 RNG 现场
newtask = LoRANewTask(...)          # super().__init__() 与基线 NewTask 同一前缀 → 共享参数逐元素相等
                                    # 适配器构造在 isolated_cpu_rng() 内 → 退出即恢复现场
rng_after = torch.get_rng_state()   # 应等于"构造基线 NewTask 之后"的端点
actual_instance_identity_report(...)  # 隔离 RNG 的审计：把 RNG 拨回 rng_before 构造基线头并逐键比较（见下）
optimizer / 训练循环                 # 无 shuffle、无 dropout、NewTask 无随机层 → 不消耗全局 RNG
```

结论（由运行期审计与单元测试两重验证）：

- **共享参数（`projection/gate/tower/env_embedding/new_env_idx`）与基线 NewTask 逐元素相等**（同一 RNG 前缀 + 同一初始化顺序）。
- **全局 RNG 端点在构造之后与基线完全一致**：适配器初始化用 `isolated_cpu_rng()`（保存—恢复全局 CPU RNG），与 `protocol.make_env_ids` 用独立 Generator 是同一思路（父协议 spec 5.3.2）。**因此"额外 RNG 消耗"为 0**，不存在"只在构造之后才发生的多余随机性"。
- 该结论不靠断言而靠**运行期产物**，且审计对象是**真实构造出来、随后被训练的那个实例**：构造前记下 RNG 现场 → 构造 → 记下端点 → 在 `isolated_cpu_rng()` 内把 RNG 拨回构造前、以同一顺序构造一个基线 `NewTask` → 逐键比较共享参数与端点（`adapter.actual_instance_identity_report`）。审计本身不消耗全局 RNG。结果落在 `adapter_report.json` 的 `construction_identity`（A6 门禁）。

### 3.6 明确不做的事（防止"顺手改回去"）

- 不新增 `MPTRec.freeze_params()`；冻结仍走父协议 §4.4 的**外部参数遍历**三件套。
- 不给 `NewTask` 或 `MPTRec` 打补丁 / monkey-patch；适配器是**新增子类 + 新增子模块**。
- 不在 `torch.no_grad()` 内跑适配器（它必须可训练）；只有 `get_infos()` 留在 `no_grad` 内。

---

## 4. 预注册判据（看到结果之前写定）

### 4.1 主指标（S1，性能）

| 编号 | 判据 |
|---|---|
| **S1** | `AUC-Test-Education ≥ 0.8521`。基线 = `0.8500685307175756`（run `20260929-1735-s20260929-m1685480945-short-904f8d0`，stage1 `s1-096f8f16-m1685480945-e2-cb2094b3`）。阈值 = 基线 + 0.00203。 |

> **阈值纪律**：`0.8521` 为**用户指定**的接受阈值，在看到本实验任何结果之前写定。它不是显著性判据，见 §8.2。

### 4.2 机制门禁（A6 / G / R1，硬门禁）

| 编号 | 判据 | 依据 |
|---|---|---|
| **A6 构造恒等** | `shared_params_bit_identical == True` **且** `global_rng_endpoint_identical == True` **且** 新增状态字典键恰为 `["rep_adapter.down.weight", "rep_adapter.up.weight"]` | §3.5 |
| **G 梯度活性** | 第 1 次 backward（更新前）：`up` 梯度范数 > 0；`down` 梯度范数 **== 0**（B=0 的固有性质，`dL/dA ∝ B`）。且在**最后一次** backward（至少已更新 1 步）：`down` 梯度范数 > 0 | 证明适配器既非"接错线"也非"死参数" |
| **R1 输出占比** | 在 val 集、**注入点**上逐流统计 `ratio_i = ‖delta_i‖₂ / max(‖h_i‖₂, 1e-12)` 的样本均值；要求**每一路**都 ∈ `[0.005, 0.5]` | 下界 0.005 = 适配器不得静默（不能"等于没加"）；上界 0.5 = 不得超过冻结表征本身的一半量级，否则"残差"一词不成立 |

> **A6 的 G 中 `down` 第 1 步为 0 是预期而非缺陷**：这是零初始化 B 的数学后果。若第 1 步 `down` 梯度非零，反而说明 B 不是零初始化 → A6 的恒等性已被破坏。门禁把这一点写成显式判据，避免把固有性质误判为失败。

### 4.3 协议门禁（沿用父协议，判据不变）

A1 / A2 / A4 / A5 与 B1–B4 **由同一个 `metrics.judge` 用同一组参数算出**；A3 仍为 `on_demand`。实验门禁写在 `gate_report.json` 的 `"adapter"` 子对象里，**不并入**父协议的 `overall_pass`，以免改变协议门禁的原语义。

> **已知前置事实**：基线 run 的 **B3 已经是 FAIL**（`gate_mean = [0.9524, 0.8544]`，第一路越过 0.95 上界）。本次实验的 B3 无论通过与否都必须如实记录；若 FAIL，**不得**归因于适配器，也不得因此重跑挑结果。

---

## 5. 必须记录的量（落盘位置）

| 量 | 落盘字段 | 用途 |
|---|---|---|
| 可训练参数量 / 占比 | `adapter_report.json: params`（`adapter_params` / `newtask_head_params` / `newtask_total_params` / `adapter_ratio_of_newtask`） + `config.json` 同名键 | E3 参数效率核算 |
| 初始逐元素恒等 | `adapter_report.json: construction_identity`（对**真实实例**审计：`shared_params_bit_identical` / `global_rng_endpoint_identical` / `extra_keys` / `up_fro_at_construction` / 两侧参数量） | A6 |
| 适配器梯度 | `adapter_report.json: grad_probe`（`first` + `per_epoch` 的 `down/up` 梯度范数） | G |
| 适配器范数 | `adapter_report.json: adapter_norms_by_epoch`（`down_fro` / `up_fro` / `effective_fro = ‖scale·BA‖_F`） + 每 epoch 的 `metrics.json: stage2.epoch_records[].adapter_effective_fro` | 训练轨迹 |
| 逐流输出占比 / 余弦 | `adapter_report.json: val_stats_final` 与 `metrics.json: mechanism.adapter`（`ratio_mean` / `ratio_std` / `ratio_max` / `cos_mean` / `delta_norm_mean` / `rep_norm_mean` / `n_zero_rep`，流序 `("gen","spec_0","spec_1")`） | R1 + 解释 delta 是在"加新方向"（cos≈0）还是"缩放原方向"（cos≈±1） |
| 冻结证据 | `metrics.json` 的 `backbone_sha256_before/after`、`gate_report.json` 的 `A1` | E2 |
| 实验门禁汇总 | `adapter_report.json: gates` + `gate_report.json: adapter` | E4 |
| 与基线差值 | `adapter_report.json: auc.delta_vs_baseline` | S1 |

统计口径与 `metrics.GateStats` / `RepStats` 一致：fp64、CPU 累加、显存 O(1)，不保留中间张量。逐流统计在 val 上**额外跑一遍前向**（不改动 `metrics.evaluate_newtask`）。

---

## 6. 运行规程

### 6.1 首轮唯一一次运行（单 seed）

```powershell
# cwd = 仓库根；Stage-1 产物必须已存在（父协议内容寻址目录，只读）
.venv\Scripts\python.exe -m census_benchmark.adapter_runner `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
    --tag short --epochs 5
```

- 数据 / split seed / model seed / env seed / batch size / lr / epochs / patience 全部继承 Stage-1 产物与父协议常量（§6.1 of 父协议），**不新增任何超参**。
- `run_id` 沿用父协议格式（`...-short-<commit7>`），`SUMMARY.md` 列结构与基线逐列一致、只追加不重写；机制标识落在 `config.json: mechanism` 与 `adapter_report.json`。
- 单元测试（CPU、秒级、不读真实数据）：

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

### 6.2 止损规则（先定后跑）

1. **机制门禁（A6/G/R1）任一 FAIL** → 机制没有真正生效 → 如实记录并**停止**，不得进入多 seed，不得改 rank/alpha 重跑挑结果。
2. **S1 FAIL** → 未达接受阈值 → 如实记录（`SUMMARY.md` 一行 + `adapter_report.json`）并**停止**；按父协议 §6.3 第 4 条处置本分支。
3. **S1 PASS 且机制门禁全过** → 才允许谈扩展；扩展的 seed 列表由用户显式指定，本文件不预设。
4. 无论结果如何，**不得**只报告成功的 run，也不得删除已写入 `SUMMARY.md` 的行。

---

## 7. 与已知 LoRA / adapter 文献的关系（明确不主张新颖性）

本实验所用的机制**没有任何新意**，属于既有方法的直接套用：

| 本文用到的做法 | 已有工作 |
|---|---|
| `h + f(h)` 形式的**残差适配器**、近恒等初始化 | Rebuffi, Bilen, Vedaldi, *Learning multiple visual domains with residual adapters*, NeurIPS 2017 |
| 冻结主干 + 训练小瓶颈模块（**adapter**） | Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019 |
| 低秩残差 `W + BA`、`B` 零初始化、`alpha/r` 缩放（**LoRA**） | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 |
| 把上述几类方法统一为同一框架 | He et al., *Towards a Unified View of Parameter-Efficient Transfer Learning*, ICLR 2022 |
| 适配器在多任务间的组合 / 共享 | Pfeiffer et al., *AdapterFusion* (2021)；*MAD-X* (2020) |
| 冻结特征 + 旁路可训练通路（同类思想） | Sung et al., *Ladder Side-Tuning* (CVPR 2022) 等 |

**在推荐 / CTR 领域**，低秩适配也已被用于多任务与序列推荐模型的参数高效迁移（本仓库未做系统检索；**引用具体文献前必须核实**）。

**本次实验可报告的唯一内容**是：在 **本仓库这套严格冻结协议**下，`rank=4` 的残差适配对 Education 新任务是否有效、代价多少参数、机制是否真的在工作。**不得**把它写成方法贡献，也**不得**在任何写作中使用"我们提出"的措辞。

> 说明：本次会话无网络检索权限，上表为基于既有知识的判断，**未做系统文献调研**；正式写作前需补齐检索与逐条核实（含上面的"需核实"项）。

---

## 8. 已知局限与可证伪性

### 8.1 局限

1. **单 seed、5 epoch**：只用于筛查，不构成统计结论；run 间噪声水平未知（无重复跑）。
2. **B 类门禁在极小数据/随机标签下无意义**：本分支的单元测试用小夹具，只验结构与 A 类，不验性能。
3. **"占比带"是工程判据**：`[0.005, 0.5]` 的上下界是量级判断，不是理论推导；换任务/换维度需重新论证。
4. **baseline runner 与 adapter runner 是两份训练循环**：已通过复用 `build_census_loaders` / `build_mptrec` / `metrics.evaluate_newtask` / `metrics.judge` 把差异限制在"新任务头 + 探针"，但两份循环仍可能随分支演化漂移（由静态守卫与文档约束）。

### 8.2 什么结果会证伪本次实验的解释

| 观察 | 解释 |
|---|---|
| A6 FAIL（初始不等同 / RNG 端点变了） | 实现有误 → 任何 AUC 都不可用 |
| G FAIL（适配器拿不到梯度，或第 1 步 `down` 梯度非零） | 机制没接上 / 零初始化被破坏 → 不是"低秩适配无效"，而是实验无效 |
| R1 FAIL（占比 < 0.005） | 适配器几乎没动 → "无效"这一结论**不能**归因于低秩假设本身 |
| R1 FAIL（占比 > 0.5） | 残差已与表征同量级 → 不能再叫残差适配，结论需重新表述 |
| S1 FAIL 但机制门禁全过 | 这才是一次**有效的阴性结果**：rank=4 的冻结表征残差修正不足以提升 Education AUC |
| S1 PASS 但仅比基线高 < 0.002 | 落在单 seed 噪声的可能范围内 → 不得宣称有效，只能作为扩展多 seed 的动机 |

---

## 9. 与父协议的接口（改动清单）

| 文件 | 处置 |
|---|---|
| `multitaskrec/model.py`、`config.py` | **零改动**（测试：与分支起点逐字节一致） |
| `census_benchmark/protocol.py`、`census_benchmark/metrics.py`、`run_census_benchmark.py` | **零改动**；`adapter_runner` 只 import 复用 |
| `census_benchmark/adapter.py` | 新增（机制 + 诊断 + 实验门禁） |
| `census_benchmark/adapter_runner.py` | 新增（runner 变体） |
| `census_benchmark/tests/test_adapter.py` | 新增（T1–T19，CPU 秒级、不读真实数据集） |
| `artifacts/census_stage2/{stage1,runs,splits}` | 只读 / 只追加；`SUMMARY.md` 追加一行（格式不变） |

---

## 10. 结果（已登记，2026-09-30）

> 本节为**运行后追加**的登记内容。§1–§9（含 §4 全部阈值）在运行前写定，本次登记**逐字未改**。

### 10.1 运行标识与配置

| 项 | 值 |
|---|---|
| 运行日期 | 2026-09-30 |
| `run_id` | `20260930-1940-s20260929-m1685480945-short-87afe03` |
| commit | `87afe03` |
| Stage-1 产物 | `s1-096f8f16-m1685480945-e2-cb2094b3`（与基线为**同一份**） |
| 基线 run | `20260929-1735-s20260929-m1685480945-short-904f8d0`（同 stage1 / 同 split / 同 model seed / 同 epochs） |
| 预注册配置 | rank 4 / alpha 4.0 / scale 1.0 / 三路共享单实例 / lr 1e-3 / epochs 5 / patience 2 —— 全部为 §3 写定值，**未做任何搜索** |
| 参数量 | `adapter_params=1024`、`newtask_head_params=27063`、`newtask_total_params=28087`、`adapter_ratio_of_newtask=0.036458147897603876`（3.65%）——与 §3.3 核算逐位一致 |
| 运行次数 | **唯一一次**单 seed（`1685480945`）短跑；无多 seed、无全量、无 rank/alpha 重跑 |

### 10.2 主指标（S1，性能）

| 量 | 值 |
|---|---|
| `AUC-Test-Education` | `0.8491310170933439` |
| 基线 `AUC-Test-Education` | `0.8500685307175756` |
| `delta_vs_baseline` | `-0.0009375136242317` |
| 预注册接受阈值 | `≥ 0.8521`（= 基线 + 0.00203） |
| **S1** | **FAIL** |

`best_val_auc` = `0.8516224188844447`，落在 epoch 5（最后一个 epoch）—— patience=2 全程未触发，且最后一个 epoch 的 val 仍是全run 最高点（该 epoch 的增量已收窄至 +0.00014）。

**与基线的逐 epoch val 对比**（同一 stage1 / split / seed）：

| epoch | adapter `auc_val_education` | 基线 `auc_val_education` | 差 |
|---|---|---|---|
| 1 | 0.8435788955057038 | 0.8412748819561441 | **+0.00230** |
| 2 | 0.8483204476371791 | 0.8473774764767228 | +0.00094 |
| 3 | 0.8501888047702446 | 0.8496795536233096 | +0.00051 |
| 4 | 0.8514818081699838 | 0.8516411115669168 | −0.00016 |
| 5 | 0.8516224188844447 | 0.8527881905614896 | **−0.00117** |

基线与适配器**同为 best_epoch = 5**。适配器在早期 epoch 略高、在第 4 个 epoch 由正转负，最终 val 与 test **双双低于基线** → 无有效增益。

### 10.3 机制门禁（A6 / G / R1）

| 编号 | 判据 | 结果 | 实测 |
|---|---|---|---|
| **A6 构造恒等** | 全过 | **PASS** | `shared_params_bit_identical=true`、`global_rng_endpoint_identical=true`、`extra_keys=["rep_adapter.down.weight","rep_adapter.up.weight"]`、`up_fro_at_construction=0.0`、`27064 → 28088` 元素 |
| **G 梯度活性** | 全过 | **PASS** | 第 1 步：`up=0.06504307687282562`、`down=0.0`（B=0 的预期后果）；第 5 epoch 末步（step 3900）：`down=0.07786145061254501`、`up=0.02799312211573124` |
| **R1 输出占比** | 三路均 ∈ `[0.005, 0.5]` | **FAIL** | `ratio_mean = [0.06397237283220941, 1.0010842046095683, 0.5789968669901538]`（流序 `gen / spec_0 / spec_1`） |

**R1 逐流判定**：`gen` = 0.0640 **PASS**；`spec_0` = 1.0011 **FAIL**（> 0.5）；`spec_1` = 0.5790 **FAIL**（> 0.5）。`ratio_max_per_stream = [0.15220239516766929, 1.308447355211623, 1.0310794166610913]`，`n_zero_rep = 0`，`count = 49881`。

**R1 失败的机制读数**（`val_stats_final`，注入点，val 集）：

| 流 | `delta_norm_mean` | `rep_norm_mean` | 比值（均值之比） | `ratio_mean`（逐样本比值的均值） | `cos_mean` |
|---|---|---|---|---|---|
| `gen` | 8.249221268203227 | 128.3266373365398 | 0.0643 | 0.06397237283220941 | −0.019940323269255947 |
| `spec_0` | 8.388407611075273 | 7.935759201504399 | 1.0570 | 1.0010842046095683 | 0.2551161421558063 |
| `spec_1` | 6.672408513369574 | 12.228059677851054 | 0.5457 | 0.5789968669901538 | −0.04235113785939911 |

（`ratio_mean` 是"逐样本比值"的均值，与"均值之比"列口径不同，量级一致。）

三路的 `delta_norm_mean` 几乎相等（6.67–8.39），而三路的 `rep_norm_mean` 相差约 16 倍（128.33 vs 7.94）：三路共用**同一个算子** `scale·BA`（`scale = 1.0`），其修正量的**绝对**量级由算子本身决定、不随输入流的量级缩放，于是相对扰动量被各流自身的范数决定 —— `gen` 仅 6.4%，而 `spec_0` ≈ 100%、`spec_1` ≈ 58%。`cos_mean` 显示 `gen` 与 `spec_1` 上 delta 近似正交（≈ 0，即"加新方向"而非"缩放原方向"），`spec_0` 轻度同向（0.255）。

`adapter_norms_by_epoch` 的 `effective_fro` 单调增长：`2.252978801727295 → 3.0969295501708984 → 3.689136028289795 → 4.119241714477539 → 4.426547527313232` —— 算子范数在 5 个 epoch 内持续变大，扰动随之加剧。

### 10.4 协议门禁（沿用父协议，判据不变）

`A1 / A2 / A4 / A5 / B1 / B2 / B4` = **PASS**；`B3` = **FAIL**（`gate_mean = [0.952401451979339, 0.8543901456350915]`）。`adapter` 子对象**未并入**父协议 `overall_pass`（§4.3）。

**B3 为继承性失败，与适配器无关**：本 run 的 `gate_mean` 与基线 run 的 `gate_mean` **逐位相同**（`[0.952401451979339, 0.8543901456350915]`），`cos_gen_spec`（`[0.3259840184960287, 0.39654519772756186]`）与 `gen_std`（`3.4769905669669257`）同样逐位相同。这三个量由 `metrics.evaluate_newtask` 直接从**冻结 backbone** 读出（`backbone.gate_networks[i](dnn_input)` 与 `get_infos()` 的原始 `gen_rep/spec_reps`，见 `census_benchmark/metrics.py:85-86`，均在适配器作用之前），因此逐位相同既是 §2.2 第 7 条的预期，也是"适配器没有渗进 backbone 通路"的运行期旁证。

冻结证据：`backbone_sha256_before == backbone_sha256_after == a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85`；`A1.detail.grads_all_none = true`。

### 10.5 结论与解释

**结论：阴性。** 适配器**确实在工作**（A6 PASS：构造时刻与基线逐元素等同、RNG 端点未变、新增键恰为两个适配器权重；G PASS：`up` 首步即有梯度、`down` 在更新一步后持续有梯度），但**没有带来任何效果增益**，且**机制门禁 R1 在两条 source 流上 FAIL**。

**解释（按 §8.2 的证伪表逐条对照）**：

1. **不能归因于"低秩假设本身无效"。** R1 FAIL 的方向是"占比 > 0.5"（§8.2 第 4 行），而不是"占比 < 0.005"（第 3 行）。适配器没有静默、也没有接错线 —— 它是**过度扰动**了。
2. **失效环节是"共享 + 绝对尺度"这一具体形式**：三路共享同一份权重、`scale = 1.0`、无任何逐流归一化，使修正量的绝对量级对三路一视同仁（`delta_norm_mean` ≈ 6.67–8.39），而冻结表征的量级天然悬殊（`spec_0` 7.94、`spec_1` 12.23，仅 `gen` 的 1/16–1/10）。结果是 `spec_0` 上 delta 与表征本身同量级（约 1.0×）、`spec_1` 上约 0.58× —— 按 §8.2 第 4 行，"残差"一词对这两路不再成立，结论必须据此重新表述。
3. **效果侧无增益**：test `0.8491310170933439` 低于基线 `0.8500685307175756`（`-0.0009375136242317`），也低于阈值 `0.8521`；与基线的 val 差由 epoch 1 的 +0.0023 一路收窄、在第 4 epoch 转负、终值 −0.0012，与"早期轻微有益、随 `effective_fro` 增长转为过扰动"的读数一致。
4. 因此 S1 的 FAIL **不是** §8.2 第 5 行那种"机制门禁全过下的有效阴性结果"；本次的可靠结论是：**在共享 + 绝对尺度 + `scale=1.0` 的预注册形式下，rank=4 的低秩残差对 Education 新任务无增益，且对小范数的 source 专属表征构成过度扰动。**

**止损执行情况（§6.2）**：机制门禁 FAIL（规则 1）与 S1 FAIL（规则 2）同时触发 → **停止**。未进入多 seed、未做全量跑、**未改 rank / alpha / scale / 注入点后重跑挑结果**、未删除或重写 `SUMMARY.md` 任何一行、未修改 §4 任何阈值。此外记录两条**未被用作辩护理由**的观察：本 run `best_epoch = 5`（patience 未触发）；基线 run 同样 `best_epoch = 5`。二者都不改变上述结论，也不允许据此扩展。

**分支处置**：`exp/stage2-lora-adapter` **保留**（不删除），**不合入 `master`**（父协议 §3.3 第 1 条只对"验证通过"的分支规定合并；本分支按 §6.3 第 4 条止损）。推送远端按父协议 §3.3 第 5 条需用户显式批准。

**测试**：`census_benchmark/tests/test_adapter.py` 的 19 项聚焦测试（T1–T19，CPU、秒级、不读真实数据集）全部通过；其中 `test_baseline_reference_matches_recorded_summary` 把 §4.1 的基线值钉在 `SUMMARY.md` 的记录上，`test_param_report_real_dimensions_match_preregistration` 把 §3.3 的参数量核算钉死。

**未做（本轮明确不做，且本次结果不构成做它们的理由）**：多 seed / 全量跑、rank–scale 调参、逐流独立适配器（§2.3 的备选）、AliCCP / ByteRec（N6）、任何把本机制写作贡献的表述（N1、§7）。

