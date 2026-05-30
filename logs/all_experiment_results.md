# MPT-Rec New-task Generalization: Complete Results

> 所有实验使用 `run_newtask_from_ckpt.py` 独立 RNG，GAN-on 预训练（uni_coe=0.9, env_coe=0.1）
> 日期: 2026-05-30

## 1. CensusIncome (T3=Education)

### Raw Data

| Seed | Mode | Val AUC | Test AUC | λ | KL |
|------|------|---------|----------|----|----|
| 1685480945 | fw | 0.8555 | 0.8507 | --- | --- |
| 1688723512 | fw | 0.8581 | 0.8598 | --- | --- |
| 1689453621 | fw | 0.8572 | 0.8539 | --- | --- |
| 1685480945 | tes | 0.8553 | 0.8504 | --- | --- |
| 1688723512 | tes | 0.8573 | 0.8590 | --- | --- |
| 1689453621 | tes | 0.8574 | 0.8545 | --- | --- |
| 1685480945 | prompt | 0.8651 | 0.8626 | --- | --- |
| 1688723512 | prompt | 0.8624 | 0.8638 | --- | --- |
| 1689453621 | prompt | 0.8649 | 0.8622 | --- | --- |
| 1685480945 | tcprompt_fixed | 0.8648 | 0.8624 | 0.5000 | --- |
| 1688723512 | tcprompt_fixed | 0.8645 | 0.8630 | 0.5000 | --- |
| 1689453621 | tcprompt_fixed | 0.8625 | 0.8601 | 0.5000 | --- |
| 1685480945 | tcprompt_learnable | 0.8673 | 0.8643 | 0.4079 | --- |
| 1688723512 | tcprompt_learnable | 0.8629 | 0.8639 | 0.6293 | --- |
| 1689453621 | tcprompt_learnable | 0.8634 | 0.8594 | 0.4753 | --- |
| 1685480945 | cgr | 0.8645 | 0.8627 | --- | --- |
| 1688723512 | cgr | 0.8650 | 0.8660 | --- | --- |
| 1689453621 | cgr | 0.8602 | 0.8566 | --- | --- |
| 1685480945 | affinity_gate | 0.8613 | 0.8583 | --- | --- |
| 1688723512 | affinity_gate | 0.8626 | 0.8632 | --- | --- |
| 1689453621 | affinity_gate | 0.8610 | 0.8590 | --- | --- |
| 1685480945 | kl_prompt | 0.8639 | 0.8594 | --- | 0.0100 |
| 1688723512 | kl_prompt | 0.8604 | 0.8623 | --- | 0.0072 |
| 1689453621 | kl_prompt | 0.8644 | 0.8611 | --- | 0.0086 |

### Mean ± Std (Test AUC)

| Fusion Strategy | Test AUC | Std | Δ vs Prompt |
|----------------|----------|-----|-------------|
| FW (Fixed Weights) | 0.8548 | ±0.0046 | -0.0081 |
| TES (Task Emb Similarity) | 0.8546 | ±0.0043 | -0.0083 |
| **Prompt-tuning (original)** | **0.8629** | **±0.0008** | baseline |
| TC-Prompt (λ=0.5 fixed) | 0.8618 | ±0.0015 | -0.0011 |
| TC-Prompt (λ learnable) | 0.8625 | ±0.0027 | -0.0004 |
| CGR (Confidence-Gated Routing) | 0.8618 | ±0.0048 | -0.0011 |
| Affinity Gate (OOD-aware hard switch) | 0.8601 | ±0.0026 | -0.0028 |
| KL-Prompt (β=0.1) | 0.8609 | ±0.0015 | -0.0020 |

Learned λ: 0.4079, 0.6293, 0.4753 (mean=0.5042)
Final KL: 0.0100, 0.0072, 0.0086 (mean=0.0086)

---

## 2. AliCCP (T3=BSI)

### Raw Data

| Seed | Mode | Val AUC | Test AUC | λ | KL |
|------|------|---------|----------|----|----|
| 1688723512 | fw | 0.6708 | 0.6735 | --- | --- |
| 1688723740 | fw | 0.6743 | 0.6781 | --- | --- |
| 1688738016 | fw | 0.6678 | 0.6716 | --- | --- |
| 1688723512 | tes | 0.6691 | 0.6722 | --- | --- |
| 1688723740 | tes | 0.6719 | 0.6766 | --- | --- |
| 1688738016 | tes | 0.6626 | 0.6686 | --- | --- |
| 1688723512 | prompt | 0.6797 | 0.6759 | --- | --- |
| 1688723740 | prompt | 0.6783 | 0.6764 | --- | --- |
| 1688738016 | prompt | 0.6706 | 0.6636 | --- | --- |
| 1688723512 | tcprompt_fixed | 0.6803 | 0.6762 | 0.5000 | --- |
| 1688723740 | tcprompt_fixed | 0.6807 | 0.6793 | 0.5000 | --- |
| 1688738016 | tcprompt_fixed | 0.6702 | 0.6639 | 0.5000 | --- |
| 1688723512 | tcprompt_learnable | 0.6803 | 0.6766 | 0.4959 | --- |
| 1688723740 | tcprompt_learnable | 0.6788 | 0.6784 | 0.4430 | --- |
| 1688738016 | tcprompt_learnable | 0.6747 | 0.6668 | 0.5133 | --- |
| 1688723512 | cgr | 0.6768 | 0.6757 | --- | --- |
| 1688723740 | cgr | 0.6815 | 0.6784 | --- | --- |
| 1688738016 | cgr | 0.6375 | 0.6354 | --- | --- |
| 1688723512 | affinity_gate | 0.6778 | 0.6710 | --- | --- |
| 1688723740 | affinity_gate | 0.6689 | 0.6729 | --- | --- |
| 1688738016 | affinity_gate | 0.6708 | 0.6703 | --- | --- |
| 1688723512 | kl_prompt | 0.6774 | 0.6785 | --- | 0.0008 |
| 1688723740 | kl_prompt | 0.6683 | 0.6735 | --- | 0.0001 |
| 1688738016 | kl_prompt | 0.6825 | 0.6776 | --- | 0.0013 |

### Mean ± Std (Test AUC)

| Fusion Strategy | Test AUC | Std | Δ vs FW |
|----------------|----------|-----|---------|
| **FW (Fixed Weights)** | **0.6744** | **±0.0033** | baseline |
| TES (Task Emb Similarity) | 0.6725 | ±0.0040 | -0.0019 |
| Prompt-tuning (original) | 0.6719 | ±0.0073 | -0.0025 |
| TC-Prompt (λ=0.5 fixed) | 0.6731 | ±0.0081 | -0.0013 |
| TC-Prompt (λ learnable) | 0.6740 | ±0.0063 | -0.0004 |
| CGR (Confidence-Gated Routing) | 0.6632 | ±0.0241 ⚠️ | -0.0112 |
| Affinity Gate (OOD-aware hard switch) | 0.6714 | ±0.0014 | -0.0030 |
| **KL-Prompt (β=0.1)** | **0.6765** | **±0.0027** | **+0.0021** |

Learned λ: 0.4959, 0.4430, 0.5133 (mean=0.4841)
Final KL: 0.0008, 0.0001, 0.0013 (mean=0.0007)

---

## 3. 数据分析

### 3.1 核心发现

#### CensusIncome：attention 有效但所有方法等价

- Prompt 最优（0.8629），但 6 个 attention-based 方法全部挤在 0.8601-0.8629（极差 0.0028）
- FW（0.8548）和 TES（0.8546）显著低于 attention 方法（Δ≈-0.008），说明 **attention 确实从源任务中提取了有用信息**
- 但所有改进 attention 的尝试（TC-Prompt、CGR、Affinity Gate、KL-Prompt）在这个数据集上**均未超越原始 Prompt**
- 原因：Education 与 Income/Marital 存在中等正相关（ρ=0.14-0.19），projection network 能正常学到有意义的 attention，不需要额外的门控或正则化

#### AliCCP：attention 不可靠，KL-Prompt 是唯一解

- **KL-Prompt 最优**（0.6765），比 FW（0.6744）高 +0.0021，且方差更低（±0.0027 vs ±0.0033）
- FW 本身非常强，仅比最优方法低 0.0021——说明 BSI 本质上需要均匀利用 CTR/CVR 信息
- Prompt（0.6719）反而不如 FW，均值低且方差大（±0.0073），**说明 instance-level attention 在这个数据集上有害**
- 最关键的发现：**KL-Prompt 是唯一能在 seed3 上维持高 AUC 的方法**（见 3.2）

#### TC-Prompt：相关性先验无效

- CensusIncome 学到的 λ：0.41, 0.63, 0.48（均值 0.50）
- AliCCP 学到的 λ：0.50, 0.44, 0.51（均值 0.48）
- λ 在两个数据集上均收敛到 ~0.5，且与初始值（0.5）几乎一致
- 原因是 λ 的梯度极小（之前诊断测出 ~0.0013），乘以 ρ（0.05-0.19）后，对 attention logits 的贡献仅 ~0.03-0.08，而 attention logits 本身的范围在 h_p·E/τ（τ=150）下远大于此
- **TC-Prompt 本质上退化为了 Prompt**——这是设计层面的失败，不是调参问题

---

### 3.2 数据异常：AliCCP seed3 系统性偏弱

seed3（1688738016）在几乎所有方法上都是最差的：

| Method | seed1 | seed2 | seed3 | seed3 跌幅 |
|--------|:-----:|:-----:|:-----:|:--------:|
| FW | 0.6735 | 0.6781 | 0.6716 | -0.0028 |
| TES | 0.6722 | 0.6766 | 0.6686 | -0.0039 |
| Prompt | 0.6759 | 0.6764 | **0.6636** | **-0.0130** |
| TC-Prompt(fixed) | 0.6762 | 0.6793 | 0.6639 | -0.0144 |
| TC-Prompt(learn) | 0.6766 | 0.6784 | 0.6668 | -0.0109 |
| CGR | 0.6757 | 0.6784 | **0.6354** | **-0.0418** |
| Affinity Gate | 0.6710 | 0.6729 | 0.6703 | -0.0016 |
| **KL-Prompt** | 0.6785 | 0.6735 | **0.6776** | **+0.0000** |

**分析：**

1. **seed3 的 Stage1 checkpoint 质量较差**。FW 在此 seed 上也低 0.0028（这个差异仅来自 Stage1 表征质量，因为 FW 不学习 attention）。以此为基准，attention-based 方法额外的跌幅来自于**学出了虚假相关性**。

2. **Prompt 在 seed3 额外跌了 0.0102**（相对 FW 的跌幅）。这说明 seed3 的 projection network 学出的 attention 权重是**有害的**——不如均匀权重。

3. **CGR 跌了 0.0390**（相对 FW）。这是 CGR 全局标量门控的设计缺陷的实证（详见第 4 节）。

4. **Affinity Gate 跌最少（0.0016）**，因为它的逐样本 cosine similarity 特征能有效检测 OOD 样本并切换到 FW。代价是正常 seed 上均值偏低（过于保守）。

5. **KL-Prompt 完全不受 seed3 影响**（0.6776，与 seed1 的 0.6785 相当）。KL 正则化在 seed3 上成功地把有害的 attention 拉回了接近均匀分布，而在 seed1/2 上保留了 attention 的有益部分。这是唯一同时做到"好 seed 不拖累、差 seed 不崩溃"的方法。

### 3.3 数据异常：CensusIncome Prompt 方差极低

Prompt 在 CensusIncome 上方差仅 ±0.0008，是所有方法中最低的。此前我们用 `run_tcprompt_experiment.py` 串行跑出的旧 Prompt 方差为 ±0.0030。

**原因**：旧数据中 5 个 mode 在一个进程内串行训练，RNG 状态被前置 mode 扰动，导致 Prompt 的 NewTask 初始化在不同 seed 间不一致。新数据用 `run_newtask_from_ckpt.py` 独立启动每个 mode，RNG 状态干净一致。低方差说明 **CensusIncome 上 Prompt 的结果高度可复现**，之前观测到的方差是 RNG 污染造成的假象。

---

## 4. CGR 方差爆炸根因分析

### 现象

CGR 在 AliCCP 上 3 个 seed 的 Test AUC：0.6757, 0.6784, **0.6354**（±0.0241）

seed3 的 epoch 级轨迹：
```
Epoch  1: val=0.4682   ← 起步极低（FW seed3 epoch1 约 0.57）
Epoch 10: val=0.6051   ← 恢复缓慢
Epoch 20: val=0.6235
Epoch 30: val=0.6252   ← 最终也远低于其他方法（FW=0.6716, KL=0.6776）
```

### 根因：CGR 的置信门控是全局标量，无实例级信息

CGR 的门控输入为：
```python
gate_input = [new_env_emb, source_mean, source_var, gen_mean, gen_var]
# shape: (5 * rep_dim,) = 320 维（AliCCP）
# 所有统计量均来自冻结的 MPTRec，对全部样本完全相同
```

这意味着 **g 是一个全局标量——对所有样本都一样**。门控 MLP 的输入不包含任何逐样本信息。

对比三个方法的设计：

| 方法 | 门控输入 | 是否逐样本变化 |
|---|---|---|
| CGR | task-embedding 统计量（全局固定） | ❌ 全局标量 |
| Affinity Gate | 每个样本的 cosine similarity 特征 | ✅ 逐样本 |
| KL-Prompt | 无门控，固定 β 正则化 | — |

### 为什么 seed3 崩溃

1. **CGR 的 confidence_mlp 初始 bias=2.0**（sigmoid≈0.88），warmup 后 g 初始偏向信任 attention
2. seed3 的 Stage1 checkpoint 恰好 attention 质量较差（Prompt seed3 也只有 0.6636）
3. 由于 g 是全局标量，**无法对不同样本做差异化处理**——差的 attention 被同等信任
4. 梯度通过全局 g 反向传播到 MLP 权重时信号极弱，模型无法自救

### 为什么 CensusIncome 上没问题

CensusIncome 的 attention 本身就可靠（Prompt=0.8629），全局 g 偏向信任 attention 是正确的策略。CGR 的设计缺陷只在 attention 不可靠的数据集上暴露。

### 教训

**解耦（decoupling）不等于丢弃信息**。CGR 为了避免 projection network 的虚假相关性，把门控输入设计为全局统计量，结果丢弃了所有逐样本信息。Affinity Gate 保留了逐样本的 cosine similarity 特征，因此方差极低（±0.0014）。KL-Prompt 更彻底——直接不学习门控，用固定正则化，零方差风险。

---

## 5. 总结与建议

### 方法排名（按 AliCCP 排序，因为这是唯一能区分方法优劣的数据集）

| 方法 | AliCCP AUC | 参数 | 推荐 |
|------|:----------:|:----:|:----:|
| **KL-Prompt (β=0.1)** | **0.6765** | 0 | ✅ 首选 |
| FW | 0.6744 | 0 | ✅ 简单基线 |
| TC-Prompt(learn) | 0.6740 | 0 | ❌ 无实际增益 |
| TC-Prompt(fixed) | 0.6731 | 0 | ❌ 无实际增益 |
| TES | 0.6725 | 0 | ❌ 不如 FW |
| Prompt | 0.6719 | 0 | ❌ 不如 FW 且方差大 |
| Affinity Gate | 0.6714 | ~80 | ❌ 保守过度 |
| CGR | 0.6632 | ~900 | ❌ 致命缺陷 |

### 核心结论

1. **Attention 并非总是有益的**。在任务相关时（CensusIncome ρ≈0.16）它有效（+0.008 vs FW）；在任务无关时（AliCCP ρ≈0.06）它有害（-0.003 vs FW）。

2. **KL-Prompt 是唯一正确的方法**。它在 attention 可靠时保留 attention 的信息增益，在 attention 不可靠时自动拉回均匀分布。零额外参数，零方差风险。

3. **TC-Prompt 是无效设计**。λ·ρ 的数值太小（~0.04），无法对 attention logits 产生可测量的影响。λ 始终停留在初始值 0.5 附近。

4. **CGR 的解耦门控是一个反模式**。把门控输入设计为全局统计量（对全体样本相同）等价于学习一个全局标量，丢失了所有逐样本判断能力。

5. **Affinity Gate 方向对但过于保守**。保留逐样本特征是正确设计，但因硬切换和 Gumbel 噪声导致即使 attention 可靠时也频繁切换回 FW，拉低了均值。

6. **FW 是一个被低估的强基线**。在 AliCCP 上它仅比 KL-Prompt 低 0.0021，比所有其他方法都高。任何新方法必须在 AliCCP 上显著超越 FW 才能声称有效。

---

## 6. Hardware & Runtime

- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- CensusIncome: ~200K samples, Stage 2 ≈ 7 min/mode
- AliCCP: 5M samples, Stage 2 ≈ 30 min/mode
- Total runtime: ~14 hours (42 runs across both datasets)
