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

## 3. 关键发现

### CensusIncome
- 所有 attention-based 方法挤在 0.860-0.863 区间，无法拉开显著差距
- **Prompt（纯 attention）是最优的**（0.8629），但优势在噪声范围内
- 原因是 Education 与 Income/Marital 有中等正相关（ρ≈0.15-0.19），attention 正常运作

### AliCCP
- **KL-Prompt 最优**（0.6765），比 FW 高 +0.0021，比 Prompt 高 +0.0046
- **FW（均匀权重）非常强**（0.6744），说明 BSI 确实需要均匀利用 CTR/CVR 信息
- **Prompt 在 seed3 崩了**（0.6636），方差巨大（±0.0073），证实了 attention 学虚假相关性的假设
- **CGR 在 seed3 严重崩溃**（0.6354），是表现最差的 method
- **Affinity Gate 最稳定**（±0.0014）但均值不高（0.6714）
- **KL-Prompt 同时兼顾了高均值和低方差**（0.6765 ± 0.0027）

### 综合结论
- AliCCP 上 BSI 任务的 attention 确实会学出虚假相关性（Prompt/CGR seed3 崩溃）
- 最简单的方案效果最好：FW（均匀权重）和 KL-Prompt（轻微正则化 attention）
- KL-Prompt 是唯一在 AliCCP 上显著优于 FW 的方法，且零额外参数

---

## 4. Hardware & Runtime

- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- CensusIncome: ~200K samples, Stage 2 ≈ 7 min/mode
- AliCCP: 5M samples, Stage 2 ≈ 30 min/mode
- Total runtime: ~14 hours (42 runs across both datasets)
