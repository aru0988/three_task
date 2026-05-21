# TC-Prompt Experiment: Complete Results

## 1. CensusIncome (T3=Education)

### Raw Data

| Seed | Mode | Val AUC | Test AUC | λ |
|------|------|---------|----------|----|
| 1685480945 | fw | 0.8569 | 0.8520 | --- |
| 1685480945 | tes | 0.8571 | 0.8525 | --- |
| 1685480945 | prompt | 0.8610 | 0.8579 | --- |
| 1685480945 | tcprompt_fixed | 0.8621 | 0.8577 | 0.5000 |
| 1685480945 | tcprompt_learnable | 0.8618 | 0.8585 | 0.4037 |
| 1688723512 | fw | 0.8562 | 0.8579 | --- |
| 1688723512 | tes | 0.8567 | 0.8584 | --- |
| 1688723512 | prompt | 0.8629 | 0.8652 | --- |
| 1688723512 | tcprompt_fixed | 0.8638 | 0.8653 | 0.5000 |
| 1688723512 | tcprompt_learnable | 0.8640 | 0.8647 | 0.6827 |
| 1689453621 | fw | 0.8565 | 0.8536 | --- |
| 1689453621 | tes | 0.8561 | 0.8530 | --- |
| 1689453621 | prompt | 0.8637 | 0.8606 | --- |
| 1689453621 | tcprompt_fixed | 0.8638 | 0.8605 | 0.5000 |
| 1689453621 | tcprompt_learnable | 0.8629 | 0.8600 | 0.5550 |

### Mean ± Std (Test AUC)

| Fusion Strategy | Test AUC | Δ vs Prompt |
|----------------|----------|-------------|
| FW (Fixed Weights) | 0.8545 ± 0.0025 | -0.0067 |
| TES (Task Emb Similarity) | 0.8546 ± 0.0026 | -0.0066 |
| **Prompt-tuning (original)** | **0.8612 ± 0.0030** | baseline |
| TC-Prompt (λ=0.5 fixed) | 0.8612 ± 0.0032 | 0.0000 |
| TC-Prompt (λ learnable) | 0.8611 ± 0.0026 | -0.0001 |

Learned λ: 0.4037, 0.6827, 0.5550 (mean=0.5471 ± 0.1140)

### Diagnostic Analysis (seed=1685480945)

| Metric | Prompt | TC-Prompt (learnable) |
|--------|--------|----------------------|
| Test AUC | 0.8626 | 0.8643 |
| Mean W_T1 | 0.6014 | 0.5716 |
| Mean W_T2 | 0.3986 | 0.4284 |
| Var W_T1 | 0.0605 | 0.0736 |
| Var W_T2 | 0.0605 | 0.0736 |
| Task emb cos_sim(T1,T2) | -0.0500 | -0.0500 |
| λ final | --- | 0.4073 |
| Mean |∂L/∂λ| | --- | ~0.0013 |

Label Pearson correlations: ρ(T1,T2)=0.178, ρ(T1,T3)=0.186, ρ(T2,T3)=0.142

---

## 2. AliCCP (T3=BSI)

### Raw Data (seeds 1-2 complete, seed 3 partial)

| Seed | Mode | Val AUC | Test AUC | λ |
|------|------|---------|----------|----|
| 1688723512 | fw | 0.6658 | 0.6686 | --- |
| 1688723512 | tes | 0.6445 | 0.6463 | --- |
| 1688723512 | prompt | 0.6246 | 0.6210 | --- |
| 1688723512 | tcprompt_fixed | 0.6269 | 0.6239 | 0.5000 |
| 1688723512 | tcprompt_learnable | 0.6257 | 0.6222 | 0.4503 |
| 1688723740 | fw | 0.6767 | 0.6789 | --- |
| 1688723740 | tes | 0.6398 | 0.6376 | --- |
| 1688723740 | prompt | 0.6148 | 0.6181 | --- |
| 1688723740 | tcprompt_fixed | 0.6149 | 0.6189 | 0.5000 |
| 1688723740 | tcprompt_learnable | 0.6158 | 0.6188 | 0.4419 |
| 1688738016 | fw | 0.6672 | 0.6746 | --- |
| 1688738016 | tes | PENDING | --- | --- |
| 1688738016 | prompt | PENDING | --- | --- |
| 1688738016 | tcprompt_fixed | PENDING | --- | --- |
| 1688738016 | tcprompt_learnable | PENDING | --- | --- |

### Mean ± Std (Test AUC, seeds 1-2 only)

| Fusion Strategy | Test AUC | Δ vs Prompt |
|----------------|----------|-------------|
| **FW (Fixed Weights)** | **0.6738 ± 0.0052** | **+0.0542** |
| TES (Task Emb Similarity) | 0.6420 ± 0.0044 | +0.0224 |
| TC-Prompt (λ=0.5 fixed) | 0.6214 ± 0.0025 | +0.0018 |
| TC-Prompt (λ learnable) | 0.6205 ± 0.0017 | +0.0009 |
| Prompt-tuning (original) | 0.6196 ± 0.0015 | baseline |

Learned λ: 0.4503, 0.4419 (mean=0.4461)

---

## 3. Cross-Dataset Comparison

| Dataset | Best Method | Worst Method | prompt AUC | TC-Prompt Δ | Instance-level wins? |
|---------|------------|--------------|------------|-------------|---------------------|
| CensusIncome | prompt (0.8612) | FW (0.8545) | 0.8612 | -0.0001 | YES |
| AliCCP | **FW (0.6738)** | prompt (0.6196) | 0.6196 | +0.0014 | **NO** |

### Key Finding

**The effectiveness of instance-level prompt fusion is dataset-dependent:**
- On CensusIncome (moderate task correlations ρ≈0.15-0.19), instance-level
  attention provides +0.0067 over fixed weights
- On AliCCP (weak BSI correlation with CTR/CVR), fixed weights outperform
  instance-level attention by +0.0542
- **TC-Prompt's correlation prior does NOT change the ranking in either case**

---

## 4. α Sensitivity (CensusIncome) — from Week 1 paper update

| α | AUC/T1 | AUC/T2 | AUC/T3 |
|---|--------|--------|--------|
| 0.01 | 0.5732 ± 0.0035 | 0.5991 ± 0.0038 | 0.6658 ± 0.0030 |
| 0.05 | 0.5735 ± 0.0032 | 0.5992 ± 0.0035 | 0.6660 ± 0.0028 |
| 0.10 | 0.5737 ± 0.0030 | 0.5993 ± 0.0033 | 0.6663 ± 0.0025 |
| 0.30 | 0.5721 ± 0.0028 | 0.6007 ± 0.0031 | 0.6682 ± 0.0026 |
| 0.50 | 0.5712 ± 0.0032 | 0.6015 ± 0.0035 | 0.6695 ± 0.0029 |

---

## 5. Silhouette Scores (CensusIncome) — from Week 1 paper update

| Model | CensusIncome | AliCCP |
|-------|-------------|--------|
| PLE | 0.312 | 0.189 |
| MPT-Rec (α=0.1) | 0.418 | 0.247 |
| MPT-Rec (α=0.3) | 0.475 | 0.291 |

---

## 6. Hardware & Runtime

- GPU: NVIDIA GeForce RTX 3060 Laptop (6 GB VRAM)
- CPU: Intel, Windows 11
- CensusIncome: ~200K samples, Stage 1 ≈ 5 min/seed, Stage 2 ≈ 7 min/mode
- AliCCP: 5M samples, Stage 1 ≈ 55 min/seed, Stage 2 ≈ 30 min/mode
- Total runtime: ~10 hours (across both datasets)
