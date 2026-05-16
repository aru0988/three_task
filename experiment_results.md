# MPT-Rec 论文复现 — 实验数据记录

## Table 3: MTL 两任务结果 ✅

| 数据集 | 模型 | AUC/T1 | AUC/T2 | Gain/T1 | Gain/T2 |
|--------|------|--------|--------|---------|---------|
| CensusIncome | Single Task | - | - | - | - |
| CensusIncome | SharedBottom | - | - | - | - |
| CensusIncome | MMOE | - | - | - | - |
| CensusIncome | PLE | - | - | - | - |
| CensusIncome | STEM | - | - | - | - |
| CensusIncome | SparseSharing | - | - | - | - |
| CensusIncome | CSRec | - | - | - | - |
| CensusIncome | **MPT-Rec** | - | - | - | - |
| AliCCP | Single Task | - | - | - | - |
| AliCCP | SharedBottom | - | - | - | - |
| AliCCP | MMOE | - | - | - | - |
| AliCCP | PLE | - | - | - | - |
| AliCCP | STEM | - | - | - | - |
| AliCCP | SparseSharing | - | - | - | - |
| AliCCP | CSRec | - | - | - | - |
| AliCCP | **MPT-Rec** | - | - | - | - |

> 用户已复现，需填入实际值

---

## Table 4: 新任务对已有任务的负迁移影响（3任务全量训练）

**数据集**: CensusIncome (T1=Income, T2=Marital, T3=Education)

| 模型 | AUC/T1 (2-task) | AUC/T1 (3-task) | Δ T1 | AUC/T2 (2-task) | AUC/T2 (3-task) | Δ T2 | Impact |
|------|-----------------|-----------------|------|-----------------|-----------------|------|--------|
| SharedBottom | 0.9484 | 0.9448±0.0013 | -0.0036 | 0.9897 | 0.9915±0.0010 | +0.0018 | −/+ |
| MMOE | | | | | | | |
| PLE | | | | | | | |
| STEM | | | | | | | |

> SharedBottom 5-seed mean: Income=0.9448, Marital=0.9915, Education=0.8645

---

## Table 5: 新任务泛化 Fine-tuning

### CensusIncome

| 方法 | AUC/T3 | #Params | %Params | #FLOPs | %FLOPs |
|------|--------|---------|---------|--------|--------|
| MAML | | | - | | - |
| SharedBottom* | | | | | |
| MMOE* | | | | | |
| PLE* | | | | | |
| MPT-Rec | | | | | |

### AliCCP

| 方法 | AUC/T3 | #Params | %Params | #FLOPs | %FLOPs |
|------|--------|---------|---------|--------|--------|
| MAML | | | - | | - |
| SharedBottom* | | | | | |
| MMOE* | | | | | |
| PLE* | | | | | |
| MPT-Rec | | | | | |

---

## Figure 4: 消融 — 表征分离对已有任务的影响

### CensusIncome

| 变体 | AUC/T1 | AUC/T2 |
|------|--------|--------|
| MPT-Rec (full) | | |
| MPT-Rec (Share-only) | | |
| MPT-Rec (Specific-only) | | |
| MPT-Rec (-GAN) | | |

### AliCCP

| 变体 | AUC/T1 | AUC/T2 |
|------|--------|--------|
| MPT-Rec (full) | | |
| MPT-Rec (Share-only) | | |
| MPT-Rec (Specific-only) | | |
| MPT-Rec (-GAN) | | |

---

## Figure 5: 消融 — 表征分离对新任务的影响

### CensusIncome

| 变体 | AUC/T3 |
|------|--------|
| MPT-Rec (full) | |
| MPT-Rec (Share-only) | |
| MPT-Rec (Specific-only) | |
| MPT-Rec (-GAN) | |

### AliCCP

| 变体 | AUC/T3 |
|------|--------|
| MPT-Rec (full) | |
| MPT-Rec (Share-only) | |
| MPT-Rec (Specific-only) | |
| MPT-Rec (-GAN) | |

---

## Figure 6: 消融 — Prompt 机制有效性

### CensusIncome + AliCCP

| 变体 | AUC/T3 (Census) | AUC/T3 (AliCCP) |
|------|-----------------|------------------|
| MPT-Rec (prompt-tuning) | | |
| MPT-Rec (FW) | | |
| MPT-Rec (TES) | | |

---

## Figure 7: t-SNE 表征可视化

- [ ] CensusIncome: MPT-Rec vs PLE 表征分离
- [ ] AliCCP: MPT-Rec vs PLE/STEM 表征分离
