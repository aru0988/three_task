# TC-Prompt Experiment Results — CensusIncome T3 (Education)

## Raw Data (3 seeds × 5 fusion modes)

| Seed | Mode | Val AUC | Test AUC | λ |
|------|------|---------|----------|---|
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

## Mean ± Std (Test AUC)

| Fusion Strategy | Test AUC |
|----------------|----------|
| FW (Fixed Weights) | 0.8545 ± 0.0025 |
| TES (Task Emb Sim) | 0.8546 ± 0.0026 |
| **Prompt-tuning (original)** | **0.8612 ± 0.0030** |
| TC-Prompt (λ=0.5 fixed) | 0.8612 ± 0.0032 |
| TC-Prompt (λ learnable) | 0.8611 ± 0.0026 |

Learned λ values: 0.4037, 0.6827, 0.5550 (mean=0.5471)

## Conclusion

TC-Prompt does NOT show statistically significant improvement over original
MPT-Rec prompt-tuning on CensusIncome T3. All three instance-level methods
(prompt, tcprompt_fixed, tcprompt_learnable) perform equivalently.

Instance-level attention (prompt/tcprompt) significantly outperforms task-level
(TES) and fixed-weight (FW) baselines by +0.0066 AUC, confirming the value of
instance-level fusion.
