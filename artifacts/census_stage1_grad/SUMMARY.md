> 说明：本表是**只追加**的门禁结论表（预注册 §4）。每个 run 的判定依据与解读见 `runs/<run_id>/report.md`；
> 同目录 `audit.json` 是逐字节可复现的原始证据（无时间戳/绝对路径），`prereg.md` 是阈值预注册快照。
> run `20260930-2351-s20260929-m1685480945-short-87afe03`（2026-09-30 正式审计）= **NO_ACTION**：
> 负余弦出现在 50% 的采样步，但平均余弦 ≈ +0.0008，冲突在均值上相互抵消（G2 要求 ≤ -0.05 未过）→ **不试 PCGrad**，主线继续。
| run_id | commit | smoke | n_steps | n_epochs | neg_cos_rate | mean_cos | nonzero_rate | status |
|---|---|---|---|---|---|---|---|---|
| 20260930-2351-s20260929-m1685480945-short-87afe03 | 87afe03 | False | 68 | 2 | 0.5000 | 0.0008 | 1.0000 | NO_ACTION |
