# AliCCP 参数高效集成：固定种子 2–5 扩展预注册

## 起点与目的

本分支 `exp/aliccp-parameter-efficient-ensemble-seeds2-5` 从方向 3 的文档基点 `61b6a2c` 建立，仅移入已审核的三臂实现 `c12d0b7`，不继承 seed1 结果文件。已知 seed1 在原方向分支的 E−B test AUC 为 +0.0252211960、E−C 为 +0.0160523635，且平均预测低于按 val 选出的较强成员；本扩展不得按新结果改结构或阈值。目标是验证跨种子效用，并区分「双头联合训练」与「推理平均」的贡献；不把已有集成方法包装为新颖方法。

## 固定运行清单

只使用 AliCCP `p2M-v500k-t1M`，共享指纹 `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`。每种子 B=原始 NewTask、C=单 rank-16 残差头、E=双 rank-8 残差头按 B→C→E 各运行一次；相同前缀、相同种子、相同冻结 Stage-1、batch=2000、Adam lr=1e-4、最多 5 epoch、patience=2。每臂独立 val AUC 选 checkpoint，test 仅一次。不得复用旧 baseline 的预测充作 B 臂，也不得在本扩展中增加训练臂、调参或改变预算。

| model seed | 固定 Stage-1 ID | backbone SHA256 |
|---|---|---|
| 1688723740 | `s1-5c060b9c-m1688723740-e3-4e1b5c6f` | `e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c` |
| 1688738016 | `s1-5c060b9c-m1688738016-e3-47619ce0` | `5728346835fb4fa4b2e9a906bd2305527524a4ebfebceb566394ac79b3a6fc7c` |
| 1688749593 | `s1-5c060b9c-m1688749593-e3-bb2b68de` | `527ff514f8ee1695e302351d030a2ead81f4719d07a4b1fc19dc8520496e4e55` |
| 1688762746 | `s1-5c060b9c-m1688762746-e3-6343490f` | `191b464e3dbaefae4b030787db7adc8bf723b275ebbdb31ad08d29e87bd456cf` |

上述 Stage-1 是已有的公平基线产物，meta 记录均为 2M/500k/1M、3 epoch/patience 2、`git.dirty=false`；以只读方式引入，不覆盖。运行前提交本文件与实现，88 项测试、smoke 和参数预算门禁通过；正式 run 必须 `git.dirty=false`。若某 Stage-1 的 meta、指纹或 SHA 不符，暂停该 seed 而非替换 checkpoint。

## 结果与归因判定

- 每种子从 raw `predictions.pt` 用独立 sklearn 重算 B/C/E 的 test/val BSI AUC、E−B、E−C、C−B、两成员预测差异、val 选中成员的 test AUC，以及平均 E 减去该 val 选中成员的 test AUC。后者是预注册诊断，不使用 test 挑成员。
- 每种子核验三臂的 seed、前缀、预算、Stage-1/backbone SHA、commit、`git.dirty=false`、样本量、标签逐项相同、val 选点、wall time、原始预测与记录数值；A 类门禁必须满足。B1/B4 若为 Stage-1 继承失败，原样记录，不能写「全部门禁通过」。任一真实性核验失败的 run 不进入汇总。
- 汇总固定五种子（seed1 原始分支 + 本分支四种子）的逐种子 delta、均值、样本标准差、正提升比例、最差种子、配对 95% t 区间（描述性，不作显著性宣称）。新分类沿用 Δtest≥+0.001 正提升；−0.02<Δtest<+0.001 无明显提升；Δtest≤−0.02 明显衰退，历史结论不改写。
- 判「组合方案值得后续研究」需五种子 E−B 与 E−C 的均值均 ≥+0.001、至少 3/5 种子各自两个差值都 ≥+0.001、平均 Δval 同向，且至少 3/5 机制未塌缩。未达成则记录不稳定并关闭本变体；无明显提升须结合机制、val 与成本判断提升空间。
- 「推理平均本身有价值」另需 E 均值相对按 val 选中成员的 test AUC 五种子平均 ≥+0.001 且至少 3/5 为正；未满足时即便组合方案有效，也只可称双头联合训练/参数化组合有效，不得归因于平均集成。

成功与失败均保存原始产物、配置、日志和判断，独立核验后回填 SUMMARY、提交并推送；确认错误的派生记录作废并从 raw/正确重跑生成，不让无效数字进入汇总。绝不合并 master。
