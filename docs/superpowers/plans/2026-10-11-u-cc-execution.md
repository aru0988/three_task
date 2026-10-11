# U 剩余实验：CC 执行计划

> 使用 executing-plans 逐项执行；本文件交给本机 CC，不要求开新任务或新分支。

**目标：** 比较一阶段是否读取 U 与二阶段 U 是否冻结，保证 B 严格复用、旧任务变化可见；先完成单种子，再按验证集证据补稳定性。

**架构：** 原始 B 不含 U。U 新任务头统一直接读取 U；trainable 模式只优化 U 与新任务头，原主干与旧任务 readout 参数冻结。预算延长仅重新训练两臂头，不假称断点恢复。

**技术栈：** PyTorch、unittest、原始 NPZ 独立 rank-AUC 验证。

## 0. 固定现场与准入

工作树 `D:\MPT-Rec-three_task\MPT-Rec-exp-universal-representation`；分支 `exp/universal-representation-bu`。
禁止切换/合并 master，禁止覆盖旧结果，禁止改 `exp/universal-representation` 四臂历史。
用户本轮只让准备代码，以下正式运行由 CC 后续执行；不沿用历史 CC 进程。

```powershell
Set-Location 'D:\MPT-Rec-three_task\MPT-Rec-exp-universal-representation'
$py = 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe'
$src = 'D:\MPT-Rec-three_task\MPT-Rec'
$cB = 'results\universal\paired-census-no-read-frozen-seed1685480945'
$cRead = 'results\universal\paired-census-read-detached-frozen-seed1685480945'
$aRead = 'results\universal\paired-aliccp-read-detached-frozen-seed1688723512'
git branch --show-current
git status --short
& $py -m unittest discover -s universal_experiment -p 'test*.py' -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
foreach ($run in @($cB, $cRead, $aRead)) {
    & $py -m universal_experiment.verify_paired $run
    if ($LASTEXITCODE -ne 0) { throw "Invalid source $run" }
}
```

- [ ] 核验没有本任务训练进程；目录存在且 .pt 完整；不要使用 PATH 中的 Python。
- [ ] 已有 untracked AliCCP 正式结果应先纳入记录/提交，不能删除或误认 smoke。
- [ ] 正式开跑前 `git status --porcelain` 必须为空。验证命令若改写 verification.json，先提交真实核验结果。
- [ ] 不改学习率/初始化/U 自监督权重。U 初始化及 mask 种子仍固定；新增模型种子只代表 base/model-seed 稳定性，不声称全部随机源独立。

每项仅运行一次；先检查输出目录，已有 report 则先核验，不重复运行。缺 report 的中断现场归档后才重新跑，不假称续训。

## 1. Census detached-read：仅改变二阶段冻结策略

```powershell
& $py -m universal_experiment.paired_run census --source $src --out results\universal\paired-census-read-detached-trainable-seed1685480945 --read-policy read_detached --u-mode trainable --seed 1685480945 --reuse-b $cB --reuse-u-stage1 $cRead
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired results\universal\paired-census-read-detached-trainable-seed1685480945
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] 复用 B 全部结果与 ReadDetached U Stage-1；只训练 U Stage-2，20/patience5。
- [ ] 对比 frozen U 与 trainable U；记录旧任务 Stage-1→Stage-2 delta，并另列 U 相对 B 的旧任务差。

## 2. AliCCP detached-read：相同冻结策略对照

```powershell
& $py -m universal_experiment.paired_run aliccp --source $src --out results\universal\paired-aliccp-read-detached-trainable-seed1688723512 --read-policy read_detached --u-mode trainable --seed 1688723512 --reuse-b $aRead --reuse-u-stage1 $aRead
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired results\universal\paired-aliccp-read-detached-trainable-seed1688723512
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] 复用现有 AliCCP B 与 U Stage-1；只训练 U Stage-2，30/patience5。

## 3. AliCCP no-read 冻结：补缺失单元

```powershell
$aNoRead = 'results\universal\paired-aliccp-no-read-frozen-seed1688723512'
& $py -m universal_experiment.paired_run aliccp --source $src --out $aNoRead --read-policy no_read --u-mode frozen --seed 1688723512 --reuse-b $aRead
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired $aNoRead
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] 此项需要 fresh no-read U Stage-1，不能拿 ReadDetached 权重冒充；B 不重训。

## 4. no-read 不冻结：补齐冻结策略对照

```powershell
& $py -m universal_experiment.paired_run census --source $src --out results\universal\paired-census-no-read-trainable-seed1685480945 --read-policy no_read --u-mode trainable --seed 1685480945 --reuse-b $cB --reuse-u-stage1 $cB
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired results\universal\paired-census-no-read-trainable-seed1685480945
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
& $py -m universal_experiment.paired_run aliccp --source $src --out results\universal\paired-aliccp-no-read-trainable-seed1688723512 --read-policy no_read --u-mode trainable --seed 1688723512 --reuse-b $aRead --reuse-u-stage1 $aNoRead
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired results\universal\paired-aliccp-no-read-trainable-seed1688723512
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] no-read 旧任务完全不使用 U，更新 U 后旧任务预测应不变；若不变性失败先查错，不进入扩种子。

## 5. 有界预算复核（只由验证曲线触发）

Census 20轮最佳仍在上限时，先复核 Stage-2 30轮；不是直接上60/100/160。
示例对当前 frozen read-U：

```powershell
& $py -m universal_experiment.paired_run census --source $src --out results\universal\paired-census-read-detached-frozen-s2budget30-seed1685480945 --read-policy read_detached --u-mode frozen --seed 1685480945 --stage1-epochs 20 --stage2-epochs 30 --reuse-stage1-pair $cRead
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
& $py -m universal_experiment.verify_paired results\universal\paired-census-read-detached-frozen-s2budget30-seed1685480945
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] 两臂头从相同原始初始化重新训练，Stage-1 不重跑；长预算 B 不得用20轮 B代替。
- [ ] 若检验 trainable 版本，使用第1项结果目录作 `--reuse-stage1-pair`，并改 `--u-mode trainable` 与唯一输出名。
- [ ] AliCCP 已30轮、B/U仍增长：本轮先保留“未充分收敛”，不擅自上40轮；提出基于末段验证曲线的预算建议，等待批准。
- [ ] Stage-2延长不能证明Stage-1已收敛，分别报告两个阶段。Stage-1延长改变base，必须fresh B/U配对，不复用旧预算B。

## 6. 判定、选型与额外种子（不得按 test 选方案）

- [ ] 每项记录 val/test AUC、U−B、旧任务前后及相对B差、best/run epoch、收敛标记、参数/优化参数数、阶段耗时和复用来源。
- [ ] 分类：Δtest≥+.001 正提升；−.02<Δtest<+.001 无明显提升；≤−.02 明显衰退。负结果必须保留。
- [ ] 选型只用新任务 val、旧任务 val 与成本，test仅作冻结方案的报告；不因test调学习率。
- [ ] 若旧任务任一 val 下降≥.001，相对B或相对自身Stage-1均需标注权衡，不称无损。验证提升不足或收敛未解，不直接宣布最优。
- [ ] 仅对选定方案补两个 canonical model seeds：Census 1685463909、1685477428；AliCCP 1688723740、1688738016。
  每个新 seed 没有匹配 B 时，去掉 reuse 参数 fresh B/U 一次；后续相同 seed 再复用该结果。
  命令保持第1/2项相同 dataset/read-policy/u-mode，只更改 --seed、唯一 --out 并删除 reuse 参数；不得自行改数据划分。
- [ ] 汇总逐seed、均值/样本标准差/配对95% t区间、最差seed与正提升比例；只有跨seed和验证方向支持才给出最终方案。

## 7. 每项核验后提交，最后推送

- [ ] 向 `docs/experiments/universal-representation-bu.md` 追加结果，不覆盖历史结论。B新seed/新预算加入 paired-baselines.json。
- [ ] 仅将对应 run 的 JSON/NPZ 与模型权重显式加入；`.pt`被忽略，需 `git add -f <具体文件>`。用户已批准权重/raw上传，但不要夹带原始数据集。
- [ ] 同方向分支提交；下一正式运行前工作树干净。不得删除真实负结果，确认错误的派生产物删除后重建。
- [ ] `git -c http.proxy=http://127.0.0.1:7897 push origin exp/universal-representation-bu`。
  推送失败本地保留，报告具体错误；不要重跑实验来解决网络问题。

## 实现核验与限制

代码入口已支持 trainable/frozen、reuse-b、reuse-u-stage1、reuse-stage1-pair 和预算参数。
不支持 optimizer 断点续训；保存的是验证选点权重。不会自动调参、自动登记基线、自动提交或启动CC。
小规模 smoke 是实现测试，不得把它的AUC写入正式汇总。复用失配直接拒绝，不会默默重训B。

准备阶段已执行32项单元测试并通过；另完成Census trainable端到端smoke、
复用B+U Stage-1的smoke、只复用两臂Stage-1的frozen smoke；均从raw核验。
现有Census canonical B准入（代码/运行环境/原始证据）通过；现有AliCCP正式结果已再次核验。
三个历史baseline commit的模型、协议、Stage-1与头初始化语义身份与当前代码一致。
这些结果只证明实现可运行，不代表新方案的正式AUC或已收敛。
