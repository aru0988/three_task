# U 后续实验执行计划（2026-10-12）

> CC 使用 executing-plans 顺序执行。代码和配置由 Codex 负责；CC 只执行、核验、整理证据，遇实现缺陷停止该项并反馈。

**目标：** 排除 Stage-2 预算截断，再检验 no-read trainable U 的稳定性。
**架构：** 保持已有模型、划分、初始化、学习率、patience=5；长预算两臂从原始初始化重训，复用 Stage-1 配对。
**技术栈：** PyTorch、PowerShell、unittest、独立 raw AUC verifier。

## 现有证据与优先级

Census 三种子 Δtest 均值 +0.0029665781，95%配对t区间 [-0.0020694384,+0.0080025946]，3/3正向；旧任务预测逐字节不变。不能宣称统计显著。Stage-2 部分预算耗尽。
AliCCP no-read trainable 在40轮 Δval +0.0215921928、Δtest +0.0029591736，但B/U均最佳40/40，末10轮仍提升。read-detached trainable 有明显旧任务退化，不作为无损候选。
现有入口支持全部本轮配置，无需修改模型代码。本计划授权固定的60轮预算复核，替代旧计划“40轮后等待预算决定”；不自动递增。

## 0. 准入

工作树 D:\MPT-Rec-three_task\MPT-Rec-exp-universal-representation，分支 exp/universal-representation-bu。单GPU串行；检查其他训练进程，不终止不明进程。先提交本计划及真实未提交结果，正式训练要求干净工作树。不要切master、覆盖历史或上传原始数据集。

```powershell
Set-Location 'D:\MPT-Rec-three_task\MPT-Rec-exp-universal-representation'
$py = 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe'
$src = 'D:\MPT-Rec-three_task\MPT-Rec'
git branch --show-current
git status --short
& $py -m unittest discover -s universal_experiment -p 'test*.py' -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
```

## 1. Census 三种子统一 Stage-2 30轮

- [ ] 顺序执行以下配置。已存在目录只核验，不重跑；不完整目录保留并报告，停止该项。

```powershell
foreach ($seed in @(1685480945,1685463909,1685477428)) {
    $prior = "results\universal\paired-census-no-read-trainable-seed$seed"
    $out = "results\universal\paired-census-no-read-trainable-s2budget30-seed$seed"
    & $py -m universal_experiment.verify_paired $prior
    if ($LASTEXITCODE -ne 0) { throw 'Source invalid' }
    if (-not (Test-Path $out)) {
        & $py -m universal_experiment.paired_run census --source $src --out $out --read-policy no_read --u-mode trainable --seed $seed --stage1-epochs 20 --stage2-epochs 30 --reuse-stage1-pair $prior
        if ($LASTEXITCODE -ne 0) { throw 'Run failed' }
    }
    & $py -m universal_experiment.verify_paired $out
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
    # 每项完成后记录结果并提交，再运行下一项，确保工作树干净。
}
```

- [ ] 检查前20轮与原记录一致；旧任务预测不变；分别记录两阶段收敛。汇总30轮三种子的配对均值、样本SD、95%t区间、最差seed。不能把20轮和30轮混成同一组。

## 2. AliCCP 单种子 Stage-2 60轮预算复核

- [ ] 核验40轮源，复用其Stage-1两臂，不复用40轮头、不假称续训。

```powershell
$prior = 'results\universal\paired-aliccp-no-read-trainable-s2budget40-seed1688723512'
$out = 'results\universal\paired-aliccp-no-read-trainable-s2budget60-seed1688723512'
& $py -m universal_experiment.verify_paired $prior
if ($LASTEXITCODE -ne 0) { throw 'Source invalid' }
if (-not (Test-Path $out)) {
    & $py -m universal_experiment.paired_run aliccp --source $src --out $out --read-policy no_read --u-mode trainable --seed 1688723512 --stage1-epochs 30 --stage2-epochs 60 --reuse-stage1-pair $prior
    if ($LASTEXITCODE -ne 0) { throw 'Run failed' }
}
& $py -m universal_experiment.verify_paired $out
if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
```

- [ ] 检查前40轮重现；记录末10轮验证曲线、best/run epoch、旧任务不变性、成本。若60轮仍预算耗尽，停止扩种子，报告未收敛；不自动上100轮。

## 3. AliCCP 稳定性：有条件执行

仅当第2项两臂Stage-2均由patience收敛、Δval>0且旧任务不变性通过，执行固定两个额外种子；不以test决定是否执行。Stage-1保持30轮，所有结果明确受此预算限制。

```powershell
foreach ($seed in @(1688723740,1688738016)) {
    $out = "results\universal\paired-aliccp-no-read-trainable-s2budget60-seed$seed"
    if (-not (Test-Path $out)) {
        & $py -m universal_experiment.paired_run aliccp --source $src --out $out --read-policy no_read --u-mode trainable --seed $seed --stage1-epochs 30 --stage2-epochs 60
        if ($LASTEXITCODE -ne 0) { throw 'Run failed' }
    }
    & $py -m universal_experiment.verify_paired $out
    if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
    # 每项整理证据并提交后才执行下一项。
}
```

## 4. 交付与下一轮边界

- [ ] 向 docs/experiments/universal-representation-bu.md 追加逐seed及统一预算汇总，保留全部负结果；登记严格匹配的新预算B，不能冒用旧预算B。
- [ ] 记录训练commit、配置、复用来源、raw核验、val/test、旧任务自身前后及相对B差、参数与耗时。复用耗时和完整训练成本分别报告。
- [ ] Stage-1预算耗尽仍需注明，本轮不宣称两阶段完全收敛。Stage-1延长需下一轮fresh B/U，不能用本轮结果证明。
- [ ] Census补到五种子、同参数量随机U对照、自监督消融均留待预算复核结果后由Codex配置；本轮不自行写代码或搜索超参。
- [ ] 将每项核验失败的命令、日志与最小复现交回Codex；禁止放宽verifier掩盖错误。本计划准备阶段仅检查入口参数与已有文档，完整测试由CC准入执行。

## 5. 实验价值准入与结果保留（本节优先于以上批量执行）

用户要求：避免无意义重复；错误或无意义结果不保存为实验成果。

- 每次正式运行前检查已有config/report/verification及训练曲线。已有相同协议的有效结果直接复用，不重复训练。仅重新读取与核验不能冒充新实验。
- 开跑前写下唯一待回答的问题、为何已有证据不足、预计新增证据、停止条件；无法写清则不运行。
- Census先只做1685477428的30轮配对复核：其B最佳位于20轮上限，是明确预算截断疑点。核验后查看B/U选点与验证曲线；再逐一检查另两个种子的既有曲线，仅在存在预算耗尽/未满足patience疑点、且需要统一预算对比时运行相应30轮配对。已有两臂明确收敛的seed不重训；若沿用其原结果，明确标注实际预算与提前停止依据，不冒称新30轮运行。
- AliCCP 60轮仅用于回答40轮截断是否影响配对结论。若已有同配置60轮有效结果则直接使用；若核验或前40轮重现失败，停止后续实验，先交Codex查错。
- 扩种子必须满足第3节条件；不为了凑实验数量扩种子。不得用test选择方案或决定预算。
- 正式结果保留条件：配置与来源匹配、独立核验通过、回答预先明确的问题。有效的零提升/退化结果属于有意义证据，保留简洁结论与必要可复核证据，防止重复探索。
- smoke、错误配置、污染数据、核验失败、重复产物不得登记baseline、进入正式汇总、提交或推送。先留下简短失败原因和排除记录，便于定位；错误产物在缺陷修复并确认无需排查后清理，不长期保存完整无效权重/预测。
- 清理只针对本轮明确识别的无效产物，确认绝对路径在该工作树结果目录内；不得删除历史有效负结果、来源权重或原因未明的中断现场。CC不得因AUC差而判定“无意义”。
