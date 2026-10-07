"""方向 3：AliCCP 参数高效集成（共享冻结 Stage-1 backbone + 共享 Stage-2 NewTask 主头 + 小残差头）。

spec: docs/superpowers/specs/2026-10-08-aliccp-parameter-efficient-ensemble-design.md
上游协议（只引用、不修改）：docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md

设计（预注册冻结）：
- 三臂：B = 原始 NewTask；C = 同一 NewTask + 一个 rank-16 残差头；E = 同一 NewTask + 两个 rank-8 残差头，
  概率均值为集成输出。C/E 可训练参数量精确相等（各 1040，主头之外），E 总量远小于两个完整 NewTask。
- 残差头 = 无偏置两层的低秩线性映射（down: rep_dim→rank，up: rank→1，up 零初始化），
  仅读取 NewTask 已形成的 fused representation，以近零（初值精确为 0）残差作用于概率 logit：
  p_i = sigmoid(z_main + r_i)。零残差时成员输出与主头逐位相同（单元测试守卫）。
- 训练：E 的两个头逐样本分别 BCE 后取平均（C 为同一目标的单头退化情形），
  损失 = mean_i BCE(p_i, y) + NewTask.get_l2_reg()；优化器 Adam(lr=protocol.LR)。
  集成只在推理（val 选点 / test 评估）时对成员概率求算术平均。
- 本模块不修改 multitaskrec/*、config.py、aliccp_benchmark/{protocol,metrics}.py；
  阶段 2 编排镜像 bench.run_stage2（复用其加载器/冻结/门禁机器），并额外落盘 raw 预测供独立解析器重算。
- 真实性守卫（独立复核加固）：config/metrics/predictions 三份产物必须对身份字段
  （run_id/tag/arm/seed/前缀/stage1_id/commit+git.dirty/指纹/backbone SHA/预算(含 val)/epochs/patience）
  逐项互证（缺失即失败，任何单方自述——含 metrics.json——不得单独通过）；raw 载荷核验长度、
  有限性、[0,1] 范围、标签二值性、成员数与「集成=成员均值」；产物树内锚定：前缀指纹文件
  自哈希、Stage-1 meta、由 backbone.pt 字节重建模型重算 backbone SHA。三臂间身份字段必须
  一致且 test/val 标签逐项相同（哈希+长度）。
- 运行产物不写 SUMMARY.md（本实验的台账行待独立复核后再处置）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitaskrec.model import MPTRec, NewTask

from . import bench, metrics, protocol

# ---- 臂定义与判定常量（spec：机制门禁与结果规则）----
ARM_RANKS = {"B": (), "C": (16,), "E": (8, 8)}
RESIDUAL_COLLAPSE_TOL = 1e-4  # 两头平均绝对预测差低于此值判机制塌缩
DELTA_POSITIVE_MIN = 0.001  # test AUC 差值 >= +0.001 为「正提升」
DELTA_DECLINE_MAX = -0.02  # 差值 <= -0.02 为「明显衰退」


class ResidualHead(nn.Module):
    """rank-r 低秩线性残差：r(x) = up(down(x))，(B, rep_dim) -> (B, 1)。

    两层均无偏置；up 层零初始化 ⇒ 初值残差恒为 0（"近零初始残差"的最强形式，零残差 ≡ 主头）。
    down 层沿用 PyTorch 默认初始化（各自独立抽样，两头不共享任何参数）。
    已知代价：第一步反传时 down 层梯度为 0（∂r/∂down ∝ up = 0），第二步起恢复（单元测试钉死）。
    """

    def __init__(self, rep_dim: int, rank: int):
        super().__init__()
        if rank < 1 or rep_dim < 1:
            raise ValueError(f"非法维度: rep_dim={rep_dim} rank={rank}")
        self.rank = int(rank)
        self.down = nn.Linear(rep_dim, rank, bias=False)
        self.up = nn.Linear(rank, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, fused_rep: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(fused_rep))


class EnsembleNewTask(NewTask):
    """主头逐位同构于原 NewTask（super().__init__ 承担全部构造），附加 n 个残差头。

    残差头输入仅为 NewTask 已形成的 fused representation；成员概率 = sigmoid(主头 logit + r_i)；
    forward 返回成员概率的算术平均（E 的集成输出；C 为单成员退化）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None, ranks=(8, 8)):
        super().__init__(
            input_size=input_size,
            rep_dim=rep_dim,
            tower_dnn_hidden_units=tower_dnn_hidden_units,
            reg_dnn=reg_dnn,
            device=device,
        )
        # 结构守卫：主头 logit 通过 tower_network.mlp 去掉末层 Sigmoid 取到（MLP 无 dropout 时末层即 Sigmoid）
        if not isinstance(self.tower_network.mlp[-1], nn.Sigmoid):
            raise AssertionError("tower_network 末层非 Sigmoid，无法取主头 logit（协议配置不应触发）")
        self.ranks = tuple(int(r) for r in ranks)
        self.residual_heads = nn.ModuleList(ResidualHead(rep_dim, r) for r in self.ranks)

    def _fused_rep_and_logit(self, dnn_input, gen_rep, spec_reps, env_embs):
        """逐字复刻 NewTask.forward 截至 fused_rep 的路径（test_ensemble 与父类 forward 逐位一致守卫）。"""
        exist_env_embs = torch.stack(env_embs, dim=1)
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)

        H_out = self.projection_network(dnn_input)
        W = torch.mm(H_out, exist_env_embs) / self.temperature
        W = F.softmax(W, dim=-1).unsqueeze(2)

        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), W).squeeze()
        env_aware_rep = new_spec_rep * new_env_emb
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze()

        logit = self.tower_network.mlp[:-1](fused_rep)  # 主头概率 logit（sigmoid 前激活）
        return fused_rep, logit

    def member_probabilities(self, dnn_input, gen_rep, spec_reps, env_embs):
        """各残差头的成员概率列表（推理时求算术平均得到集成输出）。"""
        fused_rep, logit = self._fused_rep_and_logit(dnn_input, gen_rep, spec_reps, env_embs)
        return [torch.sigmoid(logit + head(fused_rep)).squeeze() for head in self.residual_heads]

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        members = self.member_probabilities(dnn_input, gen_rep, spec_reps, env_embs)
        return torch.stack(members, dim=0).mean(dim=0)


def build_arm_head(arm, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device):
    """按臂构造头。主头构造顺序在三个臂中完全一致（同种子下逐位相同）。"""
    if arm not in ARM_RANKS:
        raise KeyError(f"未知臂: {arm}（可选 {sorted(ARM_RANKS)}）")
    ranks = ARM_RANKS[arm]
    if not ranks:
        return NewTask(
            input_size=input_size,
            rep_dim=rep_dim,
            tower_dnn_hidden_units=list(tower_dnn_hidden_units),
            reg_dnn=reg_dnn,
            device=device,
        )
    return EnsembleNewTask(
        input_size=input_size,
        rep_dim=rep_dim,
        tower_dnn_hidden_units=list(tower_dnn_hidden_units),
        reg_dnn=reg_dnn,
        device=device,
        ranks=ranks,
    )


def arm_members(head, dnn_input, gen_rep, spec_reps, env_embs):
    """统一取成员概率列表：B 臂为 [主头概率]；C/E 为各残差头成员概率。"""
    if isinstance(head, EnsembleNewTask):
        return head.member_probabilities(dnn_input, gen_rep, spec_reps, env_embs)
    return [head(dnn_input, gen_rep, spec_reps, env_embs)]


def residual_param_count(rep_dim: int, ranks) -> int:
    return sum(rep_dim * r + r for r in ranks)


def param_report(head) -> dict:
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    residual = sum(p.numel() for name, p in head.named_parameters() if name.startswith("residual_heads"))
    return {"total": total, "trainable": trainable, "main_head": total - residual, "residual": residual}


def budget_check(*, c_trainable: int, e_trainable: int, newtask_trainable: int, enforce: bool = True) -> dict:
    """spec 公平配对：C/E 可训练参数量相差 ≤5%，且 E 总量 < 两个完整 NewTask；正式运行（enforce=True，
    默认）不满足即 NO-GO。enforce=False 只记录布尔结论不抛错（tiny 规模 fixture 专用：该条件与规模
    相关，必须在正式 64/32/32 配置下验证，见 test_ensemble 的两个预算测试）。"""
    ratio = abs(c_trainable - e_trainable) / min(c_trainable, e_trainable)
    within = ratio <= 0.05
    below_two = e_trainable < 2 * newtask_trainable
    if enforce and not within:
        raise AssertionError(f"NO-GO：C/E 可训练参数量相差 {ratio:.4%} > 5%（C={c_trainable}, E={e_trainable}）")
    if enforce and not below_two:
        raise AssertionError(f"NO-GO：E 可训练参数量 {e_trainable} >= 两个完整 NewTask {2 * newtask_trainable}")
    return {
        "c_trainable": int(c_trainable),
        "e_trainable": int(e_trainable),
        "newtask_trainable": int(newtask_trainable),
        "c_vs_e_ratio": ratio,
        "c_vs_e_within_5pct": within,
        "e_less_than_two_newtasks": below_two,
    }


def mechanism_stats(member_probs, tol: float = RESIDUAL_COLLAPSE_TOL):
    """E 两头预测的机制统计：平均/最大绝对差、分位分歧、相关系数、塌缩判定。单成员返回 None。"""
    if member_probs is None or len(member_probs) < 2:
        return None
    first = member_probs[0].detach().double()
    second = member_probs[1].detach().double()
    diff = (first - second).abs()
    # torch.quantile 要求 q 与输入同 dtype；diff 已转 float64，q 默认 float32 会 RuntimeError
    quantiles = torch.quantile(diff, torch.tensor([0.5, 0.9, 0.99], dtype=diff.dtype, device=diff.device))
    std_ok = float(first.std()) > 0.0 and float(second.std()) > 0.0
    corr = float(torch.corrcoef(torch.stack([first, second]))[0, 1]) if std_ok else None
    exactly_identical = bool(torch.equal(first, second))
    mean_abs = float(diff.mean())
    return {
        "n": int(diff.numel()),
        "mean_abs_diff": mean_abs,
        "max_abs_diff": float(diff.max()),
        "q50_abs_diff": float(quantiles[0]),
        "q90_abs_diff": float(quantiles[1]),
        "q99_abs_diff": float(quantiles[2]),
        "corr": corr,
        "exactly_identical": exactly_identical,
        "collapsed": bool(exactly_identical or mean_abs < tol),
        "tol": float(tol),
    }


def screen_verdict(*, delta_eb, delta_ec, val_direction_consistent, collapsed, gates_ok) -> dict:
    """spec 结果规则：分类按 E−B；扩展资格要求 E−C 与 E−B 均 >= +0.001、val 同向、机制未塌缩、门禁全过。"""
    if delta_eb >= DELTA_POSITIVE_MIN:
        classification = "POSITIVE_IMPROVEMENT"
    elif delta_eb > DELTA_DECLINE_MAX:
        classification = "NO_CLEAR_IMPROVEMENT"
    else:
        classification = "CLEAR_DECLINE"
    conditions = {
        "delta_eb_positive": bool(delta_eb is not None and delta_eb >= DELTA_POSITIVE_MIN),
        "delta_ec_positive": bool(delta_ec is not None and delta_ec >= DELTA_POSITIVE_MIN),
        "val_direction_consistent": bool(val_direction_consistent),
        "mechanism_not_collapsed": bool(collapsed is False),  # 机制未知（None）不算未塌缩
        "gates_ok": bool(gates_ok),
    }
    return {
        "classification": classification,
        "expansion_eligible": all(conditions.values()),
        "conditions": conditions,
        "thresholds": {"positive_min": DELTA_POSITIVE_MIN, "decline_max": DELTA_DECLINE_MAX},
    }


def make_arm_run_id(now, *, prefix_tag, model_seed, tag, commit, arm) -> str:
    if arm not in ARM_RANKS:
        raise KeyError(f"未知臂: {arm}")
    base = protocol.make_run_id(now, prefix_tag=prefix_tag, model_seed=model_seed, tag=tag, commit=commit)
    return f"{base}-ens-{arm}"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate_members(head, model, loader, device) -> dict:
    """单次遍历：标签 + 各成员概率 + 臂输出（成员概率算术平均）。"""
    head.eval()
    labels, members_acc = [], None
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        members = arm_members(head, dnn_input, gen_rep, spec_reps, env_embs)
        if members_acc is None:
            members_acc = [[] for _ in members]
        for acc, member in zip(members_acc, members):
            acc.append(member.cpu())
        labels.append(y)
    member_probs = [torch.cat(acc) for acc in members_acc]
    return {
        "labels": torch.cat(labels).long(),
        "members": member_probs,
        "arm": torch.stack(member_probs, dim=0).mean(dim=0),
    }


def run_arm(
    *,
    root,
    arm,
    stage1_id,
    data_files,
    budgets,
    prefix_tag,
    model_seed,
    epochs,
    patience,
    tag,
    device,
    enforce_b=False,
    enforce_budget=True,
    require_clean=False,
    log=print,
    vocab=None,
    expert_hidden=protocol.EXPERT_HIDDEN,
    tower_hidden=protocol.TOWER_HIDDEN,
    embedding_size=protocol.EMBEDDING_SIZE,
    input_size=protocol.INPUT_SIZE,
    batch_size=protocol.BATCH_SIZE,
    lr=protocol.LR,
    reg_dnn=protocol.REG_DNN,
    newtask_rep_dim=None,
    run_id=None,
) -> dict:
    """单臂阶段 2：加载固定 Stage-1 → 真冻结 → 训练臂头 → val 选点 → test 单评 → 门禁与 raw 落盘。

    与 bench.run_stage2 同构（复用其加载器/冻结/门禁机器）；额外落盘 predictions.pt（val 逐 epoch + test
    的 raw 概率与标签）供独立解析器重算。不写 SUMMARY.md（独立复核后另行处置）。
    """
    if arm not in ARM_RANKS:
        raise KeyError(f"未知臂: {arm}（可选 {sorted(ARM_RANKS)}）")
    if require_clean:
        state = protocol.git_state()
        if state["dirty"]:
            raise RuntimeError(f"工作树有未提交的已跟踪修改（git.dirty=true），正式运行前必须提交：{state}")

    t0 = time.time()
    bench._reset_peak_vram(device)
    log(f"[ens-{arm}] 开始：stage1_id={stage1_id} tag={tag} model_seed={model_seed} ranks={list(ARM_RANKS[arm])}")
    protocol.seed_model(model_seed)  # 独立进程重播种：头初始化与阶段 1 轨迹解耦

    datasets, loaders = bench._loaders(data_files, budgets, batch_size)

    # ---- A2 / A4：指纹与标签对齐 ----
    prefix_sha_ok = True
    fp = None
    try:
        fp = protocol.load_fingerprint(root, prefix_tag)
        protocol.verify_fingerprint(fp)
    except (AssertionError, FileNotFoundError) as exc:  # noqa: BLE001
        prefix_sha_ok = False
        log(f"[ens-{arm}] A2 前缀指纹校验失败：{exc}")
        if fp is None:
            fp = {"fingerprint_sha256": None, "label_counts": {}}
    len_ok = bench._budget_of(datasets, budgets)
    counts_ok = True
    for split in ("train", "val", "test"):
        try:
            protocol.verify_label_counts(datasets[split], budgets[split], fp["label_counts"][split])
        except (AssertionError, KeyError) as exc:  # noqa: BLE001
            counts_ok = False
            log(f"[ens-{arm}] A4 标签计数失败（{split}）：{exc}")

    # ---- 加载固定 Stage-1 产物（唯一 backbone 来源）----
    art = protocol.load_stage1(root, stage1_id)
    art_meta = art["meta"]
    env_ids = art["env_ids"]
    fingerprint_sha_match = art_meta.get("fingerprint_sha256") == fp.get("fingerprint_sha256")
    env_ids_sha_match = protocol.sha256_tensor(env_ids) == art_meta.get("env_ids_sha256")

    vocab = dict(vocab) if vocab is not None else protocol.build_vocab()
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(protocol.DROPOUT),
        reg_embedding=protocol.REG_EMBEDDING,
        reg_dnn=reg_dnn,
        device=device,
    ).to(device)
    model.load_state_dict(art["backbone_state"])
    backbone_sha_loaded = protocol.backbone_sha256(model)
    backbone_sha_matches_stage1 = backbone_sha_loaded == art_meta.get("backbone_sha256")
    stage1_id_recorded = art_meta.get("stage1_id") == stage1_id

    # ---- 真冻结三件套 ----
    protocol.freeze_backbone(model)
    backbone_sha_before = protocol.backbone_sha256(model)
    log(f"[ens-{arm}] backbone 已加载并冻结：sha={backbone_sha_before[:16]}（A6 匹配={backbone_sha_matches_stage1}）")

    rep_dim = int(newtask_rep_dim) if newtask_rep_dim is not None else int(list(expert_hidden)[-1])
    if rep_dim != int(list(expert_hidden)[-1]):
        raise AssertionError("NewTask rep_dim 必须等于 expert_dnn_hidden_units[-1]（env_embs 维度约束）")
    head = build_arm_head(
        arm, input_size=input_size, rep_dim=rep_dim,
        tower_dnn_hidden_units=tower_hidden, reg_dnn=reg_dnn, device=device,
    ).to(device)
    report = param_report(head)
    budget_report = None
    if arm != "B":
        # 对照与集成的预算核对按臂定义解析计算（主头三臂同构，见 build_arm_head 构造顺序）；
        # 正式运行（enforce_budget=True，默认）下不满足即 NO-GO
        budget_report = budget_check(
            c_trainable=report["main_head"] + residual_param_count(rep_dim, ARM_RANKS["C"]),
            e_trainable=report["main_head"] + residual_param_count(rep_dim, ARM_RANKS["E"]),
            newtask_trainable=report["main_head"],
            enforce=enforce_budget,
        )
        log(f"[ens-{arm}] 预算门禁：C/E≤5%={budget_report['c_vs_e_within_5pct']} "
            f"E<2×NewTask={budget_report['e_less_than_two_newtasks']} enforce={enforce_budget}")
    log(f"[ens-{arm}] 头参数：trainable={report['trainable']}（主头 {report['main_head']} + 残差 {report['residual']}）")
    optimizer = torch.optim.Adam(params=head.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()

    best_auc, best_epoch, best_weight, earlystop_count = 0.0, None, None, 0
    epoch_records, val_history = [], []
    for epoch in range(1, epochs + 1):
        head.train()
        epoch_t0 = time.time()
        loss_sum, steps = 0.0, 0
        for _, _, y, features in loaders["train"]:
            for key in features:
                features[key] = features[key].to(device)
            with torch.no_grad():  # 计算图级冻结；backbone 恒为 eval
                dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
            members = arm_members(head, dnn_input, gen_rep, spec_reps, env_embs)
            # E 的两头逐样本分别 BCE 后取平均；C 为同一目标的单头退化（spec 训练目标）
            loss = sum(loss_func(member.cpu(), y.float()) for member in members) / len(members)
            loss = loss + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            steps += 1
        val = evaluate_members(head, model, loaders["val"], device)
        val_auc = metrics.auc_score(val["labels"], val["arm"])
        val_history.append({"arm": val["arm"], "members": val["members"]})
        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(1, steps),
                "val_auc_bsi": val_auc,
                "seconds": round(time.time() - epoch_t0, 1),
            }
        )
        log(f"[ens-{arm}] Epoch:{epoch} train_loss={loss_sum / max(1, steps):.4f} AUC-Val-BSI:{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_epoch, earlystop_count = val_auc, epoch, 0
            best_weight = copy.deepcopy(head.state_dict())
        else:
            earlystop_count += 1
            log(f"[ens-{arm}] EarlyStopping count {earlystop_count}")
            if earlystop_count == patience:
                log(f"[ens-{arm}] EarlyStopping at epoch {epoch}")
                break
    head.load_state_dict(best_weight)
    test = evaluate_members(head, model, loaders["test"], device)  # test 只评一次
    test_auc = metrics.auc_score(test["labels"], test["arm"])
    mechanism = mechanism_stats(test["members"])
    gate_mean = bench.newtask_gate_mean(head, model, loaders["val"], device)

    # ---- A1：冻结完整性 ----
    backbone_sha_after = protocol.backbone_sha256(model)
    try:
        protocol.assert_no_grads(model)
        backbone_grads_none = True
    except AssertionError as exc:  # noqa: BLE001
        backbone_grads_none = False
        log(f"[ens-{arm}] A1 失败：{exc}")

    # ---- 门禁判定（沿用评测协议 machinery，不修改）----
    a_facts = {
        "backbone_sha_before": backbone_sha_before,
        "backbone_sha_after": backbone_sha_after,
        "backbone_grads_none": backbone_grads_none,
        "prefix_sha_ok": prefix_sha_ok,
        "fingerprint_sha_match": fingerprint_sha_match,
        "len_ok": len_ok,
        "counts_ok": counts_ok,
        "env_ids_sha_match": env_ids_sha_match,
        "backbone_sha_matches_stage1": backbone_sha_matches_stage1,
        "stage1_id_recorded": stage1_id_recorded,
    }
    a_gates = metrics.evaluate_a_gates(a_facts)
    b_gates = metrics.evaluate_b_gates(
        {
            "auc_val_ctr": art_meta.get("best_val_auc_ctr", 0.0),
            "auc_val_cvr": art_meta.get("best_val_auc_cvr", 0.0),
            "auc_val_bsi_best": best_auc,
            "auc_test_bsi": test_auc,
            "gate_mean": gate_mean,
            "cluster_events": art_meta.get("cluster_events", []),
            "train_size": budgets["train"],
        }
    )
    gates = {**a_gates, **b_gates}
    passed = metrics.hard_pass(a_gates, enforce_b=enforce_b, b_gates=b_gates)

    # ---- 产物落盘（raw 预测 + 配置 + 指标 + 门禁；不写 SUMMARY）----
    # git 溯源单一来源：三份产物记录同一 git_state（commit+dirty），run_id 后缀 commit 亦取同一值，
    # 供独立解析器（rederive_arm）做 config/metrics/predictions 三方互证
    git_now = protocol.git_state()
    commit_now = git_now["commit"]  # git_state 内部即 code_commit()
    if run_id is None:
        run_id = make_arm_run_id(
            datetime.now(), prefix_tag=prefix_tag, model_seed=model_seed, tag=tag,
            commit=commit_now, arm=arm,
        )
    run_path = protocol.run_dir(root, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, run_path / "newtask.pt")
    predictions = {
        # 身份字段与 config/metrics 逐项同值落盘（rederive_arm 三方互证；缺失即判失败）
        "run_id": run_id,
        "tag": tag,
        "arm": arm,
        "model_seed": int(model_seed),
        "prefix_tag": prefix_tag,
        "stage1_id": stage1_id,
        "commit": commit_now,
        "git": git_now,
        "fingerprint_sha256": fp.get("fingerprint_sha256"),
        "backbone_sha256": backbone_sha_before,
        "budgets": dict(budgets),
        "epochs": int(epochs),
        "patience": int(patience),
        "test": {"labels": test["labels"], "arm": test["arm"], "members": test["members"]},
        "val": {
            "labels": val["labels"],  # 各 epoch 标签相同，取最后一次逐 epoch 评估的标签
            "epochs": [{"arm": record["arm"], "members": record["members"]} for record in val_history],
        },
    }
    torch.save(predictions, run_path / "predictions.pt")

    metrics_doc = {
        "run_id": run_id,
        "tag": tag,
        "arm": arm,
        "ranks": list(ARM_RANKS[arm]),
        "stage1_id": stage1_id,
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "enforce_b": bool(enforce_b),
        "best_epoch": best_epoch,
        "best_val_auc_bsi": float(best_auc),
        "test_auc_bsi": float(test_auc),
        "mechanism": mechanism,
        "gate_mean": gate_mean,
        "per_epoch": epoch_records,
        "param_report": report,
        "budget_report": budget_report,
        "backbone_sha256_loaded": backbone_sha_loaded,
        "backbone_sha256_before": backbone_sha_before,
        "backbone_sha256_after": backbone_sha_after,
        "backbone_grads_none": backbone_grads_none,
        "env_ids_sha256": protocol.sha256_tensor(env_ids),
        "fingerprint_sha256": fp.get("fingerprint_sha256"),
        "newtask_state_sha256": file_sha256(run_path / "newtask.pt"),
        "hard_pass": bool(passed),
        "commit": commit_now,
        "git": git_now,
        "versions": bench._versions(),
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1) if device.type == "cuda" else None
        ),
    }
    config_doc = {
        "run_id": run_id,
        "tag": tag,
        "arm": arm,
        "ranks": list(ARM_RANKS[arm]),
        "stage1_id": stage1_id,
        "prefix_tag": prefix_tag,
        "data_files": {k: str(v) for k, v in data_files.items()},
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "reg_dnn": float(reg_dnn),
        "newtask_rep_dim": rep_dim,
        "expert_hidden": list(expert_hidden),
        "tower_hidden": list(tower_hidden),
        "input_size": int(input_size),
        "embedding_size": int(embedding_size),
        "enforce_b": bool(enforce_b),
        "enforce_budget": bool(enforce_budget),
        "require_clean": bool(require_clean),
        "budget_report": budget_report,
        # 身份字段（与 metrics/predictions 逐项同值；rederive_arm 三方互证）
        "fingerprint_sha256": fp.get("fingerprint_sha256"),
        "backbone_sha256": backbone_sha_before,
        "commit": commit_now,
        "git": git_now,
    }
    gate_doc = {
        "run_id": run_id,
        "tag": tag,
        "arm": arm,
        "enforce_b": bool(enforce_b),
        "gates": gates,
        "hard_pass": bool(passed),
    }
    (run_path / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "config.json").write_text(json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "gate_report.json").write_text(json.dumps(gate_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[ens-{arm}] 完成：run_id={run_id} AUC-Val-BSI(best)={best_auc:.4f} AUC-Test-BSI={test_auc:.4f} "
        f"hard_pass={passed} wall={metrics_doc['wall_seconds']}s（SUMMARY 未写，待独立复核）")
    return {
        "run_id": run_id,
        "run_dir": str(run_path),
        "arm": arm,
        "gates": gates,
        "metrics": metrics_doc,
        "hard_pass": bool(passed),
    }


# ---- 独立解析器：从 raw 预测/标签/配置/checkpoint 重算（spec：正式结果独立核验）----
# 真实性守卫：不采信任何单方自述——身份字段三方（config/metrics/predictions）逐项互证；
# raw 载荷长度/有限性/[0,1] 范围/成员均值核验；产物树锚定（前缀指纹文件自哈希、Stage-1
# meta、backbone.pt 字节级重算）。检查项三态：True 通过 / False 失败 / None 不适用
# （仅限产物树不可定位、或未被声明的可选产物）。
_CROSS_ARM_IDENTITY_FIELDS = (
    # 跨臂必须逐项一致的共享标识（run_id/arm 各臂天然不同，不在此列）
    "tag", "model_seed", "prefix_tag", "stage1_id", "commit", "git_dirty",
    "fingerprint_sha256", "backbone_sha256", "budgets", "epochs", "patience",
)


def _canon(value) -> str:
    return protocol.canonical_json(value)


def _safe(fn) -> bool:
    """任一异常判 False：篡改/损坏的载荷不允许把解析器打挂，也不允许静默通过。"""
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def _safe_value(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _git_dirty(doc):
    git = doc.get("git")
    return git.get("dirty") if isinstance(git, dict) else None


def _three_way_equal(docs, field) -> bool:
    """字段在三份产物中都存在且规范序列化后等价；缺失/None 一律判 False（严格模式）。"""
    values = [doc.get(field) for doc in docs]
    if any(v is None for v in values):
        return False
    return len({_canon(v) for v in values}) == 1


def _commit_block_ok(doc) -> bool:
    git = doc.get("git") if isinstance(doc.get("git"), dict) else {}
    commit = doc.get("commit")
    return commit is not None and commit == git.get("commit")


def _hard_pass_recompute(gate_doc) -> bool:
    """由 gate_report 逐门禁 verdict 独立重算 hard_pass（不采信其 hard_pass 字段）。"""
    verdicts = gate_doc.get("gates") if isinstance(gate_doc.get("gates"), dict) else {}
    a_gates = {k: v for k, v in verdicts.items() if str(k).startswith("A")}
    b_gates = {k: v for k, v in verdicts.items() if str(k).startswith("B")}
    return metrics.hard_pass(a_gates, enforce_b=bool(gate_doc.get("enforce_b")), b_gates=b_gates)


def _numel_or_none(t):
    return int(t.numel()) if isinstance(t, torch.Tensor) else None


def _is_vec(t, n) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 1 and int(t.numel()) == int(n)


def _members_list(members):
    return list(members) if isinstance(members, (list, tuple)) else []


def _arrays_aligned(labels, arm_probs, members) -> bool:
    """arm/成员概率必须与标签同长、一维，且成员列表非空。"""
    return _safe(
        lambda: isinstance(labels, torch.Tensor) and labels.dim() == 1
        and _is_vec(arm_probs, labels.numel())
        and len(_members_list(members)) >= 1
        and all(_is_vec(m, labels.numel()) for m in _members_list(members))
    )


def _probs_finite(arm_probs, members) -> bool:
    ms = _members_list(members)
    return _safe(
        lambda: bool(ms) and isinstance(arm_probs, torch.Tensor)
        and bool(torch.isfinite(arm_probs).all())
        and all(bool(torch.isfinite(m).all()) for m in ms)
    )


def _probs_in_unit_range(arm_probs, members) -> bool:
    ms = _members_list(members)
    return _safe(
        lambda: bool(ms) and isinstance(arm_probs, torch.Tensor)
        and bool(((arm_probs >= 0) & (arm_probs <= 1)).all())
        and all(bool(((m >= 0) & (m <= 1)).all()) for m in ms)
    )


def _member_count_matches(members, arm) -> bool:
    if arm not in ARM_RANKS:
        return False
    return len(_members_list(members)) == (len(ARM_RANKS[arm]) or 1)


def _mean_matches_arm(arm_probs, members) -> bool:
    """臂输出必须逐位等于成员概率的算术平均（B/C 单成员退化；与 run_arm 落盘同式计算）。"""
    ms = _members_list(members)
    if not ms or not isinstance(arm_probs, torch.Tensor):
        return False
    return _safe(lambda: bool(torch.equal(arm_probs, torch.stack(ms, dim=0).mean(dim=0))))


def _artifact_root(run_dir: Path):
    """仅当 run_dir 位于 <root>/runs/<run_id>（protocol.run_dir 布局）下才返回 root；
    此时锚定检查为强制项（文件缺失即 False）。其他位置返回 None（锚定项记 None，不适用）。"""
    parent = run_dir.parent
    return parent.parent if parent.name == "runs" else None


def _fingerprint_anchor_ok(fp_path: Path, declared_prefix_tag, declared_sha, declared_budgets) -> bool:
    if not fp_path.exists():
        return False
    fp = json.loads(fp_path.read_text(encoding="utf-8"))
    return (
        fp.get("fingerprint_sha256") == protocol.fingerprint_digest(fp)  # 文件自哈希（防字段编辑）
        and declared_sha is not None
        and fp.get("fingerprint_sha256") == declared_sha
        and fp.get("prefix_tag") == declared_prefix_tag
        and fp.get("budgets") == declared_budgets
    )


def _stage1_meta_ok(s1_dir: Path, config, predictions) -> bool:
    meta_path = s1_dir / "meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        meta.get("stage1_id") == config.get("stage1_id") == predictions.get("stage1_id")
        and meta.get("prefix_tag") == config.get("prefix_tag")
        and meta.get("model_seed") == config.get("model_seed")
        and meta.get("budgets") == config.get("budgets")
        and meta.get("fingerprint_sha256") == config.get("fingerprint_sha256")
    )


def _backbone_recompute_ok(s1_dir: Path, declared_sha) -> bool:
    """由 Stage-1 backbone.pt 字节流重建模型并重算 backbone SHA（架构取自 meta.config，
    与 bench.run_stage1 的构造一致）；与声明值及 meta 记录比对——自述一致也逃不过字节级重算。"""
    if declared_sha is None:
        return False
    meta_path = s1_dir / "meta.json"
    backbone_path = s1_dir / "backbone.pt"
    if not meta_path.exists() or not backbone_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta.get("config") or {}
    model = MPTRec(
        num_tasks=int(cfg["num_tasks"]),
        feature_vocabulary={str(k): int(v) for k, v in (cfg.get("vocab") or {}).items()},
        embedding_size=int(cfg["embedding_size"]),
        input_size=int(cfg["input_size"]),
        expert_dnn_hidden_units=[int(h) for h in cfg["expert_hidden"]],
        tower_dnn_hidden_units=[int(h) for h in cfg["tower_hidden"]],
        dropout=[float(d) for d in cfg["dropout"]],
        reg_embedding=float(cfg["reg_embedding"]),
        reg_dnn=float(cfg["reg_dnn"]),
        device=torch.device("cpu"),
    )
    model.load_state_dict(torch.load(backbone_path, map_location="cpu"))
    return protocol.backbone_sha256(model) == declared_sha and meta.get("backbone_sha256") == declared_sha


def rederive_arm(run_dir) -> dict:
    """独立重算 + 真实性守卫（见本节顶部说明）。所有检查项 True/False/None 三态。"""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics_doc = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    gate_doc = json.loads((run_dir / "gate_report.json").read_text(encoding="utf-8"))
    predictions = torch.load(run_dir / "predictions.pt", map_location="cpu")

    docs3 = (config, metrics_doc, predictions)  # 三份互证产物
    docs4 = (config, metrics_doc, predictions, gate_doc)
    identity = {
        "run_id": predictions.get("run_id"),
        "arm": predictions.get("arm"),
        "tag": predictions.get("tag"),
        "model_seed": predictions.get("model_seed"),
        "prefix_tag": predictions.get("prefix_tag"),
        "stage1_id": predictions.get("stage1_id"),
        "commit": predictions.get("commit"),
        "git_dirty": _git_dirty(predictions),
        "fingerprint_sha256": predictions.get("fingerprint_sha256"),
        "backbone_sha256": predictions.get("backbone_sha256"),
        "budgets": predictions.get("budgets"),
        "epochs": predictions.get("epochs"),
        "patience": predictions.get("patience"),
    }

    # ---- raw 载荷字段（防错配：长度、有限性、范围、成员结构、均值自洽）----
    pred_test = predictions.get("test") if isinstance(predictions.get("test"), dict) else {}
    pred_val = predictions.get("val") if isinstance(predictions.get("val"), dict) else {}
    budgets = config.get("budgets") if isinstance(config.get("budgets"), dict) else {}
    test_labels = pred_test.get("labels")
    val_labels = pred_val.get("labels")
    test_members = pred_test.get("members")
    val_epochs_raw = pred_val.get("epochs")
    val_epochs = (
        [r if isinstance(r, dict) else {} for r in val_epochs_raw] if isinstance(val_epochs_raw, list) else []
    )
    n_test = _numel_or_none(test_labels)
    n_val = _numel_or_none(val_labels)

    # ---- 独立重算（异常一律记 None，不中断解析器）----
    auc_test = _safe_value(lambda: metrics.auc_score(test_labels, pred_test.get("arm")))
    val_aucs = [
        _safe_value(lambda: metrics.auc_score(val_labels, record.get("arm"))) for record in val_epochs
    ]
    val_aucs_complete = bool(val_aucs) and all(v is not None for v in val_aucs)
    best_epoch = int(max(range(len(val_aucs)), key=lambda i: val_aucs[i])) + 1 if val_aucs_complete else None
    best_val_auc = max(val_aucs) if val_aucs_complete else None
    mechanism = _safe_value(lambda: mechanism_stats(test_members))

    state_sha = metrics_doc.get("newtask_state_sha256")
    state_path = run_dir / "newtask.pt"
    if state_sha is None:
        state_sha_check = None  # 未声明的可选产物
    elif not state_path.exists():
        state_sha_check = False  # 声明了却缺失 ⇒ 失败
    else:
        state_sha_check = file_sha256(state_path) == state_sha

    # ---- 产物树锚定（root 可定位时为强制项）----
    root = _artifact_root(run_dir)
    fingerprint_anchor = None
    stage1_meta_check = None
    backbone_recompute_check = None
    if root is not None:
        prefix_tag = config.get("prefix_tag")
        stage1_id = config.get("stage1_id")
        declared_fp = predictions.get("fingerprint_sha256")
        declared_bb = predictions.get("backbone_sha256")
        if isinstance(prefix_tag, str):
            fp_path = protocol.splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
            fingerprint_anchor = _safe(
                lambda: _fingerprint_anchor_ok(fp_path, prefix_tag, declared_fp, budgets)
            )
        else:
            fingerprint_anchor = False
        if isinstance(stage1_id, str):
            s1_dir = protocol.stage1_dir(root, stage1_id)
            stage1_meta_check = _safe(lambda: _stage1_meta_ok(s1_dir, config, predictions))
            backbone_recompute_check = _safe(lambda: _backbone_recompute_ok(s1_dir, declared_bb))
        else:
            stage1_meta_check = False
            backbone_recompute_check = False

    backbone_vals = [
        config.get("backbone_sha256"),
        metrics_doc.get("backbone_sha256_loaded"),
        metrics_doc.get("backbone_sha256_before"),
        metrics_doc.get("backbone_sha256_after"),
        predictions.get("backbone_sha256"),
    ]
    arm_values = [d.get("arm") for d in docs4]
    checks = {
        # ---- 身份：config / metrics / predictions 三方逐项互证（缺失即失败）----
        "run_id_consistent": _three_way_equal(docs4, "run_id"),
        "tag_consistent": _three_way_equal(docs4, "tag"),
        "arm_consistent": _safe(lambda: len(set(arm_values)) == 1 and all(v in ARM_RANKS for v in arm_values)),
        "model_seed_consistent": _three_way_equal(docs3, "model_seed"),
        "prefix_tag_consistent": _three_way_equal(docs3, "prefix_tag"),
        "stage1_id_consistent": _three_way_equal(docs3, "stage1_id"),
        "commit_consistent": _three_way_equal(docs3, "commit") and all(_commit_block_ok(d) for d in docs3),
        "git_clean_all": all(_git_dirty(d) is False for d in docs3),
        "fingerprint_sha256_consistent": _three_way_equal(docs3, "fingerprint_sha256"),
        "backbone_sha256_consistent": all(v is not None for v in backbone_vals)
        and len({_canon(v) for v in backbone_vals}) == 1,
        "budgets_consistent": _three_way_equal(docs3, "budgets"),
        "epochs_consistent": _three_way_equal(docs3, "epochs")
        and isinstance(config.get("epochs"), int)
        and config["epochs"] >= 1,
        "patience_consistent": _three_way_equal(docs3, "patience")
        and isinstance(config.get("patience"), int)
        and config["patience"] >= 1,
        "hard_pass_consistent": _safe(
            lambda: gate_doc.get("hard_pass") in (True, False)
            and gate_doc.get("hard_pass") == metrics_doc.get("hard_pass")
            and gate_doc.get("hard_pass") == _hard_pass_recompute(gate_doc)
        ),
        # ---- 独立重算与 metrics.json 对照 ----
        "auc_matches_metrics_json": _safe(
            lambda: auc_test is not None and abs(auc_test - float(metrics_doc["test_auc_bsi"])) <= 1e-9
        ),
        "val_aucs_match": _safe(
            lambda: val_aucs_complete
            and len(val_aucs) == len(metrics_doc["per_epoch"])
            and all(abs(a - float(r["val_auc_bsi"])) <= 1e-9 for a, r in zip(val_aucs, metrics_doc["per_epoch"]))
        ),
        "best_epoch_matches": best_epoch is not None and best_epoch == metrics_doc.get("best_epoch"),
        "val_best_matches": _safe(
            lambda: best_val_auc is not None
            and abs(best_val_auc - float(metrics_doc["best_val_auc_bsi"])) <= 1e-9
        ),
        "epoch_records_wellformed": _safe(
            lambda: isinstance(metrics_doc.get("per_epoch"), list)
            and len(metrics_doc["per_epoch"]) >= 1
            and [r.get("epoch") for r in metrics_doc["per_epoch"]]
            == list(range(1, len(metrics_doc["per_epoch"]) + 1))
            and isinstance(metrics_doc.get("epochs"), int)
            and len(metrics_doc["per_epoch"]) <= metrics_doc["epochs"]
            and isinstance(metrics_doc.get("best_epoch"), int)
            and 1 <= metrics_doc["best_epoch"] <= len(metrics_doc["per_epoch"])
        ),
        "state_sha_matches": state_sha_check,
        # ---- raw 载荷：长度 / 二值 / 对齐 / 有限 / 范围 / 成员结构 / 均值自洽 ----
        "n_test_matches_budget": _safe(
            lambda: isinstance(test_labels, torch.Tensor)
            and budgets.get("test") is not None
            and int(test_labels.numel()) == int(budgets["test"])
        ),
        "n_val_matches_budget": _safe(
            lambda: isinstance(val_labels, torch.Tensor)
            and budgets.get("val") is not None
            and int(val_labels.numel()) == int(budgets["val"])
        ),
        "test_labels_binary": _safe(
            lambda: isinstance(test_labels, torch.Tensor)
            and test_labels.dim() == 1
            and bool(((test_labels == 0) | (test_labels == 1)).all())
        ),
        "val_labels_binary": _safe(
            lambda: isinstance(val_labels, torch.Tensor)
            and val_labels.dim() == 1
            and bool(((val_labels == 0) | (val_labels == 1)).all())
        ),
        "test_arrays_aligned": _arrays_aligned(test_labels, pred_test.get("arm"), test_members),
        "val_epochs_arrays_aligned": bool(val_epochs)
        and all(_arrays_aligned(val_labels, r.get("arm"), r.get("members")) for r in val_epochs),
        "test_probs_finite": _probs_finite(pred_test.get("arm"), test_members),
        "val_probs_finite": bool(val_epochs)
        and all(_probs_finite(r.get("arm"), r.get("members")) for r in val_epochs),
        "test_probs_in_unit_range": _probs_in_unit_range(pred_test.get("arm"), test_members),
        "val_probs_in_unit_range": bool(val_epochs)
        and all(_probs_in_unit_range(r.get("arm"), r.get("members")) for r in val_epochs),
        "member_count_matches_arm": _safe(
            lambda: bool(val_epochs)
            and _member_count_matches(test_members, predictions.get("arm"))
            and all(_member_count_matches(r.get("members"), predictions.get("arm")) for r in val_epochs)
        ),
        "test_arm_equals_member_mean": _mean_matches_arm(pred_test.get("arm"), test_members),
        "val_arms_equal_member_mean": bool(val_epochs)
        and all(_mean_matches_arm(r.get("arm"), r.get("members")) for r in val_epochs),
        "val_epoch_count_matches_per_epoch": _safe(
            lambda: bool(val_epochs)
            and isinstance(metrics_doc.get("per_epoch"), list)
            and len(val_epochs) == len(metrics_doc["per_epoch"])
        ),
        # ---- 产物树锚定（root 可定位时为强制项；None=布局不可定位，不适用）----
        "fingerprint_anchor_matches": fingerprint_anchor,
        "stage1_meta_matches": stage1_meta_check,
        "backbone_recompute_matches": backbone_recompute_check,
        # ---- 门禁自述 ----
        "gate_hard_pass": bool(gate_doc.get("hard_pass")),
    }
    applicable = [v for v in checks.values() if v is not None]
    return {
        "run_dir": str(run_dir),
        "run_id": metrics_doc.get("run_id"),
        "arm": predictions.get("arm"),
        "stage1_id": predictions.get("stage1_id"),
        "identity": identity,
        "artifact_root": str(root) if root is not None else None,
        "n_test": n_test,
        "n_val": n_val,
        "test_labels_sha256": _safe_value(lambda: protocol.sha256_tensor(test_labels)),
        "val_labels_sha256": _safe_value(lambda: protocol.sha256_tensor(val_labels)),
        "auc_test_bsi": auc_test,
        "val_aucs": val_aucs,
        "best_epoch": best_epoch,
        "best_val_auc_bsi": best_val_auc,
        "mechanism": mechanism,
        "commit": predictions.get("commit"),
        "wall_seconds": metrics_doc.get("wall_seconds"),
        "checks": checks,
        "all_ok": bool(applicable) and all(applicable),
    }


def _difference(left, right):
    return None if left is None or right is None else left - right


def compare_arms(run_dirs) -> dict:
    """三臂配对核验：逐臂独立重算 + 跨臂身份一致 + test/val 标签逐项相同 + 配对差值 + 结局分类。

    跨臂必须逐项一致的共享标识见 _CROSS_ARM_IDENTITY_FIELDS（run_id/arm 各臂天然不同，
    不在其列，仅做单臂内四方一致核验）；标签一致性按逐项字节哈希（含长度）核验，防
    「同长度不同内容」错配。不写任何文件，由调用方决定呈现。
    """
    reports = {}
    for run_dir in run_dirs:
        report = rederive_arm(run_dir)
        if report["arm"] in reports:
            raise ValueError(f"重复臂: {report['arm']}")
        reports[report["arm"]] = report

    identity_checks, label_checks = {}, {}
    if reports:
        for field in _CROSS_ARM_IDENTITY_FIELDS:
            values = [report["identity"].get(field) for report in reports.values()]
            identity_checks[field] = all(v is not None for v in values) and len({_canon(v) for v in values}) == 1
        test_shas = [report["test_labels_sha256"] for report in reports.values()]
        val_shas = [report["val_labels_sha256"] for report in reports.values()]
        n_tests = [report["n_test"] for report in reports.values()]
        n_vals = [report["n_val"] for report in reports.values()]
        label_checks = {
            "test_labels_identical": all(s is not None for s in test_shas) and len(set(test_shas)) == 1,
            "val_labels_identical": all(s is not None for s in val_shas) and len(set(val_shas)) == 1,
            "test_lengths_equal": all(v is not None for v in n_tests) and len(set(n_tests)) == 1,
            "val_lengths_equal": all(v is not None for v in n_vals) and len(set(n_vals)) == 1,
        }
    identifiers_consistent = bool(identity_checks) and all(identity_checks.values())
    labels_consistent = bool(label_checks) and all(label_checks.values())

    stage1_ids = {report["identity"]["stage1_id"] for report in reports.values()}
    stage1_consistent = len(stage1_ids) == 1 and None not in stage1_ids
    paired = set(reports) == {"B", "C", "E"}
    deltas = {}
    deltas_complete = False
    if paired:
        for left, right in (("E", "B"), ("E", "C"), ("C", "B")):
            deltas[f"{left}-{right}"] = {
                "test_auc_bsi": _difference(reports[left]["auc_test_bsi"], reports[right]["auc_test_bsi"]),
                "val_auc_bsi": _difference(reports[left]["best_val_auc_bsi"], reports[right]["best_val_auc_bsi"]),
            }
        deltas_complete = all(
            d["test_auc_bsi"] is not None and d["val_auc_bsi"] is not None for d in deltas.values()
        )
    if paired and deltas_complete:
        e_mechanism = reports["E"]["mechanism"]
        collapsed = e_mechanism["collapsed"] if e_mechanism else None
        val_ok = deltas["E-B"]["val_auc_bsi"] > 0 and deltas["E-C"]["val_auc_bsi"] > 0
        gates_ok = (
            stage1_consistent
            and identifiers_consistent
            and labels_consistent
            and all(report["all_ok"] for report in reports.values())
        )
        screen = screen_verdict(
            delta_eb=deltas["E-B"]["test_auc_bsi"],
            delta_ec=deltas["E-C"]["test_auc_bsi"],
            val_direction_consistent=val_ok,
            collapsed=collapsed,
            gates_ok=gates_ok,
        )
    else:
        reason = (
            "缺少 B/C/E 三臂之一或臂名不齐"
            if not paired
            else "差值不可用（raw 载荷或 metrics 重算失败，不得据此判定）"
        )
        screen = {
            "classification": "INCOMPLETE",
            "expansion_eligible": False,
            "conditions": {},
            "reason": reason,
        }
    return {
        "paired": paired,
        "stage1_consistent": stage1_consistent,
        "stage1_ids": sorted(str(s) for s in stage1_ids),
        "identifiers_consistent": identifiers_consistent,
        "identity_checks": identity_checks,
        "labels_consistent": labels_consistent,
        "label_checks": label_checks,
        "arms": reports,
        "deltas": deltas,
        "screen": screen,
    }
