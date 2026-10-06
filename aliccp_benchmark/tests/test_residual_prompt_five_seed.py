"""守卫：canonical 五 seed 移植的等价性（预注册 §3.2 A1″–A3″）。

事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md
CPU、秒级、不读真实数据集、不需要 GPU。

三类守卫（全部只读）：
1. 接线字节级：`bench.py` / `run_aliccp_benchmark.py` 与 pin 提交 `013e105` 逐字节相同（LF sha256）；
2. 重建等式（文本级）：`residual_prompt.py` 与 `tests/test_residual_prompt.py` == 钉死 blob 文本 +
   恰好替换（机制：docstring / `import os` / 3 个 run-reference 常量行；测试：docstring / `DOC_PATH` /
   `WHITELIST` 块 / `TestPreregConstants` 整类）；每处替换前断言旧文本恰出现 1 次；
3. 运行期 run-reference：未设 env ⇒ 三常量 `None` 且 `arm_verdict` 默认参数路径响亮失败（TypeError，
   绝不静默沿用他 seed 参照）；设 env ⇒ 逐位解析并进入 U 判定与 M0 规则文本。
"""
import hashlib
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PIN_RP_BLOB = "674213f619c5d5039811c71242a7727118348daf"
PIN_RP_LF_SHA256 = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
PIN_BENCH_BLOB = "bf3647686838157bf5fd7645615faad9f2d7185d"
PIN_BENCH_LF_SHA256 = "b063a36667173b66fc8a7ac932502cc34753c91e2bfcfdafaedfce443f1db3b4"
PIN_RUNNER_BLOB = "229c25a1c86719fa6fb6b057d6cff3f4120fa053"
PIN_RUNNER_LF_SHA256 = "2966ec3985e5b91e277f429c2f236fb551de7479d114def90f767f2d813ad285"
PIN_TEST_BLOB = "768a477b5c71af32c5c59ec6feb20c29f7873557"

DOCSTRING_ANCHOR = '"""\nfrom __future__ import annotations\n'
TEST_DOCSTRING_ANCHOR = '"""\nimport ast\n'

NEW_MODULE_DOCSTRING = '''"""AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）——canonical 五 seed 配对验证。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md
移植自 `013e105:aliccp_benchmark/residual_prompt.py`（git blob
`674213f619c5d5039811c71242a7727118348daf`）；相对钉死 blob 恰 3 组适配（预注册 §3.2 A1″–A3″）：
模块 docstring、`import os` 一行、3 个 run-reference 常量行改为运行期解析
（`RP_BASELINE_AUC_TEST` / `RP_BASELINE_AUC_VAL` / `RP_REFERENCE_PRED_STD`；未提供 ⇒ 常量 None ⇒
处理臂在 M0/效用计算处响亮失败，绝不静默沿用他 seed 参照）；其余逐字节一致（守卫测试重建等式 +
`TestMechanismPin` AST 钉死）。机制本体（公式、初始化、门控、注入点）与 seed1 实现（`79ddefa`，
blob `5dc7158ca999c9e7e6217a999c37869231411549`）及 Census 祖本（`99b9510`，blob
`107221b26382da7fd44990e167d9a61da2680d92`）逐字一致。

公式（逐样本；x = `dnn_input`，d = `rep_dim`，三路 h_s = `gen_rep` 与两个 `spec_rep`）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自上下文
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s

结构性事实（测试锁定）：
  * S1 范数受控：ratio_s = ||delta_s|| / ||h_s|| = g_eff ≤ |alpha|（||P|| ≤ sqrt(d)）；
  * S2 跨流一致：同一 x 下三路相对扰动相同，与各流自身范数无关；
  * S3 保守初值：alpha = 0 ⇒ delta ≡ 0 ⇒ 前向与基线逐位一致（共享权重相同时）。

硬约束：不改 `multitaskrec/model.py`（子类扩展）；新增模块构造在 `isolated_cpu_rng()` 内，
共享参数与全局 RNG 端点与基线逐位一致；不主张新颖性（prompt / 适配器 / 条件化调制均为既有方法族）。

用法（接线见 aliccp_benchmark.bench.run_stage2；处理臂每 seed 先由 pin 提供三个 run-reference env 变量）：

    $env:RP_BASELINE_AUC_TEST=<paired baseline test_auc>; $env:RP_BASELINE_AUC_VAL=<paired baseline best_val>;
    $env:RP_REFERENCE_PRED_STD=<reference head val pred_std>; python run_aliccp_benchmark.py stage2
    --stage1-id <sid> --variant residual-prompt --tag short --model-seed <S> --prompt-reference-newtask <baseline_run>/newtask.pt
"""'''

NEW_BASELINE_TEST_LINE = (
    'BASELINE_AUC_TEST = float(os.environ["RP_BASELINE_AUC_TEST"]) if "RP_BASELINE_AUC_TEST" in os.environ else None'
    '  # 五 seed 适配 A3″：配对基线 test（运行期显式提供；未提供=None ⇒ 处理臂响亮失败，绝不静默沿用他 seed 参照）'
)
NEW_BASELINE_VAL_LINE = (
    'BASELINE_AUC_VAL = float(os.environ["RP_BASELINE_AUC_VAL"]) if "RP_BASELINE_AUC_VAL" in os.environ else None'
    '  # 同上（AUC-Val-BSI best；同时充当 M0 期望参照 AUC）'
)
NEW_REFERENCE_STD_LINE = (
    'REFERENCE_PRED_STD = float(os.environ["RP_REFERENCE_PRED_STD"]) if "RP_REFERENCE_PRED_STD" in os.environ else None'
    '  # 参照头 val pred_std（M0/G7；每 seed pin 提供）'
)

NEW_TEST_DOCSTRING = '''"""TDD 测试：AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）——canonical 五 seed 配对验证。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md
CPU、秒级、不读真实数据集（tiny 端到端沿用 test_smoke 夹具口径）。

移植自 `013e105:aliccp_benchmark/tests/test_residual_prompt.py`（git blob
`768a477b5c71af32c5c59ec6feb20c29f7873557`）；相对钉死版恰 5 处分支适配（预注册 §3.2 A4″ 的细化，
见 §13 披露：docstring / 导入机制模块前的 run-reference 示例 env 块 / DOC_PATH / WHITELIST /
TestPreregConstants 整类），由守卫测试 `test_residual_prompt_five_seed.py` 以
"钉死文本 + 恰好替换"重建等式钉死。
"""'''

NEW_DOC_PATH_LINE = 'DOC_PATH = "docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md"'

NEW_WHITELIST_BLOCK = '''WHITELIST = {
    DOC_PATH,
    "aliccp_benchmark/residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt.py",
    "aliccp_benchmark/tests/test_residual_prompt_five_seed.py",
    "aliccp_benchmark/five_seed.py",
    "aliccp_benchmark/tests/test_five_seed.py",
    "run_aliccp_benchmark.py",
    "aliccp_benchmark/bench.py",
    "artifacts/aliccp_bench/SUMMARY.md",
    "verify_reference_residual_prompt_five_seed.py",
    "verify_residual_prompt_five_seed.py",
    "aliccp_benchmark/incremental_utility_router.py",
    "aliccp_benchmark/tests/test_incremental_utility_router.py",
    "run_incremental_utility_router.py",
    "verify_incremental_utility_router.py",
    "docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md",
}
'''

NEW_TEST_ENV_BLOCK = '''from pathlib import Path

import os

# 五 seed 适配 A3″：测试进程在导入机制模块前固定 run-reference 示例值（pin 提交时 seed2 的配对
# 基线/参照值），保证 TestArmVerdict 判定构造与 tiny 端到端（经逐字节 pin 的 bench 接线）在进程内
# 可运行；"未设 env ⇒ 三常量 None ⇒ 处理臂响亮失败"的运行期语义由
# test_residual_prompt_five_seed.py 的 subprocess 守卫覆盖。
os.environ["RP_BASELINE_AUC_TEST"] = "0.5974422649550507"
os.environ["RP_BASELINE_AUC_VAL"] = "0.5809347091990792"
os.environ["RP_REFERENCE_PRED_STD"] = "0.005217193225189258"
'''

NEW_PREREG_CLASS = '''class TestPreregConstants(unittest.TestCase):
    def test_thresholds_and_baseline_reference(self):
        self.assertEqual(RP.PROMPT_HIDDEN, 16)
        self.assertEqual(RP.AUC_TEST_DELTA_MIN, 0.0055)
        self.assertEqual(RP.AUC_VAL_DIRECTION_MIN, 0.0)
        # 五 seed 适配 A3″：run-reference 三常量运行期解析；本测试进程已在导入前固定示例值（见文件头）
        self.assertEqual(RP.BASELINE_AUC_TEST, 0.5974422649550507)
        self.assertEqual(RP.BASELINE_AUC_VAL, 0.5809347091990792)
        self.assertEqual(RP.PRED_STD_MIN_RATIO, 0.5)
        self.assertEqual(RP.REFERENCE_PRED_STD, 0.005217193225189258)
        self.assertEqual(RP.REFERENCE_IDENTITY_TOL, 1e-9)
        self.assertEqual(RP.EXPECTED_NEW_PARAMS_TOTAL, 2385)
        self.assertEqual(RP.EXPECTED_HEAD_PARAMS, 8129)
        self.assertEqual(RP.RATIO_BAND, (0.005, 0.5))
        self.assertEqual(RP.BOUND_TOL, 1e-6)
        self.assertEqual(RP.RUN_ID_SUFFIX, "-rpg")
        self.assertEqual(RP.VARIANTS, ("baseline", "residual-prompt"))

    def test_prereg_doc_tokens_and_summary_ledger(self):
        doc = (REPO / DOC_PATH).read_text(encoding="utf-8")
        for token in ["8133d32", "221580a", "79ddefa", "3e2f083", "e931cd7", "013e105", "a40836e",
                      "10ea6fa", "acba208", "428c76f", "f65412f",
                      PIN_BLOB, PIN_SHA256_LF, PIN_TEST_BLOB,
                      "5dc7158ca999c9e7e6217a999c37869231411549",
                      "674213f619c5d5039811c71242a7727118348daf",
                      "bf3647686838157bf5fd7645615faad9f2d7185d",
                      "229c25a1c86719fa6fb6b057d6cff3f4120fa053",
                      "1688723512", "1688723740", "1688738016", "1688749593", "1688762746",
                      "s1-5c060b9c-m1688723512-e3-3a30e2c0",
                      "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
                      "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9",
                      "20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg",
                      "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07",
                      "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg",
                      "0.005394543882337954", "0.00448174078674779",
                      "0.008103113327467715", "0.008622497456618139",
                      "0.0055", "0.001", "0.02", "2.7764451051977987", "2385", "8129",
                      "VALID_POSITIVE", "VALID_NEGATIVE", "MECHANISM_FAIL", "-rpg",
                      "POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION",
                      "TWENTY_EPOCH_CONDITION_SATISFIED", "TWENTY_EPOCH_CONDITION_NOT_SATISFIED",
                      "RP_BASELINE_AUC_TEST", "RP_BASELINE_AUC_VAL", "RP_REFERENCE_PRED_STD"]:
            self.assertIn(token, doc)
        summary = (REPO / "artifacts" / "aliccp_bench" / "SUMMARY.md").read_text(encoding="utf-8")
        row = [line for line in summary.splitlines() if "b2e17f9" in line and "run_id" not in line]
        self.assertEqual(len(row), 1)
        self.assertIn("0.598839", row[0])
        self.assertIn("0.578153", row[0])'''


def git_blob_text(blob: str) -> str:
    proc = subprocess.run(["git", "cat-file", "blob", blob], cwd=REPO, capture_output=True)
    if proc.returncode != 0:
        raise AssertionError(f"钉死 blob 不可得（系谱依赖，如实响亮失败）: {blob}")
    return proc.stdout.decode("utf-8").replace("\r\n", "\n")


def working_text(rel: str) -> str:
    return (REPO / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")


def lf_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"替换锚点必须恰出现 1 次（实际 {text.count(old)}）: {old[:60]!r}"
    return text.replace(old, new)


def replace_line_once(text: str, name: str, new_line: str) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(name)} = .*$")
    matches = pattern.findall(text)
    assert len(matches) == 1, f"{name} 行必须恰出现 1 次（实际 {len(matches)}）"
    return pattern.sub(lambda _m: new_line, text, count=1)


def replace_block_once(text: str, name: str, new_block: str) -> str:
    pattern = re.compile(rf"(?ms)^{re.escape(name)} = \{{.*?^\}}\n")
    matches = pattern.findall(text)
    assert len(matches) == 1, f"{name} 块必须恰出现 1 次（实际 {len(matches)}）"
    return pattern.sub(lambda _m: new_block, text, count=1)


def replace_prereg_class_once(text: str, new_class: str) -> str:
    pattern = re.compile(r"(?ms)^class TestPreregConstants\(unittest\.TestCase\):.*?(?=\n\nif __name__ == \"__main__\":)")
    matches = pattern.findall(text)
    assert len(matches) == 1, f"TestPreregConstants 整类必须恰出现 1 次（实际 {len(matches)}）"
    return pattern.sub(lambda _m: new_class, text, count=1)


def _split_docstring(pinned: str, anchor: str):
    """按 docstring 锚点切分：返回 (锚点后续首行, 其余全部)，锚点必须恰出现 1 次。"""
    assert pinned.count(anchor) == 1, f"钉死文本必须恰含 1 个 docstring 锚点: {anchor!r}"
    _head, _sep, tail = pinned.partition(anchor)
    first_line = anchor.split("\n", 1)[1]   # 锚点第二行（from __future__… / import ast）必须保留
    return first_line, tail


def rebuild_module_text(pinned: str) -> str:
    first_line, tail = _split_docstring(pinned, DOCSTRING_ANCHOR)
    tail = replace_once(tail, "import math\n", "import math\nimport os\n")
    tail = replace_line_once(tail, "BASELINE_AUC_TEST", NEW_BASELINE_TEST_LINE)
    tail = replace_line_once(tail, "BASELINE_AUC_VAL", NEW_BASELINE_VAL_LINE)
    tail = replace_line_once(tail, "REFERENCE_PRED_STD", NEW_REFERENCE_STD_LINE)
    return NEW_MODULE_DOCSTRING + "\n" + first_line + tail


def rebuild_test_text(pinned: str) -> str:
    first_line, tail = _split_docstring(pinned, TEST_DOCSTRING_ANCHOR)
    tail = replace_once(tail, "from pathlib import Path\n", NEW_TEST_ENV_BLOCK)
    tail = replace_line_once(tail, "DOC_PATH", NEW_DOC_PATH_LINE)
    tail = replace_block_once(tail, "WHITELIST", NEW_WHITELIST_BLOCK)
    tail = replace_prereg_class_once(tail, NEW_PREREG_CLASS)
    return NEW_TEST_DOCSTRING + "\n" + first_line + tail


def run_python(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    for key in ("RP_BASELINE_AUC_TEST", "RP_BASELINE_AUC_VAL", "RP_REFERENCE_PRED_STD"):
        env.pop(key, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True)


class TestWiringBytes(unittest.TestCase):
    def test_bench_and_runner_byte_identical_to_pin(self):
        self.assertEqual(lf_sha256((REPO / "aliccp_benchmark/bench.py").read_bytes()),
                         PIN_BENCH_LF_SHA256, "bench.py 必须与 pin 013e105 逐字节相同")
        self.assertEqual(lf_sha256((REPO / "run_aliccp_benchmark.py").read_bytes()),
                         PIN_RUNNER_LF_SHA256, "run_aliccp_benchmark.py 必须与 pin 013e105 逐字节相同")

    def test_pin_blobs_available(self):
        for blob in (PIN_RP_BLOB, PIN_BENCH_BLOB, PIN_RUNNER_BLOB, PIN_TEST_BLOB):
            self.assertIsInstance(git_blob_text(blob), str)


class TestReconstructionEquality(unittest.TestCase):
    def test_module_reconstruction_equality(self):
        pinned = git_blob_text(PIN_RP_BLOB)
        self.assertEqual(lf_sha256(pinned.encode("utf-8")), PIN_RP_LF_SHA256)
        self.assertEqual(rebuild_module_text(pinned), working_text("aliccp_benchmark/residual_prompt.py"),
                         "机制文件必须 == 钉死 blob + 恰好替换（docstring/import os/3 常量行）")

    def test_ported_test_reconstruction_equality(self):
        pinned = git_blob_text(PIN_TEST_BLOB)
        self.assertEqual(rebuild_test_text(pinned), working_text("aliccp_benchmark/tests/test_residual_prompt.py"),
                         "移植测试文件必须 == 钉死 blob + 恰好替换（docstring/DOC_PATH/WHITELIST/整类）")


class TestRuntimeReferenceConstants(unittest.TestCase):
    def test_unset_env_gives_none(self):
        proc = run_python(
            "import aliccp_benchmark.residual_prompt as RP;"
            "print(RP.BASELINE_AUC_TEST, RP.BASELINE_AUC_VAL, RP.REFERENCE_PRED_STD)"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "None None None")

    def test_set_env_parses_values_bitwise(self):
        proc = run_python(
            "import aliccp_benchmark.residual_prompt as RP;"
            "print(repr(RP.BASELINE_AUC_TEST), repr(RP.BASELINE_AUC_VAL), repr(RP.REFERENCE_PRED_STD))",
            env_extra={"RP_BASELINE_AUC_TEST": "0.5974422649550507",
                       "RP_BASELINE_AUC_VAL": "0.5809347091990792",
                       "RP_REFERENCE_PRED_STD": "0.005217193225189258"},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(),
                         "0.5974422649550507 0.5809347091990792 0.005217193225189258")

    def test_arm_verdict_without_reference_values_fails_loudly(self):
        code = (
            "import aliccp_benchmark.residual_prompt as RP\n"
            "try:\n"
            "    RP.arm_verdict(auc_test=0.6, auc_val=0.6, probe={}, val_stats={}, params={},\n"
            "                   reference={'val_auc': 0.5, 'pred_dispersion': {'pred_std': 0.1}})\n"
            "except TypeError as exc:\n"
            "    print('TYPEERROR', exc)\n"
        )
        proc = run_python(code)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("TYPEERROR"), proc.stdout)

    def test_arm_verdict_uses_runtime_values(self):
        code = (
            "import json\n"
            "import aliccp_benchmark.residual_prompt as RP\n"
            "probe = {'construction': {'shared_params_bit_identical': True, 'global_rng_endpoint_identical': True,\n"
            "                          'extra_keys': sorted(RP.EXTRA_PARAM_NAMES)},\n"
            "         'init_forward': {'bit_identical': True}, 'grad_probe': [], 'alpha_final': 0.0}\n"
            "val_stats = {'streams': {'ratio_max': [0.0, 0.0, 0.0], 'ratio_mean': [0.1, 0.1, 0.1],\n"
            "                         'ratio_std': [0.0, 0.0, 0.0], 'delta_norm_mean': [0.0, 0.0, 0.0],\n"
            "                         'h_norm_mean': [1.0, 1.0, 1.0], 'cos_mean': [0.0, 0.0, 0.0],\n"
            "                         'n_zero_rep': [0, 0, 0]},\n"
            "             'gate': {'geff_mean': 0.1, 'geff_std': 0.0, 'geff_min': 0.0, 'geff_max': 0.1},\n"
            "             'dispersion': {'pred_std': 0.01}}\n"
            "params = {'new_param_list': [{'name': n} for n in RP.EXTRA_PARAM_NAMES],\n"
            "          'new_params_total': 2385, 'head_params': 8129}\n"
            "reference = {'val_auc': 0.5, 'pred_dispersion': {'pred_std': 0.02}}\n"
            "arm = RP.arm_verdict(auc_test=0.6, auc_val=0.6, probe=probe, val_stats=val_stats,\n"
            "                     params=params, reference=reference)\n"
            "print(json.dumps({'baseline_test': arm['U']['U1']['observed']['baseline_auc_test'],\n"
            "                  'delta_test': arm['U']['U1']['observed']['delta_test'],\n"
            "                  'baseline_val': arm['U']['U2']['observed']['baseline_auc_val'],\n"
            "                  'delta_val': arm['U']['U2']['observed']['delta_val'],\n"
            "                  'rule_has_val': '0.31' in arm['M0']['rule']}))\n"
        )
        proc = run_python(code, env_extra={"RP_BASELINE_AUC_TEST": "0.5",
                                           "RP_BASELINE_AUC_VAL": "0.31",
                                           "RP_REFERENCE_PRED_STD": "0.02"})
        self.assertEqual(proc.returncode, 0, proc.stderr)
        import json as _json
        out = _json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["baseline_test"], 0.5)
        self.assertAlmostEqual(out["delta_test"], 0.1, places=12)
        self.assertEqual(out["baseline_val"], 0.31)
        self.assertAlmostEqual(out["delta_val"], 0.29, places=12)
        self.assertTrue(out["rule_has_val"])


if __name__ == "__main__":
    unittest.main()
