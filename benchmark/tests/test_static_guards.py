"""静态守卫：gitignore 覆盖、master 模型文件零改动、共享模块无数据集专名、无 FLOPs 依赖。

数据集适配器（datasets/*.py）允许出现数据集专名；pipeline.py 是纯编排层，不得出现任何数据集
专名或数据集 schema 词（verdict / overall_pass / split_seed / ctr / cvr / bsi）。
gates.py 例外说明：pass / verdict 是两种输出 schema 的字段名（renderer 直出），不算数据集专名，
但它同样不得出现数据集名字。
"""
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def _ignored(rel_path):
    """git check-ignore 返回 0 表示被忽略。"""
    return subprocess.run(["git", "check-ignore", "-q", rel_path], cwd=REPO).returncode == 0


class TestStaticGuards(unittest.TestCase):
    def test_gitignore_blocks_artifacts_and_keeps_summary(self):
        for rel in ("artifacts/census_stage2/stage1", "artifacts/census_stage2/runs",
                    "artifacts/census_stage2/splits",
                    "artifacts/census_stage2/stage1/s1-deadbeef-m1-e2-cafe0000/backbone.pt",
                    "artifacts/census_stage2/runs/20260929-1530-s20260929-m1685480945-short-fd75198/metrics.json",
                    "artifacts/aliccp_bench/stage1", "artifacts/aliccp_bench/runs",
                    "artifacts/aliccp_bench/splits",
                    "artifacts/aliccp_bench/stage1/s1-deadbeef-m1-e1-cafe0000/backbone.pt",
                    "artifacts/aliccp_bench/runs/20261009-1530-p2M-v500k-t1M-m1688723512-short-fd75198/metrics.json",
                    "artifacts/aliccp_bench/logs/20261009-1530-aliccp-stage1-smoke.log"):
            self.assertTrue(_ignored(rel), f"应被忽略: {rel}")
        self.assertFalse(_ignored("artifacts/census_stage2/SUMMARY.md"))     # SUMMARY 必须能入库
        self.assertFalse(_ignored("artifacts/aliccp_bench/SUMMARY.md"))

    def test_master_model_files_untouched(self):
        diff = _git("diff", "--name-only", "master", "--", "multitaskrec", "config.py",
                    "CensusIncome_MPTRec.py", "CensusIncome_NewTask.py").stdout.strip()
        self.assertEqual(diff, "", f"协议分支不得改动模型/master 文件: {diff}")

    def test_no_flops_and_split_seed_locked(self):
        cli_src = (REPO / "run_benchmark.py").read_text(encoding="utf-8")
        protocol_src = (REPO / "benchmark" / "protocol.py").read_text(encoding="utf-8")
        census_src = (REPO / "benchmark" / "datasets" / "census.py").read_text(encoding="utf-8")
        for src in (cli_src, protocol_src):
            self.assertNotIn("fvcore", src)                         # 不做 FLOPs
        self.assertIn("random_state=split_seed", census_src)        # 划分只由 split seed 决定
        self.assertNotIn("random_state=model_seed", census_src)

    def test_pipeline_free_of_dataset_tokens(self):
        src = (REPO / "benchmark" / "pipeline.py").read_text(encoding="utf-8").lower()
        for token in ("census", "aliccp", "education", "income", "marital", "verdict",
                      "overall_pass", "split_seed", "prefix_tag"):
            self.assertNotIn(token, src, f"pipeline.py 不得出现数据集专名: {token}")
        for token in ("ctr", "cvr", "bsi"):      # 短词按词边界，避免 MPTRecTrain 之类误伤
            self.assertIsNone(re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", src),
                              f"pipeline.py 不得出现数据集专名: {token}")

    def test_gates_free_of_dataset_names(self):
        src = (REPO / "benchmark" / "gates.py").read_text(encoding="utf-8").lower()
        for token in ("census", "aliccp", "education", "income", "marital"):
            self.assertNotIn(token, src, f"gates.py 不得出现数据集专名: {token}")


if __name__ == "__main__":
    unittest.main()
