"""3-seed 扩展分支的迁移等价性守卫：六个从 `exp/census-stage2-residual-prompt-seed-recheck`
@ `41feb29` 迁移的文件必须与该分支逐字节一致（blob 主钉子），并提供三层证明：

1. blob 主钉子（`git hash-object` 口径：入库内容身份，与 core.autocrlf 无关；本仓库
   工作树为 CRLF 而 blob 为 LF，hash-object 走 clean 过滤，故两者可比）——始终运行；
2. diff 守卫：`git diff --quiet 41feb29 -- <paths>` 为空——`41feb29` 对象在场时运行；
3. AST 守卫：工作树文件与 `git show 41feb29:<path>` 各自 `ast.parse` 后 `ast.dump` 相等
   ——`41feb29` 对象在场时运行（证明语义等价、不受行尾/格式影响）。

唯一事实来源：docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-3seed-expansion-design.md §3.4。

零改动文件（`87afe03` 基线，blob 钉死）在此一并复核；其 diff 形式的守卫另见迁移文件
`test_residual_prompt.py::TestStaticGuards`（不重复）。
"""
import ast
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根（与既有守卫同口径）

SOURCE_COMMIT = "41feb29"
MIGRATION_PINS = {
    "census_benchmark/residual_prompt.py": "107221b26382da7fd44990e167d9a61da2680d92",
    "census_benchmark/paired_verdict.py": "50469811c2e61e91d4508e0ddb3151a2c08dfeef",
    "run_census_benchmark.py": "e15cad7565d4487f6a61055f195c662f64eb4f15",
    "census_benchmark/tests/test_residual_prompt.py": "847097b7cb48a56c8593dd76d63a71a21a4c9e4f",
    "census_benchmark/tests/test_paired_verdict.py": "773edf5c02ebff1b76e428fbc76493eda91db449",
    "census_benchmark/tests/test_migration_equivalence.py": "2ced4274b7aef1d2d14284b1ba687b287f619490",
}
# 上游协议/模型文件：本分支相对 87afe03 零改动（blob 钉死；spec §8）
ZERO_CHANGE_PINS = {
    "multitaskrec/model.py": "052b8669063af0e65d5ce85a554bc003888ea500",
    "config.py": "d152be8ff2e591bb4a97fd40d23fc0674a3806cf",
    "census_benchmark/protocol.py": "3589c589e5b91e36fde7e38e0aa71ac332cdcb75",
    "census_benchmark/metrics.py": "609504fef0a1eac3673c5f4cd8652b5e2daaaee1",
}


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=text, encoding="utf-8")


class TestMigrationEquivalence3Seed(unittest.TestCase):
    def test_migrated_files_match_recheck_branch_blobs(self):
        for rel, pin in MIGRATION_PINS.items():
            with self.subTest(path=rel):
                out = _git("hash-object", rel)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), pin,
                                 f"{rel} 与迁移源 {SOURCE_COMMIT} 不逐字节一致（迁移漂移）")

    def test_zero_change_files_match_base_blobs(self):
        for rel, pin in ZERO_CHANGE_PINS.items():
            with self.subTest(path=rel):
                out = _git("hash-object", rel)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), pin,
                                 f"{rel} 相对 87afe03 被改动（协议/模型文件零改动约束被违反）")

    def test_diff_guard_working_tree_equals_source_branch(self):
        have = _git("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
        if have.returncode != 0:
            self.skipTest(f"{SOURCE_COMMIT} 不在本地对象库；fetch 迁移源分支后可复核")
        out = _git("diff", "--quiet", SOURCE_COMMIT, "--", *MIGRATION_PINS)
        self.assertEqual(out.returncode, 0,
                         f"工作树与 {SOURCE_COMMIT} 存在差异（diff 守卫失败）: {out.stdout}")

    def test_ast_guard_semantically_equal_to_source_branch(self):
        have = _git("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
        if have.returncode != 0:
            self.skipTest(f"{SOURCE_COMMIT} 不在本地对象库；fetch 迁移源分支后可复核")
        for rel in MIGRATION_PINS:
            with self.subTest(path=rel):
                local = (REPO / rel).read_text(encoding="utf-8")
                src = _git("show", f"{SOURCE_COMMIT}:{rel}").stdout
                self.assertEqual(ast.dump(ast.parse(local)), ast.dump(ast.parse(src)),
                                 f"{rel} 与 {SOURCE_COMMIT} 的 AST 不等价")


if __name__ == "__main__":
    unittest.main()
