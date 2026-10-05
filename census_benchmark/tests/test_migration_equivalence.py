"""迁移等价性守卫：三个从先分支迁移的文件必须与实现提交 `99b9510` 逐字节一致。

唯一事实来源：docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-seed-recheck-design.md §3.4。

主钉子 = git blob SHA-1（`git hash-object` 口径：即"入库内容"的身份，与 core.autocrlf 无关；
本仓库 autocrlf=true，工作树为 CRLF 而 blob 为 LF，hash-object 会走 clean 过滤，故两者可比）。

交叉核对（仅当 `99b9510` 对象在本地对象库时）：`git rev-parse 99b9510:<path>` 必须等于同一钉子。
对象缺失时该项显式 skip（主钉子不降级）；`git fetch origin exp/stage2-residual-prompt-gate` 后可复核。
"""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # 测试依赖 cwd = 仓库根（与既有守卫同口径）

SOURCE_COMMIT = "99b9510"
MIGRATION_PINS = {
    "census_benchmark/residual_prompt.py": "107221b26382da7fd44990e167d9a61da2680d92",
    "census_benchmark/tests/test_residual_prompt.py": "847097b7cb48a56c8593dd76d63a71a21a4c9e4f",
    "run_census_benchmark.py": "e15cad7565d4487f6a61055f195c662f64eb4f15",
}


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


class TestMigrationEquivalence(unittest.TestCase):
    def test_migrated_files_match_prior_impl_blobs(self):
        for rel, pin in MIGRATION_PINS.items():
            with self.subTest(path=rel):
                out = _git("hash-object", rel)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), pin,
                                 f"{rel} 与先分支实现 {SOURCE_COMMIT} 不逐字节一致（迁移漂移）")

    def test_pins_cross_checked_against_prior_commit_when_present(self):
        have = _git("cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}")
        if have.returncode != 0:
            self.skipTest(f"{SOURCE_COMMIT} 不在本地对象库；fetch 先分支后可复核")
        for rel, pin in MIGRATION_PINS.items():
            with self.subTest(path=rel):
                out = _git("rev-parse", f"{SOURCE_COMMIT}:{rel}")
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), pin)


if __name__ == "__main__":
    unittest.main()
