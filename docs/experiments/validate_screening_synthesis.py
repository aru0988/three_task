#!/usr/bin/env python
"""Read-only validator for the MPT-Rec screening synthesis deliverables.

Checks (all against committed git objects; no working-tree assumptions, no network):

  V0  JSON parses; both deliverable files exist; entry count sanity.
  V1  Every branch exists; every tip SHA equals the real branch tip; every
      result-commit SHA exists in the object database.
  V2  For every entry: the branch-tip SUMMARY.md and spec document exist, their
      blob hashes equal the pinned values, and every listed run id appears in
      the SUMMARY text.
  V3  Delta arithmetic: recompute arm - baseline from the full-precision AUCs
      recorded in the JSON and compare with the documented delta string within
      its printed precision (tolerance = max(10^-decimals, 1e-12)).
  V4  Append-only discipline: for result branches sharing their lineage base's
      SUMMARY file, every base data row appears unchanged and in order in the
      branch-tip SUMMARY.
  V5  MD<->JSON consistency: the set of branches parsed from the MD evidence
      tables equals the JSON entry set; each entry's classification token
      appears in the MD.
  V6  Spot-checks: each arm AUC value's 12-char prefix appears in the branch-tip
      spec document or in one of the entry's result-commit messages
      (skipped for baseline entries and entries without an arm AUC).

Usage:  <python> docs/experiments/validate_screening_synthesis.py
Exit code 0 iff every check passes.

Provenance note: this script reads committed objects only (git show / rev-parse
/ log). It deliberately does not touch artifacts/, run directories, or any
working-tree file other than the two deliverables it validates.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # docs/experiments -> repo root
JSON_PATH = HERE / "2026-10-04-mptrec-screening-synthesis.json"
MD_PATH = HERE / "2026-10-04-mptrec-screening-synthesis.md"

BASE_BRANCH = {
    "census": "infra/fair-stage2-benchmark",
    "aliccp": "infra/aliccp-fair-benchmark",
}

CHECKS: list[tuple[str, bool, str]] = []


def check(cid: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((cid, bool(ok), detail))


def git(*args: str) -> tuple[int, str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return out.returncode, out.stdout


def is_separator(line: str) -> bool:
    return set(line.strip()) <= set("|-: ")


def summary_rows(text: str) -> tuple[str, list[str]]:
    """Return (header_line, data_rows) of a SUMMARY markdown table."""
    table_lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    header = table_lines[0] if table_lines else ""
    rows = [
        ln
        for ln in table_lines
        if not is_separator(ln) and "run_id" not in ln[:12]
    ]
    return header, rows


def decimals_of(s: str) -> int:
    s = s.strip()
    if "." not in s:
        return 0
    frac = s.split(".", 1)[1]
    frac = re.sub(r"[^0-9]", "", frac)
    return len(frac)


def main() -> int:
    if not JSON_PATH.exists():
        print(f"FAIL: missing {JSON_PATH}")
        return 1
    if not MD_PATH.exists():
        print(f"FAIL: missing {MD_PATH}")
        return 1
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    entries = data["entries"]
    md_text = MD_PATH.read_text(encoding="utf-8")

    check("V0.entry_count", len(entries) == 23, f"entries={len(entries)} expected=23")

    branches_in_json = {e["branch"] for e in entries}
    check("V0.branch_unique", len(branches_in_json) == len(entries), f"{len(branches_in_json)} unique")

    # ---- V1: refs -----------------------------------------------------------
    for e in entries:
        b = e["branch"]
        rc, out = git("rev-parse", "--verify", f"refs/heads/{b}")
        if rc != 0:
            check(f"V1.branch.{b}", False, "branch missing")
            continue
        check(f"V1.branch.{b}", True)
        check(f"V1.tip.{b}", out.strip() == e["tip_commit"], f"{out.strip()} vs {e['tip_commit']}")
        for c in e["result_commits"]:
            rc2, _ = git("cat-file", "-e", c["sha"])
            check(f"V1.commit.{b}.{c['sha'][:7]}", rc2 == 0, "commit missing")

    # ---- V2: SUMMARY rows + blobs + specs ----------------------------------
    summary_cache: dict[str, str] = {}
    for e in entries:
        b = e["branch"]
        sf = e["summary_file"]
        key = f"{b}:{sf}"
        rc, text = git("show", key)
        summary_cache[key] = text
        check(f"V2.summary.{b}", rc == 0, "SUMMARY unreadable")
        if rc != 0:
            continue
        rc, blob = git("rev-parse", key)
        check(f"V2.blob.{b}", blob.strip() == e["summary_blob"], f"{blob.strip()} vs {e['summary_blob']}")
        for rid in e["summary_rows"]:
            check(f"V2.row.{b}.{rid[:16]}", rid in text, "row not found in SUMMARY")
        rc, spec_blob = git("rev-parse", f"{b}:{e['spec_path']}")
        check(f"V2.spec.{b}", rc == 0 and spec_blob.strip() == e["spec_blob"],
              f"{spec_blob.strip()} vs {e['spec_blob']}")

    # ---- V3: delta arithmetic ----------------------------------------------
    for e in entries:
        u = e["utility"]
        a, base, d = u.get("arm_auc_full"), u.get("baseline_auc_full"), u.get("delta_reported")
        if not (a and base and d):
            continue
        try:
            recomputed = Decimal(a) - Decimal(base)
            reported = Decimal(d.lstrip("+"))
        except InvalidOperation:
            check(f"V3.delta.{e['id']}", False, "unparsable numeric string")
            continue
        tol = max(Decimal(10) ** (-decimals_of(d)), Decimal("1e-12"))
        ok = abs(recomputed - reported) <= tol
        check(f"V3.delta.{e['id']}", ok,
              f"recomputed {recomputed} vs reported {reported} (tol {tol})")

    # ---- V4: append-only ----------------------------------------------------
    base_keys = {}
    for e in entries:
        if e["classification"].startswith("BASELINE"):
            base_keys[e["lineage"].split()[0]] = f"{e['branch']}:{e['summary_file']}"
    for e in entries:
        if e["classification"].startswith("BASELINE"):
            continue
        lineage = e["lineage"].split()[0]
        base_key = base_keys.get(lineage)
        if base_key is None:
            check(f"V4.append.{e['id']}", False, f"no base for lineage {lineage}")
            continue
        base_branch, base_file = base_key.split(":", 1)
        if e["summary_file"] != base_file:
            # different ledger (e.g. the stage-1 gradient audit ledger) - N/A
            continue
        base_text = summary_cache.get(base_key, "")
        tip_text = summary_cache.get(f"{e['branch']}:{e['summary_file']}", "")
        _, base_rows = summary_rows(base_text)
        _, tip_rows = summary_rows(tip_text)
        ptr, ok = 0, True
        for row in base_rows:
            try:
                idx = tip_rows.index(row, ptr)
            except ValueError:
                ok = False
                break
            ptr = idx + 1
        check(f"V4.append.{e['id']}", ok,
              f"base rows not an ordered subsequence of tip rows ({len(base_rows)} base rows)")

    # ---- V5: MD<->JSON consistency -----------------------------------------
    md_branches = set(re.findall(r"^\|\s*`((?:exp|infra)/[^`]+)`", md_text, flags=re.M))
    check("V5.md_branch_set", md_branches == branches_in_json,
          f"md={len(md_branches)} json={len(branches_in_json)} "
          f"only_md={sorted(md_branches - branches_in_json)} only_json={sorted(branches_in_json - md_branches)}")
    for e in entries:
        token = e["classification"].split()[0].rstrip(",;")
        check(f"V5.class_token.{e['id']}", token in md_text, f"token '{token}' missing from MD")

    # ---- V6: arm-value spot-check in spec or commit messages ---------------
    for e in entries:
        if e["classification"].startswith("BASELINE"):
            continue
        arm = e["utility"].get("arm_auc_full")
        if not arm:
            continue
        prefix = arm[:12]
        rc, spec_text = git("show", f"{e['branch']}:{e['spec_path']}")
        found = rc == 0 and prefix in spec_text
        if not found:
            for c in e["result_commits"]:
                rc2, msg = git("log", "-1", "--format=%B", c["sha"])
                if rc2 == 0 and prefix in msg:
                    found = True
                    break
        check(f"V6.armvalue.{e['id']}", found, f"prefix {prefix} not in spec or commit messages")

    # ---- report -------------------------------------------------------------
    failures = [(cid, detail) for cid, ok, detail in CHECKS if not ok]
    total = len(CHECKS)
    for cid, detail in failures:
        print(f"FAIL {cid}: {detail}")
    print(f"checks run: {total}; failed: {len(failures)}")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: ALL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
