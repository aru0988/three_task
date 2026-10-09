"""CLI 共享工具。tee_stdout 逐字节搬自 run_census_benchmark.py（行为不变）。"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextlib.contextmanager
def tee_stdout(stream):
    """stdout 同时写真实终端与给定文本流（文件或 StringIO）→ 落盘 stdout.log。"""
    class _Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            stream.write(text)
            return len(text)

        def flush(self):
            sys.__stdout__.flush()
            stream.flush()

    original, sys.stdout = sys.stdout, _Tee()
    try:
        yield
    finally:
        sys.stdout = original
