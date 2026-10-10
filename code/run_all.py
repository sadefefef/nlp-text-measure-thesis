# -*- coding: utf-8 -*-
"""一键复现：生成语料 -> 抽取测度 -> 运行全部实验。

用法：python run_all.py [--n 3000]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    print(f"\n>>> python {script} {' '.join(args)}")
    ret = subprocess.call([sys.executable, str(HERE / script), *args])
    if ret != 0:
        raise SystemExit(f"{script} 执行失败（exit={ret}）")


def main() -> None:
    n = "3000"
    if "--n" in sys.argv:
        n = sys.argv[sys.argv.index("--n") + 1]
    run("corpus_builder.py", "--n", n)
    run("experiment.py")
    print("\n全部完成：语料 data/reviews.csv，结果 results/")


if __name__ == "__main__":
    main()
