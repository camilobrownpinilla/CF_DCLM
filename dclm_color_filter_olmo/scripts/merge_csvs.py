#!/usr/bin/env python3
"""
merge_csvs.py

Usage
-----
python scripts/merge_csvs.py --in-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-10B --out /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-10B/combined.csv.gz --threads 64
python scripts/merge_csvs.py --in-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B --out /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B/combined.csv.gz --threads 64

Requirements
------------
pandas 2.1+  (for pyarrow engine)   pip install --upgrade pandas pyarrow
"""

from __future__ import annotations
import argparse
import gzip
import os
from pathlib import Path
from typing import List, Iterable

import pandas as pd
from joblib import Parallel, delayed   # lightweight parallelism


COLS_EXPECTED = [
    "start",
    "end",
    "id",
    "path",
    "tokens_from_csv",
    "span_length",
    "score",
    "npy_file",
]


def read_one(fp: Path) -> pd.DataFrame:
    """Read one *-scored.csv.gz → DataFrame, enforcing column order/dtypes."""
    try:
        df = pd.read_csv(
            fp,
            compression="gzip",
            header=0,                 # expect header line
            usecols=COLS_EXPECTED,    # ignore stray cols if any
            dtype={
                "start": "int64",
                "end": "int64",
                "id": "string",
                "path": "string",
                "tokens_from_csv": "int32",
                "span_length": "int32",
                "score": "float32",
                "npy_file": "string",
            },
            engine="pyarrow",         # 10–30× faster & lower RAM than C parser
        )
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed reading {fp}") from exc


def merge_and_sort(files: List[Path], n_jobs: int) -> pd.DataFrame:
    """Read → concat → sort all DataFrames."""
    dfs: Iterable[pd.DataFrame]
    if n_jobs == 1:
        dfs = (read_one(f) for f in files)
    else:
        dfs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(read_one)(f) for f in files)

    big = pd.concat(dfs, ignore_index=True, copy=False)
    big.sort_values("score", ascending=False, inplace=True, ignore_index=True)
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Directory with *-scored.csv.gz files")
    ap.add_argument("--out", required=True, help="Output .csv.gz file")
    ap.add_argument("--threads", type=int, default=1, help="Parallel readers (default 1)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*-scored.csv.gz"))
    if not files:
        ap.error(f"No *-scored.csv.gz files found under {in_dir}")

    print(f"[INFO] Found {len(files)} files – reading with {args.threads} thread(s)…")
    big_df = merge_and_sort(files, args.threads)

    print(f"[INFO] Writing {args.out} ({len(big_df):,} rows)…")
    big_df.to_csv(
        args.out,
        index=False,
        compression="gzip",
        header=True,
        na_rep="",
    )
    print("[✓] Done.")


if __name__ == "__main__":
    main()