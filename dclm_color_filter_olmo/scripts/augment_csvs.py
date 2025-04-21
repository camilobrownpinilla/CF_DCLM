#!/usr/bin/env python3
"""
augments.py

For every *.csv.gz found under --csv-dir, the script

1.  opens the file (stream‑decompressed with gzip)
2.  reads each row   start,end,id,path,token_count   (no header expected)
3.  looks up <id> in the JSON lines file indicated by <path>
4.  appends the document’s “score” field to that row
5.  writes the result to <original‑name>.scored.csv.gz in the same directory

The JSON file can be plain `.jsonl` or `.jsonl.zst`.  
Processing is streaming and path‑level caches prevent reloading the same JSONL
more than once per CSV, so it scales to very large files.

Usage
-----
python scripts/augment_csvs.py --csv-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-10B --workers 64
python scripts/augment_csvs.py --csv-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B --workers 64

Requires: pandas (for convenience), zstandard (for .zst), and tqdm (progress bars)
"""

import argparse
import csv
import gzip
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd
import zstandard as zstd
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# Helpers for streaming JSONL / JSONL.ZST                                     #
# --------------------------------------------------------------------------- #
def _jsonl_reader(path: Path) -> Iterator[Dict]:
    """Yield JSON objects from .jsonl OR .jsonl.zst."""
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)  # large window OK
        with path.open("rb") as fh:
            with dctx.stream_reader(fh) as reader:
                buf = io.BufferedReader(reader)
                for line in buf:
                    yield json.loads(line)
    else:  # plain .jsonl
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf‑8") as fh:
            for line in fh:
                yield json.loads(line)


def _index_jsonl(path: Path) -> Dict[str, float]:
    """
    Build an in‑memory index id -> score for one JSONL file.

    The JSON objects are assumed to contain keys
        "id"   : string  (matches the CSV's id column)
        "score": float
    """
    mapping: Dict[str, float] = {}
    for obj in _jsonl_reader(path):
        mapping[obj["id"]] = obj["score"]
    return mapping


# --------------------------------------------------------------------------- #
# CSV processing                                                              #
# --------------------------------------------------------------------------- #
def _process_single_csv(csv_path: Path) -> None:
    """
    Create <csv_path>.scored.csv.gz with an added 'score' column.
    """
    base_name = csv_path.name.removesuffix(".csv.gz")
    out_path = csv_path.with_name(f"{base_name}-scored.csv.gz")

    npy_path = csv_path.with_suffix("").with_suffix(".npy")

    with gzip.open(csv_path, "rt", newline="") as fin, \
        gzip.open(out_path, "wt", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # Add new header row
        writer.writerow([
            "start", "end", "id", "path", "tokens_from_csv",
            "span_length", "score", "npy_file"
        ])

        json_cache: Dict[Path, Dict[str, float]] = {}

        for start, end, doc_id, jsonl_path_str, tok_cnt in tqdm(reader,
                                                                desc=csv_path.name,
                                                                unit="rows"):
            jsonl_path = Path(jsonl_path_str)
            if jsonl_path not in json_cache:
                if jsonl_path.is_file():
                    json_cache[jsonl_path] = _index_jsonl(jsonl_path)
                elif jsonl_path.with_suffix(".jsonl.zst").is_file():
                    json_cache[jsonl_path] = _index_jsonl(jsonl_path.with_suffix(".jsonl.zst"))
                elif jsonl_path.with_suffix(".jsonl").is_file():
                    json_cache[jsonl_path] = _index_jsonl(jsonl_path.with_suffix(".jsonl"))
                else:
                    sys.stderr.write(f"[WARN] Cannot find JSONL for {jsonl_path}\n")
                    json_cache[jsonl_path] = {}

            score = json_cache[jsonl_path].get(doc_id)
            try:
                span_length = int(end) - int(start)
            except ValueError:
                span_length = None

            writer.writerow([
                start, end, doc_id, jsonl_path_str, tok_cnt,
                span_length, score, str(npy_path)
            ])

    print(f"[✓] Wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI / multiprocessing                                                       #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", required=True,
                        help="Directory containing *.csv.gz files to augment")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes (1 = disable)")
    args = parser.parse_args()

    csv_files = sorted(Path(args.csv_dir).glob("*.csv.gz"))
    if not csv_files:
        sys.exit("[ERR] No .csv.gz files found!")

    if args.workers == 1:
        for csv_f in csv_files:
            _process_single_csv(csv_f)
    else:
        with mp.Pool(args.workers) as pool:
            list(pool.imap_unordered(_process_single_csv, csv_files))


if __name__ == "__main__":
    main()
