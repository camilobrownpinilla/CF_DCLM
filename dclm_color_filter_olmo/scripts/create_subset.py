#!/usr/bin/env python3
"""
create_subset.py

Create a mem‑mapped OLMo/Dolma dataset from a scored‑and‑sorted CSV.

Supports:
  • --top TOKENS_LIMIT
  • --bottom TOKENS_LIMIT
  • --bottom TOKENS_LIMIT --random TOKENS_LIMIT [--seed S]

Only *one* of --top / --bottom is required.  --random is optional and only
valid together with --bottom.

CSV schema expected (header must exist):

    start,end,id,path,tokens_from_csv,span_length,score,npy_file

Usage:

python scripts/create_subset.py --src-csv /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B/combined.csv.gz --top 600000000 --out-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B-top600M
python scripts/create_subset.py --src-csv /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B/combined.csv.gz --bottom 600000000 --out-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B-bottom600M
python scripts/create_subset.py --src-csv /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B/combined.csv.gz --random 600000000 --out-dir /n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-3B-random600M --seed 1234
"""

from __future__ import annotations
import argparse, csv, gzip, random, sys, collections
from pathlib import Path
from typing import Dict, List, Deque, Tuple
import numpy as np
from tqdm.auto import tqdm

COLS = ["start", "end", "id", "path",
        "tokens_from_csv", "span_length", "score", "npy_file"]
Row = Dict[str, str]

# ---------------------------------------------------------------------- utils
def open_csv(p: Path):
    return gzip.open(p, "rt", newline="")

def n_tokens(row: Row) -> int:
    # prefer tokens_from_csv if present, else span_length
    return int(row["span_length"])

# ------------------------------------------------------------- select helpers
def select_top_rows(csv_path: Path, cap: int) -> List[Row]:
    out, total = [], 0
    with open_csv(csv_path) as fh:
        for row in csv.DictReader(fh):
            t = n_tokens(row)
            if total + t > cap: break
            out.append(row); total += t
    return out

def select_bottom_rows(csv_path: Path, cap: int) -> List[Row]:
    dq: Deque[Row] = collections.deque(); tot = 0
    with open_csv(csv_path) as fh:
        for row in csv.DictReader(fh):
            t = n_tokens(row)
            dq.append(row); tot += t
            while dq and tot - n_tokens(dq[0]) > cap:
                tot -= n_tokens(dq.popleft())
    return list(dq)

def sample_random_rows(csv_path: Path, cap: int,
                       rng: random.Random) -> List[Row]:
    """length‑weighted reservoir sampling"""
    sel: List[Row] = []; total = 0
    with open_csv(csv_path) as fh:
        for row in csv.DictReader(fh):
            t = n_tokens(row)
            if total < cap:
                sel.append(row); total += t
            else:
                if rng.random() < cap / (total + t):
                    idx = rng.randrange(len(sel))
                    total -= n_tokens(sel[idx])
                    sel[idx] = row; total += t
    # trim if we overshot
    sel.sort(key=lambda r: float(r["score"]), reverse=True)
    keep, run = [], 0
    for r in sel:
        t = n_tokens(r)
        if run + t > cap: break
        keep.append(r); run += t
    return keep

# ------------------------------------------------------------- dataset build
def build_dataset(rows: List[Row], out_dir: Path, dtype=np.uint16):
    out_npy = out_dir / "tokens.npy"
    out_meta = out_dir / "metadata.csv.gz"

    tot = sum(n_tokens(r) for r in rows)
    mmap = np.memmap(out_npy, mode="w+", dtype=dtype, shape=(tot,))
    cur: int = 0
    cache: Dict[Path, np.ndarray] = {}

    with gzip.open(out_meta, "wt", newline="") as fout:
        wr = csv.writer(fout)
        # wr.writerow(["start", "end", "id", "path", "token_count"])
        for r in tqdm(rows, desc="Copy", unit="doc"):
            s, e = int(r["start"]), int(r["end"])
            tcnt = e - s
            src = Path(r["npy_file"])
            if src not in cache:
                cache[src] = np.memmap(src, dtype=np.uint16)
            mmap[cur:cur+tcnt] = cache[src][s:e]
            wr.writerow([cur, cur+tcnt, r["id"], str(out_npy), tcnt])
            cur += tcnt
    mmap.flush()
    print(f"[DONE] {tot:,} tokens written to {out_npy}")

# ------------------------------------------------------------------- CLI
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-csv", required=True, help=".csv.gz with scores")
    ap.add_argument("--top",    type=int, help="Top‑score token budget")
    ap.add_argument("--bottom", type=int, help="Bottom‑score token budget")
    ap.add_argument("--random", type=int, help="Random token budget")
    ap.add_argument("--seed",   type=int, default=42, help="PRNG seed")
    ap.add_argument("--out-dir", required=True, help="Output folder")
    ap.add_argument("--dtype",  default="uint16", help="NumPy dtype")
    return ap.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.src_csv).expanduser()
    out_dir  = Path(args.out_dir).expanduser(); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows: List[Row] = []
    if args.top:
        print(f"[SELECT] top {args.top:,} tokens")
        rows += select_top_rows(csv_path, args.top)

    if args.bottom:
        print(f"[SELECT] bottom {args.bottom:,} tokens")
        rows += select_bottom_rows(csv_path, args.bottom)

    if args.random:
        print(f"[SELECT] random {args.random:,} tokens")
        rows += sample_random_rows(csv_path, args.random, rng)

    # deduplicate rows (same id could appear if budgets overlap)
    dedup, seen = [], set()
    for r in rows:
        if r["id"] not in seen:
            dedup.append(r); seen.add(r["id"])
    rows = sorted(dedup, key=lambda r: float(r["score"]), reverse=True)

    print(f"[INFO] docs: {len(rows):,} — tokens: {sum(n_tokens(r) for r in rows):,}")
    build_dataset(rows, out_dir, dtype=np.dtype(args.dtype))

if __name__ == "__main__":
    main()