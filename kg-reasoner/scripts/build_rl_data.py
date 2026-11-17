"""Sample the RL training set (Appendix A: 2,000 examples randomly sampled from the 8 datasets).

    python scripts/build_rl_data.py --source_dir data/training_set --n 2000 --out data/rl/train.jsonl

Put the original *training* splits of the eight datasets in --source_dir (file names as in
data/test_set, e.g. cwq.json, WebQSP.json, ...). Each dataset contributes an equal share.
"""
import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records, write_jsonl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default="data/training_set")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="data/rl/train.jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.source_dir, "*.json*")))
    if not files:
        sys.exit(f"No training files in {args.source_dir}; add the training splits of the datasets.")

    rng = random.Random(args.seed)
    per_file = {f: load_records(f) for f in files}
    per_file = {f: rs for f, rs in per_file.items() if rs}
    quota = max(1, args.n // len(per_file))
    sample = []
    for f, recs in per_file.items():
        sample += rng.sample(recs, min(quota, len(recs)))
    chosen = {id(r) for r in sample}
    rest = [r for recs in per_file.values() for r in recs if id(r) not in chosen]
    sample += rng.sample(rest, max(0, min(args.n - len(sample), len(rest))))
    rng.shuffle(sample)
    sample = sample[: args.n]
    write_jsonl(args.out, sample)
    print(f"wrote {len(sample)} records to {args.out}:",
          {d: sum(r["dataset"] == d for r in sample) for d in sorted({r['dataset'] for r in sample})})


if __name__ == "__main__":
    main()
