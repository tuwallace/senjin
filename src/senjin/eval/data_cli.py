"""`senjin-data fetch | stats | moves`: work with the Anthropic Interviewer dataset."""

import argparse
import collections
from pathlib import Path

from senjin.eval.dataset import SPLITS, extract_moves, fetch, load_split
from senjin.interview.exemplars import build_index

CACHE = Path("data/hf_cache")
EXEMPLARS = Path("data/exemplars/moves.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(prog="senjin-data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the dataset into data/hf_cache")
    sub.add_parser("stats", help="count transcripts and interviewer move types")
    moves = sub.add_parser("moves", help="build the embedded exemplar library")
    moves.add_argument("--per-kind", type=int, default=40)
    moves.add_argument("--out", type=Path, default=EXEMPLARS)
    args = parser.parse_args()

    if args.cmd == "fetch":
        for path in fetch(CACHE):
            print(path)
    elif args.cmd == "stats":
        kinds: collections.Counter[str] = collections.Counter()
        for split in SPLITS:
            rows = list(load_split(CACHE, split))
            print(f"{split}: {len(rows)} transcripts")
            for tid, turns in rows:
                kinds.update(m.kind for m in extract_moves(tid, split, turns))
        print(dict(kinds))
    else:
        count = build_index(CACHE, args.out, args.per_kind)
        print(f"wrote {count} exemplars to {args.out}")


if __name__ == "__main__":
    main()
