"""Tiny CLI:  python -m rag.cli ingest data/sample_docs   |   python -m rag.cli ask "question" """
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.pipeline import RAGPipeline


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest"); ing.add_argument("paths", nargs="+")
    ask = sub.add_parser("ask"); ask.add_argument("question"); ask.add_argument("--session")
    args = ap.parse_args()
    p = RAGPipeline.build()
    if args.cmd == "ingest":
        files = [f for path in map(Path, args.paths) for f in (sorted(path.iterdir()) if path.is_dir() else [path])]
        for f in files:
            r = p.ingest(f.name, f.read_bytes())
            print(f"{r.status:9} {r.filename:36} chunks={r.chunks:3} {r.message or ''}")
    else:
        print(json.dumps(p.ask(args.question, args.session).model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
