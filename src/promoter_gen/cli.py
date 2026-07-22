from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .generator import PromoterGenerator, STRENGTHS
from .io import write_csv, write_fasta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promoter-gen",
        description="Generate bacterial sigma-70 promoter DNA sequences.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate promoter sequences")
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--strength", choices=STRENGTHS, default="medium")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/promoters.fasta"),
    )
    generate.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="optional CSV path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        generator = PromoterGenerator(seed=args.seed)
        records = generator.generate_promoters(
            args.count,
            strengths=(args.strength,),
        )
        fasta_path = write_fasta(records, args.output)
        if args.metadata is not None:
            write_csv(records, args.metadata)
        print(f"Generated {len(records)} sequences: {fasta_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

