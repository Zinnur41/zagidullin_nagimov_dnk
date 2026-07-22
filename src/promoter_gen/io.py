"""Input/output helpers for generated DNA sequences."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .generator import SequenceRecord


def write_fasta(records: Iterable[SequenceRecord], path: Path) -> Path:
    """Write records to FASTA with metadata in each header."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            header = (
                f">{record.identifier} label={record.label} "
                f"strength={record.strength} score={record.architecture_score:.4f}"
            )
            stream.write(f"{header}\n{record.sequence}\n")
    return path


def write_csv(records: Sequence[SequenceRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SequenceRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)
    return path


def write_json(data: Dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return path


def read_fasta(path: Path) -> List[str]:
    """Read sequences from a simple single-line or wrapped FASTA file."""

    sequences: List[str] = []
    current: List[str] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
                continue
            current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return sequences


def dataset_summary(records: Sequence[SequenceRecord]) -> Dict[str, object]:
    if not records:
        return {
            "count": 0,
            "unique_fraction": 0.0,
            "mean_gc_content": 0.0,
            "mean_architecture_score": 0.0,
            "counts_by_strength": {},
        }

    counts_by_strength: Dict[str, int] = {}
    for record in records:
        counts_by_strength[record.strength] = (
            counts_by_strength.get(record.strength, 0) + 1
        )
    return {
        "count": len(records),
        "unique_fraction": len({record.sequence for record in records}) / len(records),
        "mean_gc_content": sum(record.gc_content for record in records) / len(records),
        "mean_architecture_score": (
            sum(record.architecture_score for record in records) / len(records)
        ),
        "counts_by_strength": counts_by_strength,
    }

