"""Biologically informed generator for bacterial sigma-70 promoters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


DNA_ALPHABET: Tuple[str, ...] = ("A", "C", "G", "T")
STRENGTHS: Tuple[str, ...] = ("strong", "medium", "weak")

# Per-base mutation probabilities for the -35 and -10 elements.
STRENGTH_MUTATION_RATES: Dict[str, Tuple[float, float]] = {
    "strong": (0.04, 0.03),
    "medium": (0.12, 0.09),
    "weak": (0.24, 0.18),
}


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters of the promoter architecture.

    Coordinates are relative to the transcription start site (TSS, +1).
    The default 81 nt window therefore covers positions -60 through +20.
    """

    window_start: int = -60
    window_end: int = 20
    minus35_consensus: str = "TTGACA"
    minus10_consensus: str = "TATAAT"
    minus35_coordinate: int = -35
    minus35_jitter: int = 1
    spacer_min: int = 15
    spacer_max: int = 19
    preferred_spacer: int = 17
    background_probabilities: Tuple[float, float, float, float] = (
        0.25,
        0.25,
        0.25,
        0.25,
    )
    up_element_start: int = -55
    up_element_end: int = -40
    up_at_probability: float = 0.78

    @property
    def length(self) -> int:
        return self.window_end - self.window_start + 1

    def validate(self) -> None:
        if self.window_end < self.window_start:
            raise ValueError("window_end must not be smaller than window_start")
        if set(self.minus35_consensus + self.minus10_consensus) - set(DNA_ALPHABET):
            raise ValueError("consensus motifs may contain only A, C, G and T")
        if self.spacer_min < 0 or self.spacer_max < self.spacer_min:
            raise ValueError("invalid spacer range")
        if not self.spacer_min <= self.preferred_spacer <= self.spacer_max:
            raise ValueError("preferred_spacer must be inside the spacer range")
        if len(self.background_probabilities) != len(DNA_ALPHABET):
            raise ValueError("exactly four background probabilities are required")
        if any(probability < 0 for probability in self.background_probabilities):
            raise ValueError("background probabilities must be non-negative")
        if not math.isclose(sum(self.background_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("background probabilities must sum to one")
        if not 0.0 <= self.up_at_probability <= 1.0:
            raise ValueError("up_at_probability must be in [0, 1]")

        latest_minus35 = self.minus35_coordinate + self.minus35_jitter
        latest_minus10 = (
            latest_minus35 + len(self.minus35_consensus) + self.spacer_max
        )
        if self.minus35_coordinate - self.minus35_jitter < self.window_start:
            raise ValueError("-35 element does not fit inside the sequence window")
        if latest_minus10 + len(self.minus10_consensus) - 1 > self.window_end:
            raise ValueError("-10 element does not fit inside the sequence window")


@dataclass(frozen=True)
class SequenceRecord:
    """One generated DNA sequence and its provenance."""

    identifier: str
    sequence: str
    label: int
    strength: str
    minus35_coordinate: Optional[int]
    minus10_coordinate: Optional[int]
    spacer_length: Optional[int]
    minus35_mutations: Optional[int]
    minus10_mutations: Optional[int]
    gc_content: float
    architecture_score: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def gc_content(sequence: str) -> float:
    """Return the GC fraction of a non-empty DNA sequence."""

    if not sequence:
        raise ValueError("sequence must not be empty")
    normalized = sequence.upper()
    invalid = set(normalized) - set(DNA_ALPHABET)
    if invalid:
        raise ValueError(f"invalid DNA symbols: {sorted(invalid)}")
    return (normalized.count("G") + normalized.count("C")) / len(normalized)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires strings of equal length")
    return sum(a != b for a, b in zip(left, right))


def architecture_score(
    sequence: str,
    config: Optional[GenerationConfig] = None,
) -> float:
    """Score the best sigma-70-like motif pair in ``sequence``.

    The score is in [0, 1]. Motif agreement contributes 90% and preference for
    a 17 nt spacer contributes 10%.
    """

    config = config or GenerationConfig()
    config.validate()
    sequence = sequence.upper()
    if len(sequence) != config.length:
        raise ValueError(f"expected a sequence of length {config.length}")
    invalid = set(sequence) - set(DNA_ALPHABET)
    if invalid:
        raise ValueError(f"invalid DNA symbols: {sorted(invalid)}")

    best = 0.0
    for minus35_coordinate in range(
        config.minus35_coordinate - config.minus35_jitter,
        config.minus35_coordinate + config.minus35_jitter + 1,
    ):
        minus35_index = minus35_coordinate - config.window_start
        observed35 = sequence[
            minus35_index : minus35_index + len(config.minus35_consensus)
        ]
        similarity35 = 1.0 - (
            hamming_distance(observed35, config.minus35_consensus)
            / len(config.minus35_consensus)
        )
        for spacer in range(config.spacer_min, config.spacer_max + 1):
            minus10_coordinate = (
                minus35_coordinate + len(config.minus35_consensus) + spacer
            )
            minus10_index = minus10_coordinate - config.window_start
            observed10 = sequence[
                minus10_index : minus10_index + len(config.minus10_consensus)
            ]
            similarity10 = 1.0 - (
                hamming_distance(observed10, config.minus10_consensus)
                / len(config.minus10_consensus)
            )
            spacing_score = math.exp(
                -0.5 * ((spacer - config.preferred_spacer) / 1.25) ** 2
            )
            score = 0.45 * similarity35 + 0.45 * similarity10 + 0.10 * spacing_score
            best = max(best, score)
    return float(min(max(best, 0.0), 1.0))


class PromoterGenerator:
    """Generate promoter and background sequences reproducibly."""

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or GenerationConfig()
        self.config.validate()
        self.seed = seed
        self._rng = random.Random(seed)

    def _background_base(self) -> str:
        return self._rng.choices(
            DNA_ALPHABET,
            weights=self.config.background_probabilities,
            k=1,
        )[0]

    def _background_sequence(self) -> List[str]:
        return [self._background_base() for _ in range(self.config.length)]

    def _mutate_motif(self, motif: str, probability: float) -> Tuple[str, int]:
        result: List[str] = []
        mutations = 0
        for base in motif:
            if self._rng.random() >= probability:
                result.append(base)
                continue
            alternatives = [candidate for candidate in DNA_ALPHABET if candidate != base]
            alternative_weights = [
                self.config.background_probabilities[DNA_ALPHABET.index(candidate)]
                for candidate in alternatives
            ]
            result.append(self._rng.choices(alternatives, alternative_weights, k=1)[0])
            mutations += 1
        return "".join(result), mutations

    def _choose_spacer(self, strength: str) -> int:
        choices = list(range(self.config.spacer_min, self.config.spacer_max + 1))
        if strength == "strong":
            weights = [
                math.exp(-0.5 * ((value - self.config.preferred_spacer) / 0.75) ** 2)
                for value in choices
            ]
        elif strength == "medium":
            weights = [
                math.exp(-0.5 * ((value - self.config.preferred_spacer) / 1.25) ** 2)
                for value in choices
            ]
        else:
            weights = [1.0] * len(choices)
        return self._rng.choices(choices, weights=weights, k=1)[0]

    def _enrich_up_element(self, sequence: List[str]) -> None:
        start = max(self.config.up_element_start, self.config.window_start)
        end = min(self.config.up_element_end, self.config.window_end)
        for coordinate in range(start, end + 1):
            if self._rng.random() < self.config.up_at_probability:
                sequence[coordinate - self.config.window_start] = self._rng.choice(("A", "T"))

    def generate_promoter(
        self,
        identifier: str,
        strength: str = "medium",
    ) -> SequenceRecord:
        if strength not in STRENGTHS:
            raise ValueError(f"strength must be one of {STRENGTHS}")

        sequence = self._background_sequence()
        self._enrich_up_element(sequence)
        minus35_coordinate = self.config.minus35_coordinate + self._rng.randint(
            -self.config.minus35_jitter,
            self.config.minus35_jitter,
        )
        spacer = self._choose_spacer(strength)
        minus10_coordinate = (
            minus35_coordinate + len(self.config.minus35_consensus) + spacer
        )
        rate35, rate10 = STRENGTH_MUTATION_RATES[strength]
        motif35, mutations35 = self._mutate_motif(
            self.config.minus35_consensus,
            rate35,
        )
        motif10, mutations10 = self._mutate_motif(
            self.config.minus10_consensus,
            rate10,
        )

        minus35_index = minus35_coordinate - self.config.window_start
        minus10_index = minus10_coordinate - self.config.window_start
        sequence[
            minus35_index : minus35_index + len(motif35)
        ] = list(motif35)
        sequence[
            minus10_index : minus10_index + len(motif10)
        ] = list(motif10)
        sequence_text = "".join(sequence)
        return SequenceRecord(
            identifier=identifier,
            sequence=sequence_text,
            label=1,
            strength=strength,
            minus35_coordinate=minus35_coordinate,
            minus10_coordinate=minus10_coordinate,
            spacer_length=spacer,
            minus35_mutations=mutations35,
            minus10_mutations=mutations10,
            gc_content=gc_content(sequence_text),
            architecture_score=architecture_score(sequence_text, self.config),
        )

    def _contains_near_consensus_pair(self, sequence: str) -> bool:
        for minus35_coordinate in range(
            self.config.minus35_coordinate - self.config.minus35_jitter,
            self.config.minus35_coordinate + self.config.minus35_jitter + 1,
        ):
            minus35_index = minus35_coordinate - self.config.window_start
            observed35 = sequence[
                minus35_index : minus35_index + len(self.config.minus35_consensus)
            ]
            if hamming_distance(observed35, self.config.minus35_consensus) > 1:
                continue
            for spacer in range(self.config.spacer_min, self.config.spacer_max + 1):
                minus10_coordinate = (
                    minus35_coordinate + len(self.config.minus35_consensus) + spacer
                )
                minus10_index = minus10_coordinate - self.config.window_start
                observed10 = sequence[
                    minus10_index : minus10_index + len(self.config.minus10_consensus)
                ]
                if hamming_distance(observed10, self.config.minus10_consensus) <= 1:
                    return True
        return False

    def generate_background(self, identifier: str) -> SequenceRecord:
        for _ in range(1_000):
            sequence = "".join(self._background_sequence())
            if not self._contains_near_consensus_pair(sequence):
                return SequenceRecord(
                    identifier=identifier,
                    sequence=sequence,
                    label=0,
                    strength="background",
                    minus35_coordinate=None,
                    minus10_coordinate=None,
                    spacer_length=None,
                    minus35_mutations=None,
                    minus10_mutations=None,
                    gc_content=gc_content(sequence),
                    architecture_score=architecture_score(sequence, self.config),
                )
        raise RuntimeError("failed to sample a valid background sequence")

    def generate_promoters(
        self,
        count: int,
        strengths: Sequence[str] = STRENGTHS,
        identifier_prefix: str = "P",
    ) -> List[SequenceRecord]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if not strengths or any(strength not in STRENGTHS for strength in strengths):
            raise ValueError(f"strengths must contain values from {STRENGTHS}")
        return [
            self.generate_promoter(
                identifier=f"{identifier_prefix}{index + 1:05d}",
                strength=strengths[index % len(strengths)],
            )
            for index in range(count)
        ]

    def generate_dataset(
        self,
        n_promoters: int,
        n_background: int,
        strengths: Sequence[str] = STRENGTHS,
    ) -> List[SequenceRecord]:
        if n_background < 0:
            raise ValueError("n_background must be non-negative")
        records = self.generate_promoters(n_promoters, strengths=strengths)
        records.extend(
            self.generate_background(f"N{index + 1:05d}")
            for index in range(n_background)
        )
        self._rng.shuffle(records)
        return records

