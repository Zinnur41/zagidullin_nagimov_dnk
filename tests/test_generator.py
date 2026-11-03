import pytest

from promoter_gen.generator import (
    DNA_ALPHABET,
    GenerationConfig,
    PromoterGenerator,
    STRENGTHS,
    architecture_score,
)


def test_generation_is_reproducible() -> None:
    first = PromoterGenerator(seed=7).generate_dataset(12, 12)
    second = PromoterGenerator(seed=7).generate_dataset(12, 12)
    assert first == second


def test_promoter_metadata_and_alphabet() -> None:
    config = GenerationConfig()
    records = PromoterGenerator(config=config, seed=11).generate_promoters(30)
    assert len(records) == 30
    for record in records:
        assert len(record.sequence) == config.length
        assert set(record.sequence) <= set(DNA_ALPHABET)
        assert record.label == 1
        assert record.strength in STRENGTHS
        assert config.spacer_min <= record.spacer_length <= config.spacer_max
        assert 0.0 <= record.architecture_score <= 1.0


def test_background_rejects_full_consensus_architecture() -> None:
    config = GenerationConfig()
    records = PromoterGenerator(config=config, seed=17).generate_dataset(0, 100)
    assert all(record.label == 0 for record in records)
    assert all(record.strength == "background" for record in records)
    assert all(record.architecture_score < 0.96 for record in records)


def test_architecture_score_prefers_consensus_pair() -> None:
    config = GenerationConfig()
    sequence = list("C" * config.length)
    index35 = config.minus35_coordinate - config.window_start
    coordinate10 = (
        config.minus35_coordinate
        + len(config.minus35_consensus)
        + config.preferred_spacer
    )
    index10 = coordinate10 - config.window_start
    sequence[index35 : index35 + 6] = config.minus35_consensus
    sequence[index10 : index10 + 6] = config.minus10_consensus
    assert architecture_score("".join(sequence), config) == pytest.approx(1.0)


def test_invalid_strength_is_rejected() -> None:
    with pytest.raises(ValueError):
        PromoterGenerator().generate_promoter("P1", strength="unknown")

