"""Интерактивные графики Plotly для отчёта ClearML."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from .generator import DNA_ALPHABET, GenerationConfig, SequenceRecord


COLORS: dict[str, str] = {
    "A": "#2ca02c",
    "C": "#1f77b4",
    "G": "#ff7f0e",
    "T": "#d62728",
    "promoter": "#6f42c1",
    "background": "#7f8c8d",
    "strong": "#2ca02c",
    "medium": "#ffbf00",
    "weak": "#d62728",
}


def _base_frequencies(records: Sequence[SequenceRecord]) -> np.ndarray:
    """Посчитать частоты нуклеотидов в каждой позиции"""
    if not records:
        raise ValueError("records must not be empty")

    length: int = len(records[0].sequence)
    frequencies = np.zeros((len(DNA_ALPHABET), length), dtype=float)
    for record in records:
        if len(record.sequence) != length:
            raise ValueError("all sequences must have equal length")

        for position, base in enumerate(record.sequence):
            frequencies[DNA_ALPHABET.index(base), position] += 1.0

    return frequencies / len(records)


def nucleotide_profile_figure(
    records: Sequence[SequenceRecord],
    config: GenerationConfig,
) -> go.Figure:
    """Сравнить позиционные частоты в промоторах и фоне"""
    promoters: list[SequenceRecord] = [record for record in records if record.label == 1]
    background: list[SequenceRecord] = [record for record in records if record.label == 0]
    coordinates: list[int] = list(range(config.window_start, config.window_end + 1))
    figure = go.Figure()

    for label, subset, dash in (
        ("promoter", promoters, "solid"),
        ("background", background, "dot"),
    ):
        frequencies = _base_frequencies(subset)
        for base_index, base in enumerate(DNA_ALPHABET):
            figure.add_trace(
                go.Scatter(
                    x=coordinates,
                    y=frequencies[base_index],
                    mode="lines",
                    name=f"{base}: {label}",
                    legendgroup=base,
                    line={"color": COLORS[base], "dash": dash},
                    opacity=1.0 if label == "promoter" else 0.50,
                    hovertemplate=(
                        "Координата %{x}<br>Частота %{y:.3f}<extra>"
                        f"{base}: {label}</extra>"
                    ),
                )
            )

    figure.add_vrect(
        x0=config.minus35_coordinate - config.minus35_jitter,
        x1=(
            config.minus35_coordinate
            + config.minus35_jitter
            + len(config.minus35_consensus)
            - 1
        ),
        fillcolor="#6f42c1",
        opacity=0.10,
        line_width=0,
        annotation_text="-35",
    )

    expected_minus10 = (
        config.minus35_coordinate
        + len(config.minus35_consensus)
        + config.preferred_spacer
    )

    figure.add_vrect(
        x0=expected_minus10 - 2,
        x1=expected_minus10 + len(config.minus10_consensus) + 2,
        fillcolor="#e83e8c",
        opacity=0.10,
        line_width=0,
        annotation_text="-10",
    )

    figure.update_layout(
        title="Позиционный профиль нуклеотидов",
        xaxis_title="Координата относительно TSS",
        yaxis_title="Частота",
        yaxis={"range": [0, 1]},
        hovermode="x unified",
        template="plotly_white",
    )

    return figure


def promoter_heatmap_figure(
    records: Sequence[SequenceRecord],
    config: GenerationConfig,
) -> go.Figure:
    """Показать частоты нуклеотидов в промоторах"""
    promoters: list[SequenceRecord] = [record for record in records if record.label == 1]
    frequencies = _base_frequencies(promoters)
    coordinates: list[int]= list(range(config.window_start, config.window_end + 1))

    figure = go.Figure(
        data=go.Heatmap(
            z=frequencies,
            x=coordinates,
            y=list(DNA_ALPHABET),
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar={"title": "Частота"},
            hovertemplate="Координата %{x}<br>%{y}: %{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Тепловая карта позиционных частот в промоторах",
        xaxis_title="Координата относительно TSS",
        yaxis_title="Нуклеотид",
        template="plotly_white",
    )

    return figure


def gc_distribution_figure(records: Sequence[SequenceRecord]) -> go.Figure:
    figure = go.Figure()
    for label, name in ((1, "promoter"), (0, "background")):
        values: list[float] = [record.gc_content for record in records if record.label == label]
        figure.add_trace(
            go.Histogram(
                x=values,
                name=name,
                opacity=0.65,
                nbinsx=30,
                marker_color=COLORS[name],
                histnorm="probability density",
            )
        )

    figure.update_layout(
        title="Распределение GC-состава",
        xaxis_title="Доля G+C",
        yaxis_title="Плотность",
        barmode="overlay",
        template="plotly_white",
    )

    return figure


def architecture_score_figure(records: Sequence[SequenceRecord]) -> go.Figure:
    figure = go.Figure()
    order: tuple[str, ...] = ("strong", "medium", "weak", "background")
    for strength in order:
        values: list[float] = [
            record.architecture_score
            for record in records
            if record.strength == strength
        ]

        figure.add_trace(
            go.Box(
                y=values,
                name=strength,
                marker_color=COLORS.get(strength, COLORS["background"]),
                boxmean=True,
            )
        )

    figure.update_layout(
        title="Архитектурный score по классу последовательности",
        xaxis_title="Класс",
        yaxis_title="Score близости к σ70-промотору",
        yaxis={"range": [0, 1.02]},
        template="plotly_white",
    )

    return figure


def roc_figure(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    roc_auc: float,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name=f"k-mer model (AUC={roc_auc:.3f})",
            line={"color": COLORS["promoter"], "width": 3},
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Случайный классификатор",
            line={"color": "#95a5a6", "dash": "dash"},
        )
    )

    figure.update_layout(
        title="ROC-кривая на hold-out выборке",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
        template="plotly_white",
    )

    return figure


def confusion_matrix_figure(confusion: np.ndarray) -> go.Figure:
    text: list[list[str]] = [[str(int(value)) for value in row] for row in confusion]
    figure = go.Figure(
        data=go.Heatmap(
            z=confusion,
            x=["background", "promoter"],
            y=["background", "promoter"],
            colorscale="Purples",
            text=text,
            texttemplate="%{text}",
            hovertemplate=(
                "Истинный: %{y}<br>Предсказанный: %{x}<br>N=%{z}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Матрица ошибок",
        xaxis_title="Предсказанный класс",
        yaxis_title="Истинный класс",
        template="plotly_white",
    )

    return figure


def cross_validation_figure(
    cross_validation: Mapping[str, Sequence[float]],
) -> go.Figure:
    figure = go.Figure()
    for metric, values in cross_validation.items():
        figure.add_trace(
            go.Bar(
                x=[f"Fold {index + 1}" for index in range(len(values))],
                y=list(values),
                name=metric,
                hovertemplate="Значение %{y:.3f}<extra>" + metric + "</extra>",
            )
        )

    figure.update_layout(
        title="Метрики 5-fold cross-validation",
        xaxis_title="Фолд",
        yaxis_title="Метрика",
        yaxis={"range": [0, 1.02]},
        barmode="group",
        template="plotly_white",
    )

    return figure


def generated_quality_figure(
    generated_records: Sequence[SequenceRecord],
    probabilities: Sequence[float],
) -> go.Figure:
    if len(generated_records) != len(probabilities):
        raise ValueError("one probability is required for each generated record")

    figure = go.Figure()
    for strength in ("strong", "medium", "weak"):
        values: list[float] = [
            float(probability)
            for record, probability in zip(generated_records, probabilities)
            if record.strength == strength
        ]

        figure.add_trace(
            go.Violin(
                y=values,
                name=strength,
                box_visible=True,
                meanline_visible=True,
                marker_color=COLORS[strength],
            )
        )

    figure.update_layout(
        title="Вероятность класса «promoter» для новых кандидатов",
        xaxis_title="Заданная сила",
        yaxis_title="Вероятность k-mer-модели",
        yaxis={"range": [0, 1.02]},
        template="plotly_white",
    )

    return figure


def diversity_figure(
    generated_records: Sequence[SequenceRecord],
) -> go.Figure:
    """Оценить уникальность и попарные различия кандидатов"""
    metrics: Dict[str, List[float]] = {
        "unique_fraction": [],
        "mean_pairwise_difference": [],
    }
    strengths: tuple[str, str, str] = ("strong", "medium", "weak")
    for strength in strengths:
        sequences: list[str] = [
            record.sequence
            for record in generated_records
            if record.strength == strength
        ]
        metrics["unique_fraction"].append(
            len(set(sequences)) / len(sequences) if sequences else 0.0
        )
        # Берём непересекающиеся соседние пары, чтобы не сравнивать все последовательности.
        differences = [
            sum(left_base != right_base for left_base, right_base in zip(left, right))
            / len(left)
            for left, right in zip(sequences[::2], sequences[1::2])
        ]

        metrics["mean_pairwise_difference"].append(
            float(np.mean(differences)) if differences else 0.0
        )

    figure = go.Figure()
    for metric, values in metrics.items():
        figure.add_trace(go.Bar(x=list(strengths), y=values, name=metric))

    figure.update_layout(
        title="Разнообразие сгенерированных кандидатов",
        xaxis_title="Заданная сила",
        yaxis_title="Доля",
        yaxis={"range": [0, 1.02]},
        barmode="group",
        template="plotly_white",
    )

    return figure


def all_figures(
    records: Sequence[SequenceRecord],
    generated_records: Sequence[SequenceRecord],
    generated_probabilities: Sequence[float],
    config: GenerationConfig,
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    roc_auc: float,
    confusion: np.ndarray,
    cross_validation: Mapping[str, Sequence[float]],
) -> Dict[str, go.Figure]:
    """Собрать все графики"""
    return {
        "01_nucleotide_profile": nucleotide_profile_figure(records, config),
        "02_promoter_heatmap": promoter_heatmap_figure(records, config),
        "03_gc_distribution": gc_distribution_figure(records),
        "04_architecture_score": architecture_score_figure(records),
        "05_roc_curve": roc_figure(
            false_positive_rate,
            true_positive_rate,
            roc_auc,
        ),
        "06_confusion_matrix": confusion_matrix_figure(confusion),
        "07_cross_validation": cross_validation_figure(cross_validation),
        "08_generated_quality": generated_quality_figure(
            generated_records,
            generated_probabilities,
        ),
        "09_generated_diversity": diversity_figure(generated_records),
    }