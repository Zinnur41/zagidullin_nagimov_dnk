from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .generator import SequenceRecord


@dataclass
class ModelResult:
    model: Pipeline
    metrics: Dict[str, float]
    cross_validation: Dict[str, List[float]]
    y_test: np.ndarray
    probability_test: np.ndarray
    false_positive_rate: np.ndarray
    true_positive_rate: np.ndarray
    confusion: np.ndarray


def build_classifier(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "kmers",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 6),
                    lowercase=False,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=3.0,
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def train_and_evaluate(
    records: Sequence[SequenceRecord],
    seed: int = 42,
    test_size: float = 0.20,
    cross_validation_folds: int = 5,
) -> ModelResult:
    if not records:
        raise ValueError("records must not be empty")
    sequences = [record.sequence for record in records]
    labels = np.asarray([record.label for record in records], dtype=int)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if set(unique_labels.tolist()) != {0, 1}:
        raise ValueError("both promoter and background labels are required")
    if int(label_counts.min()) < max(2, cross_validation_folds):
        raise ValueError("not enough examples per class for cross-validation")

    x_train, x_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    model = build_classifier(seed=seed)
    model.fit(x_train, y_train)
    probability_test = model.predict_proba(x_test)[:, 1]
    predicted_test = (probability_test >= 0.5).astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probability_test,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted_test)),
        "precision": float(precision_score(y_test, predicted_test, zero_division=0)),
        "recall": float(recall_score(y_test, predicted_test, zero_division=0)),
        "f1": float(f1_score(y_test, predicted_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability_test)),
    }

    cv_model = build_classifier(seed=seed)
    cv = StratifiedKFold(
        n_splits=cross_validation_folds,
        shuffle=True,
        random_state=seed,
    )
    raw_cv = cross_validate(
        cv_model,
        sequences,
        labels,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=1,
        return_train_score=False,
    )
    cross_validation = {
        metric: [float(value) for value in raw_cv[f"test_{metric}"]]
        for metric in metrics
    }
    return ModelResult(
        model=model,
        metrics=metrics,
        cross_validation=cross_validation,
        y_test=y_test,
        probability_test=probability_test,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        confusion=confusion_matrix(y_test, predicted_test, labels=[0, 1]),
    )


def promoter_probabilities(model: Pipeline, sequences: Sequence[str]) -> np.ndarray:
    if not sequences:
        return np.asarray([], dtype=float)
    return model.predict_proba(list(sequences))[:, 1]

