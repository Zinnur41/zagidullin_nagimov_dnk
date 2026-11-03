from promoter_gen.generator import PromoterGenerator
from promoter_gen.model import promoter_probabilities, train_and_evaluate


def test_model_training_and_probabilities() -> None:
    generator = PromoterGenerator(seed=23)
    records = generator.generate_dataset(90, 90)
    result = train_and_evaluate(records, seed=23, cross_validation_folds=3)
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0
    assert result.metrics["roc_auc"] > 0.80
    assert result.confusion.shape == (2, 2)
    candidates = generator.generate_promoters(5, strengths=("strong",))
    probabilities = promoter_probabilities(
        result.model,
        [record.sequence for record in candidates],
    )
    assert len(probabilities) == 5
    assert all(0.0 <= value <= 1.0 for value in probabilities)

