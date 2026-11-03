from promoter_gen.generator import PromoterGenerator
from promoter_gen.io import dataset_summary, read_fasta, write_csv, write_fasta


def test_fasta_and_csv_export(tmp_path) -> None:
    records = PromoterGenerator(seed=5).generate_promoters(4)
    fasta_path = write_fasta(records, tmp_path / "records.fasta")
    csv_path = write_csv(records, tmp_path / "records.csv")
    assert read_fasta(fasta_path) == [record.sequence for record in records]
    assert csv_path.read_text(encoding="utf-8").startswith("identifier,sequence,label")


def test_dataset_summary() -> None:
    records = PromoterGenerator(seed=9).generate_dataset(6, 6)
    summary = dataset_summary(records)
    assert summary["count"] == 12
    assert summary["unique_fraction"] == 1.0
    assert summary["counts_by_strength"]["background"] == 6

