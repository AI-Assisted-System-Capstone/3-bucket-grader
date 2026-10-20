import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from data_pipeline import FULL_LABELS, LABELS, build_input_text  # noqa: E402
from distilbert_model import DistilBertModel, load_distilbert, save_distilbert  # noqa: E402
from predict import load_any_model, main as predict_main, score_reports  # noqa: E402
from severity_model import bucket_probs, triage_score  # noqa: E402
from conftest import make_reports  # noqa: E402

N_GRADES = len(FULL_LABELS)
CUTOFF = 0.3
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "unit", "icu", "patient", "fall", "dose", "died", "x", "ray"]


class FakeTrainer:
    """Stands in for transformers.Trainer: save_distilbert only calls save_model."""

    def __init__(self, network):
        self.network = network

    def save_model(self, path):
        self.network.save_pretrained(path)


@pytest.fixture(scope="module")
def tiny_parts(tmp_path_factory):
    """A randomly initialised 10-class DistilBERT small enough to build in a second."""
    vocab_file = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB))
    tokenizer = transformers.BertTokenizerFast(str(vocab_file))
    torch.manual_seed(0)
    network = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(
        vocab_size=len(VOCAB), dim=16, hidden_dim=32, n_layers=1, n_heads=2,
        max_position_embeddings=32, num_labels=N_GRADES,
    ))
    triage = np.linspace(0, 1, 40)
    calibrator = LogisticRegression().fit(triage.reshape(-1, 1), (triage > 0.5).astype(int))
    return tokenizer, network, calibrator


@pytest.fixture
def saved_dir(tmp_path, tiny_parts):
    tokenizer, network, calibrator = tiny_parts
    save_distilbert(str(tmp_path), FakeTrainer(network), tokenizer, calibrator, CUTOFF, 16, 0.95)
    return tmp_path


REPORTS = ["patient fall x ray", "dose died", "unit icu patient"]


def test_save_writes_everything_load_needs(saved_dir):
    for name in ("model", "tokenizer", "calibrator.joblib", "config.json"):
        assert os.path.exists(saved_dir / name)
    cfg = json.loads((saved_dir / "config.json").read_text())
    assert cfg == {"target": "grade", "cutoff_raw_triage": CUTOFF, "target_recall": 0.95, "max_length": 16}


def test_load_restores_cutoff_and_classes(saved_dir):
    model = load_distilbert(str(saved_dir))
    assert isinstance(model, DistilBertModel)
    assert model.cutoff == CUTOFF
    assert model.max_length == 16
    np.testing.assert_array_equal(model.classes_, np.arange(N_GRADES))
    assert not model.network.training  # eval mode: dropout would make scores random


def test_predict_proba_rows_sum_to_one_and_match_the_network(saved_dir, tiny_parts):
    tokenizer, network, _ = tiny_parts
    model = load_distilbert(str(saved_dir))
    proba = model.predict_proba(REPORTS)
    assert proba.shape == (len(REPORTS), N_GRADES)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    enc = tokenizer(REPORTS, padding=True, truncation=True, max_length=16, return_tensors="pt")
    with torch.no_grad():
        expected = torch.softmax(
            network.eval()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits, dim=-1).numpy()
    np.testing.assert_allclose(proba, expected, atol=1e-5)


def test_scores_do_not_depend_on_batch_mates(saved_dir):
    # Padding to the longest report in a batch must not change a report's score.
    model = load_distilbert(str(saved_dir))
    alone = model.predict_proba([REPORTS[0]])
    together = model.predict_proba(REPORTS)[:1]
    np.testing.assert_allclose(alone, together, atol=1e-5)


def test_predict_proba_empty_input(saved_dir):
    assert load_distilbert(str(saved_dir)).predict_proba([]).shape == (0, N_GRADES)


def test_score_reports_output_matches_the_tfidf_format(saved_dir):
    model = load_distilbert(str(saved_dir))
    reports = make_reports(REPORTS).assign(**{"Event No.": ["E-1", "E-2", "E-3"]})
    out = score_reports(reports, model)

    assert list(out.columns) == ["event_no", "grade", "bucket", "p_none", "p_some", "p_serious",
                                 "triage_score_raw", "triage_probability", "flagged"]
    assert list(out["event_no"]) == ["E-1", "E-2", "E-3"]
    np.testing.assert_allclose(out[[f"p_{b}" for b in LABELS]].sum(axis=1), 1.0, rtol=1e-5)
    assert set(out["grade"]) <= set(FULL_LABELS)
    assert out["triage_probability"].between(0, 1).all()

    proba = model.predict_proba(build_input_text(reports))
    np.testing.assert_allclose(out["triage_score_raw"], triage_score(proba, model.classes_))
    np.testing.assert_allclose(out[[f"p_{b}" for b in LABELS]], bucket_probs(proba, model.classes_))
    assert list(out["grade"]) == [FULL_LABELS[i] for i in proba.argmax(axis=1)]
    assert (out["flagged"] == (out["triage_score_raw"] >= CUTOFF)).all()
    assert "event_comments" not in out.columns


def test_score_ignores_post_investigation_columns(saved_dir):
    model = load_distilbert(str(saved_dir))
    clean = make_reports(REPORTS[:2])
    leaky = clean.assign(manager_comments=["patient died", "patient died"],
                         unit_actions_taken=["dose died", "dose died"])
    pd.testing.assert_frame_equal(score_reports(clean, model), score_reports(leaky, model))


def test_score_requires_narrative(saved_dir):
    with pytest.raises(ValueError, match="event_comments"):
        score_reports(pd.DataFrame({"Location Name": ["ICU"]}), load_distilbert(str(saved_dir)))


def test_load_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="finetune_distilbert"):
        load_distilbert(str(tmp_path))


def test_load_rejects_a_bucket_target_model(saved_dir):
    cfg_path = saved_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cfg, "target": "bucket"}))
    with pytest.raises(ValueError, match="target grade"):
        load_distilbert(str(saved_dir))


def test_load_any_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown model"):
        load_any_model("gpt")


def test_cli_model_distilbert(saved_dir, tmp_path):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    make_reports(REPORTS).assign(**{"Event No.": ["E-1", "E-2", "E-3"]}).to_csv(src, index=False)

    predict_main([str(src), "-o", str(dst), "--model", "distilbert", "--model-dir", str(saved_dir)])

    out = pd.read_csv(dst)
    assert list(out["event_no"]) == ["E-1", "E-2", "E-3"]
    assert {"grade", "bucket", "triage_probability", "flagged"} <= set(out.columns)
