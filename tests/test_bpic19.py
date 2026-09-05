"""Reproduction of the paper's BPIC'19 evaluation (Section V).

Runs only when the environment variable ``WISE_BPIC19_CSV`` points to the
challenge log; the file is not distributed with the package.
"""

import os
import sys
from pathlib import Path

import pytest

import wise

CSV = os.environ.get("WISE_BPIC19_CSV")
pytestmark = pytest.mark.skipif(not CSV or not Path(CSV).exists(), reason="set WISE_BPIC19_CSV to the BPI Challenge 2019 CSV")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FOCUS = [("companyID_0000", "Packaging"), ("companyID_0000", "Logistics"), ("companyID_0003", "Real Estate")]


@pytest.fixture(scope="module")
def result():
    sys.path.insert(0, str(EXAMPLES))
    from bpic19_evaluation import load

    log = load(CSV)
    norm = wise.Norm.load(EXAMPLES / "bpic19_norm.json")
    return wise.score(log, norm)


def test_section_v_scores(result):
    assert len(result.scores) == 251_734
    assert result.applicability_density(scope=False) == pytest.approx(0.762, abs=1e-3)
    means = result.summary()["mean_score"]
    assert means["Finance"] == pytest.approx(0.819, abs=1e-3)
    assert means["Logistics"] == pytest.approx(0.817, abs=1e-3)
    assert means["Compliance"] == pytest.approx(0.871, abs=1e-3)
    assert means["Automation"] == pytest.approx(0.844, abs=1e-3)


def test_table_xi_focus_slices(result):
    by = ["case Company", "case Spend area text"]
    backlog = wise.prioritize(result, by, view="Automation", gamma=20)
    expected = {FOCUS[0]: (109_199, 0.0087, 945.7), FOCUS[1]: (5_242, 0.0561, 294.2), FOCUS[2]: (583, 0.0869, 50.6)}
    for key, (n, gap, pi) in expected.items():
        assert backlog.loc[key, "n_cases"] == n
        assert backlog.loc[key, "stable_gap"] == pytest.approx(gap, abs=5e-4)
        assert backlog.loc[key, "stable_PI"] == pytest.approx(pi, abs=1.0)


def test_table_x_and_figure_4(result):
    doc = "case Purchasing Document"
    agreement = wise.view_agreement(result, doc, k=20, gamma=20)
    assert agreement.loc[("Finance", "Compliance"), "top20_overlap"] == pytest.approx(0.481, abs=5e-3)
    assert agreement.loc[("Finance", "Automation"), "top20_overlap"] == pytest.approx(0.081, abs=5e-3)
    assert agreement.loc[("Finance", "Compliance"), "score_correlation"] == pytest.approx(0.817, abs=5e-3)
    concentration = wise.concentration(wise.prioritize(result, doc, view="Compliance", gamma=20))
    assert concentration.loc[0.8, "share_of_slices"] == pytest.approx(0.023, abs=1e-3)
    assert concentration.loc[0.95, "share_of_slices"] == pytest.approx(0.092, abs=1e-3)
