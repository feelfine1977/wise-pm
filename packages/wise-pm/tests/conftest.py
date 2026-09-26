import pytest

MARKERS = (
    "spec(id): the paper section or decision record the test implements, e.g. spec('IV-B.lag')",
    "paper: a worked example taken verbatim from the paper",
    "regression: pins a confirmed defect (xfail strict until the fix lands)",
    "property: property-based test (Hypothesis)",
    "bpic19: needs the BPI Challenge 2019 CSV via WISE_BPIC19_CSV",
    "bench: benchmark; skipped unless WISE_RUN_BENCH=1",
)


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and the warnings-as-errors policy so the packaged tests behave like the workspace run."""
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)
    config.addinivalue_line("filterwarnings", "error")


import wise
from _support.builders import evaluate, make_log


@pytest.fixture
def p2p_norm():
    return wise.running_p2p_norm()


@pytest.fixture
def p2p_log():
    return wise.running_p2p_log()


@pytest.fixture
def p2p_result(p2p_log, p2p_norm):
    return wise.score(p2p_log, p2p_norm)


@pytest.fixture
def helpers():
    return make_log, evaluate
