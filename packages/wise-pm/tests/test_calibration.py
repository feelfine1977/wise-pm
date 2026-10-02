"""Calibration records of population-dependent recipes (decision 0001, sections 3 and 4).

Values are hand-derived from the running example (``wise.running_p2p_log()``):
``n_events`` per case is A 4, B 7, C 4, D 4, E 2, so its median over all five
cases is 4; over the cases B and E alone it is (7 + 2) / 2 = 4.5. The log runs
from 2023-12-31 to 2024-01-31; B and E together from 2023-12-31 to 2024-01-13.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import wise
from wise.calibration import Calibration, CalibrationStatus, parse_calibration
from wise.errors import NormError
from wise.recipes import QuantileScaleRecipe, parse_recipe

pytestmark = pytest.mark.spec("decision:0001/3")

SCALED: dict[str, Any] = {"name": "scaled", "kind": "quantile_scale", "attribute": "n_events", "q": 0.5}
FULL_WINDOW = ("2023-12-31T00:00:00", "2024-01-31T00:00:00")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def log() -> wise.EventLog:
    return wise.running_p2p_log()


@pytest.fixture
def norm() -> wise.Norm:
    return wise.running_p2p_norm().replace(derived_attributes=(SCALED,))


def _cases(log: wise.EventLog, ids: list[str]) -> wise.EventLog:
    events = log.events
    return wise.EventLog(events[events["case"].isin(ids)], schema=log.schema)


def _fitted(divisor: float = 4.0, **changes: Any) -> Calibration:
    record = Calibration.fitted(0.5, divisor, log_fingerprint="f" * 64, size=5, window=FULL_WINDOW)
    return dataclasses.replace(record, **changes) if changes else record


# ----------------------------------------------------------------------------- the record
def test_a_fitted_record_holds_divisor_quantile_fingerprint_size_window_and_status() -> None:
    record = _fitted()
    assert (record.divisor, record.quantile, record.log_fingerprint, record.size, record.window) == (
        4.0,
        0.5,
        "f" * 64,
        5,
        FULL_WINDOW,
    )
    assert record.status is CalibrationStatus.FITTED
    assert record.is_fitted


def test_an_unfitted_record_holds_only_its_quantile() -> None:
    record = Calibration(0.95)
    assert record.status is CalibrationStatus.UNFITTED
    assert not record.is_fitted
    assert record.to_dict() == {
        "status": "unfitted",
        "quantile": 0.95,
        "divisor": None,
        "log_fingerprint": None,
        "size": None,
        "window": None,
    }


def test_the_record_is_frozen() -> None:
    record = _fitted()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.divisor = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: Calibration(1.5), "quantile must be a number between 0 and 1"),
        (lambda: Calibration(True), "quantile must be a number between 0 and 1"),  # type: ignore[arg-type]
        (lambda: Calibration(0.5, "guessed"), "status must be one of"),  # type: ignore[arg-type]
        (lambda: Calibration(0.5, divisor=4.0), "an unfitted calibration holds no divisor"),
        (lambda: Calibration(0.5, CalibrationStatus.FITTED, divisor=4.0), "needs log_fingerprint, size, window"),
        (lambda: _fitted(divisor=0.0), "divisor must be a finite number > 0"),
        (lambda: _fitted(divisor=float("inf")), "divisor must be a finite number > 0"),
        (lambda: _fitted(log_fingerprint=""), "log_fingerprint must be a non-empty string"),
        (lambda: _fitted(size=-1), "size must be an integer >= 0"),
        (lambda: _fitted(size=2.5), "size must be an integer >= 0"),
        (lambda: _fitted(window=("2024-01-01",)), r"window must be a pair \[start, end\]"),
        (lambda: _fitted(window=(1, 2)), "window must hold ISO 8601 text or null"),
    ],
)
def test_a_malformed_record_is_refused(make: Any, message: str) -> None:
    with pytest.raises(NormError, match=message):
        make()


@pytest.mark.parametrize("record", [_fitted(), Calibration(0.5)], ids=["fitted", "unfitted"])
def test_the_record_round_trips_through_the_norm_document(record: Calibration, norm: wise.Norm) -> None:
    assert parse_calibration(json.loads(json.dumps(record.to_dict()))) == record
    with_record = norm.replace(derived_attributes=({**SCALED, "calibration": record.to_dict()},))
    again = wise.Norm.loads(with_record.dumps())
    assert again.recipes[0] == QuantileScaleRecipe("scaled", "n_events", 0.5, record)
    assert again.recipes[0].calibration == record  # type: ignore[attr-defined]
    assert again.dumps() == with_record.dumps()
    assert again.recipes[0].to_dict()["calibration"] == record.to_dict()


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        ("fitted", "calibration is malformed: calibration must be a mapping"),
        ({"status": "unfitted"}, "calibration is malformed: calibration is missing 'quantile'"),
        ({"status": "unfitted", "quantile": 0.5, "seed": 1}, "calibration is malformed: calibration has unknown key 'seed'"),
        ({"status": "fitted", "quantile": 0.5, "divisor": 4.0}, "calibration is malformed: a fitted calibration needs"),
        ({"status": "unfitted", "quantile": 0.95}, "calibration quantile 0.95 differs from the recipe's q 0.5"),
    ],
)
def test_a_malformed_calibration_document_is_refused_with_the_recipe_named(calibration: Any, message: str) -> None:
    with pytest.raises(NormError, match=rf"derived attribute 'scaled' \(quantile_scale\): {message}"):
        wise.Norm.from_dict({**wise.running_p2p_norm().to_dict(), "derived_attributes": [{**SCALED, "calibration": calibration}]})


def test_a_stored_divisor_outside_a_record_is_refused() -> None:
    with pytest.raises(NormError, match=r"'scaled' \(quantile_scale\): divisor is not read; .* 'calibration'"):
        parse_recipe({**SCALED, "divisor": 4.0})


# ----------------------------------------------------------------------------- fitting
def test_fit_calibration_returns_the_norm_with_fitted_records(norm: wise.Norm, log: wise.EventLog) -> None:
    other = {"name": "gr", "kind": "count", "activities": ["GR"]}
    norm = norm.replace(derived_attributes=(other, {**SCALED, "divisor": None}))
    fitted = wise.fit_calibration(norm, log)
    record = fitted.recipes[1].calibration  # type: ignore[attr-defined]
    assert record == Calibration.fitted(0.5, 4.0, log_fingerprint=log.fingerprint(), size=5, window=FULL_WINDOW)
    assert fitted.derived_attributes == (other, {**SCALED, "calibration": record.to_dict()})
    assert norm.recipes[1].calibration is None  # type: ignore[attr-defined]
    assert norm.derived_attributes[1] == {**SCALED, "divisor": None}
    # fitted on the cases B and E only
    sub = _cases(log, ["B", "E"])
    on_sub = wise.fit_calibration(norm, sub).recipes[1].calibration  # type: ignore[attr-defined]
    assert (on_sub.divisor, on_sub.size, on_sub.window, on_sub.log_fingerprint) == (
        4.5,
        2,
        ("2023-12-31T00:00:00", "2024-01-13T00:00:00"),
        sub.fingerprint(),
    )


def test_fitting_again_replaces_the_record(norm: wise.Norm, log: wise.EventLog) -> None:
    once = wise.fit_calibration(norm, _cases(log, ["B", "E"]))
    twice = wise.fit_calibration(once, log)
    assert twice.recipes[0].calibration.divisor == 4.0  # type: ignore[attr-defined]
    assert twice == wise.fit_calibration(norm, log)


def test_a_norm_without_population_dependent_recipes_is_returned_as_it_is(log: wise.EventLog) -> None:
    norm = wise.running_p2p_norm()
    assert wise.fit_calibration(norm, log) == norm


def test_a_recipe_on_a_scaled_attribute_is_fitted_on_the_values_of_the_fitted_record(log: wise.EventLog) -> None:
    # halved: n_events / 8 clipped -> A .5, B .875, C .5, D .5, E .25; its maximum is .875
    halved = {"name": "halved", "kind": "quantile_scale", "attribute": "n_events", "q": 1.0}
    again = {"name": "again", "kind": "quantile_scale", "attribute": "halved", "q": 1.0}
    norm = wise.running_p2p_norm().replace(derived_attributes=(halved, again))
    fitted = wise.fit_calibration(norm, log)
    assert [r.calibration.divisor for r in fitted.recipes] == [7.0, 1.0]  # type: ignore[attr-defined]


def test_no_shipped_norm_carries_a_fitted_divisor() -> None:
    for path in sorted(EXAMPLES.glob("*.json")):
        for recipe in wise.Norm.load(path).recipes:
            if isinstance(recipe, QuantileScaleRecipe):
                assert recipe.calibration is None, path.name


# ----------------------------------------------------------------------------- scoring
@pytest.mark.spec("decision:0001/3")
def test_scoring_with_a_fitted_record_uses_its_divisor_whatever_cases_are_scored(norm: wise.Norm, log: wise.EventLog) -> None:
    fitted = wise.fit_calibration(norm, log)
    sub = _cases(log, ["B", "E"])
    assert wise.score(sub, fitted).cases["scaled"].tolist() == [1.0, 0.5]  # 7/4 clipped, 2/4
    assert wise.score(log, fitted).cases["scaled"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.5]
    result = wise.score(sub, fitted)
    assert result.calibrations == {"scaled": fitted.recipes[0].calibration}  # type: ignore[attr-defined]


@pytest.mark.spec("decision:0001/4")
@pytest.mark.parametrize("calibration", [None, Calibration(0.5).to_dict()], ids=["no-record", "unfitted"])
def test_without_a_fitted_record_the_legacy_profile_divides_by_the_scored_quantile_and_records_it(
    calibration: Any, log: wise.EventLog
) -> None:
    recipe = SCALED if calibration is None else {**SCALED, "calibration": calibration}
    norm = wise.running_p2p_norm().replace(derived_attributes=(recipe,))
    sub = _cases(log, ["B", "E"])
    result = wise.score(sub, norm)
    assert result.cases["scaled"].tolist() == pytest.approx([1.0, 2 / 4.5], abs=0, rel=1e-15)
    assert result.calibrations == {
        "scaled": Calibration.fitted(
            0.5, 4.5, log_fingerprint=sub.fingerprint(), size=2, window=("2023-12-31T00:00:00", "2024-01-13T00:00:00")
        )
    }
    whole = wise.score(log, norm)
    assert whole.cases["scaled"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.5]
    assert whole.calibrations["scaled"].divisor == 4.0


@pytest.mark.spec("decision:0001/4")
def test_on_the_log_it_was_fitted_on_a_fitted_record_gives_the_classic_tables(norm: wise.Norm, log: wise.EventLog) -> None:
    norm = norm.replace_constraint("c1", constraint=wise.Metric("scaled", threshold=0.6))
    classic = wise.score(log, norm)
    fitted_norm = wise.fit_calibration(norm, log)
    fitted = wise.score(log, fitted_norm)
    for table in ("cases", "violations", "in_scope", "scores"):
        pd.testing.assert_frame_equal(getattr(fitted, table), getattr(classic, table), check_exact=True)
    assert fitted.calibrations == classic.calibrations == {"scaled": fitted_norm.recipes[0].calibration}  # type: ignore[attr-defined]


def test_a_run_without_derived_attributes_records_no_calibration(norm: wise.Norm, log: wise.EventLog) -> None:
    assert wise.score(log, wise.running_p2p_norm()).calibrations == {}
    log.derive(norm.derived_attributes)
    assert wise.score(log, norm, derive=False).calibrations == {}
