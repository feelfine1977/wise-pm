"""Properties that must hold for any log and norm (randomised)."""

import numpy as np
import pandas as pd
import pytest

import wise

ACTS = ["PO", "GR", "INV", "CLR", "CINV", "CHG"]


def random_log(rng, n_cases=120):
    rows = []
    for i in range(n_cases):
        n = int(rng.integers(1, 9))
        acts = rng.choice(ACTS, size=n)
        days = np.sort(rng.uniform(0, 60, size=n))
        for a, d in zip(acts, days):
            rows.append(
                {
                    "case": f"c{i:03d}",
                    "activity": a,
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=float(d)),
                    "amount": float(rng.uniform(0, 100)),
                    "flow": rng.choice(["DF1", "DF2"]),
                    "region": rng.choice(list("XYZ")),
                }
            )
    df = pd.DataFrame(rows)
    return wise.EventLog(
        df,
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow", "region"],
        exposure_col="amount",
        exposure_agg="sum",
    )


def random_norm(rng):
    cons = [
        wise.NormConstraint("pres", "L1", wise.Presence("INV"), weight=rng.uniform(0.5, 2)),
        wise.NormConstraint(
            "lag",
            "L2",
            wise.Lag("GR", "INV", delta=5, width=10, missing_b=rng.choice(["violate", "skip", "censor"])),
            weight=rng.uniform(0.5, 2),
            applicability={"flow": ["DF1"]},
        ),
        wise.NormConstraint(
            "each", "L2", wise.Lag("GR", "INV", delta=5, width=10, activation="each"), weight=rng.uniform(0.5, 2)
        ),
        wise.NormConstraint("prec", "L2", wise.Precedence("GR", "INV"), weight=rng.uniform(0.5, 2)),
        wise.NormConstraint("sing", "L3", wise.Singularity("GR", k=1, K=2), weight=rng.uniform(0.5, 2)),
        wise.NormConstraint("excl", "L3", wise.Exclusion("CINV", after="INV"), weight=rng.uniform(0.5, 2)),
        wise.NormConstraint(
            "bal", "L4", wise.Balance("amount", "INV", "amount", "GR", tau=0.1, width=0.5), weight=rng.uniform(0.5, 2)
        ),
    ]
    layers = [wise.Layer(f"L{i}") for i in range(1, 5)]
    views = [
        wise.View("A", layer_weights={f"L{i}": float(rng.uniform(0.05, 1)) for i in range(1, 5)}),
        wise.View("B", constraint_weights={c.id: float(rng.uniform(0, 1)) for c in cons}),
    ]
    return wise.Norm(cons, layers, views)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("mode", ["flat", "layer_balanced"])
def test_score_invariants(seed, mode):
    rng = np.random.default_rng(seed)
    log, norm = random_log(rng), random_norm(rng)
    res = wise.score(log, norm, mode=mode)
    V = res.violations.to_numpy()
    assert np.nanmin(V) >= 0 and np.nanmax(V) <= 1
    S = res.scores.to_numpy()
    assert np.nanmin(S) >= -1e-12 and np.nanmax(S) <= 1 + 1e-12
    assert res.check_decomposition(1e-9) < 1e-9
    # unscored ⇔ no evaluated constraint with positive weight
    w = norm.weight_vector("B").to_numpy()
    none = (res.applicable.to_numpy() * (w > 0)).sum(axis=1) == 0
    assert (res.scores["B"].isna().to_numpy() == none).all()
    # weights are scale free
    scaled = norm.replace(
        views=(norm.views[0], wise.View("B", constraint_weights={k: 7 * v for k, v in norm.views[1].constraint_weights.items()}))
    )
    pd.testing.assert_series_equal(wise.score(log, scaled, mode=mode).scores["B"], res.scores["B"])
    # shuffling events does not change anything
    shuffled = wise.EventLog(
        log.events.sample(frac=1, random_state=seed),
        case_col="case",
        activity_col="activity",
        timestamp_col="time",
        case_attributes=["flow", "region"],
        exposure_col="amount",
        exposure_agg="sum",
    )
    pd.testing.assert_frame_equal(wise.score(shuffled, norm, mode=mode).violations, res.violations)
    # sum rule: Σ_s n_s μ_s = |Σ| μ̄  (γ = 0, case volume)
    b = wise.prioritize(res, "region", view="A")
    assert (b["n_cases"] * b["mean_score"]).sum() == pytest.approx(res.scores["A"].dropna().sum())
    assert (b["PI"] >= 0).all()
    # JSON round trip preserves scores
    pd.testing.assert_frame_equal(wise.score(log, wise.Norm.loads(norm.dumps()), mode=mode).scores, res.scores)
