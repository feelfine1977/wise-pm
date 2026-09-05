"""WISE on the running purchase-to-pay example of the paper (Tables V–VI).

Run:  python examples/quickstart.py
"""

from pathlib import Path

import pandas as pd

import wise

pd.set_option("display.width", 160)

# Phase 1 — norm and views (Table V)
norm = wise.running_p2p_norm()
print(norm)
print(norm.describe()[["layer", "type", "applicability"]], "\n")
print("Raw weights per view\n", norm.weight_table(), "\n")

# Phase 2 — score cases (Table VI)
log = wise.running_p2p_log()
print(log)
result = wise.score(log, norm)
print("Violations ν_c(σ)  (NaN = not applicable)\n", result.violations.round(3), "\n")
print("Case scores S^(p)(σ)\n", result.scores.round(4), "\n")
print("Finance layer contributions Δ_λ (sum to 1 − S)\n", result.contributions["Finance"].round(4), "\n")
result.check_decomposition()

# Phase 3 — slice backlog and explanation
backlog = wise.prioritize(result, by="company", view="Finance", gamma=1.0)
print("Backlog by company (Finance)\n", backlog.round(4), "\n")
print("Layer drivers per company\n", wise.layer_drivers(result, "company", view="Finance").round(4), "\n")
print(
    "Constraint drivers inside company B\n",
    wise.constraint_drivers(result, "Finance", {"company": "B"})[["layer", "mean_penalty", "share_violated"]].round(3),
    "\n",
)
print("Worst cases in company B\n", result.worst_cases("Finance", where={"company": "B"})[["score"]].round(4), "\n")

# The norm as a versioned artefact
path = Path(__file__).with_name("running_p2p_norm.json")
norm.dump(path)
print(f"Norm written to {path}")
