import pandas as pd
import wise

# Event log: one row per event. Column names default to the pm4py conventions
# (case:concept:name, concept:name, time:timestamp); pass others explicitly.
events = pd.read_csv("log.csv")
log = wise.EventLog(
    events,
    case_col="case",
    activity_col="activity",
    timestamp_col="time",
    case_attributes=["flow_type", "company", "spend_area", "vendor"],
    exposure_col="net_worth",  # optional: exposure-weighted priorities
)

# Norm N = (C, Λ, lay) with views; built in code or loaded from a JSON file
norm = wise.Norm(
    constraints=[
        wise.NormConstraint("c1", "completeness", wise.Presence("Record Invoice Receipt")),
        wise.NormConstraint(
            "c2", "lead_times", wise.Lag("Record Goods Receipt", "Record Invoice Receipt", delta=10, width=20, unit="D")
        ),
        wise.NormConstraint(
            "c3",
            "match",
            wise.Balance("amount", "Record Invoice Receipt", "amount", "Record Goods Receipt", tau=0.05, width=0.20),
        ),
        wise.NormConstraint("c5", "handling", wise.Singularity("Record Goods Receipt", k=2, K=3)),
        wise.NormConstraint(
            "c6", "exceptions", wise.Exclusion("Cancel Invoice Receipt"), applicability={"flow_type": ["DF1", "DF2"]}
        ),
    ],
    layers=[wise.Layer(name) for name in ["completeness", "lead_times", "match", "handling", "exceptions"]],
    views=[
        wise.View("Finance", constraint_weights={"c1": 0.20, "c2": 0.45, "c3": 0.20, "c5": 0.05, "c6": 0.10}),
        wise.View(
            "Logistics",
            layer_weights={"completeness": 0.25, "lead_times": 0.15, "match": 0.05, "handling": 0.45, "exceptions": 0.10},
        ),
    ],
)
norm.dump("norm_v1.json")  # versioned artefact; wise.Norm.load(...) reads it back

# Score
result = wise.score(log, norm)
result.violations  # cases × constraints, ν_c(σ); NaN = not applicable
result.scores  # cases × views, S^(p)(σ); NaN = unscored
result.contributions["Finance"]  # cases × layers, Δ_λ (sums to 1 − S)

# Prioritise and explain
backlog = wise.prioritize(result, by=["company", "spend_area"], view="Finance", gamma=20)
wise.layer_drivers(result, by=["company", "spend_area"], view="Finance")
wise.constraint_drivers(result, "Finance", {"company": "A", "spend_area": "Packaging"})
wise.hotspot_table(backlog, drivers=wise.layer_drivers(result, ["company", "spend_area"], view="Finance"))
wise.concentration(backlog)  # share of slices carrying 80 % / 95 % of the priority mass
wise.view_agreement(result, by=["company", "spend_area"], k=20)
result.worst_cases("Finance", where={"company": "A"})
