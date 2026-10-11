# Events without a timestamp, case attributes that vary, and duplicate events

The log schema (`wise.LogSchema`) carries three rules of the method's input
(paper Sec. IV-A) as policies. Their defaults reproduce 0.1.0, so classic
numbers do not move; decision 0001 (section 4) keys these rules on the log
schema in the same way as the semantics profile on the norm.

## What an event without a timestamp contributes

`missing_timestamps` decides whether such events are refused (`raise`, the
default), removed (`drop`) or kept (`keep`). For kept events,
`untimed_events` states what each primitive does with them:

- `legacy` (default) is the 0.1.0 rule;
- `uniform` is one rule for every primitive: an untimed event counts wherever
  no time is compared, and never defines a time or a position relative to
  another event.

The two values differ in one row: `EventLog.count_scoped` without a scope,
which the `count` recipe computes with. Under `legacy` a `count` recipe and a
`Presence` constraint on the same activity can disagree on a case whose only
occurrence has no timestamp; under `uniform` they agree.

"counted" means that the event enters the result as if it had a timestamp;
"ignored" means that the result is the one of the same log without the event.
`tests/test_untimed_events.py` holds the same table and checks every row
against the kernel.

<!-- untimed-events-table -->
| Primitive | `legacy` (default) | `uniform` |
|---|---|---|
| `CaseIndex.count`, `EventLog.count` | counted | counted |
| `EventLog.count_scoped` without `after` and `before` | ignored | counted |
| `EventLog.count_scoped` with `after` or `before`, `count_before`, `count_after` | ignored | ignored |
| `CaseIndex.aggregate`, `EventLog.total` | counted | counted |
| `CaseIndex.first`, `CaseIndex.last`, `EventLog.first_ts`, `EventLog.last_ts` | ignored | ignored |
| `EventLog.first_after` (activation and response) | ignored | ignored |
| `EventLog.activation_lags` | ignored | ignored |
| `CaseIndex.tied` | never tied | never tied |
| `CaseIndex.run_starts` | one run per case | one run per case |
| event order (`EventLog.events`, `EventLog.trace`) | last in its case | last in its case |
| case table `n_events` and `exposure` | counted | counted |
| case table `first_ts` and `last_ts` | ignored | ignored |
| case attributes (first non-null value) | read after the timed events | read after the timed events |
| `EventLog.observation_window`, `window_start`, `window_end` | ignored | ignored |

What follows for the constraints and recipes built on these primitives:
`Presence`, and `Exclusion` and `Singularity` without a scope, count untimed
events; with `after` or `before`, and in `Precedence`'s count of early `b`
events, they are ignored; `Lag` never uses them; `Balance` totals include
their attribute values. The `count_events`, `nunique`, `agg`, `cv` and
`indicator_times` recipes count them, the `lag` recipe ignores them, and the
`count` recipe follows the row of `count_scoped`.

## A case attribute that varies within a case

A case attribute (`att_β(σ)`) has one value per case, but logs record it on
events. `case_attribute_values` decides what happens when the events of one
case disagree:

| Value | Value per case | A case with two non-null values |
|---|---|---|
| `first_non_null` (default) | the first non-null value in event order | kept; the attribute and its number of such cases are recorded in `ParsedLog.varying_case_attributes` and reported by `EventLog.validate()` as `varying_within_case[<attribute>]` |
| `strict` | the single non-null value | the log is refused with a `LogSchemaError` that names each attribute and its number of cases |

Null values never count as a second value.

## Events that share case, activity and timestamp

`duplicates` decides what happens to events that share their key: the case,
the activity and the timestamp, compared as given before the timestamps are
parsed (two missing timestamps are equal).

| Value | Events that share the key | Recorded |
|---|---|---|
| `keep` (default) | all kept | nothing; `ParsedLog.n_dropped_duplicates` and `n_cases_with_dropped_duplicates` are 0 |
| `drop` (`EventLog(dedupe=True)`) | one survivor kept, the others dropped | the number of dropped events and of their cases in `ParsedLog.n_dropped_duplicates` and `ParsedLog.n_cases_with_dropped_duplicates` (also on `EventLog`), reported by `EventLog.validate()` as `dropped_duplicates` and `cases_with_dropped_duplicates`, zero when nothing was dropped |

Rows equal in the key are dropped although they differ elsewhere: two events
with different event identifiers or amounts at the same instant count as one.
The survivor does not depend on the order of the input rows, so the same rows
in any order give the same log and the same fingerprint:

1. the smallest `order_col` value, if the schema names an order column;
2. then, among the events still equal, the smallest values of the other
   columns, compared column by column in column-name order, nulls last;
3. values that do not compare (nested values) or that are equal with
   different types (`1` and `True`) compare by type name and `repr`.

Rows that are equal in every column give the same survivor whichever is kept.
A `validate()` of a log that keeps duplicates has neither row, so its report
is the one of 0.1.0.
