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

## Rows that repeat another row

`duplicates` decides what happens to rows that repeat another row of the log.
Both policies that remove rows compare the values as given, before the
timestamps are parsed (two missing timestamps are equal), keep one row of each
group, and record what they dropped in the same way.

| Value | Rows dropped | Recorded |
|---|---|---|
| `keep` (default) | none | nothing; `ParsedLog.n_dropped_duplicates` and `n_cases_with_dropped_duplicates` are 0 |
| `drop` (`EventLog(dedupe=True)` or `dedupe="drop"`) | rows that share their key, the case, the activity and the timestamp, with a survivor, although they differ in other columns | the number of dropped events and of their cases in `ParsedLog.n_dropped_duplicates` and `ParsedLog.n_cases_with_dropped_duplicates` (also on `EventLog`), reported by `EventLog.validate()` as `dropped_duplicates` and `cases_with_dropped_duplicates`, zero when nothing was dropped |
| `identical` (`EventLog(dedupe="identical")`) | rows equal to a survivor in every column the log reads: the named columns and the kept ones (`keep_columns`) | as for `drop` |

Which to use:

- `identical` when the same record was loaded twice (an export read twice, a
  join that repeated rows). Two different events of one case with the same
  activity at the same instant, with other identifiers or amounts, are both
  kept.
- `drop` only when case, activity and time are known to identify an event.
  It keeps one of such two events and drops the other.

### `drop`: the key

Rows equal in the key are dropped although they differ elsewhere: two events
with different event identifiers or amounts at the same instant count as one.
The survivor does not depend on the order of the input rows, so the same rows
in any order give the same log and the same fingerprint:

1. the smallest `order_col` value, if the schema names an order column;
2. then, among the events still equal, the smallest values of the other
   columns, compared column by column in column-name order, nulls last;
3. values that do not compare (nested values) or that are equal with
   different types (`1` and `True`) compare by type name and `repr`;
   `0.0` comes before `-0.0`.

Rows that are equal in every column give the same survivor whichever is kept.

### `identical`: every column read

Two cells are equal when they hold the same type and the same value: `1`,
`1.0`, `True` and `"1"` differ, and so do `0.0` and `-0.0`. Nested cells (a
mapping, list, set or array) compare by their canonical form, the one the
fingerprint covers: the order of the keys of a mapping or of the items of a
set does not matter, the order of a list does. Two missing values in the same
column are equal (`None`, `NaN`, `NaT` and `pd.NA` alike). A column the log
does not keep (`keep_columns`) is not compared.

Since the rows of a group are equal, the log does not depend on which one
stays; where two missing values of different kinds are equal, the survivor is
chosen by the rule of `drop`, so the log and its fingerprint are the same for
every order of the input rows. The dropped rows were copies of rows the log
kept, so the fingerprint of such a log is the one of the log built with
`keep` from the rows it kept; it differs from the fingerprint of a log built
with `keep` from all the rows, and from one built with `drop`.

A `validate()` of a log that keeps duplicates has neither row, so its report
is the one of 0.1.0.
