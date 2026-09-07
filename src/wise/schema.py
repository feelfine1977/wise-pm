"""What the norm catalogue actually supports, read off the implementation.

A proposal — from a reviewer, a form, or a language model — is only worth
validating against the checks that really exist. This module publishes them,
and it derives almost everything from the maintained code rather than from a
second, hand-written list that would drift the first time a parameter moved:

* the constraint types come from :data:`wise.constraints.CONSTRAINT_TYPES` and
  their aliases from :data:`wise.constraints.TYPE_ALIASES`;
* each type's parameters, defaults and optionality come from its dataclass
  fields, and their shapes from the field annotations;
* the recipe kinds and their keys come from :data:`wise.derive.KINDS`,
  :data:`wise.derive.REQUIRED_RECIPE_KEYS` and
  :data:`wise.derive.OPTIONAL_RECIPE_KEYS`;
* the time units come from the same table the constraints parse.

The one thing that cannot be read off a dataclass is a *bound* — that a
presence needs ``m >= 1``, that a balance tolerance is a share in ``[0, 1]`` —
because those live inside ``__post_init__``. They are declared in
:data:`NUMERIC_BOUNDS`, and :func:`verify_catalogue` fails if that table and
the implementation ever disagree: every numeric parameter must be bounded and
every bound must name a parameter that exists. The catalogue builders call it,
so a drifted table cannot be published.

Usage::

    from wise.schema import constraint_catalogue, recipe_catalogue

    constraint_catalogue()["lag"].parameter("delta").minimum   # 0.0
    recipe_catalogue()["quantile_scale"].required              # ('attribute',)

Nothing here evaluates a proposal or contacts anything. It reports what the
library supports; :mod:`wise.llm.drafts` is what decides whether a particular
untrusted candidate may be previewed.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .constraints import _UNITS, CONSTRAINT_TYPES, TYPE_ALIASES, Constraint
from .derive import KINDS, OPTIONAL_RECIPE_KEYS, REQUIRED_RECIPE_KEYS, UNSAFE_RECIPE_KINDS
from .errors import NormError
from .norm import _TOP_KEYS

#: Version of the published catalogue shape. Bumped when a field changes meaning.
CATALOGUE_VERSION = "wise-catalogue/1"

#: The aggregations a ``where``-filtered ``agg`` recipe accepts, as documented
#: in :mod:`wise.derive` and passed straight to pandas.
RECIPE_AGGREGATIONS = ("sum", "mean", "max", "min", "std", "first", "last")

#: The comparison keys a recipe ``where`` clause accepts, from
#: :func:`wise.derive._where_mask`.
WHERE_OPERATORS = ("in", "not_in", "regex", "eq")


@dataclass(frozen=True)
class Bound:
    """A numeric parameter's admissible range, as ``__post_init__`` enforces it.

    ``minimum``/``maximum`` are inclusive unless the matching ``exclusive_``
    flag says otherwise. ``integer`` marks a parameter that is coerced to
    :class:`int`. ``unit`` names what the number counts, where it counts
    anything.
    """

    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    integer: bool = False
    unit: str = ""

    def contains(self, value: float) -> bool:
        """Whether ``value`` satisfies this bound (finiteness checked elsewhere).

        >>> Bound(minimum=1, integer=True).contains(0)
        False
        >>> Bound(minimum=0.0, maximum=1.0).contains(0.5)
        True
        """
        below = self.minimum is not None and (value < self.minimum or (self.exclusive_minimum and value == self.minimum))
        above = self.maximum is not None and (value > self.maximum or (self.exclusive_maximum and value == self.maximum))
        not_whole = self.integer and float(value) != int(value)
        return not (below or above or not_whole)

    def describe(self) -> str:
        """A short human reading of the range.

        >>> Bound(minimum=0.0, exclusive_minimum=True).describe()
        '> 0.0'
        """
        parts = []
        if self.minimum is not None:
            parts.append(f"{'>' if self.exclusive_minimum else '>='} {self.minimum}")
        if self.maximum is not None:
            parts.append(f"{'<' if self.exclusive_maximum else '<='} {self.maximum}")
        if self.integer:
            parts.append("integer")
        return ", ".join(parts) if parts else "finite"

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "exclusive_minimum": self.exclusive_minimum,
            "exclusive_maximum": self.exclusive_maximum,
            "integer": self.integer,
            "unit": self.unit,
        }


#: The numeric ranges the constraint constructors enforce, keyed by
#: ``(type, parameter)``. This is the one declared table in the module, and
#: :func:`verify_catalogue` holds it against the dataclasses: a numeric
#: parameter with no entry, or an entry naming a parameter that does not
#: exist, is a hard error rather than a quietly missing check.
NUMERIC_BOUNDS: dict[tuple[str, str], Bound] = {
    ("presence", "m"): Bound(minimum=1, integer=True, unit="occurrences"),
    ("singularity", "k"): Bound(minimum=0, integer=True, unit="occurrences"),
    ("singularity", "K"): Bound(minimum=0.0, exclusive_minimum=True, unit="occurrences"),
    ("lag", "delta"): Bound(minimum=0.0, unit="time units"),
    ("lag", "width"): Bound(minimum=0.0, unit="time units"),
    ("precedence", "k"): Bound(minimum=0, integer=True, unit="occurrences"),
    ("precedence", "K"): Bound(minimum=0.0, exclusive_minimum=True, unit="occurrences"),
    ("balance", "tau"): Bound(minimum=0.0, maximum=1.0, unit="share"),
    ("balance", "width"): Bound(minimum=0.0, unit="share"),
    ("balance", "eps"): Bound(minimum=0.0, exclusive_minimum=True, unit="amount"),
    ("metric", "threshold"): Bound(unit="attribute units"),
    ("metric", "width"): Bound(minimum=0.0, unit="attribute units"),
}

#: Parameters whose admissible values are a closed set the annotation does not
#: carry. Only ``lag.unit`` needs it, and the set is read from the units table
#: the constraints themselves parse.
_ENUMERATED: dict[tuple[str, str], tuple[str, ...]] = {("lag", "unit"): tuple(sorted(_UNITS))}

_LITERAL = re.compile(r"^Literal\[(.*)\]$")


def _split_optional(annotation: str) -> tuple[str, bool]:
    """``"float | None"`` → ``("float", True)``."""
    text = annotation.strip()
    if text.endswith("| None"):
        return text[: -len("| None")].strip(), True
    return text, False


def _parse_annotation(annotation: str) -> tuple[str, tuple[str, ...]]:
    """The declared shape of a parameter, and its choices where it has them.

    >>> _parse_annotation("Literal['high', 'low']")
    ('choice', ('high', 'low'))
    >>> _parse_annotation("Activities | None")
    ('activities', ())
    """
    base, _ = _split_optional(str(annotation))
    match = _LITERAL.match(base)
    if match is not None:
        choices = tuple(part.strip().strip("'\"") for part in match.group(1).split(",") if part.strip())
        return "choice", choices
    shape = {"Activities": "activities", "int": "integer", "float": "number", "str": "text", "bool": "boolean"}.get(base)
    if shape is None:
        raise NormError(
            f"schema catalogue: parameter annotation {annotation!r} is not one this module understands; "
            "teach wise.schema the new shape rather than letting a proposal be validated against nothing"
        )
    return shape, ()


@dataclass(frozen=True)
class ParameterSpec:
    """One constraint parameter: its shape, optionality, default and range."""

    name: str
    shape: str
    required: bool
    nullable: bool
    default: Any = None
    choices: tuple[str, ...] = ()
    bound: Bound | None = None

    @property
    def numeric(self) -> bool:
        return self.shape in ("integer", "number")

    @property
    def minimum(self) -> float | None:
        return None if self.bound is None else self.bound.minimum

    @property
    def maximum(self) -> float | None:
        return None if self.bound is None else self.bound.maximum

    @property
    def unit(self) -> str:
        return "" if self.bound is None else self.bound.unit

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "required": self.required,
            "nullable": self.nullable,
            "default": self.default,
            "choices": list(self.choices),
            "bound": None if self.bound is None else self.bound.to_dict(),
        }


@dataclass(frozen=True)
class ConstraintSpec:
    """One supported constraint type, as the implementation defines it."""

    type: str
    aliases: tuple[str, ...]
    summary: str
    parameters: tuple[ParameterSpec, ...]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def parameter(self, name: str) -> ParameterSpec:
        for p in self.parameters:
            if p.name == name:
                return p
        raise NormError(f"constraint type {self.type!r} has no parameter {name!r}; it has {list(self.parameter_names)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "aliases": list(self.aliases),
            "summary": self.summary,
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass(frozen=True)
class RecipeSpec:
    """One supported derived-attribute kind and the keys it reads."""

    kind: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    evaluates_expression: bool

    @property
    def keys(self) -> tuple[str, ...]:
        """Every key this kind accepts, ``name`` and ``kind`` included."""
        return ("name", "kind", *self.required, *self.optional)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "required": list(self.required),
            "optional": list(self.optional),
            "evaluates_expression": self.evaluates_expression,
        }


def _summary(cls: type[Constraint]) -> str:
    doc = (cls.__doc__ or "").strip().splitlines()
    return doc[0].strip() if doc else ""


def verify_catalogue() -> None:
    """Raise :class:`~wise.errors.NormError` if the declared table has drifted.

    Every numeric parameter of every supported constraint must carry a bound,
    every bound and every enumeration must name a parameter that exists, and
    every annotation must be one this module can read. Called by the catalogue
    builders, so a drifted table can never be published as if it were current.

    >>> verify_catalogue()
    """
    known: set[tuple[str, str]] = set()
    for type_name, cls in CONSTRAINT_TYPES.items():
        for f in dataclasses.fields(cls):
            known.add((type_name, f.name))
            shape, _ = _parse_annotation(str(f.type))
            if shape in ("integer", "number") and (type_name, f.name) not in NUMERIC_BOUNDS:
                raise NormError(
                    f"schema catalogue: {type_name}.{f.name} is numeric but has no declared bound; "
                    "add it to wise.schema.NUMERIC_BOUNDS so a proposal is checked against the real range"
                )
    for key in NUMERIC_BOUNDS:
        if key not in known:
            raise NormError(f"schema catalogue: bound declared for {key[0]}.{key[1]}, which is not a parameter of that type")
    for key in _ENUMERATED:
        if key not in known:
            raise NormError(f"schema catalogue: choices declared for {key[0]}.{key[1]}, which is not a parameter of that type")
    for kind in KINDS:
        if kind not in REQUIRED_RECIPE_KEYS or kind not in OPTIONAL_RECIPE_KEYS:
            raise NormError(f"schema catalogue: recipe kind {kind!r} has no declared key table in wise.derive")


def constraint_catalogue() -> dict[str, ConstraintSpec]:
    """Every supported constraint type, read off the implementation.

    >>> catalogue = constraint_catalogue()
    >>> sorted(catalogue)
    ['balance', 'exclusion', 'lag', 'metric', 'precedence', 'presence', 'singularity']
    >>> catalogue["presence"].parameter("m").bound.describe()
    '>= 1, integer'
    >>> catalogue["lag"].parameter("unit").choices
    ('D', 'H', 'S', 'T', 'W', 'd', 'h', 'min', 'ms', 's', 'w')
    """
    verify_catalogue()
    aliases: dict[str, list[str]] = {}
    for alias, target in TYPE_ALIASES.items():
        aliases.setdefault(target, []).append(alias)
    out: dict[str, ConstraintSpec] = {}
    for type_name, cls in CONSTRAINT_TYPES.items():
        parameters = []
        for f in dataclasses.fields(cls):
            shape, choices = _parse_annotation(str(f.type))
            _, nullable = _split_optional(str(f.type))
            required = f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
            parameters.append(
                ParameterSpec(
                    name=f.name,
                    shape=shape,
                    required=required,
                    nullable=nullable,
                    default=None if required else f.default,
                    choices=_ENUMERATED.get((type_name, f.name), choices),
                    bound=NUMERIC_BOUNDS.get((type_name, f.name)),
                )
            )
        out[type_name] = ConstraintSpec(
            type=type_name,
            aliases=tuple(sorted(aliases.get(type_name, []))),
            summary=_summary(cls),
            parameters=tuple(parameters),
        )
    return out


def recipe_catalogue() -> dict[str, RecipeSpec]:
    """Every supported derived-attribute kind and the keys it reads.

    >>> recipe_catalogue()["quantile_scale"].keys
    ('name', 'kind', 'attribute', 'q')
    >>> recipe_catalogue()["eval"].evaluates_expression
    True
    """
    verify_catalogue()
    return {
        kind: RecipeSpec(
            kind=kind,
            required=tuple(REQUIRED_RECIPE_KEYS[kind]),
            optional=tuple(OPTIONAL_RECIPE_KEYS[kind]),
            evaluates_expression=kind in UNSAFE_RECIPE_KINDS,
        )
        for kind in KINDS
    }


def supported_units() -> tuple[str, ...]:
    """The time units a :class:`~wise.Lag` accepts.

    >>> "D" in supported_units() and "h" in supported_units()
    True
    """
    return tuple(sorted(_UNITS))


def norm_top_level_keys() -> tuple[str, ...]:
    """The keys a schema-2 norm document may carry, from the loader itself.

    >>> "constraints" in norm_top_level_keys()
    True
    """
    return tuple(sorted(_TOP_KEYS))


def catalogue_digest() -> str:
    """A stable digest of the published catalogue, for a draft's provenance.

    Two libraries that agree on this digest validated a candidate against the
    same supported vocabulary.

    >>> len(catalogue_digest())
    64
    """
    import hashlib
    import json

    payload = {
        "version": CATALOGUE_VERSION,
        "constraints": {k: v.to_dict() for k, v in sorted(constraint_catalogue().items())},
        "recipes": {k: v.to_dict() for k, v in sorted(recipe_catalogue().items())},
        "units": list(supported_units()),
        "norm_keys": list(norm_top_level_keys()),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def describe_catalogue() -> dict[str, Any]:
    """The whole supported vocabulary as plain JSON types.

    This is what an application shows a reviewer, or puts in front of a model
    as *the* list of things that exist. It asserts no semantics beyond the
    library's own.
    """
    return {
        "catalogue_version": CATALOGUE_VERSION,
        "catalogue_digest": catalogue_digest(),
        "constraints": [spec.to_dict() for spec in constraint_catalogue().values()],
        "recipes": [spec.to_dict() for spec in recipe_catalogue().values()],
        "units": list(supported_units()),
        "norm_top_level_keys": list(norm_top_level_keys()),
        "recipe_aggregations": list(RECIPE_AGGREGATIONS),
        "where_operators": list(WHERE_OPERATORS),
    }


def check_parameters(type_name: str, params: Mapping[str, Any]) -> list[str]:
    """Findings for one constraint's parameters, without constructing it.

    Returns a list of human-readable findings — unknown parameter, missing
    required parameter, wrong shape, value outside its declared bound, value
    outside a closed set. An empty list means the parameters are admissible as
    far as the catalogue can tell; it is not a claim that the constraint is
    *right*.

    >>> check_parameters("presence", {"activity": "Approve", "m": 0})
    ['presence.m: 0 is outside the supported range (>= 1, integer)']
    >>> check_parameters("presence", {"activity": "Approve"})
    []
    """
    catalogue = constraint_catalogue()
    key = TYPE_ALIASES.get(type_name, type_name)
    if key not in catalogue:
        return [f"unknown constraint type {type_name!r}; supported: {sorted(catalogue)}"]
    spec = catalogue[key]
    findings: list[str] = []
    for name in params:
        if name not in spec.parameter_names:
            findings.append(f"{spec.type}: unknown parameter {name!r}; supported: {list(spec.parameter_names)}")
    for parameter in spec.parameters:
        if parameter.name not in params:
            if parameter.required:
                findings.append(f"{spec.type}: required parameter {parameter.name!r} is missing")
            continue
        findings.extend(_check_value(spec.type, parameter, params[parameter.name]))
    return findings


def _check_value(type_name: str, parameter: ParameterSpec, value: Any) -> list[str]:
    where = f"{type_name}.{parameter.name}"
    if value is None:
        return [] if parameter.nullable else [f"{where}: null is not admissible here"]
    if parameter.choices:
        # a closed set closes the parameter, whatever its annotation says: the
        # units of a lag are declared in the units table, not in ``str``
        if value not in parameter.choices:
            return [f"{where}: {value!r} is not one of {list(parameter.choices)}"]
        return []
    if parameter.shape == "activities":
        labels = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else None
        if labels is None or not labels or not all(isinstance(v, str) and v for v in labels):
            return [f"{where}: expected an activity label or a list of labels"]
        return []
    if parameter.shape == "text":
        return [] if isinstance(value, str) and value else [f"{where}: expected a non-empty string"]
    if parameter.shape == "boolean":
        return [] if isinstance(value, bool) else [f"{where}: expected true or false"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return [f"{where}: expected a number, got {type(value).__name__}"]
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return [f"{where}: a non-finite number is not a parameter"]
    bound = parameter.bound
    if bound is not None and not bound.contains(number):
        return [f"{where}: {value!r} is outside the supported range ({bound.describe()})"]
    return []


__all__ = [
    "CATALOGUE_VERSION",
    "NUMERIC_BOUNDS",
    "RECIPE_AGGREGATIONS",
    "WHERE_OPERATORS",
    "Bound",
    "ConstraintSpec",
    "ParameterSpec",
    "RecipeSpec",
    "catalogue_digest",
    "check_parameters",
    "constraint_catalogue",
    "describe_catalogue",
    "norm_top_level_keys",
    "recipe_catalogue",
    "supported_units",
    "verify_catalogue",
]
