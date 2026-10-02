"""Read-only containers for the values a frozen record hands out.

:class:`FrozenDict` and :class:`FrozenList` are ``dict`` and ``list``
subclasses whose mutating methods raise :class:`TypeError`, so a value read
from a frozen record still compares, serialises to JSON and passes
``isinstance(x, dict)`` / ``isinstance(x, list)`` like the document it came
from, but writing into it cannot change the record. :func:`freeze` converts a
nested document into them (tuples keep their type, numpy arrays become
read-only copies); :func:`thaw` converts back into plain, independent
containers.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, NoReturn, SupportsIndex, TypeVar

import numpy as np

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def _refuse(obj: object) -> NoReturn:
    raise TypeError(f"'{type(obj).__name__}' is read-only; copy it (dict(...) or list(...)) to change it")


class FrozenDict(dict[K, V]):
    """A ``dict`` that refuses to be changed."""

    def __setitem__(self, key: K, value: V) -> NoReturn:
        _refuse(self)

    def __delitem__(self, key: K) -> NoReturn:
        _refuse(self)

    def __ior__(self, other: Any) -> NoReturn:  # type: ignore[misc,override]
        _refuse(self)

    def clear(self) -> NoReturn:
        _refuse(self)

    def pop(self, *args: Any) -> NoReturn:  # type: ignore[override]
        _refuse(self)

    def popitem(self) -> NoReturn:
        _refuse(self)

    def setdefault(self, *args: Any) -> NoReturn:  # type: ignore[override]
        _refuse(self)

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        _refuse(self)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return (type(self), (dict(self),))

    def __copy__(self) -> FrozenDict[K, V]:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict[K, V]:
        return type(self)({k: copy.deepcopy(v, memo) for k, v in self.items()})


class FrozenList(list[T]):
    """A ``list`` that refuses to be changed."""

    def __setitem__(self, index: Any, value: Any) -> NoReturn:
        _refuse(self)

    def __delitem__(self, index: Any) -> NoReturn:
        _refuse(self)

    def __iadd__(self, other: Any) -> NoReturn:  # type: ignore[misc,override]
        _refuse(self)

    def __imul__(self, other: Any) -> NoReturn:  # type: ignore[misc,override]
        _refuse(self)

    def append(self, value: T) -> NoReturn:
        _refuse(self)

    def extend(self, values: Any) -> NoReturn:
        _refuse(self)

    def insert(self, index: SupportsIndex, value: T) -> NoReturn:
        _refuse(self)

    def remove(self, value: T) -> NoReturn:
        _refuse(self)

    def pop(self, index: SupportsIndex = -1) -> NoReturn:
        _refuse(self)

    def clear(self) -> NoReturn:
        _refuse(self)

    def sort(self, *args: Any, **kwargs: Any) -> NoReturn:
        _refuse(self)

    def reverse(self) -> NoReturn:
        _refuse(self)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return (type(self), (list(self),))

    def __copy__(self) -> FrozenList[T]:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenList[T]:
        return type(self)(copy.deepcopy(v, memo) for v in self)


def freeze(value: Any) -> Any:
    """``value`` with every mapping, list, set and array replaced by a read-only, independent counterpart.

    Mappings become :class:`FrozenDict`, lists :class:`FrozenList`, sets ``frozenset``, numpy arrays read-only
    copies; tuples keep their type with frozen items. Any other value is deep-copied.
    """
    if isinstance(value, FrozenDict | FrozenList):
        return value
    if isinstance(value, Mapping):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return FrozenList(freeze(v) for v in value)
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(freeze(v) for v in value)
    if isinstance(value, set | frozenset):
        return frozenset(freeze(v) for v in value)
    if isinstance(value, np.ndarray):
        out = value.copy()
        out.flags.writeable = False
        return out
    return copy.deepcopy(value)


def thaw(value: Any) -> Any:
    """Plain, independent containers for a value made by :func:`freeze` (the inverse on documents)."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, list):
        return [thaw(v) for v in value]
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(thaw(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


__all__ = ["FrozenDict", "FrozenList", "freeze", "thaw"]
