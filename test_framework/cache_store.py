"""Small in-memory LRU cache store used by inference cache experiments."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Generic, Iterator, MutableMapping, Optional, Tuple, TypeVar


K = TypeVar("K")
V = TypeVar("V")


class LruCache(Generic[K, V]):
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: "OrderedDict[K, V]" = OrderedDict()
        self.evictions = 0

    def get(self, key: K) -> Optional[V]:
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: K, value: V) -> None:
        if key in self._items:
            self._items.pop(key)
        self._items[key] = value
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> Iterator[Tuple[K, V]]:
        return iter(self._items.items())

    def as_dict(self) -> MutableMapping[K, V]:
        return self._items
