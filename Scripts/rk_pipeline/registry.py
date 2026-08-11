"""Generic name -> object registry used for tasks, features, models, splits, and explainers."""

from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str):
        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"Unknown {self._kind} '{name}'. Available: {self.names()}")
        return self._items[name]

    def names(self):
        return sorted(self._items)


TASKS = Registry("task")
FEATURES = Registry("feature representation")
MODELS = Registry("model")
SPLITS = Registry("split protocol")
EXPLAINERS = Registry("explainer")
