"""
rk_pipeline -- RansKnow's config-driven experiment framework (Phase 1+).

Four small registries (tasks, feature representations, models, CV split
protocols) let an experiment be specified as four strings instead of new
code. See Scripts/run_experiment.py for the CLI entry point.

Submodules register themselves as a side effect of being imported, so
they're imported here explicitly -- referencing the shared registry
objects alone (e.g. `from .registry import MODELS`) does not trigger
`Scripts/rk_pipeline/models.py`'s `MODELS.register(...)` calls unless
that module has actually been imported somewhere.
"""

from . import augmenters as _augmenters  # noqa: F401
from . import explainers as _explainers  # noqa: F401
from . import features as _features      # noqa: F401
from . import models as _models          # noqa: F401
from . import splits as _splits          # noqa: F401
