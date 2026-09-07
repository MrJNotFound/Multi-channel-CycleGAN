"""Lazy, narrowly scoped compatibility for pre-reorganization Python imports.

No old implementation copies are kept in the models root. Importing e.g.
models.spif_model re-exports the same class objects from the organized module;
unrelated third-party imports are not intercepted.
"""

import importlib
import importlib.abc
import importlib.util
import sys

from .registry import IMPLEMENTATIONS


class _LegacyModelLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    aliases = {f"models.{name}_model": path for name, (path, _) in IMPLEMENTATIONS.items()}
    aliases["models.patchnce"] = "models.losses.patchnce"

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.aliases:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        implementation = importlib.import_module(self.aliases[module.__name__])
        module.__doc__ = f"Compatibility alias for {implementation.__name__}."
        for name, value in vars(implementation).items():
            if not name.startswith("__"):
                setattr(module, name, value)


def install_legacy_imports():
    if not any(isinstance(finder, _LegacyModelLoader) for finder in sys.meta_path):
        sys.meta_path.append(_LegacyModelLoader())
