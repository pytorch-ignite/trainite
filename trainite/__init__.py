__version__ = "0.0.1"

import sys

mock_targets = []

try:
    import torch
except ImportError:
    mock_targets.append("torch")

try:
    import ignite
except ImportError:
    mock_targets.append("ignite")

if mock_targets:
    import importlib.machinery
    from types import ModuleType
    from unittest.mock import MagicMock

    class DynamicMockModule(ModuleType):
        def __getattr__(self, name):
            return MagicMock()

    class DependencyMocker:
        def __init__(self, *ignored_packages):
            self.ignored_packages = ignored_packages

        def find_spec(self, fullname, path, target=None):
            if any(fullname == pkg or fullname.startswith(pkg + ".") for pkg in self.ignored_packages):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            return DynamicMockModule(spec.name)

        def exec_module(self, module):
            sys.modules[module.__name__] = module

            # If this is a submodule (e.g. 'torch.nn'), we must attach it to its
            # parent module ('torch') so that dot-notation syntax does not crash.
            if "." in module.__name__:
                parent_name, child_name = module.__name__.rsplit(".", 1)
                parent_module = sys.modules.get(parent_name)
                if parent_module:
                    setattr(parent_module, child_name, module)

    sys.meta_path.insert(0, DependencyMocker(*mock_targets))
