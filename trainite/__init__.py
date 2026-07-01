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
    from importlib.abc import Loader, MetaPathFinder
    from importlib.machinery import ModuleSpec
    from types import ModuleType
    from unittest.mock import MagicMock

    class MockObject(MagicMock):
        def __getattr__(self, name: str) -> "MockObject":
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            if name.startswith("_mock") or name.startswith("_spec"):
                return super().__getattr__(name)
            return MockObject()

        def __or__(self, other) -> "MockObject":
            return self

        def __ror__(self, other) -> "MockObject":
            return self

    class MockModule(ModuleType):
        def __getattr__(self, name: str) -> MockObject:
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return MockObject()

    class MockLoader(Loader):
        def __init__(self, fullname: str) -> None:
            self.fullname = fullname

        def create_module(self, spec: ModuleSpec) -> MockModule:
            return MockModule(self.fullname)

        def exec_module(self, module: ModuleType) -> None:
            pass

    class MockFinder(MetaPathFinder):
        def __init__(self, mock_prefixes: list[str]) -> None:
            self.mock_prefixes = mock_prefixes

        def find_spec(self, fullname: str, path, target=None) -> ModuleSpec | None:
            for prefix in self.mock_prefixes:
                if fullname == prefix or fullname.startswith(prefix + "."):
                    spec = ModuleSpec(fullname, MockLoader(fullname), is_package=True)
                    spec.submodule_search_locations = []
                    return spec
            return None

    sys.meta_path.insert(0, MockFinder(mock_targets))
