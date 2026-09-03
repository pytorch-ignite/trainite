import torch.utils.data
import pytest


# Store the original init method
_original_dataloader_init = torch.utils.data.DataLoader.__init__


def patched_dataloader_init(self, *args, **kwargs):
    # Override num_workers to 0 to avoid process spawning overhead during tests
    kwargs["num_workers"] = 0
    _original_dataloader_init(self, *args, **kwargs)


@pytest.fixture(autouse=True, scope="session")
def force_dataloader_num_workers_zero():
    torch.utils.data.DataLoader.__init__ = patched_dataloader_init
    yield
    torch.utils.data.DataLoader.__init__ = _original_dataloader_init


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.option.markexpr:
        return

    integration_tests = []
    regular_tests = []

    for item in items:
        if "integration" in item.keywords:
            integration_tests.append(item)
        else:
            regular_tests.append(item)

    items[:] = regular_tests
    config.hook.pytest_deselected(items=integration_tests)