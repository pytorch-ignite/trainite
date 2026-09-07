from pathlib import Path

import pytest

from trainite.config.registry import REGISTRY, DatasetSpec, ModelSpec
from trainite.shared.utils import get_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _registered_specs():
    for group_name, specs in REGISTRY.items():
        for component_name, spec in specs.items():
            yield pytest.param(spec, id=f"{group_name}/{component_name}")


def test_registry_groups_are_populated():
    assert set(REGISTRY) == {"models", "datasets", "trainers", "preprocessors"}
    for group_name, specs in REGISTRY.items():
        assert specs, f"registry group '{group_name}' has no components"


@pytest.mark.parametrize("spec", _registered_specs())
def test_registry_spec_resolves(spec):
    assert (PROJECT_ROOT / spec.implementation_path).is_file()

    if spec.readme_template_path is not None:
        assert (PROJECT_ROOT / spec.readme_template_path).is_file()

    get_target(spec.config_cls_path)

    if isinstance(spec, DatasetSpec):
        get_target(spec.dataset_config_cls_path)

    if isinstance(spec, ModelSpec) and spec.collate_fn_target is not None:
        get_target(spec.collate_fn_target)
