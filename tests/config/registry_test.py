from pathlib import Path

import pytest

import trainite
from trainite.config.registry import (
    DATASET_SPECS,
    MODEL_SPECS,
    PREPROCESSOR_SPECS,
    REGISTRY,
    TRAINER_SPECS,
    DatasetSpec,
    ModelSpec,
)
from trainite.shared.utils import get_target

REPO_ROOT = Path(trainite.__file__).resolve().parent.parent

EXPECTED_NAMES = {
    "models": {"basic-transformer", "rope-transformer"},
    "datasets": {"string-reverse", "counting", "hugging-face"},
    "trainers": {"decoder-trainer"},
    "preprocessors": {"char", "gpt2"},
}

ALL_SPECS = [(group, key, spec) for group, specs in REGISTRY.items() for key, spec in specs.items()]
ALL_SPEC_IDS = [f"{group}:{key}" for group, key, _ in ALL_SPECS]


@pytest.mark.parametrize("group", sorted(EXPECTED_NAMES))
def test_registry_group_contains_expected_names(group):
    assert set(REGISTRY[group]) == EXPECTED_NAMES[group]


def test_registry_groups_map_to_spec_dicts():
    assert REGISTRY == {
        "models": MODEL_SPECS,
        "datasets": DATASET_SPECS,
        "trainers": TRAINER_SPECS,
        "preprocessors": PREPROCESSOR_SPECS,
    }


@pytest.mark.parametrize(("group", "key", "spec"), ALL_SPECS, ids=ALL_SPEC_IDS)
def test_implementation_path_exists(group, key, spec):
    assert (REPO_ROOT / spec.implementation_path).is_file()


@pytest.mark.parametrize(("group", "key", "spec"), ALL_SPECS, ids=ALL_SPEC_IDS)
def test_readme_template_path_exists_when_provided(group, key, spec):
    if spec.readme_template_path is None:
        pytest.skip("no readme template configured")
    assert (REPO_ROOT / spec.readme_template_path).is_file()


@pytest.mark.parametrize(("group", "key", "spec"), ALL_SPECS, ids=ALL_SPEC_IDS)
def test_config_cls_path_resolves(group, key, spec):
    assert get_target(spec.config_cls_path) is spec.config_cls


@pytest.mark.parametrize(
    ("key", "spec"),
    sorted(DATASET_SPECS.items()),
    ids=sorted(DATASET_SPECS),
)
def test_dataset_config_cls_path_resolves(key, spec):
    assert isinstance(spec, DatasetSpec)
    assert get_target(spec.dataset_config_cls_path) is spec.dataset_config_cls


@pytest.mark.parametrize(
    ("key", "spec"),
    sorted(MODEL_SPECS.items()),
    ids=sorted(MODEL_SPECS),
)
def test_collate_fn_target_resolves(key, spec):
    assert isinstance(spec, ModelSpec)
    if spec.collate_fn_target is None:
        pytest.skip("no collate_fn configured")
    assert get_target(spec.collate_fn_target) is not None


@pytest.mark.parametrize(("key", "spec"), sorted(DATASET_SPECS.items()), ids=sorted(DATASET_SPECS))
def test_dataset_preprocessor_spec_name_is_registered(key, spec):
    if spec.preprocessor_spec_name is None:
        pytest.skip("no preprocessor configured")
    assert spec.preprocessor_spec_name in PREPROCESSOR_SPECS
