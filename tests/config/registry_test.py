from trainite.config.registry import REGISTRY
from trainite.shared.utils import get_target


def test_model_registry_names():
    assert "basic-transformer" in REGISTRY["models"]
    assert "rope-transformer" in REGISTRY["models"]


def test_dataset_registry_names():
    assert "string-reverse" in REGISTRY["datasets"]
    assert "counting" in REGISTRY["datasets"]
    assert "hugging-face" in REGISTRY["datasets"]


def test_trainer_registry_names():
    assert "decoder-trainer" in REGISTRY["trainers"]


def test_preprocessor_registry_names():
    assert "char" in REGISTRY["preprocessors"]
    assert "gpt2" in REGISTRY["preprocessors"]


def test_implementation_paths_exist():
    for specs in REGISTRY.values():
        for spec in specs.values():
            assert spec.implementation_path.is_file()


def test_readme_template_paths_exist():
    for specs in REGISTRY.values():
        for spec in specs.values():
            if spec.readme_template_path is not None:
                assert spec.readme_template_path.is_file()


def test_config_cls_paths_resolve():
    for specs in REGISTRY.values():
        for spec in specs.values():
            get_target(spec.config_cls_path)


def test_dataset_config_cls_paths_resolve():
    for spec in REGISTRY["datasets"].values():
        get_target(spec.dataset_config_cls_path)


def test_collate_fn_targets_resolve():
    for spec in REGISTRY["models"].values():
        if spec.collate_fn_target is not None:
            get_target(spec.collate_fn_target)
