import pytest

from trainite.config.base import ComponentConfig
from trainite.utils import get_target, instantiate


def test_get_target_valid_and_invalid():
    # valid target
    target = get_target("trainite.utils.get_target")
    assert target is get_target

    # empty target raises ValueError
    with pytest.raises(ValueError):
        get_target("")

    # invalid target path raises ImportError
    with pytest.raises(ImportError):
        get_target("trainite.nonexistent.Nope")


def test_instantiate_invalid_inputs():
    # instantiate requires a ComponentConfig instance
    with pytest.raises(ValueError):
        instantiate("not_a_config")

    # constructing a bare ComponentConfig without _target_ will raise when validating,
    # but instantiate only accepts ComponentConfig instances; ensure non-instance is rejected above.
