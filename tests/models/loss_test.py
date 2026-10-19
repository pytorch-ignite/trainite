import pytest
import torch
import torch.nn.functional as F

from trainite.config.registry import MODEL_SPECS
from trainite.shared.utils import instantiate


@pytest.mark.parametrize("model", ["basic-transformer", "rope-transformer", "gemma4-dense"])
@pytest.mark.parametrize("flatten", [False, True])
def test_model_cross_entropy_wrapper(model, flatten):
    criterion = instantiate(MODEL_SPECS[model].loss_config_cls(ignore_index=-7))
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[-7, 1, 2], [3, 4, 0]])
    expected = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-7)
    inputs = logits.reshape(-1, 5) if flatten else logits
    targets = labels.reshape(-1) if flatten else labels
    actual = criterion(inputs, targets)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
