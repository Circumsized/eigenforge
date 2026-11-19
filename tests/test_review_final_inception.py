"""Offline regression tests for the official Inception preprocessing protocol."""

import pytest
import torch
from torchvision.models import Inception_V3_Weights

from eigenforge.evaluation.evaluator import GenAIEvaluator


@pytest.mark.parametrize("height,width", [(48, 80), (80, 48)])
def test_inception_preprocess_matches_official_transform_on_rectangular_images(height, width):
    # Spatial variation exposes aspect-ratio distortion and the missing center crop.
    images = torch.linspace(0, 1, 2 * 3 * height * width).reshape(2, 3, height, width)
    expected = Inception_V3_Weights.DEFAULT.transforms()(images)

    actual = GenAIEvaluator._inception_preprocess(images)

    assert actual.shape == (2, 3, 299, 299)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 16, 24), (1, 1, 16, 24), (1, 3, 16, 24, 1)])
def test_inception_preprocess_preserves_nchw_validation(shape):
    with pytest.raises(ValueError, match="images must be N x 3 x H x W"):
        GenAIEvaluator._inception_preprocess(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_inception_preprocess_preserves_finite_validation(value):
    images = torch.zeros(1, 3, 16, 24)
    images[0, 0, 0, 0] = value

    with pytest.raises(ValueError, match="images contain non-finite values"):
        GenAIEvaluator._inception_preprocess(images)


# ----------------------------------------------- weights_only=True weight loading


def test_inception_weights_are_loaded_with_weights_only_true(monkeypatch):
    """torchvision's own ``inception_v3(weights=...)`` reaches ``torch.hub`` with its
    default ``weights_only=False``; the evaluator must request ``True`` itself."""
    from torchvision.models import inception_v3

    seen: dict = {}
    donor = inception_v3(weights=None, aux_logits=True, transform_input=False, init_weights=False)

    def fake_get_state_dict(self, *args, **kwargs):
        seen.update(kwargs)
        return donor.state_dict()

    monkeypatch.setattr(Inception_V3_Weights, "get_state_dict", fake_get_state_dict)
    monkeypatch.setenv("EIGENFORGE_ALLOW_UNSAFE_TORCH", "1")
    model = GenAIEvaluator._load_inception("cpu")

    assert seen.get("weights_only") is True
    assert seen.get("check_hash") is True
    assert model.transform_input is False  # preprocessing is the official transform's job
    assert not model.training
    torch.testing.assert_close(
        model.Mixed_7c.branch1x1.conv.weight, donor.Mixed_7c.branch1x1.conv.weight, rtol=0, atol=0
    )


def test_evaluator_never_builds_inception_through_the_unsafe_weights_path():
    import inspect

    import eigenforge.evaluation.evaluator as evaluator_module

    source = inspect.getsource(evaluator_module)
    assert "inception_v3(weights=Inception_V3_Weights" not in source
    assert "transform_input=True" not in source


def test_clean_fid_score_is_actually_logged(monkeypatch):
    """loguru uses ``{}`` formatting; a ``%.4f`` template is emitted literally."""
    import sys
    import types

    from eigenforge.common.logger import logger

    fake = types.ModuleType("cleanfid")
    fake.fid = types.SimpleNamespace(compute_fid=lambda *args, **kwargs: 12.3456)
    monkeypatch.setitem(sys.modules, "cleanfid", fake)
    captured: list[str] = []
    handler_id = logger.add(lambda message: captured.append(message.record["message"]), level="INFO")
    try:
        score = GenAIEvaluator("cpu").compute_clean_fid("real", "fake")
    finally:
        logger.remove(handler_id)

    assert score == pytest.approx(12.3456)
    assert any("12.3456" in message and "%" not in message for message in captured), captured
