import numpy as np
import pytest

from src.features import FEATURE_DIM, augment, mirror_x, normalize


def _hand(offset=(0.2, 0.3, 0.0), scale=0.4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((21, 3)) * scale + np.array(offset)).astype(np.float32)


def test_shape_and_no_nan():
    v = normalize(_hand())
    assert v.shape == (FEATURE_DIM,)
    assert np.isfinite(v).all()


def test_translation_invariance():
    a, b = normalize(_hand(offset=(0.1, 0.1, 0))), normalize(_hand(offset=(0.7, 0.6, 0)))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_scale_invariance():
    a, b = normalize(_hand(scale=0.2)), normalize(_hand(scale=0.8))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_wrist_at_origin():
    assert np.allclose(normalize(_hand()).reshape(21, 3)[0], 0, atol=1e-6)


def test_rejects_nan():
    bad = _hand();
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        normalize(bad)


def test_mirror_twice_is_identity():
    lm = _hand()
    np.testing.assert_allclose(mirror_x(mirror_x(lm)), lm, atol=1e-6)


def test_augment_shape_finite():
    out = augment(normalize(_hand()), np.random.default_rng(0))
    assert out.shape == (FEATURE_DIM,) and np.isfinite(out).all()
