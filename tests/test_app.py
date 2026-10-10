import numpy as np


def test_app_imports_and_classify_callable():
    import app as A
    assert callable(A.classify) and callable(A.add_letter)


def test_add_letter_backspace_logic():
    import app as A
    assert A.add_letter("", "H") == "H"
    assert A.add_letter("H", "I") == "HI"
    assert A.add_letter("HI", "space") == "HI "
    assert A.add_letter("HI", "del") == "H"
    assert A.add_letter("HI", "nothing") == "HI"
    assert A.add_letter("HI", "uncertain") == "HI"  # rejected predictions never append
    assert A.add_letter("HI", None) == "HI"


def test_classify_none_image_message():
    import app as A
    out = A.classify(None)
    assert "first" in out[1].lower() or "upload" in out[1].lower()


def test_clear_prediction_resets_state():
    import app as A
    assert A.clear_prediction() is None
    assert A.add_letter("HI", A.clear_prediction()) == "HI"  # stale Add is a no-op


def test_app_builds():
    import app as A
    assert A.build_app() is not None
