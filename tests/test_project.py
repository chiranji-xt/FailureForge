"""Day 1 smoke test: the package imports and exposes its version."""

import failureforge


def test_package_imports():
    assert failureforge is not None


def test_version_exposed():
    assert isinstance(failureforge.__version__, str)
    assert failureforge.__version__ == "0.1.0"
