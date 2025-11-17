"""Basic import tests to ensure package is properly installed."""

import sys
from pathlib import Path


def test_package_import():
    """Test that the main package can be imported."""
    try:
        import src
        assert hasattr(src, "__version__")
        assert hasattr(src, "__author__")
    except ImportError as e:
        pytest.fail(f"Failed to import src package: {e}")


def test_version_string():
    """Test that version is properly formatted."""
    import src
    assert isinstance(src.__version__, str)
    assert len(src.__version__) > 0
    # Version should follow semantic versioning (e.g., 1.0.0)
    parts = src.__version__.split(".")
    assert len(parts) == 3


def test_author_info():
    """Test that author information is present."""
    import src
    assert src.__author__ == "Ruslan Magana"
    assert "ruslanmv.com" in src.__email__
    assert src.__license__ == "Apache-2.0"
