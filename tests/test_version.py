from cdtools import __version__
import re


def test_version_exists():
    """Test that version is defined and not empty."""
    assert __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format():
    """Test that version follows semantic versioning format."""
    # Basic semantic versioning pattern (X.Y.Z with optional pre-release)
    pattern = r"^\d+\.\d+\.\d+(?:[-.]?(?:alpha|beta|rc|dev)\d*)?$"
    assert re.match(
        pattern, __version__
    ), f"Version '{__version__}' doesn't follow semantic versioning"
