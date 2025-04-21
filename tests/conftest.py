"""
Pytest configuration file.
"""
import pytest
import sys
import os

# Add the parent directory to the path so we can import GA modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

@pytest.fixture
def binary_encoder_basic():
    """Fixture for basic binary encoder with integer values."""
    from GA.encoding.binary_encoding import BinaryEncoderDecoder
    return BinaryEncoderDecoder(values_range=(0, 10), points_after_decimal=0)

@pytest.fixture
def binary_encoder_decimal():
    """Fixture for binary encoder with decimal values."""
    from GA.encoding.binary_encoding import BinaryEncoderDecoder
    return BinaryEncoderDecoder(values_range=(-5, 5), points_after_decimal=2)

@pytest.fixture
def binary_encoder_large_range():
    """Fixture for binary encoder with a large value range."""
    from GA.encoding.binary_encoding import BinaryEncoderDecoder
    return BinaryEncoderDecoder(values_range=(0, 1000), points_after_decimal=1)

@pytest.fixture
def gray_encoder_basic():
    """Fixture for basic gray encoder with integer values."""
    from GA.encoding.gray_encoding import GrayEncoderDecoder
    return GrayEncoderDecoder(values_range=(0, 10), points_after_decimal=0)

@pytest.fixture
def gray_encoder_decimal():
    """Fixture for gray encoder with decimal values."""
    from GA.encoding.gray_encoding import GrayEncoderDecoder
    return GrayEncoderDecoder(values_range=(-5, 5), points_after_decimal=2)

@pytest.fixture
def gray_encoder_large_range():
    """Fixture for gray encoder with a large value range."""
    from GA.encoding.gray_encoding import GrayEncoderDecoder
    return GrayEncoderDecoder(values_range=(0, 1000), points_after_decimal=1) 