"""
Test cases for BinaryEncoderDecoder class.
"""

import numpy as np
import pytest

from GA.encoding.binary_encoding import BinaryEncoderDecoder


def test_initialization(binary_encoder_basic, binary_encoder_decimal):
    """Test proper initialization of the encoder."""
    assert binary_encoder_basic.values_range == (0, 10)
    assert binary_encoder_basic.points_after_decimal == 0
    assert binary_encoder_basic.num_bits == 4  # 10 values need 4 bits

    assert binary_encoder_decimal.values_range == (-5, 5)
    assert binary_encoder_decimal.points_after_decimal == 2
    assert (
        binary_encoder_decimal.num_bits == 10
    )  # -5 to 5 with 2 decimal places: 10*100 = 1000 values, need 10 bits


def test_encode_single_value(binary_encoder_basic, binary_encoder_decimal):
    """Test encoding a single value."""
    # Test encoding value 5 (middle of range) in basic encoder
    encoded = binary_encoder_basic.encode([5])
    assert len(encoded) == 4
    assert encoded == [0, 1, 0, 1]  # Binary for 5

    # Test encoding 0.0 in decimal encoder
    encoded = binary_encoder_decimal.encode([0.0])
    assert len(encoded) == 10
    # 0.0 is 500 steps from -5.0 (middle of range), binary for 500
    assert encoded == [0, 1, 1, 1, 1, 1, 0, 1, 0, 0]


def test_encode_multiple_values(binary_encoder_basic):
    """Test encoding multiple values."""
    encoded = binary_encoder_basic.encode([1, 7])
    assert len(encoded) == 8  # 2 values * 4 bits
    assert encoded == [0, 0, 0, 1, 0, 1, 1, 1]  # Binary for 1 and 7


def test_decode_single_value(binary_encoder_basic, binary_encoder_decimal):
    """Test decoding a single value."""
    # Decode binary 0101 (5) using basic encoder
    decoded = binary_encoder_basic.decode([0, 1, 0, 1])
    assert len(decoded) == 1
    assert decoded[0] == 5.0

    # Decode a value from decimal encoder
    decoded = binary_encoder_decimal.decode(
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0]
    )  # Binary for 500
    assert len(decoded) == 1
    assert decoded[0] == pytest.approx(0.0, abs=0.01)


def test_decode_multiple_values(binary_encoder_basic):
    """Test decoding multiple values."""
    decoded = binary_encoder_basic.decode(
        [0, 0, 0, 1, 0, 1, 1, 1]
    )  # Binary for 1 and 7
    assert len(decoded) == 2
    assert decoded == [1.0, 7.0]


def test_encode_decode_roundtrip():
    """Test encoding and then decoding returns original values."""
    original_values = [1.5, 3.25, 4.75]
    encoder = BinaryEncoderDecoder(values_range=(0, 10), points_after_decimal=2)

    encoded = encoder.encode(original_values)
    decoded = encoder.decode(encoded)

    assert len(decoded) == len(original_values)
    for original, result in zip(original_values, decoded, strict=False):
        assert result == pytest.approx(original, abs=0.01)


def test_value_range_validation(binary_encoder_basic):
    """Test validation of input values against the specified range."""
    with pytest.raises(AssertionError):
        binary_encoder_basic.encode([-1])  # Below range

    with pytest.raises(AssertionError):
        binary_encoder_basic.encode([10])  # At upper bound (exclusive)


def test_binary_validation(binary_encoder_basic):
    """Test validation of binary input."""
    with pytest.raises(AssertionError):
        binary_encoder_basic.decode([0, 1, 2])  # 2 is not a binary digit

    with pytest.raises(AssertionError):
        binary_encoder_basic.decode([0, 1, 0])  # Not a multiple of num_bits


def test_internal_methods(binary_encoder_basic, binary_encoder_decimal):
    """Test internal encoding/decoding methods."""
    # Test _encode_value
    encoded = binary_encoder_basic._encode_value(5)
    assert encoded == [0, 1, 0, 1]

    # Test _decode_value
    decoded = binary_encoder_basic._decode_value([0, 1, 0, 1])
    assert decoded == 5

    # Test _scale_values
    scaled = binary_encoder_decimal._scale_values([0.0])
    assert scaled == [500.0]

    # Test _return_value_to_scale
    original = binary_encoder_decimal._return_value_to_scale(500)
    assert original == 0.0


@pytest.mark.parametrize("value", range(-1024, 1024))
def test_all_values_in_range(value):
    """Test that all values in the range are encoded and decoded correctly."""
    encoder = BinaryEncoderDecoder(values_range=(-1.024, 1.024), points_after_decimal=3)
    to_encode = np.round(value / 1000.0, 3)
    encoded = encoder.encode([to_encode])
    assert encoded == [
        int(v) for v in format(value + 1024, "b").zfill(encoder.num_bits)
    ], f"Failed for value {to_encode}"
    assert pytest.approx(encoder.decode(encoded), 0.01) == [to_encode], (
        f"Failed for value {to_encode}"
    )


@pytest.mark.parametrize("value", range(-102, 102))
def test_all_values_in_range_cutoff(value):
    """Test that all values in the range are encoded and decoded correctly."""
    encoder = BinaryEncoderDecoder(values_range=(-1.024, 1.024), points_after_decimal=2)
    to_encode = np.round(value / 100.0, 2)
    encoded = encoder.encode([to_encode])
    assert encoded == [
        int(v) for v in format(value + 102, "b").zfill(encoder.num_bits)
    ], f"Failed for value {to_encode}"
    assert encoder.decode(encoded) == pytest.approx([to_encode], 0.01), (
        f"Failed for value {to_encode}"
    )
