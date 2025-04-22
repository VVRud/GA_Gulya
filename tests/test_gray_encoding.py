"""
Test cases for GrayEncoderDecoder class.
"""

import pytest

from GA.encoding.gray_encoding import GrayEncoderDecoder


def test_initialization(gray_encoder_basic, gray_encoder_decimal):
    """Test proper initialization of the encoder."""
    assert gray_encoder_basic.values_range == (0, 10)
    assert gray_encoder_basic.points_after_decimal == 0
    assert gray_encoder_basic.num_bits == 4  # 10 values need 4 bits

    assert gray_encoder_decimal.values_range == (-5, 5)
    assert gray_encoder_decimal.points_after_decimal == 2
    assert (
        gray_encoder_decimal.num_bits == 10
    )  # -5 to 5 with 2 decimal places: 10*100 = 1000 values, need 10 bits


def test_encode_single_value(gray_encoder_basic, gray_encoder_decimal):
    """Test encoding a single value."""
    # Test encoding value 5 (middle of range) in basic encoder
    # Binary for 5 is 0101, Gray code is 0111
    encoded = gray_encoder_basic.encode([5])
    assert len(encoded) == 4
    assert encoded == [0, 1, 1, 1]

    # Test encoding 0.0 in decimal encoder
    # 0.0 is 500 steps from -5.0 (middle of range), binary for 500 is 0111110100
    # Gray code for 500 is 0100001110
    encoded = gray_encoder_decimal.encode([0.0])
    assert len(encoded) == 10
    assert encoded == [0, 1, 0, 0, 0, 0, 1, 1, 1, 0]


def test_encode_multiple_values(gray_encoder_basic):
    """Test encoding multiple values."""
    # Binary for 1 is 0001, Gray code is 0001
    # Binary for 7 is 0111, Gray code is 0100
    encoded = gray_encoder_basic.encode([1, 7])
    assert len(encoded) == 8  # 2 values * 4 bits
    assert encoded == [0, 0, 0, 1, 0, 1, 0, 0]


def test_decode_single_value(gray_encoder_basic, gray_encoder_decimal):
    """Test decoding a single value."""
    # Gray code 0111 (binary 0101 = 5) using basic encoder
    decoded = gray_encoder_basic.decode([0, 1, 1, 1])
    assert len(decoded) == 1
    assert decoded[0] == 5.0

    # Decode a value from decimal encoder
    # Gray code 0100001110 = binary 0111110100 = 500
    decoded = gray_encoder_decimal.decode([0, 1, 0, 0, 0, 0, 1, 1, 1, 0])
    assert len(decoded) == 1
    assert decoded[0] == pytest.approx(0.0, abs=0.01)


def test_decode_multiple_values(gray_encoder_basic):
    """Test decoding multiple values."""
    # Gray code for 1 (0001) and 7 (0100)
    decoded = gray_encoder_basic.decode([0, 0, 0, 1, 0, 1, 0, 0])
    assert len(decoded) == 2
    assert decoded == [1.0, 7.0]


def test_encode_decode_roundtrip():
    """Test encoding and then decoding returns original values."""
    original_values = [1.5, 3.25, 4.75]
    encoder = GrayEncoderDecoder(values_range=(0, 10), points_after_decimal=2)

    encoded = encoder.encode(original_values)
    decoded = encoder.decode(encoded)

    assert len(decoded) == len(original_values)
    for original, result in zip(original_values, decoded, strict=False):
        assert result == pytest.approx(original, abs=0.01)


def test_value_range_validation(gray_encoder_basic):
    """Test validation of input values against the specified range."""
    with pytest.raises(AssertionError):
        gray_encoder_basic.encode([-1])  # Below range

    with pytest.raises(AssertionError):
        gray_encoder_basic.encode([10])  # At upper bound (exclusive)


def test_binary_validation(gray_encoder_basic):
    """Test validation of binary input."""
    with pytest.raises(AssertionError):
        gray_encoder_basic.decode([0, 1, 2])  # 2 is not a binary digit

    with pytest.raises(AssertionError):
        gray_encoder_basic.decode([0, 1, 0])  # Not a multiple of num_bits


@pytest.fixture
def gray_encoder_for_bits():
    """Fixture for gray encoder with exact 3 bits."""
    encoder = GrayEncoderDecoder(values_range=(0, 8), points_after_decimal=0)
    encoder.num_bits = 3  # Override to use exactly 3 bits for testing
    return encoder


def test_binary_to_gray_conversion(gray_encoder_for_bits):
    """Test that binary to Gray code conversion works correctly."""
    # Compare with known binary to Gray code conversions
    # For 0-7:
    # Binary: 000, 001, 010, 011, 100, 101, 110, 111
    # Gray:   000, 001, 011, 010, 110, 111, 101, 100
    expected_gray_for_0_to_7 = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]

    for i in range(8):
        encoded = gray_encoder_for_bits.encode([i])
        assert encoded == expected_gray_for_0_to_7[i], (
            f"Gray code for {i} doesn't match expected"
        )


def test_gray_to_binary_conversion(gray_encoder_for_bits):
    """Test that Gray code to binary conversion works correctly."""
    # Test the reverse of the binary_to_gray test
    gray_for_0_to_7 = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]

    for i, gray in enumerate(gray_for_0_to_7):
        decoded = gray_encoder_for_bits.decode(gray)
        assert decoded[0] == float(i)


def test_hamming_distance_property():
    """Test the key property of Gray code: adjacent values differ by only one bit."""
    encoder = GrayEncoderDecoder(values_range=(0, 16), points_after_decimal=0)

    for i in range(15):  # Test for values 0 to 14 (comparing with next value)
        encoded_current = encoder.encode([i])
        encoded_next = encoder.encode([i + 1])

        # Calculate Hamming distance (number of bits that differ)
        hamming_distance = sum(
            bit1 != bit2
            for bit1, bit2 in zip(encoded_current, encoded_next, strict=False)
        )

        # Gray code property: adjacent values should differ by exactly one bit
        assert hamming_distance == 1, (
            f"Gray codes for {i} and {i + 1} don't differ by exactly one bit"
        )
