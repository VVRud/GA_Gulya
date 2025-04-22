from .base import BaseEncoderDecoder


class BinaryEncoderDecoder(BaseEncoderDecoder):
    def encode(self, x: list[float]) -> list[int]:
        """
        Encodes a list of float values into a flat list of binary digits (0s and 1s).

        Args:
            x: List of float values to encode

        Returns:
            List of integers (0s and 1s) representing the binary encoding
        """
        assert all(
            self.values_range[0] <= number < self.values_range[1] for number in x
        ), "Values must be within specified range"

        result = []
        for value in self._scale_values(x):
            result.extend(self._encode_value(value))
        return result

    def _encode_value(self, value: float) -> list[int]:
        """
        Encodes a single float value into a list of binary digits (0s and 1s).
        """
        binary = bin(int(value))[2:].zfill(self.num_bits)
        return [int(bit) for bit in binary]

    def decode(self, x: list[int]) -> list[float]:
        """
        Decodes a flat list of binary digits back to float values.

        Args:
            x: List of binary digits (0s and 1s)

        Returns:
            List of decoded float values
        """
        assert len(x) % self.num_bits == 0, (
            f"Input length must be a multiple of {self.num_bits}"
        )
        assert all(bit in (0, 1) for bit in x), (
            "Input must contain only binary digits (0s and 1s)"
        )

        result = []
        for i in range(0, len(x), self.num_bits):
            result.append(
                self._return_value_to_scale(
                    self._decode_value(x[i : i + self.num_bits])
                )
            )
        return result

    def _decode_value(self, value: list[int]) -> int:
        """
        Decodes a list of binary digits back to a integer value.
        """
        return int("".join(str(bit) for bit in value), 2)
