from .binary_encoding import BinaryEncoderDecoder
from typing import List

class GrayEncoderDecoder(BinaryEncoderDecoder):
    """
    Gray encoding is a binary numeral system where two successive values differ
    in only one bit. This can reduce the Hamming distance between consecutive values,
    which can be beneficial in genetic algorithms.
    """
    def encode(self, x: List[float]) -> List[int]:
        """
        Encodes a list of float values into a flat list of binary digits (0s and 1s) using Gray encoding.
        
        Args:
            x: List of float values to encode

        Returns:
            List of integers (0s and 1s) representing the Gray code encoding
        """
        assert all(self.values_range[0] <= number < self.values_range[1] for number in x), "Values must be within specified range"
        
        result = []
        scaled_values = self._scale_values(x)
        for value in scaled_values:
            # Convert to binary then Gray code
            binary = int(value)
            gray = binary ^ (binary >> 1)
            # Use the _encode_value method but pass the Gray code
            result.extend(self._encode_value(gray))
        return result
    
    def decode(self, x: List[int]) -> List[float]:
        """
        Decodes a flat list of Gray code binary digits back to float values.
        
        Args:
            x: List of binary digits (0s and 1s) in Gray code
            
        Returns:
            List of decoded float values
        """
        # Check if input length is valid
        assert len(x) % self.num_bits == 0, f"Input length must be a multiple of {self.num_bits}"
        assert all(bit in (0,1) for bit in x), "Input must contain only binary digits (0s and 1s)"
        
        result = []
        for i in range(0, len(x), self.num_bits):
            # Get the Gray code value from the binary digits
            gray = self._decode_value(x[i:i+self.num_bits])
            # Convert Gray code back to binary
            binary = gray
            mask = gray >> 1
            while mask:
                binary ^= mask
                mask >>= 1
            # Return to original scale
            result.append(self._return_value_to_scale(binary))
        return result 