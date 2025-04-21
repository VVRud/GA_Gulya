from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from functools import lru_cache
class BaseEncoderDecoder(ABC):
    """
    Base class for all encoding classes.
    """
    def __init__(self, values_range: Tuple[float, float], points_after_decimal: int = 2):
        self.values_range = (
            np.round(np.float64(values_range[0]), points_after_decimal),
            np.round(np.float64(values_range[1]), points_after_decimal)
        )
        self.points_after_decimal = points_after_decimal
        self.scale_factor = 10 ** points_after_decimal
        self.num_bits = int(np.ceil(np.log2((values_range[1] - values_range[0]) * self.scale_factor)))

    def _scale_values(self, x: List[float]) -> List[int]:
        """
        Scales values based on range and precision.
        """
        return [np.round(np.round(number - self.values_range[0], self.points_after_decimal) * self.scale_factor) for number in x]

    def _return_value_to_scale(self, value: int) -> float:
        """
        Returns the value to the original scale.
        """
        return (value / self.scale_factor) + self.values_range[0]

    @abstractmethod
    @lru_cache(maxsize=None)
    def encode(self, x: List[float]) -> List[int]:
        """
        Encodes a list of float values into a flat list of binary digits (0s and 1s).
        
        Args:
            x: List of float values to encode
        
        Returns:
            List of integers (0s and 1s) representing the binary encoding
        """
        pass

    @abstractmethod
    @lru_cache(maxsize=None)
    def decode(self, x: List[int]) -> List[float]:
        """
        Decodes a flat list of binary digits back to float values.
        
        Args:
            x: List of binary digits (0s and 1s)
        
        Returns:
            List of decoded float values
        """
        pass
    
    def decode_many(self, x: List[List[int]]) -> List[List[float]]:
        """
        Decodes a list of lists of binary digits back to float values.
        """
        return np.array([self.decode(x_i) for x_i in x])
