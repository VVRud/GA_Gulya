from .sus import SUSSelection
from GA.selection.base_lin_rank import BaseLinRank


class LinearRankSUSSelection(BaseLinRank, SUSSelection):
    """
    Linear Rank Selection with Stochastic Universal Sampling.
    
    Args:
        beta: The beta parameter for the linear rank selection.
    """
    def __init__(self, beta: float):
        super().__init__(beta)
