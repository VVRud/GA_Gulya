from GA.selection.base_lin_rank import BaseLinRank

from .sus import SUSSelection


class LinearRankSUSSelection(BaseLinRank, SUSSelection):
    """
    Linear Rank Selection with Stochastic Universal Sampling.

    Args:
        beta: The beta parameter for the linear rank selection.
    """

    def __init__(self, beta: float):
        super().__init__(beta)
