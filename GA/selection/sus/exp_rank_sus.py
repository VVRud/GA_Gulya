from GA.selection.base_exp_rank import BaseExpRank

from .sus import SUSSelection


class ExponentialRankSUSSelection(BaseExpRank, SUSSelection):
    """
    Exponential Rank Selection with Stochastic Universal Sampling.

    Args:
        c: The c parameter for the exponential rank selection.
    """

    def __init__(self, c: float):
        super().__init__(c)
