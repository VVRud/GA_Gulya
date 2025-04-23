from GA.parent_selection.base_exp_rank import BaseExpRank

from .rws import RWSSelection


class ExponentialRankRWSSelection(BaseExpRank, RWSSelection):
    """
    Exponential Rank Selection with Roulette Wheel Selection.

    Args:
        c: The c parameter for the exponential rank selection.
    """

    def __init__(self, c: float):
        super().__init__(c)
