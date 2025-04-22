from GA.selection.base_lin_rank import BaseLinRank

from .rws import RWSSelection


class LinearRankRWSSelection(BaseLinRank, RWSSelection):
    """
    Linear Rank Selection with Roulette Wheel Selection.

    Args:
        beta: The beta parameter for the linear rank selection.
    """

    def __init__(self, beta: float):
        super().__init__(beta)
