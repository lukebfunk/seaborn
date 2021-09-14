from __future__ import annotations
from .base import Stat


class Mean(Stat):

    # TODO use some special code here to group by the orient variable?
    grouping_vars = ["color", "edgecolor", "marker", "dash"]  # TODO get automatically

    def __call__(self, data):
        return data.filter(regex="x|y").mean()
