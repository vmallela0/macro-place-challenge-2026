"""exp_v40: v36 (adaptive) with seed=7777."""
from placer_exp_v36 import OptimalPlacer as Base

class OptimalPlacer(Base):
    def __init__(self, seed=7777):
        super().__init__(seed=seed)
