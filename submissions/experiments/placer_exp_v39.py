"""exp_v39: v36 (adaptive) with seed=2024."""
from placer_exp_v36 import OptimalPlacer as Base

class OptimalPlacer(Base):
    def __init__(self, seed=2024):
        super().__init__(seed=seed)
