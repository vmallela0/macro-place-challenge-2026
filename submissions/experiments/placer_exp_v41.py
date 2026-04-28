"""exp_v41: v36 with seed=99999."""
from placer_exp_v36 import OptimalPlacer as Base

class OptimalPlacer(Base):
    def __init__(self, seed=99999):
        super().__init__(seed=seed)
