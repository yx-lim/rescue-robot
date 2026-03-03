import numpy as np

class World:
    def __init__(self, walls):
        self._walls = walls
        self.rows = np.size(walls, axis=0)
        self.cols = np.size(walls, axis=1)

    def is_wall(self, row, col):
        return self._walls[row, col]