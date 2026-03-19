import numpy as np
import random

class World:
    _NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, walls,
                 fire_spread_prob=0.3,
                 fire_spread_prob_wall=0.1,
                 num_ignition_points=0):
        self._walls = walls
        self.rows = np.size(walls, axis=0)
        self.cols = np.size(walls, axis=1)

        self._fire_spread_prob = fire_spread_prob
        self._fire_spread_prob_wall = fire_spread_prob_wall
        self._num_ignition_points = num_ignition_points
        self.fire = np.zeros((self.rows, self.cols), dtype=bool)

    def is_wall(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self._walls[row, col]

    def is_fire(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.fire[row, col]

    def start_fire(self, num_ignition_points=None):
        n = num_ignition_points if num_ignition_points is not None else self._num_ignition_points
        n = min(n, self.rows * self.cols)
        if n <= 0:
            return
        indices = random.sample(range(self.rows * self.cols), n)
        for idx in indices:
            r, c = idx // self.cols, idx % self.cols
            self.fire[r, c] = True

    def step_fire(self, dt):
        for r, c in np.argwhere(self.fire):
            for dr, dc in self._NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.fire[nr, nc]:
                    continue
                prob = self._fire_spread_prob_wall if self._walls[nr, nc] else self._fire_spread_prob
                if random.random() < prob:
                    self.fire[nr, nc] = True

    def ignite_cells(self, coords):
        self._num_ignition_points = len(coords)
        for r, c in coords:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.fire[r, c] = True