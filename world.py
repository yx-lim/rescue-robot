import numpy as np
import random

class World:
    _NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, walls,
                 fire_spread_prob=0.3,
                 fire_spread_prob_wall=0.1,
                 burn_out_time=8.0,
                 num_ignition_points=0):
        self._walls = walls
        self.rows = np.size(walls, axis=0)
        self.cols = np.size(walls, axis=1)

        # Fire (optional): set num_ignition_points > 0 and call start_fire() to enable
        self._fire_spread_prob = fire_spread_prob
        self._fire_spread_prob_wall = fire_spread_prob_wall
        self._burn_out_time = burn_out_time
        self._num_ignition_points = num_ignition_points
        self._time = 0.0
        self._fire_end_time = np.full((self.rows, self.cols), np.nan)

    def is_wall(self, row, col):
        return self._walls[row, col]

    # True if the cell is currently on fire.
    def is_fire(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        t = self._fire_end_time[row, col]
        return np.isfinite(t) and t > self._time

    # Start fire at random cells. 
    def start_fire(self, num_ignition_points=None):
        n = num_ignition_points if num_ignition_points is not None else self._num_ignition_points
        n = min(n, self.rows * self.cols)
        if n <= 0:
            return
        indices = random.sample(range(self.rows * self.cols), n)
        for idx in indices:
            r, c = idx // self.cols, idx % self.cols
            self._fire_end_time[r, c] = self._time + self._burn_out_time

    # Advance fire simulation by dt seconds.
    def step_fire(self, dt):
        self._time += dt
        self._fire_end_time[self._fire_end_time <= self._time] = np.nan
        burning = np.isfinite(self._fire_end_time) & (self._fire_end_time > self._time)
        for r, c in np.argwhere(burning):
            for dr, dc in self._NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.is_fire(nr, nc):
                    continue
                prob = self._fire_spread_prob_wall if self._walls[nr, nc] else self._fire_spread_prob
                if random.random() < prob:
                    self._fire_end_time[nr, nc] = self._time + self._burn_out_time
