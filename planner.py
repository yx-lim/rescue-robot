import numpy as np
from math import inf, sqrt
import heapq
from collections import deque
from node import Node
from robot import Robot
from typing import Tuple


class PlannerDStarLite():
    def __init__(self,
                 robot: Robot, 
                 goal: Tuple[int, int],
                 lfree=None,
                 cost_uncertain=1.0,
                 fire_multiplier=5.0):
        self.robot = robot
        self.lfree = lfree if lfree else 1.5*robot.lstart
        self.cost_uncertain = cost_uncertain
        self.fire_multiplier = fire_multiplier

        self.nodes = {}
        for r in range(robot.world.rows):
            for c in range(robot.world.cols):
                self.nodes[(r, c)] = Node(r, c)

        for (r, c), node in self.nodes.items():
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.nodes:
                    node.neighbors.append(self.nodes[(nr, nc)])

        self.start = self.nodes[(robot.row, robot.col)]
        self.goal = self.nodes[goal]
        self.last = self.start
        self.km = 0

        self.open = []
        self.goal.rhs = 0
        heapq.heappush(self.open, (self.calculate_key(self.goal), self.goal))

        self.robot.sense_radar()
        for node in self.nodes.values():
            node.old_c = self.state_cost(node)

        self.compute_shortest_path()

    def heuristic(self, a, b):
        return max(abs(a.row - b.row), abs(a.col - b.col))

    def state_cost(self, node):
        # Entering node cost
        logit = self.robot.walls_logits[node.row, node.col]
        fire = self.robot.world.is_fire(node.row, node.col)

        if logit >= 0:
            return inf

        if logit < self.lfree:
            base = 1.0
        else:
            base = self.cost_uncertain

        if fire:
            base *= self.fire_multiplier

        return base

    def edge_cost(self, u, v):
        # Can optionally include diagonal distance
        step = np.sqrt(2) if (u.row != v.row and u.col != v.col) else 1.0
        return step * self.state_cost(v)

    def calculate_key(self, s):
        m = min(s.g, s.rhs)
        return (m + self.heuristic(self.start, s) + self.km, m)

    def update_vertex(self, u):
        if u != self.goal:
            u.rhs = min((self.edge_cost(u, s) + s.g for s in u.neighbors), default=inf)

        self.open = [(k, n) for (k, n) in self.open if n is not u]
        heapq.heapify(self.open)

        if u.g != u.rhs:
            heapq.heappush(self.open, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while self.open and (
            self.open[0][0] < self.calculate_key(self.start) or self.start.g != self.start.rhs
        ):
            k_old, u = heapq.heappop(self.open)
            k_new = self.calculate_key(u)

            if k_old < k_new:
                heapq.heappush(self.open, (k_new, u))
            elif u.g > u.rhs:
                u.g = u.rhs
                for p in u.neighbors:
                    self.update_vertex(p)
            else:
                u.g = inf
                self.update_vertex(u)
                for p in u.neighbors:
                    self.update_vertex(p)

    def step(self):
        if self.start == self.goal:
            return 1
        if self.start.g == inf:
            return -1

        best = min(self.start.neighbors, key=lambda s: self.edge_cost(self.start, s) + s.g)
        drow = best.row - self.robot.row
        dcol = best.col - self.robot.col

        moved = self.robot.command(drow, dcol)
        self.robot.sense_radar()

        new_start = self.nodes[(self.robot.row, self.robot.col)]

        if moved:
            self.km += self.heuristic(self.last, new_start)
            self.last = new_start
            self.start = new_start

        changed_nodes = []
        for node in self.nodes.values():
            new_c = self.state_cost(node)
            if new_c != node.old_c:
                node.old_c = new_c
                changed_nodes.append(node)

        for node in changed_nodes:
            self.update_vertex(node)
            for nbr in node.neighbors:
                self.update_vertex(nbr)

        self.compute_shortest_path()
        return 0

    def get_path(self, max_len=500):
        if self.start.g == inf:
            return []

        path = [self.start]
        curr = self.start
        visited = {(curr.row, curr.col)}

        for _ in range(max_len):
            if curr == self.goal:
                break
            if not curr.neighbors:
                return path
            curr = min(curr.neighbors, key=lambda s: self.edge_cost(curr, s) + s.g)
            if (curr.row, curr.col) in visited:
                break
            visited.add((curr.row, curr.col))
            path.append(curr)

        return path




class PlannerTemporal:
    def __init__(self,
                 robot,
                 goal,
                 horizon=20,
                 lfree=None,
                 cost_uncertain=1.0,
                 fire_multiplier=20.0,
                 wait_cost=0.5):
        self.robot = robot
        self.goal = goal                  # tuple: (goal_row, goal_col)
        self.horizon = horizon
        self.lfree = lfree if lfree is not None else 1.5 * robot.lstart
        self.cost_uncertain = cost_uncertain
        self.fire_multiplier = fire_multiplier
        self.wait_cost = wait_cost

        self.last_path = []
        self.last_true_fire = None
        self.fire_pred = None

        # 8-neighborhood + wait
        self.moves = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        # Do an initial sense so walls/fire beliefs are populated
        self.robot.sense_radar()
        current_fire = self.capture_true_fire()
        self.last_true_fire = current_fire.copy()
        self.predict_fire(current_fire)

    def heuristic(self, r, c, t):
        gr, gc = self.goal
        # Chebyshev distance is fine for 8-connected grids
        return max(abs(r - gr), abs(c - gc))

    def cell_cost(self, r, c, t):
        # Occupancy belief from robot map
        logit = self.robot.walls_logits[r, c]

        # believed wall => blocked
        if logit >= 0:
            return inf

        # free vs uncertain
        base = 1.0 if logit < self.lfree else self.cost_uncertain

        # predicted fire penalty
        if self.fire_pred[t, r, c]:
            base *= self.fire_multiplier

        return base

    def step_cost(self, dr, dc, r, c, t):
        if dr == 0 and dc == 0:
            move_len = self.wait_cost
        elif dr != 0 and dc != 0:
            move_len = sqrt(2)
        else:
            move_len = 1.0

        return move_len * self.cell_cost(r, c, t)

    def capture_true_fire(self):
        rows, cols = self.robot.world.rows, self.robot.world.cols
        fire = np.zeros((rows, cols), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                fire[r, c] = 1 if self.robot.world.is_fire(r, c) else 0
        return fire

    def predict_fire(self, current_fire, spread_prob=0.003, burnout_prob=0.0005, changed_mask=None):
        """
        Predict future fire occupancy over time.
        spread_prob: probability that fire spreads to a neighbor cell
        burnout_prob: probability that a burning cell goes out
        """
        rows, cols = self.robot.world.rows, self.robot.world.cols
        neighbors = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)]

        if self.fire_pred is None or changed_mask is None or not np.any(changed_mask):
            self.fire_pred = np.zeros((self.horizon + 1, rows, cols), dtype=np.uint8)
            self.fire_pred[0] = current_fire

            for t in range(self.horizon):
                self.fire_pred[t + 1] = self.fire_pred[t].copy()
                burning = np.argwhere(self.fire_pred[t] > 0)

                for r, c in burning:
                    if np.random.random() < burnout_prob:
                        self.fire_pred[t + 1, r, c] = 0
                        continue

                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not self.fire_pred[t + 1, nr, nc] and np.random.random() < spread_prob:
                                self.fire_pred[t + 1, nr, nc] = 1
            return

        new_pred = self.fire_pred.copy()
        new_pred[0] = current_fire
        dirty_mask = changed_mask.astype(bool)

        for t in range(self.horizon):
            if not np.any(dirty_mask):
                break

            layer = new_pred[t]
            next_layer = new_pred[t + 1].copy()
            next_dirty = np.zeros_like(dirty_mask, dtype=bool)

            for r, c in np.argwhere(dirty_mask & (layer > 0)):
                if np.random.random() < burnout_prob:
                    next_layer[r, c] = 0
                    next_dirty[r, c] = True
                else:
                    next_layer[r, c] = 1
                    next_dirty[r, c] = True

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if next_layer[nr, nc]:
                        continue
                    if np.random.random() < spread_prob:
                        next_layer[nr, nc] = 1
                        next_dirty[nr, nc] = True

            new_pred[t + 1] = next_layer
            dirty_mask = next_dirty

        self.fire_pred = new_pred

    def plan(self):
        if self.fire_pred is not None:
            self.fire_pred[:-1] = self.fire_pred[1:]
            self.fire_pred[-1] = self.fire_pred[-2]

        current_fire = self.capture_true_fire()
        if self.last_true_fire is None:
            changed_mask = None
        else:
            changed_mask = current_fire != self.last_true_fire
        need_resample = self.fire_pred is None or changed_mask is None or np.any(changed_mask)
        if need_resample:
            self.predict_fire(current_fire, changed_mask=changed_mask)
        self.last_true_fire = current_fire.copy()

        start = (self.robot.row, self.robot.col, 0)
        goal_rc = self.goal

        open_heap = []
        heapq.heappush(open_heap, (self.heuristic(*start), 0.0, start))

        parent = {}
        gscore = {start: 0.0}
        visited = set()

        best_goal_state = None

        while open_heap:
            _, g, state = heapq.heappop(open_heap)
            r, c, t = state

            if state in visited:
                continue
            visited.add(state)

            if (r, c) == goal_rc:
                best_goal_state = state
                break

            if t >= self.horizon:
                continue

            for dr, dc in self.moves:
                nr, nc, nt = r + dr, c + dc, t + 1

                if not (0 <= nr < self.robot.world.rows and 0 <= nc < self.robot.world.cols):
                    continue

                # hard block on believed walls
                if self.robot.walls_logits[nr, nc] >= 0:
                    continue

                cst = self.step_cost(dr, dc, nr, nc, nt)
                if cst == inf:
                    continue

                ng = g + cst
                nxt = (nr, nc, nt)

                if nxt not in gscore or ng < gscore[nxt]:
                    gscore[nxt] = ng
                    parent[nxt] = state
                    f = ng + self.heuristic(nr, nc, nt)
                    heapq.heappush(open_heap, (f, ng, nxt))

        if best_goal_state is None:
            self.last_path = []
            return []

        # reconstruct (row, col, t) path
        path = []
        cur = best_goal_state
        while cur in parent:
            path.append(cur)
            cur = parent[cur]
        path.append(start)
        path.reverse()

        self.last_path = path
        return path

    def step(self):
        # already at goal
        if (self.robot.row, self.robot.col) == self.goal:
            return 1

        # update sensing before planning
        self.robot.sense_radar()

        path = self.plan()

        # no feasible path
        if len(path) <= 1:
            return -1

        # take the next temporal step as a real robot move that actually changes position
        next_move = None
        for nr, nc, _ in path[1:]:
            if (nr, nc) != (self.robot.row, self.robot.col):
                next_move = (nr, nc)
                break

        if next_move is None:
            return -1

        nr, nc = next_move
        drow = nr - self.robot.row
        dcol = nc - self.robot.col
        moved = self.robot.command(drow, dcol)

        # refresh sensing after motion attempt
        self.robot.sense_radar()

        if (self.robot.row, self.robot.col) == self.goal:
            return 1

        # if move failed but planner suggested something, just continue next cycle
        return 0 if moved or len(path) > 1 else -1

    def get_path(self):
        """
        Return path in a visualization-friendly format: list of (row, col).
        """
        if not self.last_path:
            self.plan()
        return [(r, c) for (r, c, t) in self.last_path]


class PlannerLPAStar:
    def __init__(self, robot, goal, lfree=None, cost_uncertain=1.0, fire_multiplier=5.0):
        self.robot = robot
        self.goal_pos = goal
        self.lfree = lfree if lfree is not None else 1.5 * robot.lstart
        self.cost_uncertain = cost_uncertain
        self.fire_multiplier = fire_multiplier

        self._revisit_penalty = 2.0
        self._recent_positions = deque(maxlen=10)
        self._recent_positions.append((robot.row, robot.col))
        self._last_pos = None

        self.nodes = {}
        for r in range(robot.world.rows):
            for c in range(robot.world.cols):
                self.nodes[(r, c)] = Node(r, c)

        for (r, c), node in self.nodes.items():
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in self.nodes:
                    node.neighbors.append(self.nodes[(nr, nc)])

        self.goal = self.nodes[goal]

        # Lazy-deletion open list.
        self.open = []
        self._node_version = {n: 0 for n in self.nodes.values()}

        self.robot.sense_radar()
        self._initialize_search()


    def _initialize_search(self):
        for node in self.nodes.values():
            node.g   = inf
            node.rhs = inf
            node.old_c = self.state_cost(node)
            node.parent = None

        # Forward LPA*: seed is the START, not the goal.
        self.start = self.nodes[(self.robot.row, self.robot.col)]
        self.start.rhs = 0
        self._push(self.start)
        self.compute_shortest_path()

    def _push(self, node):
        self._node_version[node] += 1
        heapq.heappush(
            self.open,
            (self.calculate_key(node), self._node_version[node], node)
        )

    def _pop(self):
        while self.open:
            key, ver, node = heapq.heappop(self.open)
            if ver == self._node_version[node]:
                return key, node
        return None, None

    def _peek_key(self):
        while self.open:
            key, ver, node = self.open[0]
            if ver == self._node_version[node]:
                return key
            heapq.heappop(self.open)
        return (inf, inf)

    def heuristic(self, a, b):
        return max(abs(a.row - b.row), abs(a.col - b.col))

    def state_cost(self, node):
        logit = self.robot.walls_logits[node.row, node.col]
        fire  = self.robot.fire[node.row, node.col]
        if logit >= 0:
            return inf
        base = 1.0 if logit < self.lfree else self.cost_uncertain
        if fire:
            base *= self.fire_multiplier
        return base

    def edge_cost(self, u, v):
        step = np.sqrt(2) if (u.row != v.row and u.col != v.col) else 1.0
        return step * self.state_cost(v)

    def calculate_key(self, s):
        # Forward search: heuristic is distance to the GOAL.
        m = min(s.g, s.rhs)
        return (m + self.heuristic(s, self.goal), m)

    def update_vertex(self, u):
        if u != self.start:
            best_rhs    = inf
            best_parent = None
            # Forward search: rhs(u) = min over predecessors p of
            #   g(p) + edge_cost(p, u)
            # On an undirected grid every neighbor is a valid predecessor.
            for p in u.neighbors:
                cand = p.g + self.edge_cost(p, u)
                if cand < best_rhs:
                    best_rhs    = cand
                    best_parent = p
            u.rhs    = best_rhs
            u.parent = best_parent

        # Invalidate existing open-list entry (lazy deletion via version bump).
        self._node_version[u] += 1

        if u.g != u.rhs:
            self._push(u)

    def compute_shortest_path(self):
        while self.open and (
            self._peek_key() < self.calculate_key(self.goal)
            or self.goal.rhs != self.goal.g
        ):
            key, u = self._pop()
            if u is None:
                break

            if u.g > u.rhs:      # overconsistent → make consistent
                u.g = u.rhs
            else:                 # underconsistent → raise g, propagate
                u.g = inf
                self.update_vertex(u)

            for s in u.neighbors:
                self.update_vertex(s)

    def step(self):
        if (self.robot.row, self.robot.col) == (self.goal.row, self.goal.col):
            return 1

        # Sync fire map with true world state.
        for r in range(self.robot.world.rows):
            for c in range(self.robot.world.cols):
                if self.robot.fire[r, c] and not self.robot.world.is_fire(r, c):
                    self.robot.fire[r, c] = 0

        self.robot.sense_radar()

        changed = False
        for n in self.nodes.values():
            new_c = self.state_cost(n)
            if new_c != n.old_c:
                n.old_c = new_c          # refresh baseline right away
                self.update_vertex(n)
                for nb in n.neighbors:   # neighbors' rhs depends on n.g
                    self.update_vertex(nb)
                changed = True

        # When the robot moves to a new cell, make the new cell the start.  
        curr_pos = (self.robot.row, self.robot.col)
        if curr_pos != (self.start.row, self.start.col):
            # Demote old start back to a normal node and let it get a real rhs.
            old_start = self.start
            old_start.rhs = inf
            self.update_vertex(old_start)

            # Promote new start: rhs = 0, no predecessors contribute.
            self.start = self.nodes[curr_pos]
            self.start.rhs = 0
            self.start.g   = 0
            self._push(self.start)
            changed = True

        if changed:
            self.compute_shortest_path()

        curr = self.nodes[(self.robot.row, self.robot.col)]
        if curr.g == inf and curr != self.goal:
            return -1
        if curr == self.goal:
            return 1

        def neighbor_score(s):
            base_cost = self.edge_cost(curr, s) + s.g
            if (s.row, s.col) in self._recent_positions:
                base_cost += self._revisit_penalty
            base_cost += 0.01 * self.heuristic(s, self.goal)
            return base_cost

        best = min(curr.neighbors, key=neighbor_score)

        prev_pos = (self.robot.row, self.robot.col)
        moved = self.robot.command(best.row - self.robot.row, best.col - self.robot.col)
        self.robot.sense_radar()

        if moved:
            self._last_pos = prev_pos
            self._recent_positions.append(prev_pos)

        return 0

    def get_path(self, max_len=500):
        """
        Reconstruct a forward path from the robot's current node by
        following the greedy policy implied by g-values, similar to
        the D* Lite planner. This is purely for visualization.
        """
        curr = self.nodes[(self.robot.row, self.robot.col)]
        if curr.g == inf:
            return []

        path = [curr]
        visited = {(curr.row, curr.col)}
        for _ in range(max_len):
            if curr == self.goal:
                break
            # Choose neighbor that minimizes edge_cost + g (forward policy).
            curr = min(
                curr.neighbors,
                key=lambda s: self.edge_cost(curr, s) + s.g
            )
            if (curr.row, curr.col) in visited:
                break
            visited.add((curr.row, curr.col))
            path.append(curr)
        return path
