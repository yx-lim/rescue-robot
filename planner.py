import numpy as np
from math import inf
import heapq
from node import Node
from robot import Robot
from typing import Tuple


class Planner():
    def __init__(self, robot: Robot, goal: Tuple[int, int]):
        self.robot = robot
        self.onDeck = []
        self.km = 0
        self.nodes = []
        for row in range(robot.world.rows):
            for col in range(robot.world.cols):
                node = Node(row, col)
                node.rhs = inf
                node.g = inf
                self.nodes.append(node)
        # Create the neighbors, being the edges between the nodes.
        for node in self.nodes:
            for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),]:
                others = [n for n in self.nodes if (n.row, n.col) == (node.row + dr, node.col + dc)]
                if len(others) > 0:
                    node.neighbors.append(others[0])
        start = (robot.row, robot.col)
        self.start = [n for n in self.nodes if (n.row, n.col) == start][0]
        self.goal = [n for n in self.nodes if (n.row, n.col) == goal][0]
        self.goal.rhs = 0
        heapq.heappush(self.onDeck, (self.calculate_key(self.goal), self.goal))
        self.last = self.start
        robot.sense_radar()
        for node in self.nodes:
            node.old_c = self.c(node)
        self.compute_shortest_path()

    def c(self, node: Node):
        return (inf if self.robot.walls[node.row][node.col] > 0.5 else 1.0)
    
    def h(self, node1: Node, node2: Node):
        return max(abs(node1.row - node2.row), abs(node1.col - node2.col))

    def calculate_key(self, s: Node):
        return (min(s.g, s.rhs) + self.h(self.start, s) + self.km, min(s.g, s.rhs))
    
    def update_vertex(self, u: Node):
        if u != self.goal:
            u.rhs = min([self.c(s_dash) + s_dash.g for s_dash in u.neighbors])
        self.onDeck = [(k, n) for k, n in self.onDeck if n is not u]
        heapq.heapify(self.onDeck)
        if u.g != u.rhs:
            heapq.heappush(self.onDeck, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        while len(self.onDeck) > 0 and (self.onDeck[0][0] < self.calculate_key(self.start) or self.start.rhs != self.start.g):
            k_old, u = heapq.heappop(self.onDeck)
            k = self.calculate_key(u)
            if k_old < k:
                heapq.heappush(self.onDeck, (k, u))
            elif u.g > u.rhs:
                u.g = u.rhs
                for s in u.neighbors:
                    self.update_vertex(s)
            else:
                u.g = inf
                for s in u.neighbors + [u]:
                    self.update_vertex(s)

    def step(self):
        if self.start == self.goal:
            return 1
        if self.start.g == inf:
            return -1
        idx = np.argmin([self.c(s_dash) + s_dash.g for s_dash in self.start.neighbors])
        next_node = self.start.neighbors[idx]
        drow = next_node.row - self.robot.row
        dcol = next_node.col - self.robot.col
        if self.robot.command(drow, dcol):
            self.start = next_node
        self.robot.sense_radar()
        changed = False
        for node in self.nodes:
            # what you'll want to do is compute the current c and compare it to the previous c
            if self.c(node) != node.old_c:
                if not changed:
                    self.km += self.h(self.last, self.start)
                    self.last = self.start
                    changed = True
                node.old_c = self.c(node)
                for neighbor in node.neighbors:
                    self.update_vertex(neighbor)
        self.compute_shortest_path()

    