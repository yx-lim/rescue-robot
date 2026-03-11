import random
import numpy as np
import world
import math

######################################################################
#
#   ROBOT SIMULATION:
#
#     robot = Robot(map_rows, map_cols, row = 0, col = 0, 
#                   pSensor = [1.0]):
#
#     robot.command(drow, dcol)
#     True/False = robot.sensor(drow, dcol)
#
#     pSensor       List of probabilities (0 to 1).  Each element is
#                   the probability that the proximity sensor will
#                   fire at a distance of (index+1 = 1,2,3,etc).
#                   The probability will be zero at greater distances.
#     kidnap        Flag - kidnap the robot (once in the simulation)
#
#     (drow, dcol)  Delta up/right/down/left/ur/ul/dr/dl: 
#                   (-1,0) (0,1) (1,0) (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)
#
#   Simulate a robot, to give us the sensor readings. Note both the command
#   and the sensor may be configured to a random probability level.
#
class Robot():
    def __init__(self, 
                 world: world.World, 
                 row = 0, 
                 col = 0, 
                 pSensor = [1.0], 
                 thetainc=math.pi / 2, 
                 lstart=-1.0,
                 lwall=0.4, 
                 lfree=0.2,
                 lbound=7.0):
        # Report.
        location = " (at %d, %d)" % (row, col)
        print("Starting robot with real" +
              " sensor probabilities = " + str(pSensor) +
              "," + str(location))

        # Save the walls, the initial location, and the probabilities.
        self.walls_logits = np.full((world.rows, world.cols), lstart)
        self.fire = np.zeros((world.rows, world.cols))
        self.lstart = lstart
        self.lwall = lwall
        self.lfree = lfree
        self.pSensor  = pSensor
        self.row = row
        self.col = col
        self.world = world
        self.thetainc = thetainc
        self.lbound = lbound
        self.steps = 0

    def adjust(self, u, v, delta):
        if (u >= 0) and (u < self.world.rows) and (v >= 0) and (v < self.world.cols):
            self.walls_logits[u, v] = np.clip(self.walls_logits[u, v] + delta, -self.lbound, self.lbound)
        else:
            print("Out of bounds (%d, %d)" % (u, v))

    def command(self, drow, dcol):
        # Check the delta.
        assert (max(abs(drow), abs(dcol)) == 1), "Bad delta"
        self.steps += 1
        # Try to move the robot the given delta.
        row = self.row + drow
        col = self.col + dcol
        if (self.walls_logits[row, col] < 0 and not self.world.is_wall(row, col)):
            self.row = row
            self.col = col
            self.walls_logits[row, col] = -self.lbound
            return True
        if self.world.is_wall(row, col):
            self.walls_logits[row, col] = self.lbound
        return False

    def sense_wall(self, drow, dcol):
        for k in range(len(self.pSensor)):
            nr = round(self.row + drow*(k+1))
            nc = round(self.col + dcol*(k+1))

            # Stop if outside map
            if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols):
                return k

            # Check wall
            if self.world.is_wall(nr, nc) and random.random() < self.pSensor[k]:
                self.adjust(nr, nc, self.lwall)  # mark wall
                return k  # stop scanning beyond wall
            self.adjust(nr, nc, -self.lfree)
        return len(self.pSensor)
    
    def sense_fire(self, drow, dcol):
        for k in range(len(self.pSensor)):
            nr = round(self.row + drow*(k+1))
            nc = round(self.col + dcol*(k+1))

            # Stop if outside map or if a wall is in the way (can't sense heat through walls)
            if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols) or self.world.is_wall(nr, nc):
                return k

            if self.world.is_fire(nr, nc):
                self.fire[nr, nc] = 1
                return k
        return len(self.pSensor)
    
    def sense_radar(self):
        angle = 0
        while angle < 2 * math.pi:
            drow = np.sin(angle)
            dcol = np.cos(angle)
            self.sense_wall(drow, dcol)
            self.sense_fire(drow, dcol)
            angle += self.thetainc