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
    def __init__(self, world: world.World, row = 0, col = 0, 
                 pSensor = [1.0], thetainc=math.pi / 2):
        # Report.
        location = " (at %d, %d)" % (row, col)
        print("Starting robot with real" +
              " sensor probabilities = " + str(pSensor) +
              "," + str(location))

        # Save the walls, the initial location, and the probabilities.
        self.walls    = np.zeros((world.rows, world.cols))
        self.pSensor  = pSensor
        self.row = row
        self.col = col
        self.world = world
        self.thetainc = thetainc

    def command(self, drow, dcol):
        # Check the delta.
        assert (max(abs(drow), abs(dcol)) == 1), "Bad delta"

        # Try to move the robot the given delta.
        row = self.row + drow
        col = self.col + dcol
        if (not self.walls[row, col] and not self.world.is_wall(row, col)):
            self.row = row
            self.col = col

    def sense_ray(self, drow, dcol):
        for k in range(len(self.pSensor)):
            nr = math.floor(self.row + drow*(k+1)) if drow < 0 else math.ceil(self.row + drow*(k+1))
            nc = math.floor(self.col + dcol*(k+1)) if dcol < 0 else math.ceil(self.col + dcol*(k+1))

            # Stop if outside map
            if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols):
                return k

            # Check wall
            if self.world.is_wall(nr, nc) and random.random() < self.pSensor[k]:
                self.walls[nr][nc] = min(1, self.walls[nr][nc]+0.2)  # mark wall
                return k  # stop scanning beyond wall
            else:
                self.walls[nr][nc] = max(0, self.walls[nr][nc]-0.1)
        return len(self.pSensor)
    
    def sense_radar(self):
        angle = 0
        while angle < 2 * math.pi:
            drow = np.sin(angle)
            dcol = np.cos(angle)
            self.sense_ray(drow, dcol)
            angle += self.thetainc