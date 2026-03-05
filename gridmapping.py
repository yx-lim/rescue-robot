"""gridlocalize.py

This is the skeleton code for the grid-based localization.

PLEASE FINISH WRITING THE CODE, especially where marked by FIXME.

"""

import numpy as np
from math import pi
from visualization import Visualization
from world import World
from robot import Robot


#
#  Define the Walls
#
w = [
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "x               xx             xx               x",
    "x                xx           xx                x",
    "x                 xx         xx                 x",
    "x        xxxx      xx       xx                  x",
    "x        x  xx      xx     xx                   x",
    "x        x   xx      xx   xx      xxxxx         x",
    "x        x    xx      xx xx     xxx   xxx       x",
    "x        x     xx      xxx     xx       xx      x",
    "x        x      xx      x      x         x      x",
    "x        x       xx           xx         xx     x",
    "x        x        x           x           x     x",
    "x        x        x           x           x     x",
    "x        x        x           x           x     x",
    "x                 xx         xx           x     x",
    "x                  x         x                  x",
    "x                  xx       xx                  x",
    "x                   xxx   xxx                   x",
    "x                     xxxxx         x           x",
    "x                                   x          xx",
    "x                                   x         xxx",
    "x            x                      x        xxxx",
    "x           xxx                     x       xxxxx",
    "x          xxxxx                    x      xxxxxx",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

walls = np.array([[1.0 * (c == "x") for c in s] for s in w])
world = World(walls)
rows = world.rows
cols = world.cols


#
#  MEASUREMENT UPDATE (CORRECTION)
#
#  Input:  prior        Grid of prior probabilities (belief)
#          probSensor   Grid of modeled probabilities that (sensor==True)
#          sensor       Actual value of sensor (True/False)
#
#  Output: post         Grid of posterior probabilities (updated belief)
#
def updateBelief(prior, probSensor, sensor):
    # Create the posterior belief.
    if sensor:  # sensor is True
        post = probSensor * prior
    else:  # sensor is False
        post = (1 - probSensor) * prior

    post /= post.sum()

    # Check the updated belief.
    if abs(np.sum(post) - 1.0) > 1e-12:
        print("WARNING: Belief does not add up to 100%")

    # Return the updated belief.
    return post


#######################################################################
#
#  Pre-compute the Modeled Sensor Probability Grid
#
#  Input:  drow, dcol   Sensor direction in row/col
#
#  Output: prob         Grid of modeled probabilities that (sensor==True)
#
def precomputeSensorProbability4(drow, dcol):
    # Prepare an empty probability grid.
    prob = np.zeros((rows, cols))
    # Pre-compute the sensor probability on the grid.
    # FIXME:
    for r in range(rows):
        for c in range(cols):
            wall = 0
            new_r, new_c = r + drow, c + dcol
            if 0 <= new_r < rows and 0 <= new_c < cols and walls[new_r][new_c]:
                wall = 1
            prob[r][c] = wall

    # Return the computed grid.
    return prob


def precomputeSensorProbability(drow, dcol, robot):
    # Prepare an empty probability grid.
    prob = np.zeros((rows, cols))
    # Pre-compute the sensor probability on the grid.
    # FIXME:
    for r in range(rows):
        for c in range(cols):
            wall = 0
            for i in range(1, len(robot.pSensor) + 1):
                new_r, new_c = r + i * drow, c + i * dcol
                if 0 <= new_r < rows and 0 <= new_c < cols and walls[new_r][new_c]:
                    wall += robot.pSensor[i - 1]
                    break
            prob[r][c] = wall

    # Return the computed grid.
    return prob


######################################################################
#
#  Main Code
#
def main():
    robot = Robot(world, row=7, col=12, pSensor=[1,1,0.8,0.8,0.5,0.5,0.5,0.5], thetainc=pi/12)

    # Initialize the figure.
    visual = Visualization(walls, robot)

    # Loop continually.
    while True:
        # Show the current belief.  Also show the actual position.
        visual.show(markRobot=True)

        # Get the command key to determine the direction.
        while True:
            key = input("Cmd (q=quit, wer // s.f // xcv)?")
            if key == "q":
                return
            elif key == "w":
                (drow, dcol) = (-1, -1)
                break  # up/left
            elif key == "e":
                (drow, dcol) = (-1, 0)
                break  # up
            elif key == "r":
                (drow, dcol) = (-1, 1)
                break  # up/right
            elif key == "s":
                (drow, dcol) = (0, -1)
                break  # left
            elif key == "f":
                (drow, dcol) = (0, 1)
                break  # right
            elif key == "x":
                (drow, dcol) = (1, -1)
                break  # down/left
            elif key == "c":
                (drow, dcol) = (1, 0)
                break  # down
            elif key == "v":
                (drow, dcol) = (1, 1)
                break  # down/right

        # Move the robot in the simulation.
        robot.command(drow, dcol)

        robot.sense_radar()

if __name__ == "__main__":
    main()
