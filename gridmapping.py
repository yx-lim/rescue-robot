"""gridlocalize.py

This is the skeleton code for the grid-based localization.

PLEASE FINISH WRITING THE CODE, especially where marked by FIXME.

"""

import time
import numpy as np
from math import pi
from visualization import Visualization
from world import World
from robot import Robot
from planner import Planner

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

w2 = w = [
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
"x                                                                  x",
"x   xxxxxxxxxxxxx        xxxxxxxxxxxxx        xxxxxxxxxxxxx        x",
"x   x           x        x           x        x           x        x",
"x   x           x        x           x        x           x        x",
"x   x           x        x           x        x           x        x",
"x   x           x        x           x        x           x        x",
"x   x           x        x           x        x           x        x",
"x   xxxxx   xxxxx        xxxxx   xxxxx        xxxxx   xxxxx        x",
"x       x   x                x   x                x   x            x",
"x       x   x                x   x                x   x            x",
"x       x   x                x   x                x   x            x",
"x       x   xxxxxxxxxxxxxxxxxx   xxxxxxxxxxxxxxxxxx   x            x",
"x       x                                                   xxxx   x",
"x       x                                                   x  x   x",
"x       x                                                   x  x   x",
"x   xxxxxxxxxxxxx      xxxxxxxxxxxxxxxxxxxxxxxxx      xxxxxxxxxx   x",
"x   x           x      x                       x      x        x   x",
"x   x           x      x                       x      x        x   x",
"x   x           x      x                       x      x        x   x",
"x   x           x      x                       x      x        x   x",
"x   x           x      x                       x      x        x   x",
"x   xxxxxxxxxxxxx      xxxxxxxxxxxxxxxxxxxxxxxxx      xxxxxxxxxx   x",
"x                                                                  x",
"x                  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx                   x",
"x                  x                           x                   x",
"x                  x                           x                   x",
"x                  x                           x                   x",
"x                  x                           x                   x",
"x                  xxxxxxxxxxxxxxxxxxxxxxxxxxxxx                   x",
"x                                                                  x",
"x                                                                  x",
"x                                                                  x",
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

walls = np.array([[1.0 * (c == "x") for c in s] for s in w])
world = World(walls)
rows = world.rows
cols = world.cols
auto = True

# How to use continuous fire:
#   1. Set num_ignition_points > 0 so fire is enabled.
#   2. In main(), world.start_fire() is called once to ignite random cells.
#   3. Each loop iteration, world.step_fire(FIRE_DT) advances fire by FIRE_DT seconds.
#   4. Adjust FIRE_DT (e.g. 0.2–0.6) to control how fast fire evolves per key press.
# Set num_ignition_points=0 to disable fire.

world = World(walls,
             fire_spread_prob=0.3,        # prob to spread to open cell per step
             fire_spread_prob_wall=0.1,   # prob to spread to wall cell per step
             burn_out_time=8.0,           # seconds before cell burns out (can reignite)
             num_ignition_points=2)       # random cells to ignite at start; 0 = no fire
rows = world.rows
cols = world.cols

# Time step for fire when stepping (seconds per loop iteration)
FIRE_DT = 0.3


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
    robot = Robot(world, row=31, col=34, pSensor=[1,1,0.8,0.8,0.5,0.5,0.5,0.5], thetainc=pi/20)
    planner = Planner(robot, (5, 14))

    if world._num_ignition_points > 0:
        world.start_fire()

    # Initialize the figure.
    visual = Visualization(walls, robot)

    # Loop continually.
    while True:
        time.sleep(0.5)
        # Advance fire simulation (manual interval)
        if world._num_ignition_points > 0:
            world.step_fire(FIRE_DT)
        # Show the current belief.  Also show the actual position.
        visual.show(markRobot=True)

        if auto:
            planner.step()    
        else:
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
