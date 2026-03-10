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

walls = np.array([[1.0 * (c == "x") for c in s] for s in w2])
world = World(walls)
rows = world.rows
cols = world.cols
auto = True

######################################################################
#
#  Main Code
#
def main():
    robot = Robot(world, row=31, col=34, pSensor=[1,1,0.8,0.8,0.5,0.5,0.5,0.5], thetainc=pi/20)
    planner = Planner(robot, (5, 14))

    # Initialize the figure.
    visual = Visualization(walls, robot)

    # Loop continually.
    while True:
        time.sleep(0.5)
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
