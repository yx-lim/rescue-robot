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
from planner import PlannerDStarLite, PlannerTemporal

#
#  Define the Walls
#
w1 = [
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

w = [
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
"x   xxxxxxxxxxxxx                                     xxxxxxxxxx   x",
"x   x           x                                     x        x   x",
"x   x           x                                     x        x   x",
"x   x           x                                     x        x   x",
"x   x           x                                     x        x   x",
"x   x           x                                     x        x   x",
"x   xxxxxxxxxxxxx                                     xxxxxxxxxx   x",
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

# How to use continuous fire:
#   1. Set num_ignition_points > 0 so fire is enabled.
#   2. In main(), world.start_fire() is called once to ignite random cells.
#   3. Each loop iteration, world.step_fire(FIRE_DT) advances fire by FIRE_DT seconds.
#   4. Adjust FIRE_DT (e.g. 0.2–0.6) to control how fast fire evolves per key press.
# Set num_ignition_points=0 to disable fire.

walls = np.array([[1.0 * (c == "x") for c in s] for s in w])
world = World(walls,
             fire_spread_prob=0.003,        # prob to spread to open cell per step
             fire_spread_prob_wall=0.006,   # prob to spread to wall cell per step
             burn_out_time=8.0,           # seconds before cell burns out (can reignite)
             num_ignition_points=50)       # random cells to ignite at start; 0 = no fire
rows = world.rows
cols = world.cols
auto = True

# Time step for fire when stepping (seconds per loop iteration)
FIRE_DT = 0.03

######################################################################
#
#  Main Code
#
def main():
    
    robot = Robot(world, row=31, col=34, pSensor=[1,1,0.8,0.8,0.7,0.7,0.7,0.5], thetainc=pi/20, lbound=10, lfree=0.2, lwall=0.5)
    # planner = PlannerDStarLite(robot, (5, 23), cost_uncertain=1, fire_multiplier=5)
    planner = PlannerTemporal(robot, (5, 23), horizon=80, cost_uncertain=1, fire_multiplier=5, wait_cost=0.5)

    if world._num_ignition_points > 0:
        world.start_fire()

    # Initialize the figure.
    visual = Visualization(walls, robot, planner)

    if isinstance(planner.goal, tuple):
        goal_pos = planner.goal
    else:
        goal_pos = (planner.goal.row, planner.goal.col)

    # Loop continually or until goal is reached.
    # while not auto or planner.start != planner.goal:
    while not auto or (robot.row, robot.col) != goal_pos:
        # Advance fire simulation (manual interval)
        if world._num_ignition_points > 0:
            world.step_fire(FIRE_DT)
        if auto:
            planner.step()  
            visual.show(markRobot=True, showPath=True)
        else:
            visual.show(markRobot=True, showPath=False)
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
    
    print(robot.steps)
    time.sleep(5)

if __name__ == "__main__":
    main()
