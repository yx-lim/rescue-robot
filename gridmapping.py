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
from planner import PlannerDStarLite, PlannerTemporal, PlannerAStarReplan

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

# A* vs D* map comparison
w1 = [
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
"x   x xxxxx   x  xx   xxxxx                                          x",
"x x x   x   x x  xx   x   x                                          x",
"x xxx xxx xxx x       xxx x                                          x",
"x   x   x   x x  xxxx   x x                                          x",
"x xxxxx xxx x x  x  xxxxx x                                          x",
"x x   x     x x  x        x                                          x",
"x xxx xxxxxxx x  x xxxxxxxx                                          x",
"x   x       x x  x x     x                                           x",
"x x xxxxxx  x x  x x xxx x                                           x",
"x x      x  x x  x x x x x                                           x",
"x x xxxx x  x x  x x x x x                                           x",
"x x x  x x  x x  x x x x x                                           x",
"x x x  x x  x x  x x x x x                                           x",
"x x xxxx x  x x  x x x x x                                           x",
"x x      x  x x  x x x x x                                           x",
"x xxxxxxxxx  x x  x xxxxx x                                          x",
"x x        x x x  x       x                                          x",
"x x  xxxx  x x x  xxxxxxx x                                          x",
"x x  x  x  x x x        x x                                          x",
"x x  x  x  x x x  xxxxx x x                                          x",
"x x  xxxx  x x x  x   x x x                                          x",
"x x        x x x  x   x x x                                          x",
"x xxxxxxxxx  x x  xxxxx x x                                          x",
"x                                                                    x",
"x        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx            x",
"x        x                                            x              x",
"x        x                                            x              x",
"x        x                                            x              x",
"x        x                                            x              x",
"x                                                                    x",
"x        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx            x",
"x                                                                    x",
"x                                                                    x",
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]

# Temporal A* vs D* map comparison
w = [
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
"x                                                                   x",
"x  xxxxxxxxxxxxxxxxxxxxx                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  xxxxx   xxxxx   xxxxx                                            x",
"x  x                   x                                            x",
"x  xxxxx   xxxxx   xxxxx                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  xxxxx   xxxxx   xxxxx                                            x",
"x  x                   x                                            x",
"x  xxxxx   xxxxx   xxxxx                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  x   x   x   x   x   x                                            x",
"x  xxxxxxxxxxxxxxxxxxxxx                                            x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"x                                                                   x",
"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
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
             num_ignition_points=0)       # random cells to ignite at start; 0 = no fire
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
    
    robot = Robot(world, row=31, col=34, pSensor=[1,1,1,1,1,1,1,1,1,1,1,1], thetainc=pi/20, lbound=10, lfree=0.2, lwall=0.5, true_walls=True)
    robot2 = Robot(world, row=31, col=34, pSensor=[1,1,1,1,1,1,1,1,1,1,1,1], thetainc=pi/20, lbound=10, lfree=0.2, lwall=0.5, true_walls=True)
    robot3 = Robot(world, row=31, col=34, pSensor=[1,1,1,1,1,1,1,1,1,1,1,1], thetainc=pi/20, lbound=10, lfree=0.2, lwall=0.5, true_walls=True)
    
    # UNCOMMENT THE PLANNER YOU WANT TO USE
    # adjust horizon (max future time considered, horizon=10 means predict 10 steps ahead)
    planner = PlannerTemporal(robot, (4, 1), horizon=80, cost_uncertain=1, fire_multiplier=100)
    planner2 = PlannerDStarLite(robot2, (4, 1), cost_uncertain=1, fire_multiplier=100, true_fire=True)
    planner3 = PlannerAStarReplan(robot3, (4, 1), cost_uncertain=1, fire_multiplier=100, true_fire=True)
    
    if world._num_ignition_points > 0:
        world.start_fire()

    manual_fire_coords = [(9, 2), (10, 2), (11, 2), (12, 2), (13, 2), (14, 2)]
    world.ignite_cells(manual_fire_coords)

    # Initialize the figure.
    visual = Visualization(walls, robot, planner)
    #visual2 = Visualization(walls, robot2, planner2)

    if isinstance(planner.goal, tuple):
        goal_pos = planner.goal
    else:
        goal_pos = (planner.goal.row, planner.goal.col)

    # Loop continually or until goal is reached.
    # while not auto or planner.start != planner.goal:
    while not auto or ((robot.row, robot.col) != goal_pos or (robot2.row, robot2.col) != goal_pos or (robot3.row, robot3.col) != goal_pos):
        # Advance fire simulation (manual interval)
        if world._num_ignition_points > 0:
            world.step_fire(FIRE_DT)
        if auto:
            if (robot.row, robot.col) != goal_pos:
                planner.step()
            if (robot2.row, robot2.col) != goal_pos:
                planner2.step()  
            if (robot3.row, robot3.col) != goal_pos:
                planner3.step()  
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
    print(planner.expanded_nodes)
    print(planner.fire_nodes)
    print(robot2.steps)
    print(planner2.expanded_nodes)
    print(planner2.fire_nodes)
    print(robot3.steps)
    print(planner3.expanded_nodes)
    print(planner3.fire_nodes)

if __name__ == "__main__":
    main()
