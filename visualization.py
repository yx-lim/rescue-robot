import matplotlib.pyplot as plt
import numpy as np
from planner import Planner
from robot import Robot


######################################################################
#
#   VISUALIZATION of the probability grid
#
#     visual = Visualization(walls, robot)
#
#     visual.show(prob)
#     visual.show(prob, msg="Message")
#     visual.show(prob, msg, markRobot=True)
#
#     walls         NumPy 2D array defining both the grid size
#                   (rows/cols) and walls (being non-zero elements).
#     robot         Robot object (see below)
#     prob          NumPy 2D array of probabilities, values 0 to 1.
#
#   visual.Show() will visualize the probabilities.  An optional
#   second argument provides a message and waits for user input.
#   An optional third argument implies the robot's actual position
#   should be overlayed with an 'x'.
#
class Visualization():
    def __init__(self, walls, robot: Robot, planner: Planner):
        # Save the walls, robot, and determine the rows/cols:
        self.walls = walls
        self.robot = robot
        self.world = robot.world
        self.planner = planner
        self.spots = np.sum(np.logical_not(walls))
        self.rows  = np.size(walls, axis=0)
        self.cols  = np.size(walls, axis=1)

        # Clear the current, or create a new figure.
        plt.clf()

        # Create two subplots: left = true world, right = robot belief.
        self.fig, (self.ax_true, self.ax_belief) = plt.subplots(1, 2, figsize=(12, 6))

        for ax in (self.ax_true, self.ax_belief):
            ax.grid(False)
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(0, self.cols)
            ax.set_ylim(self.rows, 0)

        self.ax_true.set_title("True world")
        self.ax_belief.set_title("Robot belief")

        # Add row/col numbers on the belief subplot (right).
        for row in range(0, self.rows, 2):
            self.ax_belief.text(-0.3, 0.6+row, '%d'%row,
                                verticalalignment='center',
                                horizontalalignment='right')
        for row in range(1, self.rows, 2):
            self.ax_belief.text(self.cols+0.3, 0.6+row, '%d'%row,
                                verticalalignment='center',
                                horizontalalignment='left')
        for col in range(0, self.cols, 2):
            self.ax_belief.text(0.5+col, -0.3, '%d'%col,
                                verticalalignment='bottom',
                                horizontalalignment='center')
        for col in range(1, self.cols, 2):
            self.ax_belief.text(0.5+col, self.rows+0.3, '%d'%col,
                                verticalalignment='top',
                                horizontalalignment='center')

        # Draw the grid, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            self.ax_belief.axhline(row, lw=1, color='k', zorder=1)
        for col in range(self.cols+1):
            self.ax_belief.axvline(col, lw=1, color='k', zorder=1)

        # Add the legend text on the belief subplot.
        self.ax_belief.text(0, 40, "Probability: Yellow==0%")
        self.ax_belief.text(0, 42, "     White<=0.1%, Blue, Black=100%")

        # Clear the content and mark.  Then show blank field.
        self.content_true = None
        self.content_belief = None
        self.mark    = None
        self.path    = None

        self.fig.tight_layout()
        self.show()

    def flush(self):
        # Show the plot.
        plt.pause(0.1)

    def updatemark(self, markRobot=True):
        # Clear/potentially remove the previous mark.
        if self.mark is not None:
            self.mark.remove()
            self.mark = None

        # If requested, add a new mark.
        if markRobot:
            # Grab the robot position and check.
            row = self.robot.row
            col = self.robot.col
            assert (row >= 0) and (row < self.rows), "Illegal robot row"
            assert (col >= 0) and (col < self.cols), "Illegal robot col"

            # Draw the mark.
            self.mark  = self.ax_belief.text(0.5+col, 0.5+row, 'x', color='red',
                                             verticalalignment='center',
                                             horizontalalignment='center',
                                             fontweight='bold',
                                             zorder=1)

    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))

    def updategrid_true(self):
        # True world: walls from self.walls, fire from self.world.is_fire.
        color = np.ones((self.rows, self.cols, 3))
        for row in range(self.rows):
            for col in range(self.cols):
                if self.walls[row, col]:
                    color[row, col, :] = np.array([0.0, 0.0, 0.0])  # true wall
                else:
                    color[row, col, :] = np.array([1.0, 1.0, 1.0])  # true free

                if hasattr(self.world, "is_fire") and self.world.is_fire(row, col):
                    color[row, col, :] = np.array([1.0, 0.0, 0.0])  # true fire

        if self.content_true is not None:
            self.content_true.remove()
            self.content_true = None

        self.content_true = self.ax_true.imshow(
            color,
            aspect='equal',
            interpolation='none',
            extent=[0, self.cols, self.rows, 0],
            zorder=0,
        )

    def updategrid_belief(self, showPath):
        # Robot belief: log-odds walls + perceived fire.
        prob = self.logits_to_probs(self.robot.walls_logits)
        assert np.size(prob, axis=0) == self.rows, "Inconsistent # of rows"
        assert np.size(prob, axis=1) == self.cols, "Inconsistent # of cols"

        if self.content_belief is not None:
            self.content_belief.remove()
            self.content_belief = None

        color = np.ones((self.rows, self.cols, 3))
        orange = np.array([1.0, 0.5, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                # Shades of blue. Yellow means impossible.
                p = prob[row, col]
                if p == 0:
                    color[row, col, :] = np.array([1.0, 1.0, 0.0])   # yellow = impossible
                else:
                    level = (1.0 - p)
                    color[row, col, :] = np.array([level, level, 1.0])  # deeper blue for high prob

                # Overlay perceived fire (orange tint).
                if self.robot.fire[row, col]:
                    color[row, col, :] = 0.6 * color[row, col, :] + 0.4 * orange

        if showPath:
            color[self.planner.goal.row, self.planner.goal.col, :] = np.array([0.0, 1.0, 0.0])

        self.content_belief = self.ax_belief.imshow(
            color,
            aspect='equal',
            interpolation='none',
            extent=[0, self.cols, self.rows, 0],
            zorder=0,
        )
    
    def updatepath(self):
        if self.path is not None:
            self.path.remove()
            self.path = None

        path = self.planner.get_path()
        ys = []
        xs = []

        for node in path:
            # +0.5 so it is in centre of square
            ys.append(node.row+0.5)
            xs.append(node.col+0.5)

        self.path, = self.ax_belief.plot(xs, ys, color=[0, 1, 0], linewidth=2)

    def show(self, msg = None, markRobot = False, showPath = False):
        # Update both true world and robot-belief content.
        self.updategrid_true()
        self.updategrid_belief(showPath)

        # Potentially add the mark.
        self.updatemark(markRobot)

        if showPath:
            self.updatepath()

        # Flush the figure.
        self.flush()

        # Optionally display a message and wait for confirmation.
        if msg:
            input(msg)


