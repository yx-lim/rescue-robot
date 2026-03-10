import matplotlib.pyplot as plt
import numpy as np
import random
import robot


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
    def __init__(self, walls, robot: robot.Robot):
        # Save the walls, robot, and determine the rows/cols:
        self.walls = walls
        self.robot = robot
        self.spots = np.sum(np.logical_not(walls))
        self.rows  = np.size(walls, axis=0)
        self.cols  = np.size(walls, axis=1)

        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(False)
        plt.gca().axis('off')
        plt.gca().set_aspect('equal')
        plt.gca().set_xlim(0, self.cols)
        plt.gca().set_ylim(self.rows, 0)

        # Add the row/col numbers.
        for row in range(0, self.rows, 2):
            plt.gca().text(         -0.3, 0.6+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='right')
        for row in range(1, self.rows, 2):
            plt.gca().text(self.cols+0.3, 0.6+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='left')
        for col in range(0, self.cols, 2):
            plt.gca().text(0.5+col,          -0.3, '%d'%col,
                           verticalalignment='bottom',
                           horizontalalignment='center')
        for col in range(1, self.cols, 2):
            plt.gca().text(0.5+col, self.rows+0.3, '%d'%col,
                           verticalalignment='top',
                           horizontalalignment='center')

        # Draw the grid, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            plt.gca().axhline(row, lw=1, color='k', zorder=1)
        for col in range(self.cols+1):
            plt.gca().axvline(col, lw=1, color='k', zorder=1)

        # Add the text.
        plt.gca().text(0, 29, "Probability: Yellow==0%")
        plt.gca().text(0, 31, "     White<=0.1%, Pink<1%, Violet<10%, Purple<50%, Blue=100%")

        # Clear the content and mark.  Then show blank field.
        self.content = None
        self.mark    = None
        self.show()

    def flush(self):
        # Show the plot.
        plt.pause(0.001)

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
            self.mark  = plt.gca().text(0.5+col, 0.5+row, 'x', color = 'green',
                                        verticalalignment='center',
                                        horizontalalignment='center',
                                        zorder=1)

    def updategrid(self):
        # Check the probability grid array size.
        prob = self.robot.walls
        if prob is not None:
            assert np.size(prob, axis=0) == self.rows, "Inconsistent # of rows"
            assert np.size(prob, axis=1) == self.cols, "Inconsistent # of cols"

        # Potentially remove the previous grid/content.
        if self.content is not None:
            self.content.remove()
            self.content = None

        # Create the color range.  There are clearly more elegant ways...
        color = np.ones((self.rows, self.cols, 3))
        has_fire = hasattr(self.robot.world, 'is_fire')
        orange = np.array([1.0, 0.5, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                if prob is None:
                    color[row,col,0:3] = np.array([1.0, 1.0, 1.0])   # White
                else:
                    # Shades of pink/purple/blue. Yellow means impossible.
                    p    = prob[row,col]
                    pmin = 0.9 / self.spots
                    if p == 0:
                        color[row,col,0:3] = np.array([1.0, 1.0, 0.0])
                    elif p < pmin:
                        rlevel = (1.0 - p)
                        glevel = (1.0 - p)
                        color[row,col,0:3] = np.array([rlevel, glevel, 1.0])
                    else:
                        rlevel = (1.0 - p)
                        glevel = (1.0 - p) * pmin/p
                        color[row,col,0:3] = np.array([rlevel, glevel, 1.0])
                
                # Overlay fire (orange tint) without replacing the underlying map
                if has_fire and self.robot.world.is_fire(row, col):
                    color[row, col, 0:3] = 0.6 * color[row, col, 0:3] + 0.4 * orange
    
    
        # Draw the boxes.
        self.content = plt.gca().imshow(color,
                                        aspect='equal',
                                        interpolation='none',
                                        extent=[0, self.cols, self.rows, 0],
                                        zorder=0)

    def show(self, msg = None, markRobot = False):
        # Update the content.
        self.updategrid()

        # Potentially add the mark.
        self.updatemark(markRobot)

        # Flush the figure.
        self.flush()

        # Optionally display a message and wait for confirmation.
        if msg:
            input(msg)


