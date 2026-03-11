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
        self.planner = planner
        self.spots = np.sum(np.logical_not(walls))
        self.rows  = np.size(walls, axis=0)
        self.cols  = np.size(walls, axis=1)

        self.fig1 = plt.figure(1)
        self.fig2 = plt.figure(2)
        
        self._setup_figure(self.fig1, "Robot's Perception")
        self._setup_figure(self.fig2, "True State")

        # Add the text.
        self.fig1.gca().text(0, 40, "Probability: Yellow==0%")
        self.fig1.gca().text(0, 42, "     White<=0.1%, Blue, Black=100%")

        # Clear the content and mark.  Then show blank field.
        self.content1 = None
        self.mark1    = None
        self.mark2    = None
        self.path1    = None
        self.path2    = None
        self.content2 = None
        self.show()

    def _setup_figure(self, fig, title):
        # Clear the current, or create a new figure.
        fig.clf()

        # Create a new axes, enable the grid, and set axis limits.
        fig.gca().grid(False)
        fig.gca().set_title(title, y=1.05)
        fig.gca().axis('off')
        fig.gca().set_aspect('equal')
        fig.gca().set_xlim(0, self.cols)
        fig.gca().set_ylim(self.rows, 0)

        # Add the row/col numbers.
        for row in range(0, self.rows, 2):
            fig.gca().text(         -0.3, 0.6+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='right')
        for row in range(1, self.rows, 2):
            fig.gca().text(self.cols+0.3, 0.6+row, '%d'%row,
                           verticalalignment='center',
                           horizontalalignment='left')
        for col in range(0, self.cols, 2):
            fig.gca().text(0.5+col,          -0.3, '%d'%col,
                           verticalalignment='bottom',
                           horizontalalignment='center')
        for col in range(1, self.cols, 2):
            fig.gca().text(0.5+col, self.rows+0.3, '%d'%col,
                           verticalalignment='top',
                           horizontalalignment='center')
            
        # Draw the grid, zorder 1 means draw after zorder 0 elements.
        for row in range(self.rows+1):
            fig.gca().axhline(row, lw=1, color='k', zorder=1)
        for col in range(self.cols+1):
            fig.gca().axvline(col, lw=1, color='k', zorder=1)


    def flush(self):
        # Show the plot.
        plt.pause(0.3)

    def updatemark(self, markRobot=True):
        # Clear/potentially remove the previous mark.
        if self.mark1 is not None:
            self.mark1.remove()
            self.mark1 = None
        if self.mark2 is not None:
            self.mark2.remove()
            self.mark2 = None

        # If requested, add a new mark.
        if markRobot:
            # Grab the robot position and check.
            row = self.robot.row
            col = self.robot.col
            assert (row >= 0) and (row < self.rows), "Illegal robot row"
            assert (col >= 0) and (col < self.cols), "Illegal robot col"

            # Draw the mark.
            self.mark1  = self.fig1.gca().text(0.5+col, 0.5+row, 'x', color = 'red',
                                        verticalalignment='center',
                                        horizontalalignment='center',
                                        fontweight='bold',
                                        zorder=1)
            
            self.mark2  = self.fig2.gca().text(0.5+col, 0.5+row, 'x', color = 'red',
                                        verticalalignment='center',
                                        horizontalalignment='center',
                                        fontweight='bold',
                                        zorder=1)

    def logits_to_probs(self, logits):
        return 1 / (1 + np.exp(-logits))

    def updategrid(self, showPath):
        # Check the probability grid array size.
        prob = self.logits_to_probs(self.robot.walls_logits)
        if prob is not None:
            assert np.size(prob, axis=0) == self.rows, "Inconsistent # of rows"
            assert np.size(prob, axis=1) == self.cols, "Inconsistent # of cols"

        # Potentially remove the previous grid/content.
        if self.content1 is not None:
            self.content1.remove()
            self.content1 = None

        # Create the color range.  There are clearly more elegant ways...
        color = np.ones((self.rows, self.cols, 3))
        # has_fire = hasattr(self.robot.world, 'is_fire')
        orange = np.array([1.0, 0.5, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                if prob is None:
                    color[row,col,0:3] = np.array([1.0, 1.0, 1.0])   # White
                else:
                    # Shades of blue. Yellow means impossible.
                    p    = prob[row,col]
                    if p == 0:
                        color[row,col,0:3] = np.array([1.0, 1.0, 0.0])   # yellow = impossible
                    else:
                        level = (1.0 - p)
                        color[row,col,0:3] = np.array([level, level, 1])  # deeper blue for high prob
                
                # Overlay fire (orange tint) without replacing the underlying map
                # ill keep for now only showing the robot's perception of fire, but will add 
                # the true state later (it's in one of the to-dos)
                if self.robot.fire[row, col]:
                    color[row, col, 0:3] = 0.6 * color[row, col, 0:3] + 0.4 * orange
                #if has_fire and self.robot.world.is_fire(row, col):
                #    color[row, col, 0:3] = 0.6 * color[row, col, 0:3] + 0.4 * orange
    
        if showPath:
            color[self.planner.goal.row, self.planner.goal.col, 0:3] = np.array([0.0, 1.0, 0.0])
    
        # Draw the boxes.
        self.content1 = self.fig1.gca().imshow(color,
                                        aspect='equal',
                                        interpolation='none',
                                        extent=[0, self.cols, self.rows, 0],
                                        zorder=0)
        
    def update_true(self, showPath):
        # Check the probability grid array size.
        prob = self.logits_to_probs(self.robot.walls_logits)
        if prob is not None:
            assert np.size(prob, axis=0) == self.rows, "Inconsistent # of rows"
            assert np.size(prob, axis=1) == self.cols, "Inconsistent # of cols"

        # Potentially remove the previous grid/content.
        if self.content2 is not None:
            self.content2.remove()
            self.content2 = None

        # Create the color range.  There are clearly more elegant ways...
        color = np.ones((self.rows, self.cols, 3))
        orange = np.array([1.0, 0.5, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                if self.robot.world.is_wall(row, col):
                    color[row,col,0:3] = np.array([0.0, 0.0, 0.0]) # Black
                else:
                    color[row,col,0:3] = np.array([1.0, 1.0, 1.0])   # White
                # Overlay fire (orange tint) without replacing the underlying map
                if self.robot.world.is_fire(row, col):
                    color[row, col, 0:3] = 0.6 * color[row, col, 0:3] + 0.4 * orange
                #if has_fire and self.robot.world.is_fire(row, col):
                #    color[row, col, 0:3] = 0.6 * color[row, col, 0:3] + 0.4 * orange
    
        if showPath:
            color[self.planner.goal.row, self.planner.goal.col, 0:3] = np.array([0.0, 1.0, 0.0])
    
        # Draw the boxes.
        self.content2 = self.fig2.gca().imshow(color,
                                        aspect='equal',
                                        interpolation='none',
                                        extent=[0, self.cols, self.rows, 0],
                                        zorder=0)
    
    def updatepath(self):
        # Clear both paths
        if self.path1 is not None:
            self.path1.remove()
            self.path1 = None
        if self.path2 is not None:
            self.path2.remove()
            self.path2 = None

        route = self.planner.get_path()
        ys = [node.row + 0.5 for node in route]
        xs = [node.col + 0.5 for node in route]

        plt.figure(self.fig1.number)
        self.path1, = plt.plot(xs, ys, color=[0, 1, 0], linewidth=2)

        plt.figure(self.fig2.number)
        self.path2, = plt.plot(xs, ys, color=[0, 1, 0], linewidth=2)


    def show(self, msg = None, markRobot = False, showPath = False):
        plt.figure(1)
        self.updategrid(showPath)
        self.updatemark(markRobot)

        plt.figure(2)
        self.update_true(showPath)

        if showPath:
            self.updatepath()

        # Flush the figure.
        self.flush()

        # Optionally display a message and wait for confirmation.
        if msg:
            input(msg)


