import matplotlib.pyplot as plt
import numpy as np
from planner import PlannerDStarLite, PlannerTemporal
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
    def __init__(self, walls, robot: Robot, planner: PlannerDStarLite | PlannerTemporal):
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
        ax = fig.gca()
        ax.grid(False)
        ax.set_title(title, y=1.05)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)

        for row in range(0, self.rows, 2):
            ax.text(-0.3, 0.6 + row, f"{row}",
                    verticalalignment="center",
                    horizontalalignment="right")
        for row in range(1, self.rows, 2):
            ax.text(self.cols + 0.3, 0.6 + row, f"{row}",
                    verticalalignment="center",
                    horizontalalignment="left")
        for col in range(0, self.cols, 2):
            ax.text(0.5 + col, -0.3, f"{col}",
                    verticalalignment="bottom",
                    horizontalalignment="center")
        for col in range(1, self.cols, 2):
            ax.text(0.5 + col, self.rows + 0.3, f"{col}",
                    verticalalignment="top",
                    horizontalalignment="center")

        for row in range(self.rows + 1):
            ax.axhline(row, lw=1, color="k", zorder=1)
        for col in range(self.cols + 1):
            ax.axvline(col, lw=1, color="k", zorder=1)

    def flush(self):
        plt.pause(0.05)

    def logits_to_probs(self, logits):
        return 1.0 / (1.0 + np.exp(-logits))

    def _goal_pos(self):
        goal = self.planner.goal
        if hasattr(goal, "row") and hasattr(goal, "col"):
            return goal.row, goal.col
        if isinstance(goal, tuple) and len(goal) >= 2:
            return goal[0], goal[1]
        raise ValueError("Planner goal format not supported.")

    def _route_to_xy(self, route):
        xs, ys = [], []

        for state in route:
            if hasattr(state, "row") and hasattr(state, "col"):
                r, c = state.row, state.col
            elif isinstance(state, tuple):
                if len(state) >= 2:
                    r, c = state[0], state[1]
                else:
                    continue
            else:
                continue

            xs.append(c + 0.5)
            ys.append(r + 0.5)

        return xs, ys

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

            assert 0 <= row < self.rows, "Illegal robot row"
            assert 0 <= col < self.cols, "Illegal robot col"

            self.mark1 = self.fig1.gca().text(
                0.5 + col, 0.5 + row, "x",
                color="red",
                verticalalignment="center",
                horizontalalignment="center",
                fontweight="bold",
                zorder=3
            )

            self.mark2 = self.fig2.gca().text(
                0.5 + col, 0.5 + row, "x",
                color="red",
                verticalalignment="center",
                horizontalalignment="center",
                fontweight="bold",
                zorder=3
            )

    def updategrid(self, showPath):
        prob = self.logits_to_probs(self.robot.walls_logits)

        assert np.size(prob, axis=0) == self.rows, "Inconsistent # of rows"
        assert np.size(prob, axis=1) == self.cols, "Inconsistent # of cols"

        if self.content1 is not None:
            self.content1.remove()
            self.content1 = None

        # Create the color range.  There are clearly more elegant ways...
        color = np.ones((self.rows, self.cols, 3))
        # has_fire = hasattr(self.robot.world, 'is_fire')
        orange = np.array([1.0, 0.5, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                p = prob[row, col]

                if p == 0:
                    color[row, col, :] = np.array([1.0, 1.0, 0.0])   # yellow
                else:
                    level = 1.0 - p
                    color[row, col, :] = np.array([level, level, 1])  # blue tint

                if self.robot.fire[row, col]:
                    color[row, col, :] = 0.6 * color[row, col, :] + 0.4 * orange

        if showPath:
            goal_row, goal_col = self._goal_pos()
            color[goal_row, goal_col, :] = np.array([0.0, 1.0, 0.0])

        self.content1 = self.fig1.gca().imshow(
            color,
            aspect="equal",
            interpolation="none",
            extent=[0, self.cols, self.rows, 0],
            zorder=0
        )

    def update_true(self, showPath):
        if self.content2 is not None:
            self.content2.remove()
            self.content2 = None

        color = np.ones((self.rows, self.cols, 3))
        orange = np.array([1.0, 0.5, 0.0])

        for row in range(self.rows):
            for col in range(self.cols):
                if self.robot.world.is_wall(row, col):
                    color[row, col, :] = np.array([0.0, 0.0, 0.0])
                else:
                    color[row, col, :] = np.array([1.0, 1.0, 1.0])

                if self.robot.world.is_fire(row, col):
                    color[row, col, :] = 0.6 * color[row, col, :] + 0.4 * orange

        if showPath:
            goal_row, goal_col = self._goal_pos()
            color[goal_row, goal_col, :] = np.array([0.0, 1.0, 0.0])

        self.content2 = self.fig2.gca().imshow(
            color,
            aspect="equal",
            interpolation="none",
            extent=[0, self.cols, self.rows, 0],
            zorder=0
        )

    def updatepath(self):
        if self.path1 is not None:
            self.path1.remove()
            self.path1 = None
        if self.path2 is not None:
            self.path2.remove()
            self.path2 = None

        route = self.planner.get_path()
        if not route:
            return

        xs, ys = self._route_to_xy(route)
        if len(xs) == 0:
            return

        self.path1, = self.fig1.gca().plot(xs, ys, color=[0, 1, 0], linewidth=2, zorder=2)
        self.path2, = self.fig2.gca().plot(xs, ys, color=[0, 1, 0], linewidth=2, zorder=2)

    def show(self, msg=None, markRobot=False, showPath=False):
        plt.figure(self.fig1.number)
        self.updategrid(showPath)
        self.updatemark(markRobot)

        plt.figure(self.fig2.number)
        self.update_true(showPath)
        self.updatemark(markRobot)

        if showPath:
            self.updatepath()

        self.flush()

        if msg:
            input(msg)