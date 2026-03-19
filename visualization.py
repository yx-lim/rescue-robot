import matplotlib.pyplot as plt
import numpy as np
from planner import PlannerDStarLite, PlannerTemporal, PlannerAStarReplan
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
    def __init__(self, walls, robot: Robot, planner: PlannerDStarLite | PlannerTemporal | PlannerAStarReplan):
        # Save the walls, robot, and determine the rows/cols:
        self.walls = walls
        self.robot = robot
        self.world = robot.world
        self.planner = planner
        self.spots = np.sum(np.logical_not(walls))
        self.rows  = np.size(walls, axis=0)
        self.cols  = np.size(walls, axis=1)

        # Single figure with two subplots side-by-side:
        # left = true world, right = robot belief.
        plt.close(1)
        plt.close(2)
        self.fig, (self.ax_true, self.ax_belief) = plt.subplots(1, 2, figsize=(12, 6))

        self._setup_axes(self.ax_true, "True State")
        self._setup_axes(self.ax_belief, "Robot's Perception")

        # Add the text.
        self.ax_belief.text(0, 40, "Probability: Yellow==0%")
        self.ax_belief.text(0, 42, "     White<=0.1%, Blue, Black=100%")

        # Clear the content and mark.  Then show blank field.
        self.content_belief = None
        self.content_true = None
        self.mark_belief = None
        self.mark_true = None
        self.path_belief = None
        self.path_true = None

        self.fig.tight_layout()
        self.show()

    def _setup_axes(self, ax, title):
        ax.grid(False)
        ax.set_title(title, y=1.05)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(0, self.cols)
        ax.set_ylim(self.rows, 0)

        # Add row/col numbers only on the belief subplot (right).
        if ax is self.ax_belief:
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
        if self.mark_belief is not None:
            self.mark_belief.remove()
            self.mark_belief = None
        if self.mark_true is not None:
            self.mark_true.remove()
            self.mark_true = None

        # If requested, add a new mark.
        if markRobot:
            # Grab the robot position and check.
            row = self.robot.row
            col = self.robot.col
            assert (row >= 0) and (row < self.rows), "Illegal robot row"
            assert (col >= 0) and (col < self.cols), "Illegal robot col"

            self.mark_belief = self.ax_belief.text(
                0.5 + col, 0.5 + row, "x",
                color="red",
                verticalalignment="center",
                horizontalalignment="center",
                fontweight="bold",
                zorder=3
            )

            self.mark_true = self.ax_true.text(
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

        if self.content_belief is not None:
            self.content_belief.remove()
            self.content_belief = None

        # Create the color range.  There are clearly more elegant ways...
        color = np.ones((self.rows, self.cols, 3))
        # has_fire = hasattr(self.robot.world, 'is_fire')
        orange = np.array([1.0, 0.0, 0.0])
        for row in range(self.rows):
            for col in range(self.cols):
                p = prob[row, col]

                if p == 0:
                    color[row, col, :] = np.array([1.0, 1.0, 0.0])  # yellow
                else:
                    level = 1.0 - p
                    color[row, col, :] = np.array([level, level, 1.0])  # blue tint

                if isinstance(self.planner, PlannerTemporal):
                    fire_timeline = self.planner.fire_pred[:20, row, col]
                    burning_steps = np.where(fire_timeline > 0)[0]

                    if len(burning_steps) > 0:
                        first_t = burning_steps[0]
                        fire_fraction = len(burning_steps) / 20
                        normalized_t = first_t / 20

                        # Red = fire soon and sustained, yellow = late or brief
                        fire_color = np.array([
                            1.0,
                            0.5 + normalized_t * 0.2 + (1.0 - fire_fraction) * 0.3,  # G: starts at 0.5 minimum, trends yellow
                            0.0
                        ])

                        # Stronger blend when fire arrives sooner
                        opacity = 0.25 + 0.5 * (1.0 - normalized_t)
                        color[row, col, :] = (1 - opacity) * color[row, col, :] + opacity * fire_color

                if self.robot.world.is_fire(row, col):
                    color[row, col, :] = 0.6 * color[row, col, :] + 0.4 * orange

        if showPath:
            goal_row, goal_col = self._goal_pos()
            color[goal_row, goal_col, :] = np.array([0.0, 1.0, 0.0])

        self.content_belief = self.ax_belief.imshow(
            color,
            aspect="equal",
            interpolation="none",
            extent=[0, self.cols, self.rows, 0],
            zorder=0
        )

    def update_true(self, showPath):
        if self.content_true is not None:
            self.content_true.remove()
            self.content_true = None

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

        self.content_true = self.ax_true.imshow(
            color,
            aspect="equal",
            interpolation="none",
            extent=[0, self.cols, self.rows, 0],
            zorder=0
        )

    def updatepath(self):
        if self.path_belief is not None:
            self.path_belief.remove()
            self.path_belief = None
        if self.path_true is not None:
            self.path_true.remove()
            self.path_true = None

        route = self.planner.get_path()
        if not route:
            return

        xs, ys = self._route_to_xy(route)
        if len(xs) == 0:
            return

        self.path_belief, = self.ax_belief.plot(xs, ys, color=[0, 1, 0], linewidth=2, zorder=2)
        self.path_true, = self.ax_true.plot(xs, ys, color=[0, 1, 0], linewidth=2, zorder=2)

    def show(self, msg=None, markRobot=False, showPath=False):
        self.updategrid(showPath)
        self.updatemark(markRobot)
        self.update_true(showPath)
        self.updatemark(markRobot)

        if showPath:
            self.updatepath()

        self.flush()

        if msg:
            input(msg)