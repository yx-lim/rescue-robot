from math import inf

class Node:
    # Initialization
    def __init__(self, row, col):
        # Save the matching state.
        self.row = row
        self.col = col

        # Clear the list of neighbors (used for the full graph).
        self.neighbors = []

        # Clear the parent (used for the search tree), as well as the
        # actual cost to reach (via the parent).
        self.parent = None  # No parent
        self.g = inf  # Unable to reach = infinite cost
        self.rhs = inf
        self.old_c = 1.0

    # Define the Chebyshev distance to another node.
    def distance(self, other):
        return max(abs(self.row - other.row), abs(self.col - other.col))

    # Define the "less-than" to enable sorting by cost.
    def __lt__(self, other):
        return id(self) < id(other)

    # Print (for debugging).
    def __str__(self):
        return f"({self.row:2f},{self.col:2f},{self.g:2f},{self.rhs})"

    def __repr__(self):
        return f"<Node ({self.row},{self.col}) cost={self.g} rhs={self.rhs}>"