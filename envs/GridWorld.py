from do_not_touch.contracts import MDPEnv
import numpy as np


class GridWorld(MDPEnv):
    def __init__(self, lines: int, columns: int):
        self.lines = lines
        self.columns = columns
        self.cells_count = lines * columns
        self.negative_terminal = 4
        self.positive_terminal = self.cells_count - 1
        self.__states = np.arange(self.cells_count)
        self.__actions = np.array([0, 1, 2, 3])
        self.__rewards = np.array([-1, 0, 1])
        self.probality = self.probality_setup()

    def probality_setup(self):
        p = np.zeros((len(self.__states), len(self.__actions), len(self.__states), len(self.__rewards)))
        for line in range(0, self.lines):
            for column in range(0, self.columns - 1):
                s = line * self.columns + column
                if s != self.negative_terminal and s != self.positive_terminal:
                    if s + 1 == self.positive_terminal:
                        p[s, 1, s + 1, 2] = 1.0
                    elif s + 1 == self.negative_terminal:
                        p[s, 1, s + 1, 0] = 1.0
                    else:
                        p[s, 1, s + 1, 1] = 1.0

            for column in range(1, self.columns):
                s = line * self.columns + column
                if s != self.positive_terminal and s != self.negative_terminal:
                    if s - 1 == self.negative_terminal:
                        p[s, 0, s - 1, 0] = 1.0
                    elif s - 1 == self.positive_terminal:
                        p[s, 0, s - 1, 2] = 1.0
                    else:
                        p[s, 0, s - 1, 1] = 1.0

        for column in range(0, self.columns):
            for line in range(0, self.lines - 1):
                s = self.columns * line + column
                s2 = self.columns * (line + 1) + column
                # up
                if s2 != self.positive_terminal and s2 != self.negative_terminal:
                    if s == self.negative_terminal:
                        p[s2, 2, s, 0] = 1.0
                    elif s == self.positive_terminal:
                        p[s2, 2, s, 2] = 1.0
                    else:
                        p[s2, 2, s, 1] = 1.0

                # down
                if s != self.negative_terminal and s != self.positive_terminal:
                    if s2 == self.positive_terminal:
                        p[s, 3, s2, 2] = 1.0
                    elif s2 == self.negative_terminal:
                        p[s, 3, s2, 0] = 1.0
                    else:
                        p[s, 3, s2, 1] = 1.0
        return p

    def states(self) -> np.ndarray:
        return self.__states

    def actions(self) -> np.ndarray:
        return self.__actions

    def rewards(self) -> np.ndarray:
        return self.__rewards

    def is_state_terminal(self, s: int) -> bool:
        return s == self.positive_terminal or s == self.negative_terminal

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.probality[s, a, s_p, r]

    def view_state(self, s: int):
        pass