from do_not_touch.contracts import MDPEnv
import numpy as np


class LineWorld(MDPEnv):
    def __init__(self, cells_count: int):
        self.cells_count = cells_count
        self.__states = np.arange(self.cells_count)
        self.__actions = np.array([0, 1])
        self.__rewards = np.array([-1, 0, 1])
        self.probality = self.probality_setup()

    def probality_setup(self):
        p = np.zeros((len(self.__states), len(self.__actions), len(self.__states), len(self.__rewards)))
        for s in range(1, self.cells_count - 2):
            p[s, 1, s + 1, 1] = 1.0

        for s in range(2, self.cells_count - 1):
            p[s, 0, s - 1, 1] = 1.0

        p[self.cells_count - 2, 1, self.cells_count - 1, 2] = 1.0
        p[1, 0, 0, 0] = 1.0
        return p

    def states(self) -> np.ndarray:
        return self.__states

    def actions(self) -> np.ndarray:
        return self.__actions

    def rewards(self) -> np.ndarray:
        return self.__rewards

    def is_state_terminal(self, s: int) -> bool:
        return s == self.cells_count - 1 or s == 0

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.probality[s, a, s_p, r]

    def view_state(self, s: int):
        pass

