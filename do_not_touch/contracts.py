import numpy as np


class MDPEnv:
    def states(self) -> np.ndarray:
        pass

    def actions(self) -> np.ndarray:
        pass

    def rewards(self) -> np.ndarray:
        pass

    def is_state_terminal(self, s: int) -> bool:
        pass

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        pass

    def view_state(self, s: int):
        pass


class SingleAgentEnv:
    def state_id(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def reset_random(self):
        pass
