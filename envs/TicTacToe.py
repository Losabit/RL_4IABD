import random

from do_not_touch.contracts import SingleAgentEnv
import numpy as np


class TicTacToe(SingleAgentEnv):
    def __init__(self):
        self.cases = [-1] * 9
        self.agent_pos = 0
        self.game_over = False
        self.player_turn = True
        self.player_value = 1
        self.random_player_value = 0
        self.current_score = 0.0
        self.reset()

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        if self.cases[action_id] != -1:
            print(self.cases)
            print(action_id)
            print(self.available_actions_ids())
        assert (action_id < len(self.cases))
        assert (self.cases[action_id] == -1)
        assert (not self.game_over)

        self.agent_pos = action_id
        if self.player_turn:
            self.cases[action_id] = self.player_value
        else:
            self.cases[action_id] = self.random_player_value
        self.player_turn = not self.player_turn

        if self.tictactoe_ended(self.player_value):
            self.game_over = True
            self.current_score = 1.0
        elif self.tictactoe_ended(self.random_player_value):
            self.game_over = True
            self.current_score = -1.0
        elif -1 not in self.cases:
            self.game_over = True
            self.current_score = 0.0
        elif not self.player_turn:
            rand = random.randint(0, 8)
            while self.cases[rand] != -1:
                rand = random.randint(0, 8)
            self.act_with_action_id(rand)

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        available_actions = []
        for i in range(len(self.cases)):
            if self.cases[i] == -1:
                available_actions.append(i)
        return np.array(available_actions)

    def reset(self):
        self.game_over = False
        self.current_score = 0.0
        self.agent_pos = 0
        self.player_turn = True
        self.cases = [-1] * 9

    def line_checked(self, cases) -> bool:
        return (0 in cases and 1 in cases and 2 in cases) or \
               (3 in cases and 4 in cases and 5 in cases) or \
               (6 in cases and 7 in cases and 8 in cases)

    def column_checked(self, cases) -> bool:
        return (0 in cases and 3 in cases and 6 in cases) or \
               (1 in cases and 4 in cases and 7 in cases) or \
               (2 in cases and 5 in cases and 8 in cases)

    def diagonal_checked(self, cases) -> bool:
        return (0 in cases and 4 in cases and 8 in cases) or \
               (2 in cases and 4 in cases and 6 in cases)

    def tictactoe_ended(self, player_indice) -> bool:
        player_indice_cases = []
        for i, case in enumerate(self.cases):
            if case == player_indice:
                player_indice_cases.append(i)

        if self.line_checked(player_indice_cases) or self.column_checked(player_indice_cases) or self.diagonal_checked(player_indice_cases):
            return True
        return False
