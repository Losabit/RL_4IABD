import random
from do_not_touch.contracts import SingleAgentEnv
import numpy as np
import pygame
import math


def get_best_tic_tac_toe_play(available_actions, q, S, round_counter):
    if len(available_actions) == 1:
        return available_actions[0]

    for i in range(len(list(q[S[round_counter]].keys())) - 1, 0, -1):
        best_action_value = np.sort(list(q[S[round_counter]].values()))[i]
        best_action = list(q[S[round_counter]].keys())[list(q[S[round_counter]].values()).index(best_action_value)]
        if best_action in available_actions:
            return best_action


def tic_tac_toe_env(pi, q):
    env = TicTacToe()
    X = 600
    Y = 600
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((X, Y))
    background = pygame.image.load('assets/tictactoe/background.png')
    x = pygame.image.load('assets/tictactoe/x.png')
    o = pygame.image.load('assets/tictactoe/o.png')
    game_finished = False

    game_won = 0
    game_counter = 0

    if pi and q:
        monte_carlo_playing = True
        S = []
        round_counter = 0

    while True:
        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))

        for indice_case, case in enumerate(env.cases):
            column = indice_case % 3
            line = indice_case // 3
            if case == 0:
                screen.blit(o, (column * X / 3, line * Y / 3))
            elif case == 1:
                screen.blit(x, (column * X / 3, line * Y / 3))

        if env.is_game_over() and not game_finished:
            game_finished = True
            game_counter += 1
            won = env.score() == 1
            if won:
                game_won += 1
            win_rate = int((game_won / game_counter) * 100)
            pygame.display.set_caption("Tic Tac Toe - [winrate {}%]".format(win_rate))

        if monte_carlo_playing and not game_finished:
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()

            best_action = get_best_tic_tac_toe_play(available_actions, q, S, round_counter)

            env.act_with_action_id(best_action)
            round_counter = round_counter + 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONUP and not game_finished:
                if not monte_carlo_playing:
                    pos = pygame.mouse.get_pos()
                    column = int(pos[0] // (X / 3))
                    line = int(pos[1] // (Y / 3))

                    env.act_with_action_id(column + line * 3)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and game_finished:  # restart game
                game_finished = False
                env.reset()
                if monte_carlo_playing:
                    S = []
                    round_counter = 0

        pygame.display.update()


def init_tic_tac_toe_dict():
    dict = {}
    all_possible_states = 9
    for s in range(all_possible_states):
        dict[s] = {}
        for a in range(all_possible_states):
            dict[s][a] = 0
    return dict


class TicTacToe(SingleAgentEnv):
    def __init__(self):
        self.cases = [-1] * 9
        self.game_state = 0
        self.game_over = False
        self.player_turn = True
        self.player_value = 1
        self.random_player_value = 0
        self.current_score = 0.0
        self.reset()

    def state_id(self) -> int:
        sum = 0
        available_actions_size = 2
        for i in range(len(self.cases)):
            case = self.cases[i]
            if case == self.player_value:
                sum += pow(available_actions_size, i)
            elif case == self.random_player_value:
                sum += pow(available_actions_size, len(self.cases) + i)
        return sum

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

        if self.player_turn:
            self.cases[action_id] = self.player_value
        else:
            self.cases[action_id] = self.random_player_value

        self.player_turn = not self.player_turn
        self.game_state = self.state_id()

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
        self.game_state = 0
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
