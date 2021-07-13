from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2
from envs.TicTacToe import TicTacToe
import pygame
import random
import numpy as np


def get_best_tic_tac_toe_play(available_actions, q, S, round_counter):
    for i in range(len(list(q[S[round_counter]].keys()))-1, 0, -1):
        best_action_value = np.sort(list(q[S[round_counter]].values()))[i]
        best_action = list(q[S[round_counter]].keys())[list(q[S[round_counter]].values()).index(best_action_value)]
        if best_action in available_actions:
            return best_action


def tic_tac_toe_env(pi, q):
    env = TicTacToe()
    X = 600
    Y = 600
    pygame.init()
    screen = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Tic Tac Toe')
    background = pygame.image.load('assets/tictactoe/background.png')
    x = pygame.image.load('assets/tictactoe/x.png')
    o = pygame.image.load('assets/tictactoe/o.png')
    turn = 1
    game_finished = False

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONUP and turn == 0 and not game_finished:
                if monte_carlo_playing:
                    s = env.state_id()
                    S.append(s)
                    available_actions = env.available_actions_ids()

                    best_action = get_best_tic_tac_toe_play(available_actions, q, S, round_counter)

                    env.act_with_action_id(best_action)
                    round_counter = round_counter + 1
                else:
                    pos = pygame.mouse.get_pos()
                    column = int(pos[0] // (X / 3))
                    line = int(pos[1] // (Y / 3))

                    env.act_with_action_id(column + line * 3)
                turn = 1
                if env.is_game_over() and not game_finished:
                    game_finished = True

        if turn == 1 and not game_finished:
            rand = random.randint(0, 8)
            while env.cases[rand] != -1:
                rand = random.randint(0, 8)
            env.act_with_action_id(rand)
            turn = 0
            if env.is_game_over() and not game_finished:
                game_finished = True

        pygame.display.update()


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe()
    max_episodes_count = 1000
    gamma = 0.9

    pi = {}
    q = {}
    returns = {}

    for ep in range(max_episodes_count):
        env.reset()

        S = []
        A = []
        R = []

        G = 0
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()

            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = available_actions[np.random.randint(len(available_actions))]
            A.append(chosen_action)

            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

            for t in reversed(range(len(S))):
                G = gamma * G + R[t]

                found = False
                for prev_s, prev_a in zip(S[:t], A[:t]):
                    if prev_s == S[t] and prev_a == A[t]:
                        found = True
                        break
                if found:
                    continue

                if A[t] not in returns[S[t]]:
                    returns[S[t]][A[t]] = []

                returns[S[t]][A[t]].append(G)
                q[S[t]][A[t]] = np.mean(returns[S[t]][A[t]])

                for a_key in pi[S[t]].keys():
                    pi[S[t]][a_key] = np.argmax(q[S[t]][a_key])
    return pi, q


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    epsilon = 0.1
    max_episodes_count = 1000
    gamma = 0.9

    pi = {}
    q = {}
    returns = {}

    for ep in range(max_episodes_count):
        env.reset()

        S = []
        A = []
        R = []

        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]
            A.append(chosen_action)

            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]

            found = False
            for prev_s, prev_a in zip(S[:t], A[:t]):
                if prev_s == S[t] and prev_a == A[t]:
                    found = True
                    break
            if found:
                continue

            returns[S[t]][A[t]].append(G)
            q[S[t]][A[t]] = np.mean(returns[S[t]][A[t]])

            best_action = list(q[S[t]].keys())[np.argmax(
                list(q[S[t]].values())
            )]

            for a_key in pi[S[t]].keys():
                if a_key == best_action:
                    pi[S[t]][a_key] = 1 - epsilon + epsilon / len(pi[S[t]])
                else:
                    pi[S[t]][a_key] = epsilon / len(pi[S[t]])

    return pi, q


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    # TODO
    pass


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def demo():
    pi, q = monte_carlo_es_on_tic_tac_toe_solo()
    tic_tac_toe_env(pi, q)

    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

# print(monte_carlo_es_on_secret_env2())
# print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
# print(off_policy_monte_carlo_control_on_secret_env2())
