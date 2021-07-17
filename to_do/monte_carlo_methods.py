from tqdm import tqdm

from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2
from envs.TicTacToe import TicTacToe
import pygame
import numpy as np


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


def algo_monte_carlo_es(env) -> PolicyAndActionValueFunction:
    max_episodes_count = 1000
    gamma = 0.85

    pi = {}
    q = {}
    returns = {}

    for ep in tqdm(range(max_episodes_count)):
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
    return PolicyAndActionValueFunction(pi, q)


def algo_on_policy_monte_carlo(env) -> PolicyAndActionValueFunction:
    epsilon = 0.1
    max_episodes_count = 10000
    gamma = 0.9

    pi = {}
    q = {}
    returns = {}

    for it in tqdm(range(max_episodes_count)):
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

            chosen_action = available_actions[np.random.randint(len(available_actions))]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

            G = 0

            for t in reversed(range(len(S))):
                G = gamma * G + R[t]
                s_t = S[t]
                a_t = A[t]
                found = False
                for p_s, p_a in zip(S[:t], A[:t]):
                    if s_t == p_s and a_t == p_a:
                        found = True
                        break
                if found:
                    continue

                if a_t not in returns[s_t]:
                    returns[s_t][a_t] = []

                returns[s_t][a_t].append(G)
                q[s_t][a_t] = np.mean(returns[s_t][a_t])
                optimal_a_t = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_actions_t_count = len(q[s_t])
                for a_key, q_s_a in q[s_t].items():
                    if a_key == optimal_a_t:
                        pi[s_t][a_key] = 1 - epsilon + epsilon / available_actions_t_count
                    else:
                        pi[s_t][a_key] = epsilon / available_actions_t_count

    return PolicyAndActionValueFunction(pi, q)


def algo_off_policy_monte_carlo(env) -> PolicyAndActionValueFunction:
    max_episodes_count = 10000
    gamma = 0.90

    Q = {}
    C = {}
    pi = {}

    for it in tqdm(range(max_episodes_count)):
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
                Q[s] = {}
                C[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    Q[s][a] = 0.0
                    C[s][a] = 0.0

            chosen_action = available_actions[np.random.randint(len(available_actions))]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

            G = 0
            W = 1

            for t in reversed(range(len(S))):
                G = gamma * G + R[t]

                s_t = S[t]
                a_t = A[t]

                if a_t not in C[s_t]:
                    C[s_t][a_t] = 0.0

                if a_t not in Q[s_t]:
                    Q[s_t][a_t] = 0.0

                C[s_t][a_t] += W
                Q[s_t][a_t] += (W / (C[s_t][a_t])) * (G - Q[s_t][a_t])

                for a_key in pi[s_t].keys():
                    pi[s_t][a_key] = np.argmax(Q[s_t][a_key])

                optimal_a_t = list(Q[s_t].keys())[np.argmax(list(Q[s_t].values()))]
                if chosen_action != optimal_a_t:
                    break

                W *= 1. / (available_actions[np.random.randint(len(available_actions))] + 1)

    return PolicyAndActionValueFunction(pi, Q)


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe()
    return algo_monte_carlo_es(env)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    return algo_on_policy_monte_carlo(env)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    return algo_off_policy_monte_carlo(env)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    return algo_monte_carlo_es(env)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    return algo_on_policy_monte_carlo(env)


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    return algo_off_policy_monte_carlo(env)


def demo():
    trained = off_policy_monte_carlo_control_on_tic_tac_toe_solo()
    tic_tac_toe_env(trained.pi, trained.q)

    # print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    # print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())
