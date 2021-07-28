from tqdm import tqdm
from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env3
from envs.TicTacToe import TicTacToe, tic_tac_toe_env
from to_do.monte_carlo_methods import max_dict
import numpy as np


def algo_q_learning(env) -> PolicyAndActionValueFunction:
    alpha = 0.1
    epsilon = 1.0
    gamma = 0.9
    max_iter = 10000

    pi = {}  # learned greedy policy
    b = {}  # behaviour epsilon-greedy policy
    q = {}  # action-value function of pi

    for it in tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                b[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)

            # actions disponibles differents selon les states
            available_actions_count = len(available_actions)
            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1 - epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    q[s_p] = {}
                    b[s_p] = {}
                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / len(next_available_actions)
                        q[s_p][a] = 0.0
                        b[s_p][a] = 1.0 / len(next_available_actions)
                q[s][chosen_action] += alpha * (r + gamma * np.max(list(q[s_p].values())) - q[s][chosen_action])

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)


def get_epsilon_best_action(epsilon, available_actions, Q, s):
    available_actions_len = len(available_actions)
    if available_actions_len == 1:
        return available_actions[0]
    elif available_actions_len == 0:
        action_values = list(Q[s].values())
        if len(action_values) > 0:
            best_action_value = np.sort(action_values)[len(action_values)-1]
            best_action = list(Q[s].keys())[list(Q[s].values()).index(best_action_value)]
            return best_action
        else:
            return np.random.randint(8)

    if np.random.uniform(0, 1) > epsilon:
        return available_actions[np.random.randint(available_actions_len)]
    else:
        for i in range(len(list(Q[s].keys())) - 1, 0, -1):
            best_action_value = np.sort(list(Q[s].values()))[i]
            best_action = list(Q[s].keys())[list(Q[s].values()).index(best_action_value)]
            if best_action in available_actions:
                return best_action
        return available_actions[np.random.randint(available_actions_len)]


def algo_sarsa(env) -> PolicyAndActionValueFunction:
    max_episodes_count = 10000
    alpha = 0.85
    gamma = 0.95
    epsilon = 0.1

    Q = {}
    pi = {}

    for ep in tqdm(range(max_episodes_count)):

        env.reset()
        S = []
        A = []
        R = []

        s_1 = env.state_id()
        available_actions = env.available_actions_ids()
        if s_1 not in Q:
            pi[s_1] = {}
            Q[s_1] = {}
            for a in available_actions:
                pi[s_1][a] = 1.0 / len(available_actions)
                Q[s_1][a] = 0.0
        action_1 = get_epsilon_best_action(epsilon, available_actions, Q, s_1)

        while not env.is_game_over():
            S.append(s_1)
            available_actions = env.available_actions_ids()

            if s_1 not in Q:
                pi[s_1] = {}
                Q[s_1] = {}
                for a in available_actions:
                    pi[s_1][a] = 1.0 / len(available_actions)
                    Q[s_1][a] = 0.0

            A.append(action_1)

            old_score = env.score()
            env.act_with_action_id(action_1)
            r = env.score() - old_score
            R.append(r)

            s_2 = env.state_id()
            available_actions = env.available_actions_ids()

            if s_2 not in Q:
                Q[s_2] = {}
                pi[s_2] = {}
                for a in available_actions:
                    Q[s_2][a] = 0.0
                    pi[s_2][a] = 1.0 / len(available_actions)

            action_2 = get_epsilon_best_action(epsilon, available_actions, Q, s_2)

            if action_2 not in Q[s_2]:
                Q[s_2][action_2] = 0.0

            target = r + gamma * Q[s_2][action_2]
            Q[s_1][action_1] += alpha * (target - Q[s_1][action_1])

            #for a_key in pi[s_1].keys():
            #    max = np.argmax(Q[s_1][a_key])
            #    pi[s_1][a_key] = max
            s_1 = s_2
            action_1 = action_2

    for s in Q.keys():
        max = max_dict(Q[s])
        pi[s][max[0]] = max[1]
        probabilities = np.array(list(pi[s].values()))
        probabilities /= probabilities.sum()
        for i in range(len(probabilities)):
            pi[s][i] = probabilities[i]

    return PolicyAndActionValueFunction(pi, Q)


def algo_expected_sarsa(env) -> PolicyAndActionValueFunction:
    alpha = 0.1
    epsilon = 1.0
    gamma = 0.9
    max_iter = 10000

    pi = {}  # learned greedy policy
    b = {}  # behaviour epsilon-greedy policy
    q = {}  # action-value function of pi

    for it in tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                b[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)

            # actions disponibles differents selon les states
            available_actions_count = len(available_actions)
            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1 - epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            s_p = env.state_id()
            next_available_actions = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    q[s_p] = {}
                    b[s_p] = {}
                    for a in next_available_actions:
                        pi[s_p][a] = 1.0 / len(next_available_actions)
                        q[s_p][a] = 0.0
                        b[s_p][a] = 1.0 / len(next_available_actions)
                sum = 0
                for a in pi[s_p]:
                    sum += pi[s_p][a] * q[s_p][a]
                q[s][chosen_action] += alpha * (r + gamma * sum - q[s][chosen_action])

    for s in q.keys():
        optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    return PolicyAndActionValueFunction(pi, q)


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    return algo_sarsa(env)


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    return algo_q_learning(env)


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    return algo_expected_sarsa(env)


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def demo():
    choice = 0
    print("Choisissez un mode de jeu pour TicTacToe :")
    print("1. Joueur vs Random")
    print("2. Random vs Monte Carlo")
    # print("3. Joueur vs Monte Carlo")
    while choice != 1 and choice != 2 and choice != 3:
        choice = int(input())

    number_of_games = "s"
    if choice == 2:
        print("Entrez un nombre de games à réaliser (possibilité d'appuyer sur R pour recommencer): ")
        while not number_of_games.isdigit():
            number_of_games = input()
        number_of_games = int(number_of_games)
    else:
        number_of_games = 1

    if choice == 1:
        tic_tac_toe_env(None, None)
    else:
        algo_choice = 0
        print("Choisissez un algorithme de MonteCarlo : ")
        print("1. Q Learning")
        print("2. Sarsa")
        print("3. Expected Sarsa")
        while algo_choice != 1 and algo_choice != 2 and algo_choice != 3:
            algo_choice = int(input())

        if algo_choice == 1:
            trained = q_learning_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)
        elif algo_choice == 2:
            trained = sarsa_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)
        elif algo_choice == 3:
            trained = expected_sarsa_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)

    #print(expected_sarsa_on_tic_tac_toe_solo())
    # print(sarsa_on_tic_tac_toe_solo())
    # print(q_learning_on_tic_tac_toe_solo())
    # print(expected_sarsa_on_tic_tac_toe_solo())

    # print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())
