from tqdm import tqdm
from do_not_touch.result_structures import PolicyAndActionValueFunction
from do_not_touch.single_agent_env_wrapper import Env2
from envs.TicTacToe import TicTacToe, tic_tac_toe_env
import numpy as np


def max_dict(d):
  # returns the argmax (key) and max (value) from a dictionary
  max_key = None
  max_val = float('-inf')
  for k, v in d.items():
    if v > max_val:
      max_val = v
      max_key = k
  return max_key, max_val


def algo_monte_carlo_es(env) -> PolicyAndActionValueFunction:
    max_episodes_count = 10000
    gamma = 0.85

    pi = {}
    q = {}
    returns = {}

    for ep in tqdm(range(max_episodes_count)):
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
                pi[S[t]] = list(q[S[t]].keys())[np.argmax(list(q[S[t]].values()))]

                #max = max_dict(q[s])
                #pi[s][max[0]] = max[1]

                #optimal_a_t = list(q[S[t]].keys())[np.argmax(list(q[S[t]].values()))]
                # pi[S[t]][optimal_a_t] = np.argmax(q[S[t]][optimal_a_t])
                #for a_key in pi[S[t]].keys():
                    # pi[S[t]][a_key] = np.argmax(q[S[t]][a_key])
                    # pi[S[t]][a_key] = np.argmax(q[S[t]][optimal_a_t])

    #for s in pi.keys():
    #    probabilities = np.array(list(pi[s].values()))
    #    probabilities /= probabilities.sum()
    #    for i in range(len(probabilities)):
    #        pi[s][i] = probabilities[i]

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

                max = max_dict(q[s])
                pi[s][max[0]] = max[1]
                # for a_key in pi[s_t].keys():
                #    pi[s_t][a_key] = np.argmax(Q[s_t][a_key])

                optimal_a_t = list(Q[s_t].keys())[np.argmax(list(Q[s_t].values()))]
                if chosen_action != optimal_a_t:
                    break

                W *= 1. / (available_actions[np.random.randint(len(available_actions))] + 1)

    for s in pi.keys():
        probabilities = np.array(list(pi[s].values()))
        probabilities /= probabilities.sum()
        for i in range(len(probabilities)):
            pi[s][i] = probabilities[i]

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
    choice = 0
    print("Choisissez un mode de jeu pour TicTacToe :")
    print("1. Joueur vs Random")
    print("2. Random vs Monte Carlo")
    #print("3. Joueur vs Monte Carlo")
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
        print("1. Explorating Start")
        print("2. Off Policy")
        print("3. On Policy")
        while algo_choice != 1 and algo_choice != 2 and algo_choice != 3:
            algo_choice = int(input())

        if algo_choice == 1:
            trained = monte_carlo_es_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)
        elif algo_choice == 2:
            trained = off_policy_monte_carlo_control_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)
        elif algo_choice == 3:
            trained = on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo()
            tic_tac_toe_env(trained.pi, trained.q, number_of_games)



    # print(monte_carlo_es_on_secret_env2())
    # print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    # print(off_policy_monte_carlo_control_on_secret_env2())
