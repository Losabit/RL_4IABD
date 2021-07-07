from do_not_touch.mdp_env_wrapper import Env1
from do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from envs.LineWorld import LineWorld
from envs.GridWorld import GridWorld
import numpy as np


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = LineWorld(7)
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())

    theta = 0.00001
    gamma = 1.0

    V = np.zeros((len(env.states()),))
    while True:
        delta = 0
        for s in env.states():
            old_v = V[s]
            V[s] = 0.0
            for a in env.actions():
                for s_next in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break

    return dict(enumerate(V.flatten(), 1))


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = LineWorld(7)
    V = np.zeros((len(env.states()),))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())

    theta = 0.00001
    gamma = 1.0

    while True:
        while True:
            delta = 0
            for s in env.states():
                old_v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    for s_next in env.states():
                        for r_idx, r in enumerate(env.rewards()):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (r + gamma * V[s_next])
                delta = max(delta, abs(V[s] - old_v))

            if delta < theta:
                break

        policy_stable = True
        for s in env.states():
            old_policy = pi[s, :]

            best_a = None
            best_a_value = None
            for a in env.actions():
                a_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        a_value += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_value is None or best_a_value < a_value:
                    best_a_value = a_value
                    best_a = a

            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if not np.array_equal(pi[s], old_policy):
                policy_stable = False

        if policy_stable:
            break

    final_pi = {}
    for indice, value in enumerate(pi):
        final_pi[indice] = dict(enumerate(value.flatten(), 1))

    return PolicyAndValueFunction(final_pi, dict(enumerate(V.flatten(), 1)))


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = LineWorld(7)
    V = np.zeros((len(env.states()),))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())
    pi2 = pi.copy()

    theta = 0.00001
    gamma = 1.0

    while True:
        delta = 0
        for s in env.states():
            old_v = V[s]
            V[s] = 0.0
            best_a_value = None
            best_a = None
            for a in env.actions():
                a_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        pre_a_value = env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                        a_value += pre_a_value
                        V[s] += pi[s, a] * pre_a_value
                if best_a_value is None or best_a_value < a_value:
                    best_a_value = a_value
                    best_a = a

            delta = max(delta, abs(V[s] - old_v))
            pi2[s, :] = 0.0
            pi2[s, best_a] = 1.0

        if delta < theta:
            break

    final_pi = {}
    for indice, value in enumerate(pi2):
        final_pi[indice] = dict(enumerate(value.flatten(), 1))

    return PolicyAndValueFunction(final_pi, dict(enumerate(V.flatten(), 1)))


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = GridWorld(5, 5)
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())

    theta = 0.00001
    gamma = 1.0

    V = np.zeros((len(env.states()),))
    while True:
        delta = 0
        for s in env.states():
            old_v = V[s]
            V[s] = 0.0
            for a in env.actions():
                for s_next in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break

    return dict(enumerate(V.flatten(), 1))


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = GridWorld(5, 5)
    V = np.zeros((len(env.states()),))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())

    theta = 0.00001
    gamma = 1.0

    while True:
        while True:
            delta = 0
            for s in env.states():
                old_v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    for s_next in env.states():
                        for r_idx, r in enumerate(env.rewards()):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (r + gamma * V[s_next])
                delta = max(delta, abs(V[s] - old_v))

            if delta < theta:
                break

        policy_stable = True
        for s in env.states():
            old_policy = pi[s, :]

            best_a = None
            best_a_value = None
            for a in env.actions():
                a_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        a_value += env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                if best_a_value is None or best_a_value < a_value:
                    best_a_value = a_value
                    best_a = a

            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if not np.array_equal(pi[s], old_policy):
                policy_stable = False

        if policy_stable:
            break

    final_pi = {}
    for indice, value in enumerate(pi):
        final_pi[indice] = dict(enumerate(value.flatten(), 1))

    return PolicyAndValueFunction(final_pi, dict(enumerate(V.flatten(), 1)))


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = GridWorld(5, 5)
    V = np.zeros((len(env.states()),))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())
    pi2 = pi.copy()

    theta = 0.00001
    gamma = 1.0

    while True:
        delta = 0
        for s in env.states():
            old_v = V[s]
            V[s] = 0.0
            best_a_value = None
            best_a = None
            for a in env.actions():
                a_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        pre_a_value = env.transition_probability(s, a, s_p, r_idx) * (r + gamma * V[s_p])
                        a_value += pre_a_value
                        V[s] += pi[s, a] * pre_a_value
                if best_a_value is None or best_a_value < a_value:
                    best_a_value = a_value
                    best_a = a

            delta = max(delta, abs(V[s] - old_v))
            pi2[s, :] = 0.0
            pi2[s, best_a] = 1.0

        if delta < theta:
            break

    final_pi = {}
    for indice, value in enumerate(pi2):
        final_pi[indice] = dict(enumerate(value.flatten(), 1))

    return PolicyAndValueFunction(final_pi, dict(enumerate(V.flatten(), 1)))


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())

    theta = 0.00001
    gamma = 1.0

    V = np.zeros((len(env.states()),))
    while True:
        delta = 0
        for s in env.states():
            old_v = V[s]
            V[s] = 0.0
            for a in env.actions():
                for s_next in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (r + gamma * V[s_next])
            delta = max(delta, abs(V[s] - old_v))

        if delta < theta:
            break

    return dict(enumerate(V.flatten(), 1))


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
