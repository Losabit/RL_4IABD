import os

from envs.Deep.PacMan import PacMan, pac_man_env

import tqdm
from do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv

import tensorflow as tf
from numba import jit
import numpy as np
from collections import deque
import random

from envs.Deep.TicTacToe import TicTacToe, tic_tac_toe_env

batch_size = 32


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, available_action, reward, next_state, next_available_action, done):
        self.buffer.append([state, action, available_action, reward, next_state, next_available_action, done])

    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, available_actions, rewards, next_states, next_available_actions, done = map(np.asarray,
                                                                                                     zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, available_actions, rewards, next_states, next_available_actions, done

    def size(self):
        return len(self.buffer)


def get_q_inputs(available_actions, state, state_description_length, max_actions_count):
    q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
    for i, a in enumerate(available_actions):
        q_inputs[i] = np.hstack([state, tf.keras.utils.to_categorical(a, max_actions_count)])

    return q_inputs


def deep_q_learning(env: DeepSingleAgentWithDiscreteActionsEnv):
    epsilon = 0.1
    gamma = 0.95
    eps_decay = 0.995
    eps_min = 0.01
    max_episodes_count = 100

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation=tf.keras.activations.relu,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(16, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    target_model = model

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    target_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    target_model.set_weights(model.get_weights())

    buffer = ReplayBuffer()

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        done, total_reward = False, 0
        s = env.state_description()
        env.reset()

        while not env.is_game_over():
            available_actions = env.available_actions_ids()

            # get model action
            epsilon *= eps_decay
            epsilon = max(epsilon, eps_min)
            if np.random.random() < epsilon:
                action = np.random.choice(available_actions)
            else:
                all_q_inputs = get_q_inputs(available_actions, s, state_description_length, max_actions_count)
                all_q_values = np.squeeze(model.predict(all_q_inputs))
                action = available_actions[np.argmax(all_q_values)]

            # step env
            previous_score = env.score()
            env.act_with_action_id(action)
            r = env.score() - previous_score
            next_available_action = env.available_actions_ids()

            next_s = env.state_description()
            buffer.put(s, action, available_actions, r * 0.01, next_s, next_available_action, env.is_game_over())
            total_reward += r
            s = next_s

            if buffer.size() >= batch_size:
                for _ in range(10):
                    states, actions, available_actions, rewards, next_states, next_available_actions, done = buffer.sample()
                    targets = []
                    next_q_values = []
                    for x in range(len(states)):
                        all_q_inputs = get_q_inputs(available_actions[x], states[x], state_description_length,
                                                    max_actions_count)
                        targets.append(target_model.predict(all_q_inputs).max(axis=1))

                        next_all_q_inputs = get_q_inputs(next_available_actions[x], next_states[x],
                                                         state_description_length,
                                                         max_actions_count)
                        if len(next_all_q_inputs) == 0:
                            next_q_values.append(np.array([0.0]))
                        else:
                            test = target_model.predict(next_all_q_inputs).max(axis=1)
                            next_q_values.append(test)

                    targets[range(batch_size), actions] = rewards + (1 - done) * next_q_values * gamma
                    model.fit(states, targets, epochs=1, verbose=0)

            target_model.set_weights(model.get_weights())

            print(f"Episode[{episode_id}] => Reward : {total_reward}")
    return model


def episodic_semi_gradient_sarsa(env: DeepSingleAgentWithDiscreteActionsEnv):
    epsilon = 0.25
    gamma = 0.9
    max_episodes_count = 100 if not isinstance(env, PacMan) else 2
    pre_warm = (max_episodes_count / 10) if not isinstance(env, PacMan) else 1

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    q = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])

    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        env.reset()
        round_counter = 0

        while not env.is_game_over():
            round_counter += 1
            s = env.state_description()
            available_actions = env.available_actions_ids()

            if (episode_id < pre_warm) or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                all_q_inputs = get_q_inputs(available_actions, s, state_description_length, max_actions_count)
                all_q_values = np.squeeze(q.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                q.train_on_batch(np.array([q_inputs]), np.array([target]))
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = q.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
            next_chosen_action_q_value = q.predict(np.array([next_q_inputs]))[0][0]

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            q.train_on_batch(np.array([q_inputs]), np.array([target]))

    return q


def demo():
    env = PacMan()
    # episodic_semi_gradient_sarsa_jit = jit(episodic_semi_gradient_sarsa)
    q = episodic_semi_gradient_sarsa(env)

    #env = TicTacToe()
    # episodic_semi_gradient_sarsa_jit = jit()(episodic_semi_gradient_sarsa)
    # q = episodic_semi_gradient_sarsa_jit(env)
    #q = deep_q_learning(env)
    print(q)
    pac_man_env(1, q)
    #tic_tac_toe_env(1, q)
