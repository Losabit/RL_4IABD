import os

from envs.Deep.PacMan import PacMan, pac_man_env

import tqdm
from do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv
from envs.Deep.TicTacToe import TicTacToe, tic_tac_toe_env

import tensorflow as tf
import numpy as np


def episodic_semi_gradient_sarsa(env: DeepSingleAgentWithDiscreteActionsEnv):
    epsilon = 0.1
    gamma = 0.9
    max_episodes_count = 100 if not isinstance(env, PacMan) else 10
    pre_warm = (max_episodes_count / 10) if not isinstance(env, PacMan) else 3

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
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])

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
    q = episodic_semi_gradient_sarsa(env)
    print(q)
    pac_man_env(1, q)
