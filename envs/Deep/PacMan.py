import numpy as np
import pygame
import tensorflow as tf

from do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv
from utils.timeCapsule import TimeCapsule


def add_wall(cases, start_x, start_y, x, y):
    for i in range(start_x, start_x + x):
        for j in range(start_y, start_y + y):
            cases[i][j] = -1
    return cases


# -1 mur / 0 vide / 1 dot / 2 mega dot
def initiate_map():
    cases = []
    for line in range(29):
        cases.append([1] * 26)

    cases[2][0] = 2
    cases[22][0] = 2
    cases[2][25] = 2
    cases[22][25] = 2

    for i in range(4):
        cases[i][12] = -1
        cases[i][13] = -1

    for i in range(8, 19):
        for j in range(6, 20):
            cases[i][j] = 0

    for i in range(0, 5):
        cases[13][i] = 0
    for i in range(21, 26):
        cases[13][i] = 0

    cases = add_wall(cases, 1, 1, 3, 4)
    cases = add_wall(cases, 1, 21, 3, 4)
    cases = add_wall(cases, 1, 6, 3, 5)
    cases = add_wall(cases, 1, 15, 3, 5)
    cases = add_wall(cases, 5, 1, 2, 4)
    cases = add_wall(cases, 5, 21, 2, 4)
    cases = add_wall(cases, 11, 9, 5, 8)

    cases = add_wall(cases, 8, 0, 5, 5)
    cases = add_wall(cases, 14, 0, 5, 5)
    cases = add_wall(cases, 8, 21, 5, 5)
    cases = add_wall(cases, 14, 21, 5, 5)

    cases = add_wall(cases, 5, 6, 8, 2)
    cases = add_wall(cases, 8, 7, 2, 4)
    cases = add_wall(cases, 5, 18, 8, 2)
    cases = add_wall(cases, 8, 15, 2, 4)

    cases = add_wall(cases, 14, 6, 5, 2)
    cases = add_wall(cases, 14, 18, 5, 2)

    cases = add_wall(cases, 5, 9, 2, 8)
    cases = add_wall(cases, 7, 12, 3, 2)
    cases = add_wall(cases, 17, 9, 2, 8)
    cases = add_wall(cases, 19, 12, 3, 2)
    cases = add_wall(cases, 23, 9, 2, 8)
    cases = add_wall(cases, 25, 12, 3, 2)

    cases = add_wall(cases, 20, 15, 2, 5)
    cases = add_wall(cases, 20, 6, 2, 5)

    cases = add_wall(cases, 20, 1, 2, 4)
    cases = add_wall(cases, 22, 3, 3, 2)
    cases = add_wall(cases, 20, 21, 2, 4)
    cases = add_wall(cases, 22, 21, 3, 2)

    cases = add_wall(cases, 23, 0, 2, 2)
    cases = add_wall(cases, 23, 24, 2, 2)

    cases = add_wall(cases, 26, 1, 2, 10)
    cases = add_wall(cases, 23, 6, 3, 2)
    cases = add_wall(cases, 26, 15, 2, 10)
    cases = add_wall(cases, 23, 18, 3, 2)

    return cases


def get_best_pac_man_play(available_actions, q, cases):
    all_q_inputs = np.zeros((len(available_actions), len(cases) + 4)) # 9 + 9
    for i, a in enumerate(available_actions):
        all_q_inputs[i] = np.hstack([cases, tf.keras.utils.to_categorical(a, 4)]) # 9

    all_q_values = np.squeeze(q.predict(all_q_inputs))
    #print(all_q_values)
    chosen_action = available_actions[np.argmax(all_q_values)]
    #print(available_actions)
    return chosen_action


def pac_man_env(pi, q):
    env = PacMan()
    X = 600
    Y = 600
    screen = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Pacman')
    background = pygame.image.load('assets/pacman/background.png')
    background = pygame.transform.scale(background, (600, 600))
    dot = pygame.image.load('assets/pacman/dot.png')
    dot = pygame.transform.scale(dot, (40, 40))
    pacman = pygame.image.load('assets/pacman/pacman_left.png')
    ghost_colors = [
        pygame.image.load('assets/pacman/ghosts/inky.png'),
        pygame.image.load('assets/pacman/ghosts/clyde.png'),
        pygame.image.load('assets/pacman/ghosts/pinky.png'),
        pygame.image.load('assets/pacman/ghosts/blinky.png')
    ]
    blue_ghost = pygame.image.load('assets/pacman/ghosts/blue_ghost.png')
    energizer = pygame.image.load('assets/pacman/energizer.png')

    algo_playing = False
    if pi and q:
        algo_playing = True

    while True:
        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))

        for i in range(len(env.cases)):
            line = env.cases[i]
            for j in range(len(line)):
                if line[j] == 1:
                    screen.blit(dot, (15 + 21.3 * j, 10 + 19.3 * i))
                elif line[j] == 2:
                    screen.blit(energizer, (27 + 21.3 * j, 20 + 19.3 * i))

        for i in range(4):
            if not env.ghosts[i]['dead']:
                if env.take_energizer:
                    screen.blit(blue_ghost, (25 + 21.3 * env.ghosts[i]['x'], 20 + 19.3 * env.ghosts[i]['y']))
                else:
                    screen.blit(ghost_colors[i], (25 + 21.3 * env.ghosts[i]['x'], 20 + 19.3 * env.ghosts[i]['y']))

        screen.blit(pacman, (25 + 21.3 * env.pacman_position['x'], 20 + 19.3 * env.pacman_position['y']))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN and not env.game_over:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3

                if env.move_time.can_execute() and action >= 0:
                    env.act_with_action_id(action)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and env.game_over:  # restart game
                env.reset()

        if algo_playing and not env.game_over:
            if env.move_time.can_execute():
                chosen_action = get_best_pac_man_play(env.available_actions_ids(), q, env.state_description())
                env.act_with_action_id(chosen_action)

        pygame.display.update()


class PacMan(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self):
        pygame.init()
        self.cases = initiate_map()
        self.game_over = False
        self.current_score = 0.0
        self.round_counter = 0
        self.move_time = TimeCapsule(0.3)
        self.pacman_position = {'x': 13, 'y': 22}
        self.ghost_spawn_position = [
            [13, 10],
            [8, 14],
            [17, 14],
            [17, 10]
        ]
        self.ghosts = [
            {'x': self.ghost_spawn_position[i][0], 'y': self.ghost_spawn_position[i][1], 'dead': False, 'time_to_respawn': TimeCapsule(3)} for i in range(4)
        ]
        self.take_energizer = False
        self.energizer_time = TimeCapsule(5.0)
        self.reset()

    def state_description(self) -> np.ndarray:
        complete_cases = self.get_complete_cases()
        return np.hstack(complete_cases)

    def get_complete_cases(self):
        cases_copy = np.array(self.cases)
        for i in range(4):
            cases_copy[self.ghosts[i]['y'], self.ghosts[i]['x']] = i + 4
        cases_copy[self.pacman_position['y']][self.pacman_position['x']] = 3
        return cases_copy

    def state_description_length(self) -> int:
        return 29 * 26

    def max_actions_count(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (3 >= action_id >= 0)
        assert (not self.game_over)

        if self.take_energizer and self.energizer_time.can_execute():
            self.take_energizer = False

        x = self.pacman_position['x']
        y = self.pacman_position['y']

        if action_id == 0:
            x -= 1
        elif action_id == 1:
            x += 1
        elif action_id == 2:
            y -= 1
        elif action_id == 3:
            y += 1

        self.move_ghosts()
        if 0 <= x < 26 and 0 <= y < 29:
            if self.cases[y][x] != -1:
                self.pacman_position['x'] = x
                self.pacman_position['y'] = y

                case_value = self.cases[y][x]
                if case_value == 1:
                    self.current_score += 100
                    self.cases[y][x] = 0
                elif case_value == 2:
                    self.cases[y][x] = 0
                    self.take_energizer = True
                    self.energizer_time.restart()

        if self.check_game_ended():
            self.current_score += 10000
            self.game_over = True

        #self.current_score -= 5
        self.round_counter += 1

    def check_game_ended(self):
        dot_counter = 0
        for i in range(len(self.cases)):
            for j in range(len(self.cases[i])):
                if self.cases[i][j] == 1:
                    dot_counter += 1

        if self.round_counter % 10 == 0:
            print(f"Dots left : {dot_counter}")
            print(f"Score : {self.current_score}")
        return dot_counter == 0

    def move_ghosts(self):

        if self.game_over:
            return

        for i in range(4):

            if self.ghosts[i]['dead']:
                if self.ghosts[i]['time_to_respawn'].can_execute():
                    self.ghosts[i]['dead'] = False
                    self.ghosts[i]['x'] = 13
                    self.ghosts[i]['y'] = 10
                continue

            ghost_available_action = self.get_available_actions(i)
            random_action = ghost_available_action[np.random.randint(len(ghost_available_action))]

            new_x = self.ghosts[i]['x']
            new_y = self.ghosts[i]['y']

            if random_action == 0:
                new_x -= 1
            elif random_action == 1:
                new_x += 1
            elif random_action == 2:
                new_y -= 1
            elif random_action == 3:
                new_y += 1

            if 0 <= new_x < 26 and 0 <= new_y < 29:
                if self.pacman_position['x'] == new_x and self.pacman_position['y'] == new_y:
                    if self.take_energizer:
                        self.current_score += 10000
                        self.ghosts[i] = {'x': 13, 'y': 4, 'dead': True, 'time_to_respawn': TimeCapsule(3)}
                    else:
                        self.current_score -= 0 # 100000
                        self.game_over = True
                        return
                elif self.cases[new_y][new_x] != -1:
                    can_swap_places = True
                    for j in range(4):
                        if i != j:
                            can_swap_places = self.ghosts[j]['x'] != new_x or self.ghosts[j]['y'] != new_y
                            if not can_swap_places:
                                continue

                    if can_swap_places:
                        self.ghosts[i]['x'] = new_x
                        self.ghosts[i]['y'] = new_y

    def score(self) -> float:
        return self.current_score

    def get_available_actions(self, ghost_id):
        if self.game_over:
            return np.array([], dtype=np.int)

        available_actions = []
        if ghost_id is not None:
            pos_x, pos_y = self.ghosts[ghost_id]['x'], self.ghosts[ghost_id]['y']
        else:
            pos_x, pos_y = self.pacman_position['x'], self.pacman_position['y']

        if 0 <= (pos_x - 1) < 26 and self.cases[pos_y][pos_x - 1] != -1:
            available_actions.append(0)
        if 0 <= (pos_x + 1) < 26 and self.cases[pos_y][pos_x + 1] != -1:
            available_actions.append(1)
        if 0 <= pos_y - 1 < 29 and self.cases[pos_y - 1][pos_x] != -1:
            available_actions.append(2)
        if 0 <= pos_y + 1 < 29 and self.cases[pos_y + 1][pos_x] != -1:
            available_actions.append(3)

        return np.array(available_actions, dtype=np.int)

    def available_actions_ids(self) -> np.ndarray:
        return self.get_available_actions(None)

    def reset(self):
        pygame.init()
        self.cases = initiate_map()
        self.game_state = 0
        self.game_over = False
        self.current_score = 0.0
        self.move_time = TimeCapsule(0.3)
        self.pacman_position = {'x': 13, 'y': 22}
        self.ghost_spawn_position = [
            [13, 10],
            [8, 14],
            [17, 14],
            [17, 10]
        ]
        self.ghosts = [
            {'x': self.ghost_spawn_position[i][0], 'y': self.ghost_spawn_position[i][1], 'dead': False,
             'time_to_respawn': TimeCapsule(3)} for i in range(4)
        ]
        self.take_energizer = False
