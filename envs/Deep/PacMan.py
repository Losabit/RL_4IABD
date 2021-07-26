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
    all_q_inputs = np.zeros((len(available_actions), 9 + 9))
    for i, a in enumerate(available_actions):
        all_q_inputs[i] = np.hstack([cases, tf.keras.utils.to_categorical(a, 9)])

    all_q_values = np.squeeze(q.predict(all_q_inputs))
    chosen_action = available_actions[np.argmax(all_q_values)]
    return chosen_action

def pac_man_env(pi, q):
    env = PacMan()
    X = 600
    Y = 600
    pygame.init()
    screen = pygame.display.set_mode((X, Y))
    pygame.display.set_caption('Pacman')
    background = pygame.image.load('assets/pacman/background.png')
    background = pygame.transform.scale(background, (600, 600))
    dot = pygame.image.load('assets/pacman/dot.png')
    dot = pygame.transform.scale(dot, (40, 40))
    pacman = pygame.image.load('assets/pacman/pacman_left.png')
    inky = pygame.image.load('assets/pacman/ghosts/inky.png')
    clyde = pygame.image.load('assets/pacman/ghosts/clyde.png')
    pinky = pygame.image.load('assets/pacman/ghosts/pinky.png')
    blinky = pygame.image.load('assets/pacman/ghosts/blinky.png')
    blue_ghost = pygame.image.load('assets/pacman/ghosts/blue_ghost.png')
    energizer = pygame.image.load('assets/pacman/energizer.png')

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
                elif line[j] == 3:
                    screen.blit(pacman, (25 + 21.3 * j, 20 + 19.3 * i))
                elif line[j] >= 4:
                    if env.take_energizer:
                        screen.blit(blue_ghost, (25 + 21.3 * j, 20 + 19.3 * i))
                    else:
                        if line[j] == 4:
                            screen.blit(blinky, (25 + 21.3 * j, 20 + 19.3 * i))
                        elif line[j] == 5:
                            screen.blit(clyde, (25 + 21.3 * j, 20 + 19.3 * i))
                        elif line[j] == 6:
                            screen.blit(inky, (25 + 21.3 * j, 20 + 19.3 * i))
                        elif line[j] == 7:
                            screen.blit(pinky, (25 + 21.3 * j, 20 + 19.3 * i))

        if env.take_energizer and env.energizer_time.can_execute():
            env.take_energizer = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if algo_playing:

                if env.move_time.can_execute():
                    chosen_action = get_best_pac_man_play(env.available_actions_ids(), q, env.cases)
                    env.act_with_action_id(chosen_action)

            elif event.type == pygame.KEYDOWN:
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

        pygame.display.update()


class PacMan(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self):
        self.cases = initiate_map()
        self.game_state = 0
        self.game_over = False
        self.current_score = 0.0
        self.move_time = TimeCapsule(0.3)
        self.pacman_position = {'x': 13, 'y': 22}
        self.ghosts = [
            {'x': 13, 'y': 10, 'dead': False, 'time_to_respawn': TimeCapsule(3)} for i in range(4)
        ]
        self.take_energizer = False
        self.energizer_time = TimeCapsule(5.0)
        self.reset()

    def state_id(self) -> int:
        sum = 0
        available_actions_size = 2
        for i in range(len(self.cases)):
            for j in range(len(self.cases[i])):
                case = self.cases[i][j]
                if case == 1:  # dot
                    sum += pow(available_actions_size, i)
                elif case == 2:  # mega dot
                    sum += pow(available_actions_size, len(self.cases) + i)
                elif case == 3:  # player position ?
                    sum += pow(available_actions_size, len(self.cases) * 2 + i)
                elif case >= 4:  # enemy position ?
                    sum += pow(available_actions_size, len(self.cases) * 4 + i)
        return sum

    def state_description(self) -> int:
        return np.hstack(self.cases)

    def state_description_length(self) -> int:
        return 29*26

    def max_actions_count(self) -> int:
        return 4

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (3 >= action_id >= 0)
        assert (not self.game_over)

        x = self.pacman_position['x']
        y = self.pacman_position['y']
        prev_x, prev_y = x, y

        if action_id == 0:
            x -= 1
        elif action_id == 1:
            x += 1
        elif action_id == 2:
            y -= 1
        elif action_id == 3:
            y += 1

        if 0 <= x < 26 and 0 <= y < 29:
            if self.cases[y][x] != -1:
                self.pacman_position['x'] = x
                self.pacman_position['y'] = y

                # Update cases with player position
                self.cases[prev_y][prev_x] = 0
                self.cases[y][x] = 3

        self.move_ghosts()

        case_value = self.cases[self.pacman_position['y']][self.pacman_position['x']]
        if case_value == 1:
            self.cases[self.pacman_position['y']][self.pacman_position['x']] = 0
            self.current_score += 100
        elif case_value == 2:
            self.cases[self.pacman_position['y']][self.pacman_position['x']] = 0
            self.take_energizer = True
            self.energizer_time.restart()

        self.game_state = self.state_id()

    def move_ghosts(self):
        for i in range(4):

            if self.ghosts[i]['dead']:
                if self.ghosts[i]['time_to_respawn'].can_execute():
                    self.ghosts[i]['dead'] = False
                    self.ghosts[i]['x'] = 13
                    self.ghosts[i]['y'] = 10
                    self.cases[10, 13] = i
                continue

            random_action = np.random.randint(1, 4)
            new_x = self.ghosts[i]['x']
            new_y = self.ghosts[i]['y']
            prev_x, prev_y = new_x, new_y
            if random_action == 1:
                new_x -= 1
            elif random_action == 2:
                new_x += 1
            elif random_action == 3:
                new_y -= 1
            elif random_action == 4:
                new_y += 1

            if 0 <= new_x < 26 and 0 <= new_y < 29:
                if self.cases[new_y][new_x] != -1:
                    self.ghosts[i]['x'] = new_x
                    self.ghosts[i]['y'] = new_y

                    #Update case value with ghosts
                    self.cases[new_y][new_x] = i
                    self.cases[prev_y][prev_x] = 0

                elif self.cases[new_y][new_x] == 3:
                    if self.take_energizer:
                        self.current_score += 1000
                        prev_x, prev_y = self.ghosts[i]['x'], self.ghosts[i]['y']
                        self.ghosts[i] = {'x': 13, 'y': 4, 'dead': True, 'time_to_respawn': TimeCapsule(3)}
                        self.cases[13][4] = i
                        self.cases[prev_x, prev_y] = 3
                    else:
                        self.current_score -= 10000
                        self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        return np.array([0, 1, 2, 3], dtype=np.int)

    def reset(self):
        self.cases = initiate_map()
        self.game_state = 0
        self.game_over = False
        self.current_score = 0.0
        self.move_time = TimeCapsule(0.3)
        self.pacman_position = {'x': 13, 'y': 22}
        self.ghosts = [
            {'x': 13, 'y': 10, 'dead': False, 'time_to_respawn': TimeCapsule(3)} for i in range(4)
        ]
        self.take_energizer = False
