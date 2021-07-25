import pygame
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


def demo():
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

    move_time = TimeCapsule(0.3)
    pacman_position = {'x': 13, 'y': 22}
    blinky_position = {'x': 13, 'y': 10}
    inky_position = {'x': 11, 'y': 14}
    clyde_position = {'x': 15, 'y': 14}
    pinky_position = {'x': 13, 'y': 14}

    cases = initiate_map()
    action = 0  # 1 = left / 2 = right / 3 = up / 4 = down
    current_action = 0
    score = 0
    take_energizer = False
    energizer_time = TimeCapsule(5.0)
    # x : 26 // decalage : 15 // difference : 21.3
    # y : 29 // decalage : 10 // difference : 19.3
    while True:
        screen.fill((255, 255, 255))
        screen.blit(background, (0, 0))

        for i in range(len(cases)):
            line = cases[i]
            for j in range(len(line)):
                if line[j] == 1:
                    screen.blit(dot, (15 + 21.3 * j, 10 + 19.3 * i))
                elif line[j] == 2:
                    screen.blit(energizer, (27 + 21.3 * j, 20 + 19.3 * i))

        screen.blit(pacman, (25 + 21.3 * pacman_position['x'], 20 + 19.3 * pacman_position['y']))

        if take_energizer:
            screen.blit(blue_ghost, (25 + 21.3 * blinky_position['x'], 20 + 19.3 * blinky_position['y']))
            screen.blit(blue_ghost, (25 + 21.3 * clyde_position['x'], 20 + 19.3 * clyde_position['y']))
            screen.blit(blue_ghost, (25 + 21.3 * inky_position['x'], 20 + 19.3 * inky_position['y']))
            screen.blit(blue_ghost, (25 + 21.3 * pinky_position['x'], 20 + 19.3 * pinky_position['y']))
            if energizer_time.can_execute():
                take_energizer = False
        else:
            screen.blit(blinky, (25 + 21.3 * blinky_position['x'], 20 + 19.3 * blinky_position['y']))
            screen.blit(clyde, (25 + 21.3 * clyde_position['x'], 20 + 19.3 * clyde_position['y']))
            screen.blit(inky, (25 + 21.3 * inky_position['x'], 20 + 19.3 * inky_position['y']))
            screen.blit(pinky, (25 + 21.3 * pinky_position['x'], 20 + 19.3 * pinky_position['y']))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_UP:
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 4

        if move_time.can_execute() and action != 0:
            x = pacman_position['x']
            y = pacman_position['y']
            if action == 1:
                x -= 1
            elif action == 2:
                x += 1
            elif action == 3:
                y -= 1
            elif action == 4:
                y += 1

            if 0 <= x < 26 and 0 <= y < 29:
                if cases[y][x] != -1:
                    pacman_position['x'] = x
                    pacman_position['y'] = y

        if cases[pacman_position['y']][pacman_position['x']] == 1:
            cases[pacman_position['y']][pacman_position['x']] = 0
            score += 100
        elif cases[pacman_position['y']][pacman_position['x']] == 2:
            cases[pacman_position['y']][pacman_position['x']] = 0
            take_energizer = True
            energizer_time.restart()

        pygame.display.update()
