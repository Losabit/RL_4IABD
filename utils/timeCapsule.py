import pygame
import time


class TimeCapsule:
    def __init__(self, time):
        self.time_to_end = time
        self.time_spend = time
        self.getTicksLastFrame = 0

    def can_execute(self):
        t = pygame.time.get_ticks()
        self.time_spend += (t - self.getTicksLastFrame) / 1000.0
        self.getTicksLastFrame = t
        if self.time_spend >= self.time_to_end:
            self.time_spend = 0
            return True
        return False

    def modify_time(self, time):
        self.time_to_end = time
        self.time_spend = 0

    def restart(self):
        self.time_spend = 0
        self.getTicksLastFrame = pygame.time.get_ticks()
