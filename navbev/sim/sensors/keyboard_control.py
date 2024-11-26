'''
    Date: August 19, 2023
    Description: Script for manual control in CARLA
'''

import carla
import numpy
import pygame
import time
from pygame.locals import K_w
from pygame.locals import K_s
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_q
from pygame.locals import K_UP
from pygame.locals import K_DOWN
from pygame.locals import K_RIGHT
from pygame.locals import K_LEFT
from pygame.locals import K_SPACE

class KeyboardControl(object):    
    def __init__(self, player, clock):
        self.parent = player
        self.world = self.parent.get_world()
        self.control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.clock = clock    
        
    def parse_vehicle_keys(self, keys, milliseconds):        
        if keys[K_UP] or keys[K_w]:
            self.control.throttle = min(self.control.throttle + 0.1, 1.00)
        else:
            self.control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1)
        else:
            self.control.brake = 0       

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.control.steer = round(self._steer_cache, 1)
        self.control.hand_brake = keys[K_SPACE]        
        
        if keys[K_q]:
            self.control.gear = 1 if self.control.reverse else -1    
            
    def runstep(self):
        keys = pygame.key.get_pressed()
        milliseconds = self.clock.get_time()
        print("Milliseconds: ", milliseconds)
        self.parse_vehicle_keys(keys, milliseconds)