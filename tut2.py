#!/usr/bin/env python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import matplotlib.pyplot as plt
import glob
import os
import sys
import logging
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
except IndexError:
    pass
import traceback
import carla
import random
import time
import pygame
import numpy as np
import argparse
import cv2
plt.ion()
actor_list = []


try:
    client = carla.Client("localhost",2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    pygame.init()



except:
    print("Server not started!!")



class Car(object):
    
    def camera_callback(self,image):
        image_np = np.frombuffer(image.raw_data,dtype=np.dtype('uint8'))
        image_np = np.reshape(image_np,(image.height,image.width,4))
        image_rgb = image_np[:,:,:3]
        # image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
        
        self.frame_buff = image_rgb

    def __init__(self):
        bp = blueprint_library.filter("model3")[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        self.vehicle = world.spawn_actor(bp, spawn_point)
        actor_list.append(self.vehicle)
        self.vehicle.set_autopilot(False)


        camera_bp = blueprint_library.find('sensor.camera.rgb')
        self.camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=0.8,z=1.7)), attach_to=self.vehicle)
        actor_list.append(self.camera)

        self.camera.listen(self.camera_callback)

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')

        self.frame_buff = np.zeros((600,800,3))

if __name__ == '__main__':

   
    try:
        car = Car()
        

        for i in range(10000000000000):

            

            cv2.imshow('frame_bonnet',car.frame_buff)


            cv2.waitKey(24)





        
    except:
        traceback.print_exc()
        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()
        print("Cleaned Up!")

    finally:

        cv2.destroyAllWindows()
        for actor in actor_list:
            actor.destroy()
        print("Cleaned Up!")

