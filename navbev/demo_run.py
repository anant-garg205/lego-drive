'''
    Date: January 28, 2024
    Description: Script to run inference on the end-2-end pipeline
'''

import numpy as np
import cv2
import pygame
import os
import carla
import time
import matplotlib.pyplot as plt

import torch

from navbev.config import Configs as cfg
from navbev.perception.glc.model import GLCInference

from navbev.sim.inference.planner import Planner
from navbev.sim.sensors.bev_local import BEV_Local
from navbev.sim.utils.carla_sync_mode import CarlaSyncMode
from navbev.sim.utils.transforms import *
from navbev.sim.utils.get_planner_data import GetPlannerData
from navbev.sim.utils.load_scene_from_json import LoadScenario
from navbev.sim.utils.visualization import Visualize


def main():

    __SAVE_RESULTS__ = True
    __PLOT_ONCE__ = True
    __PLAN_ONCE__ = True
    __PREDICT_ONCE__ = True

    pygame.init()
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    glc_inf = GLCInference()

    ## Load Scenario from JSON file
    scenario_json_file = f''
    load_scenario = LoadScenario(world=None, client)
    actor_list = []
    load_scenario.create_scenario(scenario_json_file, actor_list)
    
    ## Add sensors to the ego-agent
    ego_agent = load_scenario.ego_agent
    get_planner_data = GetPlannerData(ego_agent)
    sensor_dict = add_sensors(ego_agent, actor_list)

    ## Initialize BEV for traffic actors information
    bev_dims = [cfg.sensors().image.bevDim*2, cfg.sensors().image.bevDim*2]
    bevLocal = BEV_Local(bev_dims, world, sr.ego_agent)

    clock = pygame.time.Clock()
    visualize = Visualize(world) 

    with CarlaSyncMode(world,
        sensor_dict['CAM_FRONT'].sensor,
        sensor_dict['CAM_DRIVE'].sensor,
        sensor_dict['CAM_BEV'].sensor,
        fps=20
    ) as sync_mode:

        while(True):
            if step_count < 10:
                world.tick()
                step_count += 1
                continue
            clock.tick_busy_loop(60)
            world.tick()
            snapshot, sensor_dict['CAM_FRONT'].data, \
            sensor_dict['CAM_DRIVE'].data, \
            sensor_dict['CAM_BEV'].data = sync_mode.tick(timeout=2.0)

            sensor_dict['CAM_FRONT'].callback(sensor_dict['CAM_FRONT'].data)
            sensor_dict['CAM_DRIVE'].callback(sensor_dict['CAM_DRIVE'].data)
            sensor_dict['CAM_BEV'].callback(sensor_dict['CAM_BEV'].data)

            front_image = np.array(sensor_dict['CAM_FRONT'].data)
            drive_image = np.array(sensor_dict['CAM_DRIVE'].data)
            bev_image = np.array(sensor_dict['CAM_BEV'].data)

            images = [front_image, None, None, None]
            
            ## Predict goal based Front-Camera Image and Language Command
            with torch.no_grad():
                if __PREDICT_ONCE__:
                    lang_command = input(f'Input the language command: ')
                    pred_goal_2d = glc.inf(front_image, lang_command)
                    __PREDICT_ONCE__ = False

                goal_world = convert_local_to_global(sr.ego_agent.get_transform(), pred_goal_2d)[0]
                get_planner_data.runstep(target=goal_world, traffic_vehicles=bevLocal.traffic_vehicles, images)

                if __PLAN_ONCE__:
                    planner.get_control(input_data=get_planner_data.planning_data, tState=goal_local_point[:2])
                    for waypoint in planner.wayPoionts:
                        visualize.draw_path(loc1=waypoint, draw_type='string', color=[255, 0, 0])

            combined_image = visualize.combined_visualization(drive_image, front_image, bev_image)
            cv2.imshow("Combined_Image", combined_image)
            cv2.waitKey(1)
            step_count += 1

            if dist_to_goal < (cfg.carla().pure_pursuit.look_ahead+2):
                visualize.draw_x(center=random_goal_local_points[1], size=1, color=[50, 0, 0])
                goal_pt_index += 1
                planner.plan_once = True
                print("GOAL REACHED")

            world.tick()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f'\nCancelled by the User. Thank you!')