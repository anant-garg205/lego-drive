import glob
import os
import sys
import random
import queue
import numpy as np
import cv2
import pygame
import pickle
import time

import carla

from navbev.sim.data_collection.agents.navigation.basic_agent import BasicAgent
from navbev.sim.configs.configs import data_collect_configs as config
from navbev.config import Configs as cfg
from navbev.sim.utils.carla_sync_mode import CarlaSyncMode
from navbev.sim.utils.sensors_nav_bev import add_sensors
from navbev.sim.sensors.bev_local import BEV_Local
from navbev.sim.sensors.bev_local import VehicleClass
from navbev.sim.utils.visualization import Visualize

def should_quit():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			return True
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_ESCAPE:
				return True
	return False

def get_ego_center_lane(ego_agent, route_trace):

	res = cfg.carla().data_collection.ego_center_lane_res
	ego_center_lane = []
	dist_list = []	
	ego_agent_transform = ego_agent.get_transform()
	for i, waypoint in enumerate(route_trace):
		dist = np.hypot(ego_agent_transform.location.x - waypoint[0].transform.location.x, 
				  		ego_agent_transform.location.y - waypoint[0].transform.location.y)
		dist_list.append(dist)
	dist_list = np.array(dist_list)

	e_index = np.argmax(dist_list > (cfg.carla().data_collection.dist_to_waypoint_threshold-40))
	s_index = np.argmin(dist_list)

	p_distance = 50
	n = int(p_distance // res)
	for i in range(1, n):
		p_waypoint = route_trace[s_index][0].previous((n-i)*res)
		ego_center_lane.append([p_waypoint[0].transform.location.x, p_waypoint[0].transform.location.y])

	for i in range(s_index, e_index):
		n = int(2 // res)
		for j in range(1,n):
			n_waypoint = route_trace[i][0].next(j*res)
			ego_center_lane.append([n_waypoint[0].transform.location.x, n_waypoint[0].transform.location.y])
	
	return ego_center_lane

def make_directories():

	save_base_folder = cfg.carla().data_collection.save_base_folder
	town = cfg.carla().data_collection.town
	folders = ["carla_map", town, town+"_processed"]
	for folder in folders:
		path = os.path.join(save_base_folder, folder)
		os.makedirs(path, exist_ok=True)

def main():

	actor_list = []
	pygame.init()

	clock = pygame.time.Clock()

	client = carla.Client('localhost', 2000)
	client.set_timeout(10.0)
	# world = client.load_world(cfg.carla().data_collection.town)
	world = client.load_world('Town02')

	visualize = Visualize(world)

	try:
		# make_directories()
		map = world.get_map()

		spawn_points = map.get_spawn_points()
		spawn_point = random.choice(spawn_points)

		blueprint_library = world.get_blueprint_library()
		
		vehicle_bp = blueprint_library.filter('vehicle.tesla.*')[1]
		vehicle_bp.set_attribute('role_name', 'ego')

		ego_vehicle = world.spawn_actor(
			vehicle_bp,
			spawn_point
		)

		actor_list.append(ego_vehicle)
		ego_vehicle.set_simulate_physics(True)

		sensor_dict = add_sensors(ego_vehicle, actor_list)
		agent = BasicAgent(ego_vehicle)

		bev_display = pygame.display.set_mode(
			(400, 400),
			pygame.HWSURFACE | pygame.DOUBLEBUF)
		bev_dims = [400, 400]
		bev_Local = BEV_Local(bev_dims, world, ego_vehicle)

		c_waypoint = map.get_waypoint(spawn_point.location)
		# n_waypoint = c_waypoint.next(config["next_waypoint_distance"])[0]
		n_waypoint = c_waypoint.next(cfg.carla().data_collection.next_waypoint_distance)[0]

		n_waypoint = random.choice(spawn_points)

		agent.set_destination(
			[n_waypoint.location.x,
			n_waypoint.location.y,
			n_waypoint.location.z]
		)

		p_location = np.array([0, 0, 0])

		p_time = 0

		waypoint_list = []

		# np.save(os.path.join(config["save_base_folder"], "carla_map", "carla_waypoints" + config["Town"] + ".npy"), bev_Local.map_image.carla_map_waypoints)
		# np.save(os.path.join(config["save_base_folder"], "carla_map", "carla_boundary_points" + config["Town"] + ".npy"), bev_Local.map_image.boundary_points)

		Path = []
		folder_count = 5

		with CarlaSyncMode(
			world, 
			sensor_dict["CAM_FRONT"].sensor, 
			sensor_dict["CAM_LEFT"].sensor, 
			sensor_dict["CAM_RIGHT"].sensor,
			sensor_dict["CAM_BACK"].sensor,
			sensor_dict["CAM_SEG"].sensor,
			fps=20
		) as sync_mode:
			
			step_count = 10000

			while True:
				if should_quit(): return
				clock.tick()
				snapshot, sensor_dict["CAM_FRONT"].data, \
				sensor_dict["CAM_LEFT"].data, sensor_dict["CAM_RIGHT"].data, \
				sensor_dict["CAM_BACK"].data, sensor_dict["CAM_SEG"].data = sync_mode.tick(timeout=2.0)

				sensor_dict["CAM_FRONT"].callback(sensor_dict["CAM_FRONT"].data)
				sensor_dict["CAM_LEFT"].callback(sensor_dict["CAM_LEFT"].data)
				sensor_dict["CAM_RIGHT"].callback(sensor_dict["CAM_RIGHT"].data)
				sensor_dict["CAM_BACK"].callback(sensor_dict["CAM_BACK"].data)
				sensor_dict["CAM_SEG"].callback(sensor_dict["CAM_SEG"].data)

				front_image = np.array(sensor_dict["CAM_FRONT"].image)
				left_image = np.array(sensor_dict["CAM_LEFT"].image)
				right_image = np.array(sensor_dict["CAM_RIGHT"].image)
				back_image = np.array(sensor_dict["CAM_BACK"].image)
				seg_image = np.array(sensor_dict["CAM_SEG"].image)

				bev_Local.render()

				

				cv2.imshow("Front Camera", front_image)
				cv2.imshow("Seg Camera", seg_image)
				# cv2.imshow("BEV Local", bev_Local.bev_small)
				# cv2.imshow("BEV_MAP", bev_Local.map_small)
				cv2.waitKey(1)

				#################################################
				cLocation = ego_vehicle.get_transform().location
				vec2nWayPoint = [
					cLocation.x - n_waypoint.location.x,
					cLocation.y - n_waypoint.location.y,
					cLocation.z - n_waypoint.location.z
				]
				dist = np.linalg.norm(vec2nWayPoint)
				if (dist < cfg.carla().data_collection.dist_to_waypoint_threshold):
					c_waypoint = map.get_waypoint(cLocation)
					# n_waypoint = c_waypoint.next(cfg.carla().data_collection.next_waypoint_distance)[-1]


					##################################
					n_waypoint = random.choice(spawn_points)
					##################################


					agent.set_destination([
						n_waypoint.location.x,
						n_waypoint.location.y,
						n_waypoint.location.z
					])
				#################################################
				

				#################################################
				control = agent.run_step()
				control.manual_gear_shift = False
				ego_vehicle.apply_control(control)
				#################################################

				## Make All traffic Lights GREEN
				list_actors = world.get_actors()
				for actor in list_actors:
					if isinstance(actor, carla.TrafficLight):
						actor.set_state(carla.TrafficLightState.Green)

				ego_vehicle_obj = VehicleClass()
				ego_vehicle_obj.location = [
					ego_vehicle.get_location().x, 
					ego_vehicle.get_location().y, 
					ego_vehicle.get_location().z
				]	
				ego_vehicle_obj.rotation = [
					ego_vehicle.get_transform().rotation.roll, 
					ego_vehicle.get_transform().rotation.pitch, 
					ego_vehicle.get_transform().rotation.yaw
				]
				ego_vehicle_obj.velocity = [
					ego_vehicle.get_velocity().x,
					ego_vehicle.get_velocity().y,
					ego_vehicle.get_velocity().z
				]
				#################################################
				cTime = time.time()
				dist = np.hypot(cLocation.x - p_location[0], cLocation.y - p_location[1])
				if (cTime - p_time) > 0.2:
				
				# if dist > 0.5:
					
					print("Step Count: ", step_count)
					
					waypoint_list.append([cLocation.x, cLocation.y])
					p_time = cTime

					data = {}
					data["EGO_AGENT"] = ego_vehicle_obj
					data["CAM_FRONT"] = front_image
					# data["CAM_LEFT"] = left_image
					# data["CAM_RIGHT"] = right_image
					# data["CAM_BACK"] = back_image
					data["TRAFFIC_AGENTS"] = bev_Local.traffic_vehicles
					data["BEV"] = bev_Local.bev_small
					data["BEV_MAP"] = bev_Local.map_small
					data["EGO_CENTER_LANE"] = get_ego_center_lane(ego_vehicle, agent.route_trace) 

					#####################

					# for ego_lane_point in data["EGO_CENTER_LANE"]:
					# 	visualize.draw_path(loc1=ego_lane_point, loc2=None, type="string2")

					#####################
					# os.makedirs(f'{cfg.carla().data_collection.save_base_folder}/{str(folder_count)}', exist_ok=True)
					os.makedirs(f'{cfg.carla().data_collection.save_base_folder}/rgb/', exist_ok=True)
					os.makedirs(f'{cfg.carla().data_collection.save_base_folder}/seg/', exist_ok=True)
					if len(agent.route_trace) > 10:
						# pkl_file = os.path.join(config["save_base_folder"], config["Town"], str(step_count) + ".pkl")
						# pkl_file = os.path.join(cfg.carla().data_collection.save_base_folder, cfg.carla().data_collection.town, str(step_count) + ".pkl")
						# pkl_file = f'{cfg.carla().data_collection.save_base_folder}/{str(folder_count)}/{str(step_count)}.pkl'
						# with open(pkl_file, 'wb') as f:
						# 	pickle.dump(data, f)

						# Path.append([cLocation.x, cLocation.y, ego_vehicle.get_transform().rotation.yaw])
						np.save(f'{config["save_base_folder"]}.npy', Path)
						cv2.imwrite(f'{config["save_base_folder"]}/rgb/{str(step_count)}.jpg', front_image)
						cv2.imwrite(f'{config["save_base_folder"]}/seg/{str(step_count)}.jpg', seg_image)

						step_count += 1
					
					p_location = [cLocation.x, cLocation.y]
				#################################################
						
				world.set_weather(carla.WeatherParameters.CloudySunset)

				if (step_count > 20000):
					break
				world.tick()
	finally:
		print("Destroying Actors.")
		for actor in actor_list:
			try:
				actor.sensor.destroy()
			except:
				actor.destroy()
		pygame.quit()
		print("Done.")

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nCancelled by User. Bye!")

