'''
Date: January 21, 2024
Description: Script to load a scene from a json file
'''

import json
import random

import carla

'''
NOTE: IN JSON FILE, RELATIVE X-Y OF TRAFFIC AGENTS ARE INTERCHANGED.
'''

class LoadScenario:
	
	def __init__(self, world, client):
		self.ego_actor = None
		self.world = world
		self.client = client

		self.SpawnActor = carla.command.SpawnActor
		self.SetAutopilot = carla.command.SetAutopilot
		self.SetVehicleLightState = carla.command.SetVehicleLightState
		self.FutureActor = carla.command.FutureActor

		self.get_traffic_manager()

	def convert_local_to_global(self, trafficLocalTransforms, egoTransform):
		"""
		Function to convert traffic-vehicles transforms which are in ego-agent's
		local frame to CARLA world frame
		"""
		tWorldTransformList = []
		def transformPose(trafficTransform, egoTransform):
			"""
			Function to Convert Traffic Transform in Location frame to World Frame
			"""
			## Convert the Traffic Location to the world frame
			trafficLocation = carla.Location(x=trafficTransform.location.x, y=trafficTransform.location.y, z=2)
			egoTransform.transform(trafficLocation)

			## Convert the Traffic Orientation to the world frame
			tYaw = trafficTransform.rotation.yaw + egoTransform.rotation.yaw
			trafficOrientation = carla.Rotation(roll=0, pitch=0, yaw=tYaw)
			return trafficLocation, trafficOrientation
		
		## Convert Traffic Transforms to World Frame
		for trafficAgentTransform in trafficLocalTransforms:
			t_location, tOrientation = transformPose(trafficAgentTransform, egoTransform)
			tWorldTransform = carla.Transform(t_location, tOrientation)
			tWorldTransformList.append(tWorldTransform)
		return tWorldTransformList


	def destroy_existing_actors(self):
		for actor in self.world.get_actors():
			actor.destroy()


	def get_traffic_manager(self):
		self.traffic_manager = self.client.get_trafficmanager(8000)
		self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
		self.traffic_manager.set_hybrid_physics_mode(True)


	def create_scenario(self, json_file, actor_list):
		with open(json_file) as f:
			scenario_data = json.load(f)

		# Extract relevant scenario details from the JSON
		self.vehicles = scenario_data.get('vehicles', [])
		self.weather = scenario_data.get('weather', None)

		self.old_local_goal = scenario_data.get('old_local_goal', None)
		self.new_local_goal = scenario_data.get('new_local_goal', None)

		self.delta_d = scenario_data.get('delta_d', None)

		# Set the weather if specified
		if self.weather:
			weather_settings = carla.WeatherParameters(**self.weather)
			self.world.set_weather(weather_settings)

		tLocalTransformList = []
		for i, vehicle_data in enumerate(self.vehicles):
			tLocalTransform = carla.Transform(
				carla.Location(**vehicle_data['location']),
				carla.Rotation(**vehicle_data['rotation'])
			)
			tLocalTransformList.append(tLocalTransform)
		tWorldTransformList = self.convert_local_to_global(tLocalTransformList[1:], tLocalTransformList[0])
		tWorldTransformList.insert(0, tLocalTransformList[0])

		# Spawn vehicles
		batch = []
		for i, vehicle_data in enumerate(self.vehicles):
			blueprint = self.world.get_blueprint_library().find(vehicle_data['blueprint'])
			spawn_point = carla.Transform(
				carla.Location(**vehicle_data['location']),
				carla.Rotation(**vehicle_data['rotation'])
			)
			map_wp = self.world.get_map().get_waypoint(spawn_point.location)
			spawn_point = map_wp.transform
			spawn_point.location.z = 2

			blueprint.set_attribute('color', vehicle_data['color'])
			if i != 0:
				continue
			vehicle = self.world.spawn_actor(blueprint, spawn_point)
			actor_list.append(vehicle)
			if i == 0:
				self.ego_agent = vehicle

		print("Scenario loaded successfully.")