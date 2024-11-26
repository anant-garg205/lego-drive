"""
	Date: January 24, 2024
	Description: Script to process the collect CARLA data
"""

import pickle
import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation
import sys
import os
from navbev.sim.data_collection.extract_road import RoadExtraction
from navbev.sim.data_collection.transform_data import DataTransform

sys.path.append("/home/rrc/Videos/navigate-bev/navbev/src/simulation")

class VehicleClass:

	def __init__(self, vehicle=None):
		self.location = None
		self.rotation = None
		self.bounding_box = None
		self.velocity = None
		self.gt_trajectory = None

		if vehicle:
			self.location = np.array(vehicle.location)
			self.rotation = np.array(vehicle.rotation)
			self.bounding_box = np.array(vehicle.bounding_box)
			self.velocity = np.array(vehicle.velocity)


class DataProcess():
	"""
	Class for data processing. Originally, the data collected in CARLA
	is in world coordinates of the CARLA. Every data-point is converted to the local-frame
	of the Ego-Agent
	"""
	def __init__(self, base_data_folder=None):

		self.base_data_folder = base_data_folder		
		self.data_transform = DataTransform(base_data_folder)
		self.road_extract = RoadExtraction()
		self.ego_center_lane = None


	def get_ego_center_lane(self, c_location):
		def calc_l2_dist(wp1, wp2):
			dist = np.hypot(wp1[0]-wp2[0], wp1[1]-wp2[1])
			return dist

		c_location = c_location[:2]
		waypoint_list = np.load('/home/anant.garg/Videos/develop_nav_bev/navigate-bev/misc_files/Town10Path.npy')
		dist_list = []
		for waypoint in waypoint_list:
			dist = calc_l2_dist(c_location, waypoint)
			dist_list.append(dist)
		n_index = np.argmin(dist_list)
		ego_lane_length = 50

		ego_center_lane = []
		ego_center_lane.append([waypoint_list[n_index][0], waypoint_list[n_index][1]])
		while (calc_l2_dist(c_location, waypoint_list[n_index]) < 50):
			n_index += 1
			n_index = n_index % len(waypoint_list)
			ego_center_lane.append([waypoint_list[n_index][0], waypoint_list[n_index][1]])
		return np.array(ego_center_lane)


	
	def check_point_validity(self, point):
		"""
		Function to check if point lies in the BEV
		"""
		pixel_coordinate = [int(point[0]*2+100), int(point[1]*2+100)]
		if ((pixel_coordinate[0] > 0 and pixel_coordinate[0] < 200) and
					(pixel_coordinate[1] > 0 and pixel_coordinate[1] < 200)):
			return True
		return False


	def transform_traffic_agents(self):
		"""
			Function to transform the traffic agents to Ego-Agent's
			local frame
		"""
		self.traffic_vehicles = []
		for t_vehicle in self.data_transform.traffic_vehicles:
			t_vehicle = self.data_transform.tf_tv_2_local(t_vehicle)
			self.traffic_vehicles.append(t_vehicle)

	def get_GT_trajectory(self, index):
		"""
		Function to get GT Trajectory of the Ego-Vehicle.
		Considers 2-previous frame, 1-current frame and 6-future frames.
		"""
		self.GT_trajectory = []
		def read_pickle_file(index):
			pickleFilePath = os.path.join(self.base_data_folder, str(index) + ".pkl") 
			with open(pickleFilePath, 'rb') as pkl_file:
				data = pickle.load(pkl_file)
			return data
		for i in range(index-2, index+7):
			data = read_pickle_file(i)
			gtLocation  = data["EGO_AGENT"].location
			gtLocation = self.data_transform.tf_world_2_local(gtLocation)
			self.GT_trajectory.append(gtLocation)
	
	def fillAndSaveData(self, savePath, index):
		"""
		Function to save data the processed data
		"""
		index = f'{(index-2):04d}'
		pkl_file = os.path.join(savePath, index + ".pkl")
		traffic_vehicles = [VehicleClass(vehicle) for vehicle in self.traffic_vehicles]
		ego_agent = VehicleClass(self.data_transform.loaded_data["EGO_AGENT"])
		ego_agent.gt_trajectory = np.array(self.GT_trajectory)[:,0,:2]

		self.loaded_data = {}
		self.loaded_data["EGO_AGENT"] = ego_agent
		self.loaded_data["TRAFFIC_AGENTS"] = traffic_vehicles
		self.loaded_data["EGO_CENTER_LANE"] = np.array(self.ego_center_lane[:,:2])
		self.loaded_data["BEV"] = cv2.flip(self.data_transform.bev_image, 0)
		self.loaded_data["BEV_MAP"] = cv2.flip(self.data_transform.loaded_data["BEV_MAP"], 0)
		self.loaded_data["BEV_MAP_BINARY_EGO_LANE"] = self.road_extract.road
		self.loaded_data["CAM_FRONT"] = self.data_transform.loaded_data["CAM_FRONT"]
		self.loaded_data["CAM_LEFT"] = self.data_transform.loaded_data["CAM_LEFT"]
		self.loaded_data["CAM_RIGHT"] = self.data_transform.loaded_data["CAM_RIGHT"]
		self.loaded_data["CAM_BACK"] = self.data_transform.loaded_data["CAM_BACK"]

		with open(pkl_file, 'wb') as f:
			pickle.dump(self.loaded_data, f)

	def check_point_on_road(self, image, point):
		if self.check_point_validity(point):
			pixel_coordinate = [int(point[0]*2+100), int(point[1]*2+100)]
			if image[pixel_coordinate[0], pixel_coordinate[1]] > 0:
				return True
		return False
	
	def check_all_points_on_road(self, image):
		mask = []
		for t_vehicle in self.traffic_vehicles:
			mask.append(self.check_point_on_road(image, t_vehicle.location))
		self.traffic_vehicles = np.array([elem for elem, mask_value in zip(self.traffic_vehicles, mask) if mask_value])

		n_indices = self.get_N_closest_agent_indices(self.traffic_vehicles, self.data_transform.ego_vehicle_location, N=5)
		self.traffic_vehicles = self.traffic_vehicles[n_indices[:5]]

	def get_N_closest_agent_indices(self, traffic_vehicles, egoLocation, N):
			dist_list = []
			for t_vehicle in traffic_vehicles:
				dist = np.hypot(t_vehicle.location[0] - egoLocation[0], 
								t_vehicle.location[1] - egoLocation[1])
				dist_list.append(dist)
			n_indices = np.argsort(dist_list)
			n_indices = n_indices[:N]
			return np.array(n_indices)

	
	def plot_check(self):
		"""
		Function to verify the conversions of the data by plotting it
		"""
		self.bev_image = self.data_transform.bev_image.copy()
		self.front_image = self.data_transform.front_image.copy()
		bev_image = cv2.flip(self.bev_image, 0)

		for waypoint in self.ego_center_lane:
			cv2.circle(bev_image, (int(waypoint[1]*2+100), int(waypoint[0]*2+100)), 1, (0, 0, 255), -1)

		for gtPoint in self.GT_trajectory:
			cv2.circle(bev_image, (int(gtPoint[0][1]*2+100), int(gtPoint[0][0]*2+100)), 2, (120, 120, 255), -1)

		for t_vehicle in self.traffic_vehicles:
			cv2.circle(bev_image, (int(t_vehicle.location[1]*2+100), int(t_vehicle.location[0]*2+100)), 2, (0, 0, 255), -1)

			vStart = (int(t_vehicle.location[1]*2+100), int(t_vehicle.location[0]*2+100))
			vEnd = (int(t_vehicle.location[1]*2+100 + t_vehicle.velocity[1]*2), int(t_vehicle.location[0]*2+100 + t_vehicle.velocity[0]*2))

			cv2.arrowedLine(bev_image, vStart, vEnd, (255, 0, 0), 2)

		bev_image = cv2.resize(bev_image, (500, 500))
		cv2.imshow("BEV", bev_image)
		cv2.imshow("Front Cam", self.front_image)
		cv2.waitKey(1)

def main():

	base_data_folder = "/media/anant.garg/Extreme SSD/navigate_bev/carla_data/town_10_new_data"
	Town = "Town10HD"
	base_data_folder = os.path.join(base_data_folder, Town)
	data = DataProcess(base_data_folder)
	save_path = os.path.join(base_data_folder + "_processed")

	for i in range(3, 2003):
		print(f"Data Index: {i}")
		index = i
		data.data_transform.get_transformation_mat(index=index)
		data.transform_traffic_agents()
		
		ego_center_lane = data.get_ego_center_lane(data.data_transform.loaded_data["EGO_AGENT"].location)
		data.ego_center_lane = data.data_transform.tf_world_2_local(ego_center_lane)
		
		data.road_extract.get_center_and_boundary_lane(data.ego_center_lane)
		data.road_extract.extract_road()
		data.check_all_points_on_road(data.road_extract.road)
		data.get_GT_trajectory(index)
		data.fillAndSaveData(save_path, index)
		data.plot_check()	

if __name__ == "__main__":
	main()