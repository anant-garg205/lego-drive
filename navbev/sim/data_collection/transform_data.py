"""
	Date: February 03, 2024
	Description: Script to Transform Carla data from Carla-frame
				Ego-Agent's local frame
"""

import numpy as np
from scipy.spatial.transform import Rotation
import pickle
import os

class DataTransform:
	"""
	Class to transform the carla data to Ego-Agent's
	local frame
	"""
	def __init__(self, base_data_folder=None, inference=False):
		self.base_data_folder = base_data_folder
		self.inference = inference

	def get_transformation_mat(self, ego_vehicle=None, index=None):
		"""
		Function to get the transformation matrix for Ego-Agent
		At Inference:
			Ego-Agent from CARLA Simulator is used
		At Offline data processing:
			Ego-Agent information is loaded from collected data
		"""
		self.ego_transform_matrix = np.zeros((3,3))
		## This is for Live Inference
		if ego_vehicle:
			self.ego_vehicle_rotation = [ego_vehicle.get_transform().rotation.roll,
										ego_vehicle.get_transform().rotation.pitch,
										ego_vehicle.get_transform().rotation.yaw]
			
			self.ego_vehicle_location = [ego_vehicle.get_location().x,
										ego_vehicle.get_location().y]

			self.ego_rot_mat = Rotation.from_euler('xyz', np.deg2rad(self.ego_vehicle_rotation)).as_matrix()

			self.ego_transform_matrix[:2, :2] = self.ego_rot_mat[:2, :2]
			self.ego_transform_matrix[:2, 2] = np.array(self.ego_vehicle_location).T
			self.ego_transform_matrix[2, 2] = 1
			self.ego_inverse_mat = np.linalg.inv(self.ego_transform_matrix)
			return

		## This is for offline data processing
		self.read_datapoint(index)
		self.ego_rot_mat = Rotation.from_euler('xyz', np.deg2rad(self.ego_vehicle_rotation)).as_matrix()
		self.ego_transform_matrix[:2, :2] = self.ego_rot_mat[:2, :2]
		self.ego_transform_matrix[:2, 2] = self.ego_vehicle_location.T
		self.ego_transform_matrix[2, 2] = 1
		self.ego_inverse_mat = np.linalg.inv(self.ego_transform_matrix)
		return
	
	def read_datapoint(self, index):
		"""
		Function to get data-point from pickle file
		"""
		def read_pickle_file(index):
			"""
			Function to read pickle file
			"""
			pickleFilePath = os.path.join(self.base_data_folder, str(index) + ".pkl") 
			with open(pickleFilePath, 'rb') as pkl_file:
				data = pickle.load(pkl_file)
			return data
		self.index = index
		self.loaded_data = read_pickle_file(index)

		self.ego_vehicle_location = np.array(self.loaded_data["EGO_AGENT"].location)[:2]
		self.ego_vehicle_rotation = np.array(self.loaded_data["EGO_AGENT"].rotation)
		self.bev_image = self.loaded_data["BEV"]
		self.front_image = self.loaded_data["CAM_FRONT"]
		self.left_image = self.loaded_data["CAM_LEFT"]
		self.right_image = self.loaded_data["CAM_RIGHT"]
		self.back_image = self.loaded_data["CAM_BACK"]
		self.traffic_vehicles = self.loaded_data["TRAFFIC_AGENTS"][1:]

	def tf_world_2_local(self, points):
		"""
		Function to convert points in Carla Frame to Ego-Agent's
		local frame
		"""
		points = np.array(points)
		if points.ndim < 2:
			points = np.array([points])

		points = points[:,:2].T
		points = np.vstack((points, np.ones((1, points.shape[1]))))
		points = self.ego_inverse_mat @ points
		return points.T * 0.9
	
	def tf_tv_2_local(self, t_vehicle):
		"""
		Function to convert traffic-vehicle to Ego-Agent's
		local frame
		"""
		t_vehicle.location = self.tf_world_2_local(t_vehicle.location)[0]
		t_vehicle.rotation[2] += self.ego_vehicle_rotation[2]
		t_vehicle.bounding_box = self.tf_world_2_local(np.array(t_vehicle.bounding_box)).T
		t_vehicle.velocity = self.ego_inverse_mat[:2, :2] @ np.array(t_vehicle.velocity[:2]).T
		return t_vehicle