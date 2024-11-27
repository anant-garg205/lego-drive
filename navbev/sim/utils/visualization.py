"""
	Date: January 29, 2024
	Description: Script for Visualization
"""

import numpy as np
import math
import cv2
import carla

class Visualize:
	"""
	Class for Visualization
	"""
	def __init__(self, world):
		
		self.ego_agent_init_location = None
		self.ego_agent_init_rotation = None
		self.world = world

	def get_init_ego_transform(self, ego_agent_transform):
		"""
		Function to get the Initial Transform of the ego-agent.
		Because the Trajectory given by the planner and trajectory followed are plotted
		on the Initial BEV (at time=0)
		"""
		self.ego_agent_init_location = [ego_agent_transform.location.x, ego_agent_transform.location.y]
		self.ego_agent_init_rotation = np.deg2rad(ego_agent_transform.rotation.yaw)

	def plot_local_2_BEV(self, point, bev, color=[0, 255, 0], radius=1):
		"""
		Function to Plot a point in local frame (Ego-Agent's Frame) on BEV. 
		BEV is pointing upwards. Up = Ego-agent forward
		"""
		px = 100 - int(point[0]*2)
		py = 100 - int(point[1]*2)
		# cv2.circle(bev, [py, px], radius, color, -1)
		cv2.circle(bev, (py, px), radius, color, -1)


	def plot_world_2_BEV_static(self, point, bev, color=[255, 0, 0], radius=2):
		"""
		Function to plot a point in world frame on BEV
		"""
		rot_mat = np.zeros((3, 3))
		yaw = self.ego_agent_init_rotation
		cst = np.cos(yaw)
		sst = np.sin(yaw)

		rot_mat[0][0] = cst
		rot_mat[0][1] = -sst
		rot_mat[1][0] = sst
		rot_mat[1][1] = cst
		rot_mat[2][2] = 1

		## NOTE: -ve sign is taken for Y because carla uses Left-Handed Coordinate System
		rot_mat[:, 2] = [self.ego_agent_init_location[0], self.ego_agent_init_location[1], 1]
		point = np.linalg.inv(rot_mat) @ np.array([point[0], point[1], 1]).T
		point[1] *= -1	## Added on Feb 12, 2024		
		self.plot_local_2_BEV(point, bev, color, radius)

	@staticmethod
	def draw_cross(image, center, length=3, thickness=5, color=(0,0,255)):
		cv2.line(image, (int(center[0]) - length, int(center[1]) - length), (int(center[0]) + length, int(center[1]) + length), color, thickness)
		cv2.line(image, (int(center[0]) - length, int(center[1]) + length), (int(center[0]) + length, int(center[1]) - length), color, thickness)


	def draw_path(self, loc1, loc2=None, draw_type=None, size=0.05, color=[255, 0, 0]):

		if loc1 is not None:
			loc1 = carla.Location(x=loc1[0], y=loc1[1], z=0.1)
		if loc2 is not None:
			loc2 = carla.Location(x=loc2[0], y=loc2[1], z=0.1)

		color = carla.Color(r=color[0], g=color[1], b=color[2])

		if draw_type == "line":
			self.world.debug.draw_line(loc1, loc2, thickness=0.8,
			color=color, life_time=0.5, persistent_lines=True)
		elif draw_type == "string2":
			self.world.debug.draw_point(location=loc1, size=size, color=color,
								life_time=0.3, persistent_lines=True)
		else:
			self.world.debug.draw_point(loc1, size=size, color=color, 
										life_time=0)



	def draw_goal(self, loc1, z=0.5, color=[255, 0, 0]):
		if loc1 is not None:
			loc1 = carla.Location(x=loc1[0], y=loc1[1], z=0.1)

		color = carla.Color(r=color[0], g=color[1], b=color[2], a=10)
		begin = loc1 + carla.Location(z=z)
		angle = 0
		end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
		self.world.debug.draw_arrow(begin=begin, end=end, arrow_size=0.3, life_time=0, color=color)


	
	def draw_x(self, center, size=10, thickness=0.1, color=[255, 0, 0]):
		"""
		Draw an 'X' shape in CARLA world.
		:param world: CARLA world object
		:param center: Center point of the 'X' (carla.Location)
		:param size: Size of the 'X' (default is 10)
		:param thickness: Thickness of the lines (default is 0.1)
		:param color: Color of the lines (default is red with alpha 100)
		"""
		color = carla.Color(r=color[0], g=color[1], b=color[2], a=0)
		# Draw first diagonal line of the 'X'
		start_point_1 = carla.Location(x=center[0] - size / 2, y=center[1] - size / 2, z=0.1)
		end_point_1 = carla.Location(x=center[0] + size / 2, y=center[1] + size / 2, z=0.1)
		self.world.debug.draw_line(start_point_1, end_point_1, thickness=thickness, color=color)

		# Draw second diagonal line of the 'X'
		start_point_2 = carla.Location(x=center[0] + size / 2, y=center[1] - size / 2, z=0.1)
		end_point_2 = carla.Location(x=center[0] - size / 2, y=center[1] + size / 2, z=0.1)
		self.world.debug.draw_line(start_point_2, end_point_2, thickness=thickness, color=color)

	
	def combined_visualization(self, tpv, fv, bev):

		combined_image = tpv.copy()
		tpv_dims = [tpv.shape[0]//3, tpv.shape[1]//3]
		fv_resized = cv2.resize(fv, (tpv_dims[1], tpv_dims[0]))
		bev = cv2.resize(bev, (bev.shape[1]*2, bev.shape[0]*2))
		combined_image[0:tpv_dims[0], 0:tpv_dims[1]] = fv_resized
		combined_image[0:bev.shape[1], tpv.shape[1]-bev.shape[1]:tpv.shape[1]] = bev
		combined_image = cv2.resize(combined_image, (800, 600))
		return combined_image



