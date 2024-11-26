"""
	Date: January 26, 2024
	Description: Controller script for both longitudenal and lateral control
"""

from collections import deque
import numpy as np
import math
import carla
from navbev.config import Configs as cfg

class PIDController(object):
	"""
	Class for PID Controller
	"""
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D
		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0
		return self._K_P * error + self._K_I * integral + self._K_D * derivative
	

class PurePursuitController:
	"""
	Class for Pure Pursuit Controller for Lateral Control
	"""
	def __init__(self):
		self.look_ahead = cfg.carla().pure_pursuit.look_ahead
		self.wheel_base = cfg.carla().vehicle.wheel_base
		self.sK_ = cfg.carla().pure_pursuit.sK
	
	def controller(self, c_pose, t_pose):
		"""
		Function to calculate the steer command based on current pose and target pose
		"""
		ego_loc = c_pose
		v_vec = [np.cos(c_pose[2]), np.sin(c_pose[2])]
		v_vec = np.array([v_vec[0], v_vec[1], 0.0])
		w_loc = t_pose

		w_vec = np.array([w_loc[0] - ego_loc[0],
						  w_loc[1] - ego_loc[1],
						  0.0])
		
		wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
		if wv_linalg == 0:
			_dot = 1
		else:
			_dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
		_cross = np.cross(v_vec, w_vec)
		if _cross[2] < 0:
			_dot *= -1.0
		return _dot
	
	def get_target_waypoint(self, waypoints, c_pose, c_speed):
		"""
		Function to get next target waypoint based on the current pose
		of the ego-agent and its speed.

		Distance of the next waypoint is specified by a "look ahead distance"
		and factor multipled with current speed.
		"""
		self.waypoints = waypoints
		self.c_pose = c_pose
		self.c_speed = c_speed

		dist_list = []
		for waypoint in self.waypoints:
			dist = np.hypot(waypoint[0] - c_pose[0], waypoint[1] - c_pose[1])
			dist_list.append(dist)

		self.n_index = np.argmin(dist_list)
		waypoints = waypoints[self.n_index+1:]
		dist2NWP = dist_list[self.n_index]
		look_ahead = self.look_ahead + self.sK_ * self.c_speed

		while (look_ahead > dist2NWP):
			self.n_index += 1
			if self.n_index >= len(self.waypoints):
				self.n_index = len(self.waypoints)-1
				break
			dist2NWP = dist_list[self.n_index]
		return self.waypoints[self.n_index]

class Controller:
	"""
	Class for Controller. It includes both longitudenal and lateral control.
	"""
	def __init__(self, ego_agent, cfgs=None):
		self.ego_agent = ego_agent
		self.steer_PID_controller = PIDController()
		self.speed_PID_controller = PIDController()
		self.pp_controller = PurePursuitController()
		self.reached_goal = False
		self.set_target_speed()
	
	def get_waypoints(self, waypoints):
		"""
		Function to get waypoints for the path given by the planner
		"""
		self.waypoints = waypoints

	def set_target_waypoint(self, t_pose=None):
		"""
		Function to set the target waypoint.
		It can either be specified explicitly or can be calculated using the "get_target_waypoint" function
		"""		
		self.c_pose = [self.ego_agent.get_transform().location.x, 
				 self.ego_agent.get_transform().location.y,
				 np.deg2rad(self.ego_agent.get_transform().rotation.yaw)]
		self.c_speed = np.linalg.norm([self.ego_agent.get_velocity().x, self.ego_agent.get_velocity().y])
		if t_pose:
			self.t_pose = t_pose
		else:
			self.t_pose = self.pp_controller.get_target_waypoint(self.waypoints, self.c_pose, self.c_speed)

	def set_target_speed(self, t_speed=2):
		"""
		Function to set the target speed for the longitudenal controller
		"""
		self.t_speed = t_speed


	def runstep(self):
		"""
		Function to get the control command. It calls all the required functions
		"""
		max_steer = cfg.carla().vehicle.max_steer
		tSteer = np.clip(self.pp_controller.controller(self.c_pose, self.t_pose), -max_steer, max_steer) / max_steer

		vDelta = self.t_speed - self.c_speed
		throttle = self.speed_PID_controller.step(vDelta)
		control = carla.VehicleControl()

		if throttle > 0:
			control.throttle = throttle
			control.brake = 0
		else:
			control.throttle = 0
			control.brake = throttle

		if self.pp_controller.n_index == (len(self.pp_controller.waypoints)-1):
			self.pp_controller.n_index = (len(self.pp_controller.waypoints)-1)
			self.reached_goal = True
			throttle = 0.0
			tSteer = 0
			control.brake = 1
			control.throttle = throttle
		control.steer = tSteer
		control.manual_gear_shift = False
		return control