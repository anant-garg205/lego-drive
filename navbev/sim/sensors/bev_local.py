import glob
import os
import sys
import pygame
import hashlib
import math
import numpy as np
from scipy.spatial.transform import Rotation

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import cv2

# COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4 = pygame.Color(0, 0, 0)
COLOR_ALUMINIUM_5 = pygame.Color(255, 255, 255)

# COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_0 = pygame.Color(0, 0, 255)

# COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_0 = pygame.Color(255, 0, 0)
# COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

PIXELS_PER_METER = 2
PIXELS_AHEAD_VEHICLE = 0

class Util(object):

	@staticmethod
	def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
		"""Function that renders the all the source surfaces in a destination source"""
		for surface in source_surfaces:
			destination_surface.blit(surface[0], surface[1], rect, blend_mode)

class MapImage(object):

	def __init__(self, carla_world, carla_map, pixels_per_meter):
		
		self.carla_world = carla_world
		self.carla_map = carla_map

		waypoints = carla_map.generate_waypoints(2)
		margin = 50
		max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
		max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
		min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
		min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

		self.width = max(max_x - min_x, max_y - min_y)
		self._world_offset = (min_x, min_y)

		self._pixels_per_meter = pixels_per_meter
		width_in_pixels = int(self._pixels_per_meter * self.width)
		
		self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

		self.draw_road_map(self.big_map_surface, self.carla_map, PIXELS_PER_METER)

		self.surface = self.big_map_surface
		self.scale = 1

	def transform_distance(self, angle, point, dist=1.75):

		rot_mat = np.zeros((2,2))
		rot_mat[0][0] = np.cos(angle)
		rot_mat[0][1] = -np.sin(angle)
		rot_mat[1][0] = np.sin(angle)
		rot_mat[1][1] = np.cos(angle)

		dist_shifted = rot_mat @ np.array([dist, 0]).T
		point[0] += dist_shifted[0]
		point[1] += dist_shifted[1]
		return point

	def get_boundary_points(self, waypoint):

		angle = np.deg2rad(waypoint.transform.rotation.yaw) + np.pi/2
		point = [waypoint.transform.location.x, waypoint.transform.location.y]
		boundary_point = self.transform_distance(angle, point)
		return boundary_point

	def draw_road_map(self, map_surface, carla_map, world_to_pixel):
	
		map_surface.fill(COLOR_ALUMINIUM_4)
		precision = 0.05
		self.carla_map_waypoints = []
		self.boundary_points = []

		def lateral_shift(transform, shift):
			"""Makes a lateral shift of the forward vector of a transform"""
			transform.rotation.yaw += 90
			return transform.location + shift * transform.get_forward_vector()
		
		def draw_topology(carla_topology, index):

			
			topology = [x[index] for x in carla_topology]
			topology = sorted(topology, key=lambda w: w.transform.location.z)
			set_waypoints = []
			for waypoint in topology:
				waypoints = [waypoint]

				# Generate waypoints of a road id. Stop when road id differs
				nxt = waypoint.next(precision)
				if len(nxt) > 0:
					nxt = nxt[0]
					while nxt.road_id == waypoint.road_id:
						waypoints.append(nxt)
						nxt = nxt.next(precision)
						if len(nxt) > 0:
							nxt = nxt[0]
						else:
							break
				set_waypoints.append(waypoints)

				self.carla_map_waypoints.append([waypoint.transform.location.x,
											waypoint.transform.location.y])

			for waypoints in set_waypoints:
				waypoint = waypoints[0]

				#######################################
				for w in waypoints:
					self.carla_map_waypoints.append([w.transform.location.x, w.transform.location.y])
					if w.is_junction == False:
						bPoint = self.get_boundary_points(w)
						self.boundary_points.append([bPoint[0], bPoint[1]])
				#######################################

				road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
				road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

				polygon = road_left_side + [x for x in reversed(road_right_side)]
				polygon = [self.world_to_pixel(x) for x in polygon]

				if len(polygon) > 2:
					pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
					pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

		topology = carla_map.get_topology()
		draw_topology(topology, 0)

	def world_to_pixel(self, location, offset=(0, 0)):
		"""Converts the world coordinates to pixel coordinates"""
		self.scale = 1
		x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
		y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
		return [int(x - offset[0]), int(y - offset[1])]

	def world_to_pixel_width(self, width):
		"""Converts the world units to pixel units"""
		return int(self.scale * self._pixels_per_meter * width)

	def scale_map(self, scale):
		"""Scales the map surface"""
		if scale != self.scale:
			self.scale = scale
			width = int(self.big_map_surface.get_width() * self.scale)
			self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))

class VehicleClass:

	def __init__(self):
		self.location = 1
		self.rotation = 1
		self.bounding_box = 1
		self.velocity = 1
	
class BEV_Local(object):

	def __init__(self, dims, world, ego_agent):
		self.client = None
		self.server_fps = 0.0
		self.simulation_time = 0
		self.server_clock = pygame.time.Clock()
		self.world = world
		self.town_map = world.get_map()
		self.ego_agent = ego_agent
		self.traffic_vehicles = []
		
		self.width = dims[0]
		self.height = dims[1]

		self.map_image = MapImage(
			carla_world=self.world,
			carla_map=self.town_map,
			pixels_per_meter=PIXELS_PER_METER)
		
		self.surface_size = self.map_image.big_map_surface.get_width()
		
		self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
		self.actors_surface.set_colorkey(COLOR_BLACK)

		self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
		self.result_surface.set_colorkey(COLOR_BLACK)

		self.original_surface_size = min(self.width, self.height)
		scaled_original_size = self.original_surface_size * (1.0  / 0.9)
		self.ego_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()
		self.map_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()

		self.actors_with_transforms = []

		
	def _split_actors(self):
		"""Splits the retrieved actors by type id"""
		vehicles = []
		traffic_lights = []
		speed_limits = []
		walkers = []

		for actor_with_transform in self.actors_with_transforms:
			actor = actor_with_transform[0]
			if 'vehicle' in actor.type_id:
				vehicles.append(actor_with_transform)
			elif 'traffic_light' in actor.type_id:
				traffic_lights.append(actor_with_transform)
			elif 'speed_limit' in actor.type_id:
				speed_limits.append(actor_with_transform)
			elif 'walker.pedestrian' in actor.type_id:
				walkers.append(actor_with_transform)

		return (vehicles, traffic_lights, speed_limits, walkers)
	
	def _render_vehicles(self, surface, list_v, world_to_pixel, ego_agent):
		"""Renders the vehicles' bounding boxes"""
		ego_transform = None
		
		self.traffic_vehicles.clear()

		for v in list_v:

			traffic_vehicle = VehicleClass()
			
			color = COLOR_SKY_BLUE_0

			# Compute bounding box points
			if v[0].attributes['role_name'] == 'ego':
				ego_transform = v[1]
				color = COLOR_CHAMELEON_0

			bb = v[0].bounding_box.extent
			corners = [carla.Location(x=-bb.x, y=-bb.y),
					   carla.Location(x=bb.x - 0.8, y=-bb.y),
					   carla.Location(x=bb.x - 0.8, y=bb.y),
					   carla.Location(x=-bb.x, y=bb.y)
					   ]
			v[1].transform(corners)

			vehicle_corners = []
			for corner in corners:
				vehicle_corners.append([corner.x, corner.y, corner.z])

			traffic_vehicle.location = [v[1].location.x, v[1].location.y, v[1].location.z]
			traffic_vehicle.rotation = [v[1].rotation.roll, v[1].rotation.pitch, v[1].rotation.yaw]
			traffic_vehicle.bounding_box = vehicle_corners
			traffic_vehicle.velocity = [v[0].get_velocity().x, v[0].get_velocity().y, v[0].get_velocity().z]
			self.traffic_vehicles.append(traffic_vehicle)

			corners = [world_to_pixel(p) for p in corners]
			pygame.draw.polygon(surface, color, corners)

	def render_actors(self, surface, vehicles, ego_agent):
		"""Renders all the actors"""
		self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel, ego_agent)

	def clip_surfaces(self, clipping_rect):
		"""Used to improve perfomance. Clips the surfaces in order to render only the part of the surfaces that are going to be visible"""
		self.actors_surface.set_clip(clipping_rect)
		self.result_surface.set_clip(clipping_rect)

	def render(self):
		
		actors = self.world.get_actors()

		self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]

		if self.actors_with_transforms is None:
			return
		self.result_surface.fill(COLOR_BLACK)

		vehicles, _, _, _ = self._split_actors()

		self.actors_surface.fill(COLOR_BLACK)
		self.render_actors(
			self.actors_surface,
			vehicles,
			self.ego_agent)
		
		surfaces = ((self.map_image.surface, (0, 0)),
					(self.actors_surface, (0, 0)),
					)
		
		if self.ego_agent is not None:

			self.ego_transform = self.ego_agent.get_transform()
			ego_location_screen = self.map_image.world_to_pixel(self.ego_transform.location)
			ego_front = self.ego_transform.get_forward_vector()
			translation_offset = (ego_location_screen[0] - self.ego_surface.get_width() / 2 + ego_front.x * PIXELS_AHEAD_VEHICLE,
								  (ego_location_screen[1] - self.ego_surface.get_height() / 2 + ego_front.y * PIXELS_AHEAD_VEHICLE))
			
			clipping_rect = pygame.Rect(translation_offset[0],
										translation_offset[1],
										self.ego_surface.get_width(),
										self.ego_surface.get_height())
			
			self.clip_surfaces(clipping_rect)

			Util.blits(self.result_surface, surfaces)

			self.ego_surface.fill(COLOR_ALUMINIUM_4)
			self.ego_surface.blit(self.result_surface, (-translation_offset[0],
														 -translation_offset[1]))
			
			self.map_surface.fill(COLOR_ALUMINIUM_4)
			self.map_surface.blit(self.map_image.surface, (-translation_offset[0],
														 -translation_offset[1]))
			
			angle = self.ego_transform.rotation.yaw + 90
			rotated_result_surface = pygame.transform.rotozoom(self.ego_surface, angle, 0.9).convert()
			rotated_result_map_surface = pygame.transform.rotozoom(self.map_surface, angle, 0.9).convert()

			center = (self.width / 2, self.height / 2)
			rotation_pivot = rotated_result_surface.get_rect(center=center)

			bev_rotated = pygame.surfarray.array3d(rotated_result_surface)
			map_rotated = pygame.surfarray.array3d(rotated_result_map_surface)

			center_bev = [bev_rotated.shape[0] // 2 , bev_rotated.shape[1] // 2]
			bev_dims = np.array([self.width/2, self.height/2])
			bev_dims = bev_dims.astype(np.int32)

			bev_dims[0] = bev_dims[0]/2
			bev_dims[1] = bev_dims[1]/2
			bev_small = bev_rotated[center_bev[0]-bev_dims[0]:center_bev[0]+bev_dims[0], center_bev[1]-bev_dims[1]:center_bev[1]+bev_dims[1]]
			bev_map_small = map_rotated[center_bev[0]-bev_dims[0]:center_bev[0]+bev_dims[0], center_bev[1]-bev_dims[1]:center_bev[1]+bev_dims[1]]

			self.bev_small = cv2.transpose(bev_small)
			self.map_small = cv2.transpose(bev_map_small)