import carla
from carla import ColorConverter as cc
import numpy as np

class RGBCamera(object):
	def __init__(self, player, params) -> None:
		self.parent = player
		self.world = self.parent.get_world()
		self.transform = carla.Transform(carla.Location(x = params['x'],
														y = params['y'],
														z = params['z']),
										carla.Rotation(pitch = params["pitch"],
														roll = params["roll"],
														yaw = params["yaw"])
										)
		camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
		self.name = params["sensor_name"]
		self.width = params["width"]
		self.height = params["height"]

		self.calibration = np.eye(3)
		self.calibration[0, 2] = params["width"] / 2.0
		self.calibration[1, 2] = params["height"] / 2.0
		self.calibration[0, 0] = params["width"] / (2.0 * np.tan(params["fov"] * np.pi / 360.0))
		self.calibration[1, 1] = params["width"] / (2.0 * np.tan(params["fov"] * np.pi / 360.0))

		camera_blueprint.set_attribute("image_size_x", str(params["width"]))
		camera_blueprint.set_attribute("image_size_y", str(params["height"]))
		camera_blueprint.set_attribute("fov", str(params["fov"]))

		self.sensor = self.world.spawn_actor(camera_blueprint, self.transform, attach_to = self.parent)
		self.sensor.calibration = self.calibration
		self.image = np.ones((params["height"], params["width"], 3))
		self.data = None

	def callback(self, data):
		array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
		array = np.reshape(array, (data.height, data.width, 4))
		array = array[:, :, :3]
		self.image = array


class SemanticCamera(object):
	def __init__(self, player, params) -> None:
		self.parent = player
		self.world = self.parent.get_world()
		self.transform = carla.Transform(carla.Location(x = params['x'],
														y = params['y'],
														z = params['z']),
										carla.Rotation(pitch = params["pitch"],
														roll = params["roll"],
														yaw = params["yaw"])
										)
		camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
		self.name = params["sensor_name"]
		self.width = params["width"]
		self.height = params["height"]

		self.calibration = np.eye(3)
		self.calibration[0, 2] = params["width"] / 2.0
		self.calibration[1, 2] = params["height"] / 2.0
		self.calibration[0, 0] = params["width"] / (2.0 * np.tan(params["fov"] * np.pi / 360.0))
		self.calibration[1, 1] = params["width"] / (2.0 * np.tan(params["fov"] * np.pi / 360.0))

		camera_blueprint.set_attribute("image_size_x", str(params["width"]))
		camera_blueprint.set_attribute("image_size_y", str(params["height"]))
		camera_blueprint.set_attribute("fov", str(params["fov"]))

		self.sensor = self.world.spawn_actor(camera_blueprint, self.transform, attach_to = self.parent)
		self.sensor.calibration = self.calibration
		self.image = np.ones((params["height"], params["width"], 3))
		self.data = None

	def callback(self, data):
		data.convert(cc.CityScapesPalette)
		array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
		array = np.reshape(array, (data.height, data.width, 4))
		array = array[:, :, :4]
		self.image = array


class DepthCamera(object):
	def __init__(self, player, params) -> None:
		self.parent = player
		self.world = self.parent.get_world()
		self.transform = carla.Transform(carla.Location(x = params['x'],
														y = params['y'],
														z = params['z']),
										carla.Rotation(pitch = params["pitch"],
														roll = params["roll"],
														yaw = params["yaw"]))
		camera_blueprint = self.world.get_blueprint_library().find("sensor.camera.depth")
		self.name = params["sensor_name"]
		self.width = params["width"]
		self.height = params["height"]

		self.calibration = np.eye(3)
		camera_blueprint.set_attribute("image_size_x", str(params["width"]))
		camera_blueprint.set_attribute("image_size_y", str(params["height"]))
		camera_blueprint.set_attribute("fov", str(params["fov"]))

		self.sensor	= self.world.spawn_actor(camera_blueprint, self.transform, attach_to = self.parent)
		self.sensor.calibration = self.calibration
		self.image	= np.ones((params["height"], params["width"], 3))
		self.data = None

	def callback(self, data):
		data.convert(cc.Raw)
		array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
		array = np.reshape(array, (data.height, data.width, 4))
		array = array[:, :, :3]
		self.image = array
