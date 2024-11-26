import numpy as np
from scipy.interpolate import splev, splprep

from navbev.config import Configs as cfg


def tf_bev_stp(points):
    bx = np.array([-50.0 + 0.5, -50.0 + 0.5])
    dx = np.array([0.5, 0.5])
    return (points - bx) / dx


def untf_bev_stp(points):
    bx = np.array([-50.0 + 0.5 / 2.0, -50.0 + 0.5 / 2.0])
    dx = np.array([0.5, 0.5])
    return (points * dx) + bx


def tf_bev_carla(points):
    if points.ndim < 2:
        return 2 * points[::-1] + 100
    else:
        assert points.ndim == 2
    return 2 * np.flip(points, axis=1) + 100


def uniform_resample(data):
    # data = tf_bev_carla(data)
    x = np.sort(data[:, 0])
    x = data[:, 0]
    y = data[:, 1]

    tck, _ = splprep([x, y], s=0)
    u_new = np.linspace(0, 1, cfg.planner().ref_path_res)
    return np.column_stack(splev(u_new, tck))

def polyfit(points, degree=3):
    x = points[:, 0]
    y = points[:, 1]
    coeffs = np.polyfit(x, y, degree)
    smoothed_y = np.polyval(coeffs, x)
    return np.column_stack((x, smoothed_y))


def project_pixel_goal_2_local(goal2d, ground_normal=(0, 0, 1), ground_point=(0, 0, 0)):
    """
    "width": 400,
    "height": 300,
    "bevDim": 200,
    "fov": 100,

    "front_cam_trans": {
            "x": 1.3,
            "y": 0.0,
            "z": 2.3,
            "pitch": 0,
            "roll": 0,
            "yaw": 0
    },
    """
    # camera_matrix = np.array(
    #     [
    #         [167.8199, 0.0000, 200.0000],
    #         [0.0000, 167.8199, 150.0000],
    #         [0.0000, 0.0000, 1.0000],
    #     ]
    # )

    camera_matrix = np.array(
        [
            [800.0, 0.0000, 800.0000],
            [0.0000, 800.0, 600.0000],
            [0.0000, 0.0000, 1.0000],
        ]
    )

    extrinsics_cam_to_world = np.array(
        [
            [0.0000, 0.0000, 1.0000, 1.3000],
            [-1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, -1.0000, 0.0000, 1.900],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    u = goal2d[0]
    v = goal2d[1]
    pixel_coords = np.array([u, v, 1.0])
    normalized_coords = np.linalg.inv(camera_matrix) @ pixel_coords
    normalized_coords /= normalized_coords[2]
    direction_vector_camera = normalized_coords / np.linalg.norm(normalized_coords)
    rotation_matrix = extrinsics_cam_to_world[0:3, 0:3]
    camera_position = extrinsics_cam_to_world[0:3, 3]
    direction_vector_world = rotation_matrix @ direction_vector_camera
    # direction_vector_world = direction_vector_world / np.linalg.norm(direction_vector_world)
    t = np.dot(ground_normal, (ground_point - camera_position)) / np.dot(
        ground_normal, direction_vector_world
    )
    point = camera_position + t * direction_vector_world
    point[1] *= -1
    return point[:2]


def project_pixel_goal_2_local_arr(
    goals, ground_normal=(0, 0, 1), ground_point=(0, 0, 0)
):
    """
    "width": 400,
    "height": 300,
    "bevDim": 200,
    "fov": 100,

    "front_cam_trans": {
            "x": 1.3,
            "y": 0.0,
            "z": 2.3,
            "pitch": 0,
            "roll": 0,
            "yaw": 0
    },
    """
    camera_matrix = np.array(
        [
            [800.0, 0.0000, 800.0000],
            [0.0000, 800.0, 600.0000],
            [0.0000, 0.0000, 1.0000],
        ]
    )

    # camera_matrix = np.array(
    #     [
    #         [167.8199, 0.0000, 200.0000],
    #         [0.0000, 167.8199, 150.0000],
    #         [0.0000, 0.0000, 1.0000],
    #     ]
    # )
    extrinsics_cam_to_world = np.array(
        [
            [0.0000, 0.0000, 1.0000, 1.3000],
            [-1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, -1.0000, 0.0000, 1.90],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )
    
    goal_tf = []
    for goal2d in goals:
        # u = int(goal2d[0])
        # v = int(goal2d[1])
        u = goal2d[0]
        v = goal2d[1]
        pixel_coords = np.array([u, v, 1.0])
        normalized_coords = np.linalg.inv(camera_matrix) @ pixel_coords
        normalized_coords /= normalized_coords[2]
        direction_vector_camera = normalized_coords / np.linalg.norm(normalized_coords)
        rotation_matrix = extrinsics_cam_to_world[0:3, 0:3]
        camera_position = extrinsics_cam_to_world[0:3, 3]
        direction_vector_world = rotation_matrix @ direction_vector_camera
        # direction_vector_world = direction_vector_world / np.linalg.norm(direction_vector_world)
        t = np.dot(ground_normal, (ground_point - camera_position)) / np.dot(
            ground_normal, direction_vector_world
        )
        point = camera_position + t * direction_vector_world
        point[1] *= -1  # Since BEV is Horizontally flipped in the data collected
        goal_tf.append(point[:2])

    return np.array(goal_tf)
