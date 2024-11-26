from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from rich.progress import track

from navbev.planner.src import misc
from navbev.planner.src.dataloader import CarlaDataLoader
from navbev.planner.src.helper import get_obs_state
from navbev.planner.utils.transforms import (
    project_pixel_goal_2_local,
    project_pixel_goal_2_local_arr,
    uniform_resample
)
from navbev.perception.glc import transforms


class VehicleClass:
    pass


@dataclass(frozen=True)
class LoadDataset:
    ego_state: Dict[int, np.ndarray]
    bev_img: Dict[int, Any]
    bev_ref: Dict[int, Any]
    gt_traj: Dict[int, np.ndarray]
    gt_goal: Dict[int, np.ndarray]
    glc_goal: np.ndarray
    traffic: Dict[int, Any]
    ego_lane: Dict[int, Any]
    fcam: Dict[int, Any]
    N_data: int

    @classmethod
    def get_carla_data(cls, path, cfg, goals, normalize_goal, start):
        (
            ego_state,
            bev_img,
            bev_ref,
            gt_traj,
            gt_goal,
            glc_goal,
            traffic,
            ego_lane,
            fcam,
            N_data,
        ) = _load_carla_data(path, cfg, goals, normalize_goal, start)
        return cls(
            ego_state,
            bev_img,
            bev_ref,
            gt_traj,
            gt_goal,
            glc_goal,
            traffic,
            ego_lane,
            fcam,
            N_data,
        )


def _load_carla_data(path, cfg, goals, normalize_goal, start):
    dataset = CarlaDataLoader(path)

    traffic = {}
    bev_img, bev_ref = {}, {}
    gt_traj, gt_goal = {}, {}
    ego_state, ego_lane = {}, {}
    lcam, fcam, rcam = {}, {}, {}

    if start != 0:
        N_data = len(goals) + 1
    else:
        N_data = len(goals)
    
    glc_goal_px = goals[:N_data]

    if normalize_goal:
        glc_goal_px = transforms.unnormalize_goal(cfg, glc_goal_px)

    glc_goal = project_pixel_goal_2_local_arr(glc_goal_px)

    for j in track(range(start, N_data), description="Loading Data"):
        if start != 0:
            i = j - 1
        else:
            i = j
        ego_state[i] = np.hstack(
            (
                dataset[j]["EGO_AGENT"].location[:2],
                np.linalg.norm(dataset[j]["EGO_AGENT"].velocity[:2]),
                misc.normalize(dataset[j]["EGO_AGENT"].rotation[-1]),
            )
        )
        bev_ref[i] = dataset[j]["BEV"]
        bev_img[i] = dataset[j]["BEV_MAP"]
        gt_traj[i] = dataset[j]["GT_TRAJECTORY"][:, :2]
        gt_goal[i] = dataset[j]["GOAL"][:2]  # in local
        traffic[i] = get_obs_state(
            dataset[j]["TRAFFIC_AGENTS"], n_closest=cfg.planner().n_closest_obstacle
        )
        ego_lane[i] = uniform_resample(dataset[j]["EGO_CENTER_LANE"])
        fcam[i] = dataset[j]["CAM_FRONT"]
        lcam[i] = dataset[j]["CAM_LEFT"]
        rcam[i] = dataset[j]["CAM_RIGHT"]

    return (
        ego_state,
        bev_img,
        bev_ref,
        gt_traj,
        gt_goal,
        glc_goal,
        traffic,
        ego_lane,
        fcam,
        N_data,
    )