import os
import pickle

import numpy as np
import torch
from navbev.planner.utils import transforms
from torch.utils.data import Dataset


class VehicleClass:
    def __init__(self):
        self.location = None
        self.rotation = None
        self.bounding_box = None
        self.velocity = None
        self.gt_trajectory = None


class CarlaDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __getitem__(self, index):
        index = str(index).zfill(4)
        pkl = os.path.join(self.data_path, f"{index}.pkl")
        data = self.load_data_from_pickle(pkl)

        output = {"index": index}

        for key, value in data.items():
            output[key] = value

        ego_agent = data.get("EGO_AGENT")
        output["GT_TRAJECTORY"] = ego_agent.gt_trajectory
        output["INITIAL_LOCATION"] = ego_agent.location
        output["INITIAL_ROTATION"] = ego_agent.rotation
        output["GOAL"] = output["GT_TRAJECTORY"][-1]

        output["EGO_LANE"] = data.get("EGO_CENTER_LANE")

        return output

    def load_data_from_pickle(self, pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)


class TrajDataset(Dataset):
    def __init__(
        self,
        inp,
        init_state_ego,
        goal_des,
        closest_obs,
        v_obs,
        obstacle_state,
        y_lb,
        y_ub,
        fimg,
        bev,
        ego_lane,
        ego_x,
    ):
        self.inp = inp
        self.init_state_ego = init_state_ego
        self.goal_des = goal_des
        self.closest_obs = closest_obs
        self.v_obs = v_obs
        self.obstacle_state = obstacle_state
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.fimg = fimg
        self.bev = bev
        self.ego_lane = ego_lane
        self.ego_x = ego_x

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = torch.tensor(self.inp[idx]).float()
        init_state_ego = torch.tensor(self.init_state_ego[idx]).float()
        goal_des = torch.tensor(self.goal_des[idx]).float()
        closest_obs = torch.tensor(self.closest_obs[idx]).float()
        v_obs = torch.tensor(self.v_obs[idx]).float()
        obstacle_state = torch.tensor(self.obstacle_state[idx]).float()
        y_lb = torch.tensor(self.y_lb[idx]).float()
        y_ub = torch.tensor(self.y_ub[idx]).float()
        fimg = torch.tensor(self.fimg[idx])
        bev = torch.tensor(self.bev[idx])
        ego_lane = torch.tensor(self.ego_lane[idx]).float()
        ego_x = torch.tensor(self.ego_x[idx]).float()

        return (
            inp,
            init_state_ego,
            goal_des,
            closest_obs,
            v_obs,
            obstacle_state,
            y_lb,
            y_ub,
            fimg,
            bev,
            ego_lane,
            ego_x,
        )
