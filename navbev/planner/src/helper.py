import numpy as np
import torch
from typing import Dict
from navbev.planner.src import misc
from navbev.planner.utils import frenet
from navbev.planner.utils.transforms import tf_bev_carla


def _pol_matrix_comp(t):
    num = len(t)
    delt = abs(t[1] - t[0])[0]

    Ad = np.array([[1, delt, 0.5 * delt**2], [0, 1, delt], [0, 0, 1]])
    Bd = np.array([[1 / 6 * delt**3, 0.5 * delt**2, delt]]).T

    P = np.zeros((num - 1, num - 1))
    Pdot = np.zeros((num - 1, num - 1))
    Pddot = np.zeros((num - 1, num - 1))

    Pint = np.zeros((num, 3))
    Pdotint = np.zeros((num, 3))
    Pddotint = np.zeros((num, 3))

    for i in range(0, num - 1):
        for j in range(0, i):
            temp = np.dot(np.linalg.matrix_power(Ad, (i - j)), Bd)

            P[i][j] = temp[0]
            Pdot[i][j] = temp[1]
            Pddot[i][j] = temp[2]

    for i in range(0, num):
        temp = np.linalg.matrix_power(Ad, i)

        Pint[i] = temp[0]
        Pdotint[i] = temp[1]
        Pddotint[i] = temp[2]

    P = np.vstack((np.zeros((1, num - 1)), P))
    Pdot = np.vstack((np.zeros((1, num - 1)), Pdot))
    Pddot = np.vstack((np.zeros((1, num - 1)), Pddot))

    P = np.hstack((Pint, P))
    Pdot = np.hstack((Pdotint, Pdot))
    Pddot = np.hstack((Pddotint, Pddot))

    return P, Pdot, Pddot


def init_planner(cfg):
    device = cfg.globals().device

    t_fin = cfg.planner().horizon
    num = cfg.planner().n_interp

    tot_time = torch.linspace(0, t_fin, num)
    tot_time_copy = tot_time.reshape(num, 1)

    P_np, Pdot_np, Pddot_np = _pol_matrix_comp(tot_time_copy)

    num_lower = int(0.3 * num)
    num_upper = int(0.6 * num)

    P = torch.from_numpy(P_np).float().to(device)
    Pdot = torch.from_numpy(Pdot_np).float().to(device)
    Pddot = torch.from_numpy(Pddot_np).float().to(device)

    P_diag = torch.block_diag(P, P).to(device)
    Pddot_diag = torch.block_diag(Pddot, Pddot).to(device)

    return P_np, P, Pdot, Pddot, P_diag, Pddot_diag, t_fin, num_lower, num_upper


def get_obs_state(agent, n_closest):
    n_states = 5
    obs_state = {}

    if len(agent) > n_closest:
        agent = agent[: n_closest - len(agent)]

    for k, obs in enumerate(agent):
        obs_xy = obs.location[:2]
        obs_vxy = obs.velocity[:2]
        obs_yaw = obs.rotation[-1]
        obs_state[k] = np.append(np.append(obs_xy, obs_vxy), obs_yaw)
    if len(agent) < n_closest:
        if len(agent) == 0:
            return np.zeros((n_closest, n_states))
        return np.concatenate(
            (
                list(obs_state.values()),
                np.tile(obs_state[0], (n_closest - len(agent), 1)),
            )
        )
    else:
        return np.array(list(obs_state.values()))


class DataProcessor:
    """Converts dictionary objects to ndarray
    
    Contains:
        obstacle_state
        obs_xy, obs_v, obs_yaw
        ref_path
        front_cam
        bev_ref
        ego_lane
    """

    def __init__(
        self,
        ego_state: Dict,
        goal: Dict,
        traffic: Dict,
        gt_traj: Dict,
        fcam: Dict,
        bev_ref,
        ego_lane: Dict,
    ):
        self.ego_state = self._to_array(ego_state)
        self.desired_goal = (
            self._to_array(goal) if not isinstance(goal, np.ndarray) else goal
        )
        self.obstacle_state = self._to_array(traffic)
        self.obs_xy = np.array([xy.T.flatten() for xy in self.obstacle_state[..., :2]])
        self.obs_v = np.array([v.T.flatten() for v in self.obstacle_state[..., 2:-1]])
        self.obs_yaw = np.array([t for t in self.obstacle_state[..., -1]])
        self.ref_path = self._to_array(gt_traj)
        self.front_cam = self._to_array(fcam)
        self.bev_ref = self._to_array(bev_ref)
        self.ego_lane = self._to_array(ego_lane)

    def _to_array(self, dict_obj):
        return np.array(list(dict_obj.values()))


class Frenetify:
    def __init__(
        self,
        ego_state,
        obstacle_state,
        ego_lane,
        goal,
        bev_size,
        bev_res,
        lane_width,
    ):
        self.bev_size = bev_size
        self.bev_res = bev_res
        self.ego_lane = ego_lane
        self.goal = goal
        self.obstacle_state = obstacle_state
        self.ego_state = ego_state
        self.lane_width = lane_width

        self.obstacle_state_frenet = []
        self.ego_state_frenet = []
        self.ref_path_exp = []
        self.goal_frenet = []
        self.arc_vecs = []
        self.fx_dots = []
        self.fy_dots = []

        self.ego_xs = []
        self._process_data()
        self._compute_lane_bounds()

    def _compute_lane_bounds(self):
        self.lane_lb = -(self.lane_width / 2) * np.ones(np.shape(self.ego_state)[0])
        self.lane_ub = 2 * self.lane_width - np.abs(self.lane_lb)
        self.lane_bounds = misc.merge_1d(self.lane_lb, self.lane_ub)

    def _process_data(self):
        for i in range(np.shape(self.goal)[0]):
            ego_center_lane = self.ego_lane[i]
            x_path, y_path = ego_center_lane[:, 0], ego_center_lane[:, 1]
            x_goal, y_goal = self.goal[i]

            Fx_dot, Fy_dot, _, _, arc_vec, kappa, _ = frenet.compute_path_parameters(x_path, y_path)
            
            ego = (self.ego_state[i][2], self.ego_state[i][-1],)

            (
                x_ego_frenet,
                y_ego_frenet,
                vx_ego_frenet,
                vy_ego_frenet,
                _,
                _,
                psi_ego_frenet,
                _,
                _,
            ) = frenet.global_to_frenet(
                x_path,
                y_path,
                ego,
                arc_vec,
                Fx_dot,
                Fy_dot,
                kappa,
                self.bev_res,
            )

            x_goal_frenet, y_goal_frenet, _, _, _ = frenet.global_to_frenet_obs(
                x_goal,
                y_goal,
                0.0,
                0.0,
                0.0,
                x_path,
                y_path,
                arc_vec,
                Fx_dot,
                Fy_dot,
                kappa,
                self.bev_res,
            )
            
            obs_frenet = []
            for obs in self.obstacle_state[i]:
                x_obs, y_obs = obs[:2]
                vx_obs, vy_obs = obs[2:-1]
                psi_obs = misc.normalize(obs[-1])

                (
                    x_obs_frenet,
                    y_obs_frenet,
                    vx_obs_frenet,
                    vy_obs_frenet,
                    _,
                ) = frenet.global_to_frenet_obs(
                    x_obs,
                    y_obs,
                    vx_obs,
                    vy_obs,
                    psi_obs,
                    x_path,
                    y_path,
                    arc_vec,
                    Fx_dot,
                    Fy_dot,
                    kappa,
                    self.bev_res,
                )

                obs_frenet.append(
                    [
                        x_obs_frenet,
                        y_obs_frenet,
                        vx_obs_frenet,
                        vy_obs_frenet,
                        psi_ego_frenet,
                    ]
                )
            obs_frenet = np.array(obs_frenet)
            obs_frenet[:, 0][obs_frenet[:, 0] < x_ego_frenet+3] = 1e6
            
            self.obstacle_state_frenet.append(obs_frenet)
            self.ego_state_frenet.append(
                [
                    x_ego_frenet,
                    y_ego_frenet,
                    np.sqrt(vx_ego_frenet**2 + vy_ego_frenet**2),
                    np.deg2rad(psi_ego_frenet),
                ]
            )
            self.ref_path_exp.append(np.array([x_path, y_path]))
            self.goal_frenet.append(
                np.array([x_goal_frenet, y_goal_frenet])
            )
            self.arc_vecs.append(arc_vec)
            self.fx_dots.append(Fx_dot)
            self.fy_dots.append(Fy_dot)
            self.ego_xs.append(x_ego_frenet)

        self.ego_xs = np.array(self.ego_xs)
        self.ego_state_frenet = np.array(self.ego_state_frenet)
        self.obstacle_state_frenet = np.array(self.obstacle_state_frenet)
        self.goal_frenet = np.array(self.goal_frenet)

        self.obstacle_state = np.transpose(
            self.obstacle_state_frenet[..., :-1], (0, 2, 1)
        )
        self.obstacle_state_frenet_xy = misc.restack_to_1d(
            self.obstacle_state_frenet[..., :2]
        )
        self.obstacle_state_frenet_v = misc.restack_to_1d(
            self.obstacle_state_frenet[..., 2:]
        )