import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from navbev.config import Configs as cfg

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

device = cfg.globals().device


class PointNet(nn.Module):
    def __init__(self, inp_channel=1, emb_dims=512, output_channels=20):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(
            inp_channel, 64, kernel_size=1, bias=False
        )  # input_channel = 3
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.linear1 = nn.Linear(emb_dims, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(256, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # x = F.adaptive_max_pool1d(x, 1)
        # breakpoint()
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        x = self.linear1(x)
        # if x.ndim == 1: x = x.unsqueeze(0)
        x = F.relu(self.bn6(x))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class mlp_model_planner(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(mlp_model_planner, self).__init__()

        # MC Dropout Architecture
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        out = self.mlp(x)
        return out


class diff_planner_ad(nn.Module):
    def __init__(
        self,
        P,
        Pdot,
        Pddot,
        mlp,
        point_net,
        num_batch,
        t_fin,
        num_obs,
        num_lower,
        num_upper,
    ):
        super(diff_planner_ad, self).__init__()

        # BayesMLP
        self.mlp = mlp
        self.point_net = point_net

        self.num_obs = num_obs

        # P Matrices
        self.P = P.to(device)
        self.Pdot = Pdot.to(device)
        self.Pddot = Pddot.to(device)

        # No. of Variables
        self.nvar = P.size(dim=1)
        self.num = P.size(dim=0)
        self.num_batch = num_batch

        self.num_obs = num_obs
        self.a_obs = 6.0
        self.b_obs = 3.2

        ################# boundary conditions for the main optimizer

        A_var_mat = torch.eye(self.nvar, device=device)

        self.num_partial_xy = num_upper - num_lower

        A_var_partial = A_var_mat[num_lower:num_upper]

        self.A_eq_x = torch.vstack(
            [self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1], A_var_partial]
        )
        self.A_eq_y = torch.vstack(
            [
                self.P[0],
                self.Pdot[0],
                self.Pddot[0],
                self.P[-1],
                self.Pdot[-1],
                A_var_partial,
            ]
        )

        ################## constraint parameters
        self.rho_ineq = 1.0
        self.rho_obs = 1.0
        self.rho_lane = 1

        # t_fin = 6.0

        self.tot_time = torch.linspace(0, t_fin, self.num, device=device)

        self.t = t_fin / self.num

        self.v_min = 0.1
        self.v_max = 30

        self.a_max = 18.0
        self.wheel_base = 2.5
        self.steer_max = 0.6
        self.kappa_max = 0.3

        self.v_des = 20.0

        ########################33 different constraint matrices
        self.A_vel = self.Pdot
        self.A_acc = self.Pddot

        self.A_obs = torch.tile(self.P, (self.num_obs, 1))
        self.A_lane = torch.vstack((self.P, -self.P))

        ############# unroll iterations
        self.maxiter = 30  # 20

        # Smoothness
        self.weight_smoothness = 100
        self.cost_smoothness = self.weight_smoothness * (
            torch.mm(self.Pddot.T, self.Pddot)
            + 0.1 * torch.eye(self.nvar, device=device)
        )
        self.weight_aug = 1.0
        self.vel_scale = 1e-3
        self.weight_vel_track = 1.0
        self.weight_y_track = 1.0

        ########################################

        # RCL Loss
        self.rcl_loss = nn.MSELoss()

        ##############################

        self.num_lamda_input = 2 * self.nvar
        self.num_partial_solution = 2 * self.num_partial_xy

    def compute_obs_trajectories(self, closest_obs, v_obs):
        x_obs = closest_obs[:, : self.num_obs]
        y_obs = closest_obs[:, self.num_obs : 2 * self.num_obs]

        vx_obs = v_obs[:, 0 : self.num_obs]
        vy_obs = v_obs[:, self.num_obs : 2 * self.num_obs]

        # Batch Obstacle Trajectory Predictionn
        x_obs_inp_trans = x_obs.reshape(self.num_batch, 1, self.num_obs)
        y_obs_inp_trans = y_obs.reshape(self.num_batch, 1, self.num_obs)

        vx_obs_inp_trans = vx_obs.reshape(self.num_batch, 1, self.num_obs)
        vy_obs_inp_trans = vy_obs.reshape(self.num_batch, 1, self.num_obs)

        x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time.unsqueeze(1)
        y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time.unsqueeze(1)

        x_obs_traj = x_obs_traj.permute(0, 2, 1)
        y_obs_traj = y_obs_traj.permute(0, 2, 1)

        x_obs_traj = x_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
        y_obs_traj = y_obs_traj.reshape(self.num_batch, self.num_obs * self.num)

        return x_obs_traj, y_obs_traj

    # Boundary Vectors for initialization
    def compute_boundary_layer(
        self, init_state_ego, fin_state_ego, c_x_partial, c_y_partial
    ):
        x_init_vec = init_state_ego[:, 0].reshape(self.num_batch, 1)
        y_init_vec = init_state_ego[:, 1].reshape(self.num_batch, 1)

        v_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
        psi_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)

        vx_init_vec = v_init_vec * torch.cos(psi_init_vec)
        vy_init_vec = v_init_vec * torch.sin(psi_init_vec)

        ax_init_vec = torch.zeros([self.num_batch, 1], device=device)
        ay_init_vec = torch.zeros([self.num_batch, 1], device=device)

        x_fin_vec = fin_state_ego[:, 0].reshape(self.num_batch, 1)
        y_fin_vec = fin_state_ego[:, 1].reshape(self.num_batch, 1)

        vx_fin_vec = fin_state_ego[:, 2].reshape(self.num_batch, 1)

        vx_fin_vec = torch.clip(
            vx_fin_vec,
            self.v_min * torch.ones((self.num_batch, 1), device=device),
            self.v_max * torch.ones((self.num_batch, 1), device=device),
        )
        vy_fin_vec = torch.zeros((self.num_batch, 1), device=device)

        b_eq_x = torch.hstack(
            [
                x_init_vec,
                vx_init_vec,
                ax_init_vec,
                x_fin_vec,
                c_x_partial.reshape(self.num_batch, self.num_partial_xy),
            ]
        )
        b_eq_y = torch.hstack(
            [
                y_init_vec,
                vy_init_vec,
                ay_init_vec,
                y_fin_vec,
                vy_fin_vec,
                c_y_partial.reshape(self.num_batch, self.num_partial_xy),
            ]
        )

        return b_eq_x, b_eq_y

    def compute_mat_inv_init(self):
        cost_x = self.cost_smoothness
        cost_y = self.cost_smoothness

        cost_mat_x = torch.vstack(
            [
                torch.hstack([cost_x, self.A_eq_x.T]),
                torch.hstack(
                    [
                        self.A_eq_x,
                        torch.zeros(
                            (self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device
                        ),
                    ]
                ),
            ]
        )
        cost_mat_y = torch.vstack(
            [
                torch.hstack([cost_y, self.A_eq_y.T]),
                torch.hstack(
                    [
                        self.A_eq_y,
                        torch.zeros(
                            (self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device
                        ),
                    ]
                ),
            ]
        )

        cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
        cost_mat_inv_y = torch.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y

    def qp_layer_initialization(
        self, init_state_ego, fin_state_ego, c_x_partial, c_y_partial
    ):
        # Inverse Matrices
        cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_init()

        # Boundary conditions
        b_eq_x, b_eq_y = self.compute_boundary_layer(
            init_state_ego, fin_state_ego, c_x_partial, c_y_partial
        )

        lincost_x = torch.zeros([self.num_batch, self.nvar], device=device)
        lincost_y = torch.zeros([self.num_batch, self.nvar], device=device)

        sol_x = torch.mm(cost_mat_inv_x, torch.hstack([-lincost_x, b_eq_x]).T).T
        sol_y = torch.mm(cost_mat_inv_y, torch.hstack([-lincost_y, b_eq_y]).T).T

        c_x = sol_x[:, 0 : self.nvar]
        c_y = sol_y[:, 0 : self.nvar]

        # Solution
        primal_sol = torch.hstack([c_x, c_y])

        return primal_sol

    def compute_mat_inv_optim(self):
        cost_x = (
            self.cost_smoothness
            + self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc)
            + self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel)
            + self.rho_obs * torch.mm(self.A_obs.T, self.A_obs)
        )

        cost_y = (
            self.cost_smoothness
            + self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc)
            + self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel)
            + self.rho_obs * torch.mm(self.A_obs.T, self.A_obs)
            + self.rho_lane * torch.mm(self.A_lane.T, self.A_lane)
        )

        cost_mat_x = torch.vstack(
            [
                torch.hstack([cost_x, self.A_eq_x.T]),
                torch.hstack(
                    [
                        self.A_eq_x,
                        torch.zeros(
                            (self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device
                        ),
                    ]
                ),
            ]
        )
        cost_mat_y = torch.vstack(
            [
                torch.hstack([cost_y, self.A_eq_y.T]),
                torch.hstack(
                    [
                        self.A_eq_y,
                        torch.zeros(
                            (self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device
                        ),
                    ]
                ),
            ]
        )

        cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
        cost_mat_inv_y = torch.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y

    def compute_alph_d_init(
        self, primal_sol, lamda_x, lamda_y, x_obs_traj, y_obs_traj, y_ub, y_lb
    ):
        primal_sol_x = primal_sol[:, 0 : self.nvar]
        primal_sol_y = primal_sol[:, self.nvar : 2 * self.nvar]

        x = torch.mm(self.P, primal_sol_x.T).T
        xdot = torch.mm(self.Pdot, primal_sol_x.T).T
        xddot = torch.mm(self.Pddot, primal_sol_x.T).T

        y = torch.mm(self.P, primal_sol_y.T).T
        ydot = torch.mm(self.Pdot, primal_sol_y.T).T
        yddot = torch.mm(self.Pddot, primal_sol_y.T).T

        ########################################################## Obstacle update

        x_extend = torch.tile(x, (1, self.num_obs))
        y_extend = torch.tile(y, (1, self.num_obs))

        wc_alpha = x_extend - x_obs_traj
        ws_alpha = y_extend - y_obs_traj

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs)
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs)

        alpha_obs = torch.atan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
        c1_d = (
            1.0
            * self.rho_obs
            * (
                self.a_obs**2 * torch.cos(alpha_obs) ** 2
                + self.b_obs**2 * torch.sin(alpha_obs) ** 2
            )
        )
        c2_d = (
            1.0
            * self.rho_obs
            * (
                self.a_obs * wc_alpha * torch.cos(alpha_obs)
                + self.b_obs * ws_alpha * torch.sin(alpha_obs)
            )
        )

        d_temp = c2_d / c1_d
        d_obs = torch.maximum(
            torch.ones((self.num_batch, self.num * self.num_obs), device=device), d_temp
        )

        ###############################################33
        wc_alpha_vx = xdot
        ws_alpha_vy = ydot

        alpha_v = torch.atan2(ws_alpha_vy, wc_alpha_vx)

        c1_d_v = (
            1.0 * self.rho_ineq * (torch.cos(alpha_v) ** 2 + torch.sin(alpha_v) ** 2)
        )
        c2_d_v = (
            1.0
            * self.rho_ineq
            * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
        )

        d_temp_v = c2_d_v / c1_d_v

        d_v = torch.clip(
            d_temp_v,
            torch.tensor(self.v_min).to(device),
            torch.tensor(self.v_max).to(device),
        )

        #####################################################################

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = torch.atan2(ws_alpha_ay, wc_alpha_ax)

        c1_d_a = (
            1.0 * self.rho_ineq * (torch.cos(alpha_a) ** 2 + torch.sin(alpha_a) ** 2)
        )
        c2_d_a = (
            1.0
            * self.rho_ineq
            * (wc_alpha_ax * torch.cos(alpha_a) + ws_alpha_ay * torch.sin(alpha_a))
        )

        d_temp_a = c2_d_a / c1_d_a
        d_a = torch.clip(
            d_temp_a,
            torch.zeros((self.num_batch, self.num), device=device),
            torch.tensor(self.a_max).to(device),
        )

        ############################### Lane

        # Extending Dimension
        y_ub = y_ub[:, None]
        y_lb = y_lb[:, None]

        b_lane = torch.hstack(
            (
                y_ub * torch.ones((self.num_batch, self.num), device=device),
                -y_lb * torch.ones((self.num_batch, self.num), device=device),
            )
        )
        s_lane = torch.maximum(
            torch.zeros((self.num_batch, 2 * self.num), device=device),
            -torch.mm(self.A_lane, primal_sol_y.T).T + b_lane,
        )
        res_lane_vec = torch.mm(self.A_lane, primal_sol_y.T).T - b_lane + s_lane

        #########################################3

        res_ax_vec = xddot - d_a * torch.cos(alpha_a)
        res_ay_vec = yddot - d_a * torch.sin(alpha_a)

        res_vx_vec = xdot - d_v * torch.cos(alpha_v)
        res_vy_vec = ydot - d_v * torch.sin(alpha_v)

        res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
        res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)

        lamda_x = (
            lamda_x
            - self.rho_ineq * torch.mm(self.A_acc.T, res_ax_vec.T).T
            - self.rho_ineq * torch.mm(self.A_vel.T, res_vx_vec.T).T
            - self.rho_ineq * torch.mm(self.A_obs.T, res_x_obs_vec.T).T
        )

        lamda_y = (
            lamda_y
            - self.rho_ineq * torch.mm(self.A_acc.T, res_ay_vec.T).T
            - self.rho_ineq * torch.mm(self.A_vel.T, res_vy_vec.T).T
            - self.rho_ineq * torch.mm(self.A_obs.T, res_y_obs_vec.T).T
            - self.rho_lane * torch.mm(self.A_lane.T, res_lane_vec.T).T
        )

        return alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, alpha_obs, d_obs, s_lane

    def compute_x(
        self,
        cost_mat_inv_x,
        cost_mat_inv_y,
        b_eq_x,
        b_eq_y,
        x_obs_traj,
        y_obs_traj,
        lamda_x,
        lamda_y,
        alpha_obs,
        d_obs,
        alpha_a,
        d_a,
        alpha_v,
        d_v,
        y_ub,
        y_lb,
        s_lane,
    ):
        b_ax_ineq = d_a * torch.cos(alpha_a)
        b_ay_ineq = d_a * torch.sin(alpha_a)

        b_vx_ineq = d_v * torch.cos(alpha_v)
        b_vy_ineq = d_v * torch.sin(alpha_v)

        temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
        b_obs_x = x_obs_traj + temp_x_obs

        temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs
        b_obs_y = y_obs_traj + temp_y_obs

        # Extending Dimension
        y_ub = y_ub[:, None]
        y_lb = y_lb[:, None]

        b_lane = torch.hstack(
            [
                y_ub * torch.ones((self.num_batch, self.num), device=device),
                -y_lb * torch.ones((self.num_batch, self.num), device=device),
            ]
        )
        b_lane_aug = b_lane - s_lane

        lincost_x = (
            -lamda_x
            - self.rho_obs * torch.mm(self.A_obs.T, b_obs_x.T).T
            - self.rho_ineq * torch.mm(self.A_acc.T, b_ax_ineq.T).T
            - self.rho_obs * torch.mm(self.A_vel.T, b_vx_ineq.T).T
        )
        lincost_y = (
            -lamda_y
            - self.rho_obs * torch.mm(self.A_obs.T, b_obs_y.T).T
            - self.rho_ineq * torch.mm(self.A_acc.T, b_ay_ineq.T).T
            - self.rho_obs * torch.mm(self.A_vel.T, b_vy_ineq.T).T
            - self.rho_lane * torch.mm(self.A_lane.T, b_lane_aug.T).T
        )

        sol_x = torch.mm(cost_mat_inv_x, torch.hstack((-lincost_x, b_eq_x)).T).T
        sol_y = torch.mm(cost_mat_inv_y, torch.hstack((-lincost_y, b_eq_y)).T).T

        primal_sol_x = sol_x[:, 0 : self.nvar]
        primal_sol_y = sol_y[:, 0 : self.nvar]

        primal_sol = torch.hstack([primal_sol_x, primal_sol_y])

        return primal_sol

    def compute_alph_d(
        self,
        primal_sol,
        lamda_x,
        lamda_y,
        d_a,
        alpha_a,
        x_obs_traj,
        y_obs_traj,
        y_ub,
        y_lb,
    ):
        primal_sol_x = primal_sol[:, 0 : self.nvar]
        primal_sol_y = primal_sol[:, self.nvar : 2 * self.nvar]

        x = torch.mm(self.P, primal_sol_x.T).T
        xdot = torch.mm(self.Pdot, primal_sol_x.T).T
        xddot = torch.mm(self.Pddot, primal_sol_x.T).T

        y = torch.mm(self.P, primal_sol_y.T).T
        ydot = torch.mm(self.Pdot, primal_sol_y.T).T
        yddot = torch.mm(self.Pddot, primal_sol_y.T).T

        ########################################################## Obstacle update

        x_extend = torch.tile(x, (1, self.num_obs))
        y_extend = torch.tile(y, (1, self.num_obs))

        wc_alpha = x_extend - x_obs_traj
        ws_alpha = y_extend - y_obs_traj

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs)
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs)

        alpha_obs = torch.atan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
        c1_d = (
            1.0
            * self.rho_obs
            * (
                self.a_obs**2 * torch.cos(alpha_obs) ** 2
                + self.b_obs**2 * torch.sin(alpha_obs) ** 2
            )
        )
        c2_d = (
            1.0
            * self.rho_obs
            * (
                self.a_obs * wc_alpha * torch.cos(alpha_obs)
                + self.b_obs * ws_alpha * torch.sin(alpha_obs)
            )
        )
        d_temp = c2_d / c1_d
        d_obs = torch.maximum(
            torch.ones((self.num_batch, self.num * self.num_obs), device=device), d_temp
        )
        wc_alpha_vx = xdot
        ws_alpha_vy = ydot

        alpha_v = torch.atan2(ws_alpha_vy, wc_alpha_vx)

        alpha_v = torch.clip(
            alpha_v,
            -(torch.pi / 4) * torch.ones((self.num_batch, self.num), device=device),
            (torch.pi / 4) * torch.ones((self.num_batch, self.num), device=device),
        )

        c1_d_v = (
            1.0 * self.rho_ineq * (torch.cos(alpha_v) ** 2 + torch.sin(alpha_v) ** 2)
        )
        c2_d_v = (
            1.0
            * self.rho_ineq
            * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
        )

        d_temp_v = c2_d_v / c1_d_v

        temp = torch.abs(d_a * torch.abs(torch.sin(alpha_a - alpha_v)) / self.kappa_max)

        temp = torch.clip(
            temp,
            torch.tensor(self.v_min**2).to(device),
            torch.tensor(self.v_max**2).to(device),
        )

        v_min_aug = torch.pow(temp + 0.001, 0.5)

        v_min_aug = torch.clip(
            v_min_aug,
            self.v_min * torch.ones((self.num_batch, self.num), device=device),
            self.v_max * torch.ones((self.num_batch, self.num), device=device),
        )
        d_v = torch.clip(d_temp_v, v_min_aug, torch.tensor(self.v_max).to(device))

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = torch.atan2(ws_alpha_ay, wc_alpha_ax)

        c1_d_a = (
            1.0 * self.rho_ineq * (torch.cos(alpha_a) ** 2 + torch.sin(alpha_a) ** 2)
        )
        c2_d_a = (
            1.0
            * self.rho_ineq
            * (wc_alpha_ax * torch.cos(alpha_a) + ws_alpha_ay * torch.sin(alpha_a))
        )

        d_temp_a = c2_d_a / c1_d_a
        a_max_aug = (
            (d_v**2)
            * (self.kappa_max)
            / (torch.abs(torch.sin(alpha_a - alpha_v)) + 0.00001)
        )

        d_a = torch.clip(
            d_temp_a, torch.zeros((self.num_batch, self.num), device=device), a_max_aug
        )
        # Extending Dimension... lane constraints
        y_ub = y_ub[:, None]
        y_lb = y_lb[:, None]

        b_lane = torch.hstack(
            (
                y_ub * torch.ones((self.num_batch, self.num), device=device),
                -y_lb * torch.ones((self.num_batch, self.num), device=device),
            )
        )
        s_lane = torch.maximum(
            torch.zeros((self.num_batch, 2 * self.num), device=device),
            -torch.mm(self.A_lane, primal_sol_y.T).T + b_lane,
        )
        res_lane_vec = torch.mm(self.A_lane, primal_sol_y.T).T - b_lane + s_lane

        curvature = (d_a * torch.abs(torch.sin(alpha_a - alpha_v))) / (
            d_v**2 + 0.00001
        )

        curvature_der = torch.diff(curvature, dim=1)
        curvature_dder = torch.diff(curvature_der, dim=1)

        curvature_penalty = torch.maximum(
            curvature - self.kappa_max,
            torch.zeros((self.num_batch, self.num), device=device),
        )

        res_ax_vec = xddot - d_a * torch.cos(alpha_a)
        res_ay_vec = yddot - d_a * torch.sin(alpha_a)

        res_vx_vec = xdot - d_v * torch.cos(alpha_v)
        res_vy_vec = ydot - d_v * torch.sin(alpha_v)

        res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
        res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)

        res_vel_vec = torch.hstack([res_vx_vec, res_vy_vec])
        res_acc_vec = torch.hstack([res_ax_vec, res_ay_vec])
        res_obs_vec = torch.hstack([res_x_obs_vec, res_y_obs_vec])

        ###### primal constraint residuals

        res_norm_batch_temp = (
            torch.linalg.norm(res_acc_vec, dim=1)
            + torch.linalg.norm(res_vel_vec, dim=1)
            + torch.linalg.norm(curvature_penalty, axis=1)
            + torch.linalg.norm(res_obs_vec, axis=1)
            + torch.linalg.norm(res_lane_vec, axis=1)
        )
        curvature_der_smoothness = torch.linalg.norm(curvature_der, dim=1)
        curvature_dder_smoothness = torch.linalg.norm(curvature_dder, dim=1)
        res_norm_batch = res_norm_batch_temp

        lamda_x = (
            lamda_x
            - self.rho_ineq * torch.mm(self.A_acc.T, res_ax_vec.T).T
            - self.rho_ineq * torch.mm(self.A_vel.T, res_vx_vec.T).T
            - self.rho_obs * torch.mm(self.A_obs.T, res_x_obs_vec.T).T
        )

        lamda_y = (
            lamda_y
            - self.rho_ineq * torch.mm(self.A_acc.T, res_ay_vec.T).T
            - self.rho_ineq * torch.mm(self.A_vel.T, res_vy_vec.T).T
            - self.rho_obs * torch.mm(self.A_obs.T, res_y_obs_vec.T).T
            - self.rho_lane * torch.mm(self.A_lane.T, res_lane_vec.T).T
        )

        return (
            alpha_a,
            d_a,
            lamda_x,
            lamda_y,
            alpha_v,
            d_v,
            res_norm_batch,
            alpha_obs,
            d_obs,
            s_lane,
        )

    def custom_forward(
        self,
        init_state_ego,
        fin_state_ego,
        lamda_x,
        lamda_y,
        c_x_partial,
        c_y_partial,
        primal_sol,
        x_obs_traj,
        y_obs_traj,
        y_ub,
        y_lb,
    ):
        b_eq_x, b_eq_y = self.compute_boundary_layer(
            init_state_ego, fin_state_ego, c_x_partial, c_y_partial
        )

        cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_optim()

        accumulated_res_primal = 0

        (
            alpha_a,
            d_a,
            lamda_x,
            lamda_y,
            alpha_v,
            d_v,
            alpha_obs,
            d_obs,
            s_lane,
        ) = self.compute_alph_d_init(
            primal_sol, lamda_x, lamda_y, x_obs_traj, y_obs_traj, y_ub, y_lb
        )

        for i in range(0, self.maxiter):
            primal_sol = self.compute_x(
                cost_mat_inv_x,
                cost_mat_inv_y,
                b_eq_x,
                b_eq_y,
                x_obs_traj,
                y_obs_traj,
                lamda_x,
                lamda_y,
                alpha_obs,
                d_obs,
                alpha_a,
                d_a,
                alpha_v,
                d_v,
                y_ub,
                y_lb,
                s_lane,
            )

            (
                alpha_a,
                d_a,
                lamda_x,
                lamda_y,
                alpha_v,
                d_v,
                res_norm_batch,
                alpha_obs,
                d_obs,
                s_lane,
            ) = self.compute_alph_d(
                primal_sol,
                lamda_x,
                lamda_y,
                d_a,
                alpha_a,
                x_obs_traj,
                y_obs_traj,
                y_ub,
                y_lb,
            )

            accumulated_res_primal += res_norm_batch

        accumulated_res_primal = accumulated_res_primal / self.maxiter

        return primal_sol, accumulated_res_primal

    def decoder_function(
        self,
        inp_norm,
        init_state_ego,
        obstacle_state_scaled,
        x_obs_traj,
        y_obs_traj,
        y_ub,
        y_lb,
    ):
        pcd_features = self.point_net(obstacle_state_scaled)

        inp_features = torch.cat([pcd_features, inp_norm], dim=1)

        neural_output_batch = self.mlp(inp_features)

        lamda_init = neural_output_batch[:, 0 : self.num_lamda_input]
        partial_sol = neural_output_batch[
            :, self.num_lamda_input : self.num_lamda_input + self.num_partial_solution
        ]
        fin_state_ego = neural_output_batch[
            :,
            self.num_lamda_input
            + self.num_partial_solution : self.num_lamda_input
            + self.num_partial_solution
            + 4,
        ]

        lamda_x = lamda_init[:, 0 : self.nvar]
        lamda_y = lamda_init[:, self.nvar : 2 * self.nvar]

        c_x_partial = partial_sol[:, 0 : self.num_partial_xy]
        c_y_partial = partial_sol[:, self.num_partial_xy : self.num_partial_solution]

        primal_sol = self.qp_layer_initialization(
            init_state_ego, fin_state_ego, c_x_partial, c_y_partial
        )

        primal_sol, accumulated_res_primal = self.custom_forward(
            init_state_ego,
            fin_state_ego,
            lamda_x,
            lamda_y,
            c_x_partial,
            c_y_partial,
            primal_sol,
            x_obs_traj,
            y_obs_traj,
            y_ub,
            y_lb,
        )

        return primal_sol, accumulated_res_primal
    
    
    def smoothness_cost(self, tensor_x, tensor_y):
        dx = torch.diff(tensor_x, dim=1)
        dy = torch.diff(tensor_y, dim=1)
        out = torch.sum(dx**2 + dy**2, dim=0)
        return torch.sum(out)
    

    def smoothness_cost(self, tensor_x, tensor_y):
        dx = torch.diff(tensor_x, dim=1)
        dy = torch.diff(tensor_y, dim=1)
        out = torch.sum(dx**2 + dy**2, dim=0)
        return torch.sum(out)

    def ss_loss(self, accumulated_res_primal, predict_traj, desired_goal, predict_acc):
        predict_acc_norm = torch.linalg.norm(predict_acc, dim=1)
        primal_loss = 0.5 * (torch.mean(accumulated_res_primal))

        predict_traj_x = predict_traj[:, 0 : self.num]
        predict_traj_y = predict_traj[:, self.num : 2 * self.num]

        predict_goal = torch.vstack([predict_traj_x[:, -1], predict_traj_y[:, -1]]).T
        acc_loss = 0.5 * (torch.mean(predict_acc_norm))

        goal_loss = self.rcl_loss(predict_goal, desired_goal)

        smoothness_loss = self.smoothness_cost(predict_traj_x, predict_traj_y)

        w = cfg.planner().train.cost_weights

        loss = w[0] * primal_loss + w[1] * goal_loss + w[2] * smoothness_loss

        return loss, primal_loss, goal_loss

    def forward(
        self,
        inp,
        init_state_ego,
        obstacle_state,
        closest_obs,
        v_obs,
        y_ub,
        y_lb,
        P_diag,
        Pddot_diag,
    ):
        inp_norm = (inp - inp.mean()) / inp.std()

        x_obs_traj, y_obs_traj = self.compute_obs_trajectories(closest_obs, v_obs)

        min_pcd, max_pcd = obstacle_state.min(), obstacle_state.max()
        obstacle_state_scaled = (obstacle_state - min_pcd) / (max_pcd - min_pcd)

        # Decode y
        primal_sol, accumulated_res_primal = self.decoder_function(
            inp_norm,
            init_state_ego,
            obstacle_state_scaled,
            x_obs_traj,
            y_obs_traj,
            y_ub,
            y_lb,
        )

        predict_traj = (P_diag @ primal_sol.T).T
        predict_acc = (Pddot_diag @ primal_sol.T).T

        return primal_sol, accumulated_res_primal, predict_traj, predict_acc