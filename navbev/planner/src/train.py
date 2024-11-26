import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from navbev.config import Configs as cfg
from navbev.planner.src.dataloader import TrajDataset, VehicleClass
from navbev.planner.src.dataset import LoadDataset
from navbev.planner.src.helper import DataProcessor, Frenetify, init_planner
from navbev.planner.src.model import PointNet, diff_planner_ad, mlp_model_planner
from navbev.planner.utils.plot_utils import plot_loss_curves

torch.cuda.empty_cache()

device = cfg.globals().device

carla_data_path = f"{cfg.ROOT}/dataset/planner_data/pickles"
goals_path = f"{cfg.ROOT}/dataset/planner_data/goals_centroid.npy"
goals = np.load(goals_path)

dataset = LoadDataset.get_carla_data(
    path=carla_data_path, cfg=cfg, goals=goals, normalize_goal=False, start=0
)

data = DataProcessor(
    ego_state=dataset.ego_state,
    goal=dataset.glc_goal,
    traffic=dataset.traffic,
    gt_traj=dataset.gt_traj,
    fcam=dataset.fcam,
    bev_ref=dataset.bev_ref,
    ego_lane=dataset.ego_lane,
)

fre = Frenetify(
    ego_state=data.ego_state,
    obstacle_state=data.obstacle_state,
    ego_lane=dataset.ego_lane,
    goal=data.desired_goal,
    bev_size=cfg.carla().bev_size,
    bev_res=cfg.carla().bev_res,
    lane_width=cfg.carla().lane_width,
)

fre.goal_frenet[:, -1] *= -1
fre.obstacle_state[:, 2, :] *= -1
fre.obstacle_state_frenet_xy[:, fre.obstacle_state.shape[-1]:] *= -1

input_vec = np.hstack((fre.ego_state_frenet, fre.goal_frenet, fre.lane_bounds))

traj_dataset = TrajDataset(
    inp=input_vec,
    init_state_ego=fre.ego_state_frenet,
    goal_des=fre.goal_frenet,
    closest_obs=fre.obstacle_state_frenet_xy,
    v_obs=fre.obstacle_state_frenet_v,
    obstacle_state=fre.obstacle_state,
    y_lb=fre.lane_bounds[:, 0],
    y_ub=fre.lane_bounds[:, 1],
    fimg=data.front_cam,
    bev=data.bev_ref,
    ego_lane=data.ego_lane,
    ego_x=fre.ego_xs,
)

train_loader = DataLoader(
    dataset=traj_dataset,
    batch_size=cfg.planner().train.batch_size,
    shuffle=False,
    num_workers=cfg.planner().train.n_workers,
    drop_last=True,
)

P_np, P, Pdot, Pddot, P_diag, Pddot_diag, t_fin, num_lower, num_upper = init_planner(
    cfg
)

w = cfg.planner().mlp.out_weight
mlp_inp_dim = 8 + cfg.planner().pointnet.out_dim
mlp_hidden_dim = cfg.planner().mlp.hidden_dim * 2
mlp_out_dim = w[0] * np.shape(P_np)[1] + w[1] * (num_upper - num_lower) + w[2]

mlp_planner = mlp_model_planner(
    inp_dim=mlp_inp_dim, hidden_dim=mlp_hidden_dim, out_dim=mlp_out_dim
)

pointnet = PointNet(
    inp_channel=cfg.planner().pointnet.input_dim,
    emb_dims=cfg.planner().pointnet.hidden_dim,
    output_channels=cfg.planner().pointnet.out_dim,
)

print("\nInitiating...")

planner_model = diff_planner_ad(
    P=P,
    Pdot=Pdot,
    Pddot=Pddot,
    mlp=mlp_planner,
    point_net=pointnet,
    num_batch=cfg.planner().train.batch_size,
    t_fin=cfg.planner().horizon,
    num_obs=cfg.planner().n_closest_obstacle,
    num_lower=num_lower,
    num_upper=num_upper,
).to(device)


avg_train_loss, avg_aug_loss_primal, avg_goal_loss = [], [], []

optimizer = torch.optim.AdamW(
    planner_model.parameters(),
    lr=cfg.planner().train.optim_lr,
    weight_decay=cfg.planner().train.optim_wd,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=cfg.planner().train.schd_step
)

step = 0

# Training Loop
planner_model.train()
for epoch in range(cfg.planner().train.n_epochs):
    losses_train, aug_losses_primal, goal_losses = [], [], []

    for batch_idx, data in enumerate(
        track(
            train_loader,
            description=f"Epoch- {epoch+1}/{cfg.planner().train.n_epochs}:",
        )
    ):
        (
            inp,
            init_state_ego,
            goal_des,
            closest_obs,
            v_obs,
            obstacle_state,
            y_lb,
            y_ub,
            _,
            _,
            ego_lane,
            _,
        ) = data
        inp, init_state_ego, goal_des = map(
            lambda x: x.to(device), [inp, init_state_ego, goal_des]
        )
        closest_obs, v_obs, obstacle_state, y_lb, y_ub = map(
            lambda x: x.to(device), [closest_obs, v_obs, obstacle_state, y_lb, y_ub]
        )

        primal_sol, accumulated_res_primal, predict_traj, predict_acc = planner_model(
            inp,
            init_state_ego,
            obstacle_state,
            closest_obs,
            v_obs,
            y_ub,
            y_lb,
            P_diag,
            Pddot_diag,
        )

        loss, aug_loss_primal, goal_loss = planner_model.ss_loss(
            accumulated_res_primal, predict_traj, goal_des, predict_acc
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.detach().cpu().numpy())
        aug_losses_primal.append(aug_loss_primal.detach().cpu().numpy())
        goal_losses.append(goal_loss.detach().cpu().numpy())

    if (epoch + 1) % 5 == 0:
        print(
            f"Train Loss: {np.average(losses_train):.3f} |\
            Primal Loss: {np.average(aug_losses_primal):.3f} |\
            Goal loss: {np.average(goal_losses):.3f}\n"
        )

    step += 0.15
    scheduler.step()
    avg_train_loss.append(np.average(losses_train))
    avg_aug_loss_primal.append(np.average(aug_losses_primal))
    avg_goal_loss.append(np.average(goal_losses))


plot_loss_curves(
    train_loss=avg_train_loss,
    primal_loss=avg_aug_loss_primal,
    goal_loss=avg_goal_loss,
    save=True,
)


print(f"\nModel Saved to {cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/planner/")

torch.save(
    planner_model.state_dict(),
    f"{cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/planner/planner_nobev_{cfg.planner().train.n_epochs}.pth",
)