import matplotlib.pyplot as plt
import matplotlib as m

m.use("Agg")
plt.style.use("ggplot")

from navbev.config import Configs as cfg


def plot_inf_result(
    front_cam,
    frenet_traj,
    ego_state,
    desired_goal,
    obstacles,
    iter,
    batch,
    save=False,
    show_by_frame=False,
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(front_cam)
    ax1.grid(False)

    ax2.plot(frenet_traj[0], frenet_traj[1])

    ax2.axhline(y=-1.75, color="k", label="Lane LB: -1.75m")
    ax2.axhline(y=5.25, color="k", label="Lane UB: 5.25m")
    ax2.axhline(y=ego_state[1], color="orange", linestyle="--", label="Ref Path")

    ax2.scatter(
        desired_goal[0], 
        desired_goal[1], 
        marker="X", 
        s=100, 
        c="r", 
        label="GLC Goal"
    )

    if obstacles is not None:
        ax2.scatter(
            obstacles[: cfg.planner().n_closest_obstacle],
            obstacles[cfg.planner().n_closest_obstacle :],
            s=100,
            c='b',
        )

    ax2.scatter(ego_state[0], ego_state[1], s=200, c="g")

    desired_goal = desired_goal.numpy()
    ax2.set_title(f'Goal: {desired_goal} \nDist: {(desired_goal[0] - ego_state[0]):.2f} m')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-20, 20)
    ax2.legend()

    if save:
        plt.savefig(
            f"{cfg.ROOT}/{cfg.globals().paths.output_path}/out_{iter}_{batch}.png",
            dpi=100,
        )

    if show_by_frame:
        plt.show()
    plt.close("all")


def plot_loss_curves(train_loss, primal_loss, goal_loss, save=False):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

    ax1.plot(train_loss)
    ax1.set_title("Train Loss")

    ax2.plot(primal_loss)
    ax2.set_title("Primal Loss")

    ax3.plot(goal_loss)
    ax3.set_title("Goal Loss")

    if save:
        plt.savefig(
            f"{cfg.ROOT}/{cfg.globals().paths.output_path}/loss_curves.png", dpi=100
        )