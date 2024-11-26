import numpy as np
from scipy.interpolate import CubicSpline


def path_spline(x_path, y_path):
    x_diff = np.diff(x_path)
    y_diff = np.diff(y_path)

    phi = np.unwrap(np.arctan2(y_diff, x_diff))
    phi_init = phi[0]
    phi = np.hstack((phi_init, phi))

    arc = np.cumsum(np.sqrt(x_diff**2 + y_diff**2))
    arc_length = arc[-1]

    arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])

    cs_x_path = CubicSpline(arc_vec, x_path)
    cs_y_path = CubicSpline(arc_vec, y_path)
    cs_phi_path = CubicSpline(arc_vec, phi)

    return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec


def compute_path_parameters(x_path, y_path):
    Fx_dot = np.diff(x_path)
    Fy_dot = np.diff(y_path)

    Fx_dot = np.hstack((Fx_dot[0], Fx_dot))

    Fy_dot = np.hstack((Fy_dot[0], Fy_dot))

    Fx_ddot = np.diff(Fx_dot)
    Fy_ddot = np.diff(Fy_dot)

    Fx_ddot = np.hstack((Fx_ddot[0], Fx_ddot))

    Fy_ddot = np.hstack((Fy_ddot[0], Fy_ddot))

    arc = np.cumsum(np.sqrt(Fx_dot**2 + Fy_dot**2))
    arc_vec = np.hstack((0, arc[0:-1]))

    arc_length = arc_vec[-1]

    kappa = (Fy_ddot * Fx_dot - Fx_ddot * Fy_dot) / (
        ((Fx_dot**2 + Fy_dot**2) ** (1.5)) + 1e-06
    )

    return Fx_dot, Fy_dot, Fx_ddot, Fy_ddot, arc_vec, kappa, arc_length


def global_to_frenet_obs(
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
    bev_res,
):
    v_obs = np.sqrt(vx_obs**2 + vy_obs**2)

    idx_closest_point = np.argmin(
        np.sqrt((x_path - x_obs) ** 2 + (y_path - y_obs) ** 2)
    )
    closest_point_x, closest_point_y = (
        x_path[idx_closest_point],
        y_path[idx_closest_point],
    )

    x_init = arc_vec[idx_closest_point]

    kappa_interp = np.interp(x_init, arc_vec, kappa)

    Fx_dot_interp = np.interp(x_init, arc_vec, Fx_dot)
    Fy_dot_interp = np.interp(x_init, arc_vec, Fy_dot)

    normal_x = -Fy_dot_interp
    normal_y = Fx_dot_interp

    normal = np.hstack((normal_x, normal_y))
    vec = np.asarray([x_obs - closest_point_x, y_obs - closest_point_y])
    y_init = (1 / (np.linalg.norm(normal))) * np.dot(normal, vec)

    psi_init = psi_obs - np.arctan2(Fy_dot_interp, Fx_dot_interp)
    psi_init = np.arctan2(np.sin(psi_init), np.cos(psi_init))

    vx_init = v_obs * np.cos(psi_init) / (1 - y_init * kappa_interp)
    vy_init = v_obs * np.sin(psi_init)

    return x_init, y_init, vx_init, vy_init, psi_init


def global_to_frenet(
    x_path, y_path, initial_state, arc_vec, Fx_dot, Fy_dot, kappa, bev_res
):
    (
        x_global_init,
        y_global_init,
        v_global_init,
        psi_global_init,
    ) = initial_state

    vdot_global_init = 0.0
    psidot_global_init = 0.0

    idx_closest_point = np.argmin(
        np.sqrt((x_path - x_global_init) ** 2 + (y_path - y_global_init) ** 2)
    )
    closest_point_x, closest_point_y = (
        x_path[idx_closest_point],
        y_path[idx_closest_point],
    )

    x_init = arc_vec[idx_closest_point]

    kappa_interp = np.interp(x_init, arc_vec, kappa)
    kappa_pert = np.interp(x_init + 0.001, arc_vec, kappa)

    kappa_prime = (kappa_pert - kappa_interp) / 0.001

    Fx_dot_interp = np.interp(x_init, arc_vec, Fx_dot)
    Fy_dot_interp = np.interp(x_init, arc_vec, Fy_dot)

    normal_x = -Fy_dot_interp
    normal_y = Fx_dot_interp

    normal = np.hstack((normal_x, normal_y))
    vec = np.asarray([x_global_init - closest_point_x, y_global_init - closest_point_y])
    y_init = (1 / (np.linalg.norm(normal))) * np.dot(normal, vec)

    psi_init = psi_global_init - np.arctan2(Fy_dot_interp, Fx_dot_interp)
    psi_init = np.arctan2(np.sin(psi_init), np.cos(psi_init))

    vx_init = v_global_init * np.cos(psi_init) / (1 - y_init * kappa_interp)
    vy_init = v_global_init * np.sin(psi_init)

    psidot_init = psidot_global_init - kappa_interp * vx_init

    ay_init = (
        vdot_global_init * np.sin(psi_init)
        + v_global_init * np.cos(psi_init) * psidot_init
    )

    ax_init_part_1 = (
        vdot_global_init * np.cos(psi_init)
        - v_global_init * np.sin(psi_init) * psidot_init
    )
    ax_init_part_2 = -vy_init * kappa_interp - y_init * kappa_prime * vx_init

    ax_init = (
        ax_init_part_1 * (1 - y_init * kappa_interp)
        - (v_global_init * np.cos(psi_init)) * (ax_init_part_2)
    ) / ((1 - y_init * kappa_interp) ** 2)

    psi_fin = 0.0

    return (
        x_init,
        y_init,
        vx_init,
        vy_init,
        ax_init,
        ay_init,
        psi_init,
        psi_fin,
        psidot_init,
    )


def frenet_to_global(y_frenet, ref_x, ref_y, dx_by_ds, dy_by_ds):
    normal_x = -1 * dy_by_ds
    normal_y = dx_by_ds

    norm_vec = np.sqrt(normal_x**2 + normal_y**2)
    normal_unit_x = (1 / norm_vec) * normal_x
    normal_unit_y = (1 / norm_vec) * normal_y

    global_x = ref_x + y_frenet * normal_unit_x
    global_y = ref_y + y_frenet * normal_unit_y

    psi_global = np.arctan2(np.diff(global_y), np.diff(global_x))

    return global_x, global_y, psi_global


def extrapolate_path(path, extend=50, res=0.25):
    x_path, y_path = path[:, 0], path[:, 1]
    cs_x_path, cs_y_path, _, arc_length, arc_vec = path_spline(x_path, y_path)

    num_p = int(arc_length / res)
    arc_vec = np.linspace(0, arc_length, num_p)

    x_path = cs_x_path(arc_vec)
    y_path = cs_y_path(arc_vec)

    num_p = int(extend / res)

    m = (y_path[-1] - y_path[-2]) / (x_path[-1] - x_path[-2])

    if y_path[-1] > y_path[-2]:
        y_linspace_forward = np.linspace(y_path[-1], y_path[-1] + extend, num_p)
    else:
        y_linspace_forward = np.linspace(y_path[-1], y_path[-1] - extend, num_p)

    intercept = y_path[-1] - m * x_path[-1]
    x_linspace_forward = (y_linspace_forward - intercept) / m

    m = (y_path[1] - y_path[0]) / (x_path[1] - x_path[0])

    if y_path[1] > y_path[0]:
        y_linspace_backward = np.linspace(y_path[0] - extend, y_path[0], num_p)
    else:
        y_linspace_backward = np.linspace(y_path[0] + extend, y_path[0], num_p)

    intercept = y_path[0] - m * x_path[0]
    x_linspace_backward = (y_linspace_backward - intercept) / m

    y_path = np.hstack((y_linspace_backward, y_path, y_linspace_forward))
    x_path = np.hstack((x_linspace_backward, x_path, x_linspace_forward))

    return x_path, y_path