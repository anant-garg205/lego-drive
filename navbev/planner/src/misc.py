import numpy as np
from scipy.interpolate import interp1d


def interpolate(array, n_pts=100):
    x = np.linspace(0, 1, len(array))
    x_ = np.linspace(0, 1, n_pts)

    arr_interp = np.zeros((n_pts, array.shape[1]))

    for col in range(array.shape[1]):
        interp_func = interp1d(
            x, array[:, col], kind="linear", fill_value="extrapolate"
        )
        arr_interp[:, col] = interp_func(x_)
    return arr_interp


def normalize(theta):
    return np.rad2deg((theta + np.pi) % 2 * np.pi - np.pi)


def restack_to_1d(array, abs_bool=False):
    if abs_bool:
        return np.vstack(
            [np.concatenate((np.abs(point[:, 0]) + 10, point[:, 1])) for point in array]
        )
    return np.vstack([np.concatenate((point[:, 0], point[:, 1])) for point in array])


def merge_1d(arr0, arr1):
    return np.stack((arr0, arr1), axis=1)