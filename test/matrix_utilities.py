# -*- coding: utf-8 -*-
"""
Python module for matrix manipulation.
Name: matrix_utilities.py
"""
__all__ = [
    'slice_3ds', 'level_plain', 'align_row_diff_median', 'topo_extent',
    'get_line_r', 'get_line_g', 'coor_to_idx', 'idx_to_coor', 'get_idx_r',
    'is_in_poly'
]

from typing import Any, Union
# from matplotlib.pyplot import grid
import numpy as np
from scipy.optimize import curve_fit


def slice_3ds(f: Any, bias: float = 0, full: bool = False):
    dim = f.header['Grid dim']
    points = f.header['Points']

    sweep = np.linspace(round(f.Parameters[0][0], 3),
                        round(f.Parameters[0][1], 3), f.header['Points'])

    # z
    z = np.zeros((dim[1], dim[0]))
    for row in range(dim[1]):
        for col in range(dim[0]):
            z[row][col] = f.Parameters[(dim[1] - row - 1) * dim[0] + col][4]

    # di/dv
    if full:
        sliced_data = np.zeros((points, dim[1], dim[0]))
        for i in range(points):  # points of sweep for each position
            for row in range(dim[1]):  # number of row for matplotlib
                for col in range(dim[0]):  # number of column for matplotlib
                    sliced_data[i][row][col] = (
                        f.data[(dim[1] - row - 1) * dim[0] + col][1][i] +
                        f.data[(dim[1] - row - 1) * dim[0] +
                               col][4][i]) / 2  # re-distribution of data
        return sweep, sliced_data, z
    else:
        sliced_data = np.zeros((dim[1], dim[0]))
        bias_idx = find_nearest(sweep, bias)
        for row in range(dim[1]):
            for col in range(dim[0]):
                sliced_data[row][col] = (
                    f.data[(dim[1] - row - 1) * dim[0] + col][1][bias_idx] +
                    f.data[(dim[1] - row - 1) * dim[0] + col][4][bias_idx]) / 2
        return sweep[bias_idx], sliced_data, z


def find_nearest(array: np.ndarray, value: float):
    """[summary]

    Args:
        array (np.ndarray): An array of numbers.
        value (float): A specified value.

    Returns:
        int: serial number of the element, which is the nearest value.
    """
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx


def level_plain(raw_topo: np.ndarray) -> np.ndarray:
    """Level data by mean plane subtraction.

    Args:
        raw_topo (np.ndarray): Raw data of 2-dimensional topographic image.

    Returns:
        np.ndarray: Linear-background-removed 2-dimensional topographic image. 
    """

    vector: list = []
    x: np.ndarray = np.zeros((raw_topo.ndim, raw_topo.size))
    topo_plain: np.ndarray = np.zeros(raw_topo.shape)
    for i in range(raw_topo.ndim):
        vector.append(np.linspace(0, raw_topo.shape[i] - 1, raw_topo.shape[i]))
    mesh_grid: list[np.ndarray] = np.meshgrid(*vector)
    for i in range(raw_topo.ndim):
        x[i] = mesh_grid[i].flatten()

    if raw_topo.ndim == 2:
        parameter = curve_fit(linear_2d, x, raw_topo.flatten())
        topo_plain_flatten = raw_topo.flatten() - linear_2d(x, *parameter[0])
        topo_plain = topo_plain_flatten.reshape(raw_topo.shape)
        topo_plain -= topo_plain.min()
    return topo_plain


def align_row_diff_median(raw_topo: np.ndarray) -> np.ndarray:
    """_summary_
    Aligns the topography by shifting each row by the median of the row minus the median of the first row.
    This is done to remove the effect of the topography of the sample.
    Parameters
    ----------
    raw_topo : np.ndarray
        The raw topography of the sample.
    Returns
    -------
    np.ndarray
        The aligned topography of the sample.
    """
    topo_shift = np.zeros(raw_topo.shape)
    row_median = np.median(raw_topo, axis=1)
    for i in range(len(row_median)):
        topo_shift[i] = raw_topo[i] - (row_median[i] - row_median[0])
    topo_shift -= topo_shift.min()
    return topo_shift


# TODO: Create multi-dimensional linear function
def linear_2d(x: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return a * x[0] + b * x[1] + d


def get_line_r(
        init: tuple = (0, 0),
        dest: tuple = (0, 0),
        sampling: int = 1000,
        frame_size: tuple = (30, 30),  # real-space(nm)
        grid_size: tuple = (50, 50)  # grid-space
):
    """_summary_

    Args:
        init (tuple, optional): _description_. Defaults to (0, 0).
        dest (tuple, optional): _description_. Defaults to (0, 0).
        sampling (int, optional): _description_. Defaults to 1000.
        frame_size (tuple, optional): _description_. Defaults to (30, 30).

    Returns:
        _type_: _description_
    """
    coor_x = np.linspace(init[0], dest[0], sampling)
    coor_y = np.linspace(init[1], dest[1], sampling)
    idx = []
    coor_grid = []
    for i in range(sampling):
        order = coor_to_idx((coor_x[i], coor_y[i]),
                            frame_size=frame_size,
                            grid_size=grid_size,
                            mode='r')
        if order not in idx:
            idx.append(order)
    for i in range(len(idx)):
        coor_grid.append(
            idx_to_coor(idx[i],
                        frame_size=frame_size,
                        grid_size=grid_size,
                        mode='g'))
    return idx, coor_grid


def get_line_g(init: 'tuple[int, int]' = (0, 0),
               dest: 'tuple[int, int]' = (0, 0),
               frame_size: 'tuple[float, float]' = (30, 30),
               grid_size: 'tuple[int, int]' = (50, 50)):
    """_summary_

    Args:
        init (tuple[int, int], optional): _description_. Defaults to (0, 0).
        dest (tuple[int, int], optional): _description_. Defaults to (0, 0).
        frame_size (tuple[float, float], optional): _description_. Defaults to (30, 30).
        grid_size (tuple[int, int], optional): _description_. Defaults to (50, 50).

    Returns:
        _type_: _description_
    """
    coor_x = np.linspace(init[0], dest[0], 1000)
    coor_y = np.linspace(init[1], dest[1], 1000)
    coor = []
    idx = []
    for i in range(1000):
        order = coor_to_idx((round(coor_x[i]), round(coor_y[i])),
                            grid_size=grid_size,
                            mode='g')
        if order not in idx:
            idx.append(order)
    for i in range(len(idx)):
        coor.append(
            idx_to_coor(idx[i],
                        frame_size=frame_size,
                        grid_size=grid_size,
                        mode='r'))
    return idx, coor


def topo_extent(header: dict) -> 'tuple[float, float, float, float]':
    """
    Calculate position of topograph.

    Parameter
    ---------
    header : reformed header of .sxm

    Return
    ------
    position tuple (left[X], right[X], bottom[Y], top[Y])
    """

    center_X = header['Scan>Scanfield'][0]
    center_Y = header['Scan>Scanfield'][1]
    range_X = header['Scan>Scanfield'][2]
    range_Y = header['Scan>Scanfield'][3]
    return (center_X - range_X / 2, center_X + range_X / 2,
            center_Y - range_Y / 2, center_Y + range_Y / 2)


def cal_length(coor: tuple = (0, 0), o: tuple = (0, 0)) -> float:
    return np.sqrt((coor[0] - o[0])**2 + (coor[0] - o[0])**2)


def coor_to_idx(coor: tuple = (0, 0),
                frame_size: 'tuple[float, float]' = (30, 30),
                grid_size: 'tuple[int, int]' = (50, 50),
                mode: str = 'r') -> int:
    idx: int = 0
    if mode == 'r':
        x_interval = frame_size[0] / grid_size[0]
        y_interval = frame_size[1] / grid_size[1]
        idx = int(coor[1] // y_interval * grid_size[0] + coor[0] // x_interval)
    elif mode == 'g':
        idx = int((grid_size[1] - coor[0] - 1) * grid_size[0] + coor[1])
    return idx


def idx_to_coor(idx: int = 0,
                frame_size: 'tuple[float, float]' = (30, 30),
                grid_size: 'tuple[int, int]' = (50, 50),
                mode: str = 'r') -> tuple:
    coor: Union[tuple[float, float], tuple[int, int]] = (0, 0)
    if mode == 'r':
        x_interval = frame_size[0] / grid_size[0]
        y_interval = frame_size[1] / grid_size[1]
        coor = (round(x_interval * (1 / 2 + idx % grid_size[0]),
                      2), round(y_interval * (idx // grid_size[0] + 1 / 2), 2))
    elif mode == 'g':
        coor = (grid_size[1] - idx // grid_size[0] - 1, idx % grid_size[0])
    return coor


def get_idx_r(input: list,
              frame_size: tuple = (30, 30),
              grid_size: tuple = (50, 50)):
    idx = []
    coor = []
    for i in range(len(input)):
        order = coor_to_idx(input[i],
                            frame_size=frame_size,
                            grid_size=grid_size,
                            mode='r')
        if order not in idx:
            idx.append(order)
    for i in range(len(idx)):
        coor.append(
            idx_to_coor(idx[i],
                        frame_size=frame_size,
                        grid_size=grid_size,
                        mode='g'))
    return idx, coor


def __sort_coordinates(poly):
    list_of_xy_coords = np.array(poly)
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]


def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    poly_sorted = __sort_coordinates(poly).tolist()
    is_in = False
    for i, corner in enumerate(poly_sorted):
        next_i = i + 1 if i + 1 < len(poly_sorted) else 0
        x1, y1 = corner
        x2, y2 = poly_sorted[next_i]
        if (x1 == px and y1 == py) or (x2 == px
                                       and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


# def arb_to_S(input_3ds) -> np.ndarray:
#     current_fwd = input_3ds.data[:, 0]
#     current_bwd =

#     return
