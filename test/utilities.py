# -*- coding: utf-8 -*-
__all__ = ['slice_3ds']


from .nanonis_loader import loader
import numpy as np


def slice_3ds(fname: str = '', bias: float = 0, full: bool = False):
    f = loader.read_file(fname)
    dim = f.header['Grid dim']
    points = f.header['Points']

    sweep = np.linspace(round(f.Parameters[0][0], 3),
                        round(f.Parameters[0][1], 3), f.header['Points'])

    # z
    z = np.zeros(dim)
    for row in range(dim[0]):
        for col in range(dim[1]):
            z[row][col] = f.Parameters[(dim[0] - row - 1) * dim[1] + col][4]

    # di/dv
    if full:
        sliced_data = np.zeros((points, dim[0], dim[1]))
        for i in range(points):  # points of sweep for each position
            for row in range(dim[0]):  # number of row for matplotlib
                for col in range(dim[1]):  # number of column for matplotlib
                    sliced_data[i][row][col] = (f.data[(dim[0] - row - 1) * dim[1] + col][1][i] + f.data[(dim[0] - row - 1) * dim[1] + col][4][i]) / 2  # re-distribution of data
        return sweep, sliced_data, z
    else:
        sliced_data = np.zeros(dim)
        bias_idx = find_nearest(sweep, bias)
        for row in range(dim[0]):
            for col in range(dim[1]):
                sliced_data[row][col] = (f.data[(dim[0] - row - 1) * dim[1] + col][1][bias_idx] + f.data[(dim[0] - row - 1) * dim[1] + col][4][bias_idx]) / 2
        return sweep[bias_idx], sliced_data, z


def find_nearest(array, value):
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx
