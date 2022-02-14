# -*- coding: utf-8 -*-
__all__ = ['plot_topo', 'plot_grid_slice']

from typing import Union
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utilities import level_plain, slice_3ds
from nanonis_loader import loader

# color map
cdict: dict = {
    'red': [(0.0, 0.0, 0.0), (0.34, 168 / 256, 168 / 256),
            (0.67, 243 / 256, 243 / 256), (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0), (0.34, 40 / 256, 40 / 256),
              (0.67, 194 / 256, 194 / 256), (1.0, 1.0, 1.0)],
    'blue': [(0.0, 0.0, 0.0), (0.34, 15 / 256, 15 / 256),
             (0.67, 93 / 256, 93 / 256), (1.0, 1.0, 1.0)]
}
gwyddion = LinearSegmentedColormap('gwyddion', segmentdata=cdict, N=256)


def plot_topo(input: Union[str, np.ndarray] = '',
              output: str = '',
              v_min: 'list[float]' = [],
              sigma: float = 0) -> None:
    """_summary_

    Args:
        input (Union[str, np.ndarray], optional): _description_.
        Defaults to ''.
        output (str, optional): _description_. Defaults to ''.
        v_min (list[float], optional): _description_. Defaults to [].
        sigma (float, optional): _description_. Defaults to 0.
    """
    if isinstance(input, str):
        topo: np.ndarray = np.loadtxt(input)
    elif isinstance(input, np.ndarray):
        topo = input
    ratio = topo.shape[0] / topo.shape[1]
    if ratio <= 1:
        size = (3.375 / ratio, 3.375)
    else:
        size = (3.375, 3.375 * ratio)
    fig, ax = plt.subplots(figsize=size)

    if v_min:
        ax.imshow(topo, cmap=gwyddion, vmin=v_min)
    elif sigma:
        topo_median = np.median(topo)
        topo_std = np.std(topo)
        ax.imshow(topo,
                  cmap=gwyddion,
                  vmin=[
                      topo_median - sigma * topo_std,
                      topo_median + sigma * topo_std
                  ])
    else:
        ax.imshow(topo, cmap=gwyddion)
    ax.axis('off')

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    if isinstance(input, str):
        fig.savefig(input.replace(input[-3:], 'png'), dpi=300)
    else:
        fig.savefig(output, dpi=300)


def plot_grid_slice(input: str = '',
                    output: str = '',
                    sigma: float = 0,
                    title: bool = False):

    grid = loader(input)
    if not os.path.exists(output):
        os.makedirs(output)
    grid_sweep, grid_mapping, grid_z = slice_3ds(f=grid, full=True)
    grid_mapping_stat = np.zeros((grid.header['Points'], 3))
    for i in range(len(grid_mapping_stat)):
        grid_mapping_stat[i][0] = grid_sweep[i]  # bias
        grid_mapping_stat[i][1] = np.median(grid_mapping[i])  # Median
        grid_mapping_stat[i][2] = np.std(grid_mapping[i])

    for i in range(len(grid_sweep)):
        fig, ax = plt.subplots(figsize=(3.375, 3.375))
        if sigma:
            ax.imshow(
                grid_mapping[i],
                vmin=[
                    grid_mapping_stat[i][1] - sigma * grid_mapping_stat[i][2],
                    grid_mapping_stat[i][1] + sigma * grid_mapping_stat[i][2]
                ])
        else:
            ax.imshow(grid_mapping[i])
        ax.axis('off')
        if title:
            ax.set_title(str(round(grid_mapping_stat[i][0] * 1e3, 2)) + 'mV', fontsize = 16)
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.savefig(output + str(i).zfill(3) + '_' +
                    str(round(grid_mapping_stat[i][0] * 1e3, 2)) + 'mV.png')

    fig, ax = plt.subplots(figsize=(3.375, 3.375))

    ax.imshow(level_plain(grid_z), cmap=gwyddion)
    ax.axis('off')

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.savefig(output + 'grid_z.png')
