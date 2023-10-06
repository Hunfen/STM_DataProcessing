# -*- coding: utf-8 -*-
__all__ = [
    'plot_topo', 'plot_grid_slice', 'marker_layer', 'plot_grid',
    'plot_grid_respective', 'plot_grid_by_idx'
]

import os
from typing import Union
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from matrix_utilities import level_plain, slice_3ds, topo_extent
from Nanonis_loader import loader

# color map
cdict_gwyddion: dict = {
    'red': [(0.0, 0.0, 0.0), (0.34, 168 / 256, 168 / 256),
            (0.67, 243 / 256, 243 / 256), (1.0, 1.0, 1.0)],
    'green': [(0.0, 0.0, 0.0), (0.34, 40 / 256, 40 / 256),
              (0.67, 194 / 256, 194 / 256), (1.0, 1.0, 1.0)],
    'blue': [(0.0, 0.0, 0.0), (0.34, 15 / 256, 15 / 256),
             (0.67, 93 / 256, 93 / 256), (1.0, 1.0, 1.0)]
}
gwyddion = LinearSegmentedColormap('gwyddion',
                                   segmentdata=cdict_gwyddion,
                                   N=256)

# cdict_DFit: dict = {
#     'red': [(0.0, 0.0, 0.0), (222 / 256, 168 / 256, 168 / 256),
#             (0.67, 243 / 256, 243 / 256), (1.0, 1.0, 1.0)],
#     'green': [(0.0, 0.0, 0.0), (0.34, 40 / 256, 40 / 256),
#               (0.67, 194 / 256, 194 / 256), (1.0, 1.0, 1.0)],
#     'blue': [(0.0, 0.0, 0.0), (0.34, 15 / 256, 15 / 256),
#              (0.67, 93 / 256, 93 / 256), (1.0, 1.0, 1.0)]
# }
# DFit = LinearSegmentedColormap('DFit', segmentdata=cdict_DFit, N=256)


def plot_topo(input: Union[str, np.ndarray] = '',
              output: str = '',
              v_min: 'list[float]' = [],
              sigma: float = 0,
              color_map=gwyddion) -> None:
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
        ax.imshow(topo, cmap=color_map, vmin=v_min)  # type: ignore
    elif sigma:
        topo_median = np.median(topo)
        topo_std = np.std(topo)
        ax.imshow(topo,
                  cmap=color_map,
                  vmin=[
                      topo_median - sigma * topo_std,
                      topo_median + sigma * topo_std
                  ])  # type: ignore
    else:
        ax.imshow(topo, cmap=color_map)
    ax.axis('off')

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    if output:
        fig.savefig(output, dpi=300)
    elif isinstance(input, str):
        fig.savefig(input.replace(input[-3:], 'png'), dpi=300)
    else:
        print('Pls Give a savepath.')


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
                ])  # type:ignore
        else:
            ax.imshow(grid_mapping[i])
        ax.axis('off')
        if title:
            ax.set_title(str(round(grid_mapping_stat[i][0] * 1e3, 2)) + 'mV',
                         fontsize=16)
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)
        fig.savefig(output + str(i).zfill(3) + '_' +
                    str(round(grid_mapping_stat[i][0] * 1e3, 2)) + 'mV.png')

    fig, ax = plt.subplots(figsize=(3.375, 3.375))

    ax.imshow(level_plain(grid_z), cmap=gwyddion)
    ax.axis('off')

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.savefig(output + 'grid_z.png')


def marker_layer(f_path: str = '', spec_path: str = '', output: str = ''):
    """_summary_

    Args:
        f_path (str, optional): _description_. Defaults to ''.
        spec_path (str, optional): _description_. Defaults to ''.
        output (str, optional): _description_. Defaults to ''.
    """
    topo = loader(f_path)
    specs = loader(spec_path)
    extent = topo_extent(topo.header)
    fig, ax = plt.subplots(figsize=(3.375, 3.375))

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.axis('off')

    if spec_path[-3:] == 'dat':
        ax.plot(float(specs.header['X (m)']),
                float(specs.header['Y (m)']),
                '^',
                ms=8,
                color='black')
    elif spec_path[-3:] == '3ds':
        for i in range(len(specs.Parameters)):
            ax.plot(specs.Parameters[i][2], specs.Parameters[i][3], '^', ms=8)

    ax.set_facecolor('none')
    fig.set_facecolor('none')
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.savefig(output, transparent=True, dpi=100)


def plot_grid(input_grid: str = '',
              offset_x: float = 0,
              offset_y: float = 0,
              xlim: 'list[float]' = [],
              ylim: 'list[float]' = [],
              ratio_xy: float = 1 / 2,
              output_grid: str = ''):
    """_summary_

    Args:
        input_grid (str, optional): _description_. Defaults to ''.
        offset_x (float, optional): _description_. Defaults to 0.
        offset_y (float, optional): _description_. Defaults to 0.
        xlim (list[float], optional): _description_. Defaults to [].
        ylim (list[float], optional): _description_. Defaults to [].
        output_grid (str, optional): _description_. Defaults to ''.
    """
    grid = loader(input_grid)  # acquire spectra

    # Plot spectra
    fig, ax = plt.subplots(figsize=(3.375, 3.375 / ratio_xy))

    for i in range(len(grid.Parameters)):
        if np.abs(grid.Parameters[i][0] - grid.Parameters[i][1]) >= 1:
            bias = np.linspace(grid.Parameters[i][0], grid.Parameters[i][1],
                               grid.header['Points']) - offset_x
            ax.set_xlabel('Bias [V]', fontsize=16)
        else:
            bias = np.linspace(grid.Parameters[i][0] * 1e3,
                               grid.Parameters[i][1] * 1e3,
                               grid.header['Points']) - offset_x
            ax.set_xlabel('Bias [mV]', fontsize=16)
        ax.plot(bias, (grid.data[i][1] + grid.data[i][4]) / 2 + offset_y * i,
                'v',
                ms=2)
        if offset_y:
            ax.axhline(y=offset_y * i,
                       xmin=0.48,
                       xmax=0.52,
                       ls='-',
                       lw=1,
                       color='black')

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, )

    ax.set_ylabel('dI/dV [arb. units]', fontsize=16)
    fig.tight_layout(w_pad=0, h_pad=0)
    fig.savefig(output_grid, dpi=300)


def plot_grid_respective(grid_path: str = '',
                         output: str = '',
                         offset_v: float = 0,
                         ratio_xy: float = 1 / 2):
    if not os.path.exists(output):
        os.makedirs(output)
    grid = loader(grid_path)
    fig, ax = plt.subplots(figsize=(3.375, 3.375 / ratio_xy))
    for i in range(len(grid.Parameters)):
        bias = (np.linspace(grid.Parameters[i][0], grid.Parameters[i][1],
                            grid.header['Points']) - offset_v) * 1e3
        spec = (grid.data[i][1] + grid.data[i][4]) / 2
        ax.plot(bias, spec, 'v', ms=2, color='r')
        ax.set_xlim((bias.max() + bias.min()) / 2 -
                    (bias.max() - bias.min()) / 0.9 / 2,
                    (bias.max() + bias.min()) / 2 +
                    (bias.max() - bias.min()) / 0.9 / 2)
        ax.set_ylim(0, spec.max() / 0.9)

        ax.set_xlabel('Bias [mV]', fontsize=16)
        ax.set_ylabel('dI/dV [arb. units]', fontsize=16)
        fig.tight_layout(w_pad=0, h_pad=0)
        fig.savefig(
            output + str(i).zfill(4) + '_' +
            '({0}, {1})'.format(round(grid.Parameters[i][2] * 1e9, 2),
                                round(grid.Parameters[i][3] * 1e9, 2)) + '_' +
            '({0}, {1})'.format(
                round(
                    i % grid.header['Grid dim'][0] *
                    (grid.header['Grid settings'][2] * 1e9 /
                     grid.header['Grid dim'][0]) +
                    (grid.header['Grid settings'][2] * 1e9 /
                     grid.header['Grid dim'][0]) / 2, 2),
                round((grid.header['Grid dim'][1] -
                       i // grid.header['Grid dim'][0] - 1) *
                      (grid.header['Grid settings'][3] * 1e9 /
                       grid.header['Grid dim'][1]) +
                      (grid.header['Grid settings'][3] * 1e9 /
                       grid.header['Grid dim'][1]) / 2, 2)) + '_' +
            '({0}, {1})'.format(
                grid.header['Grid dim'][1] - i // grid.header['Grid dim'][0] -
                1, i % grid.header['Grid dim'][0]) + '.png',
            dpi=100)
        ax.cla()


def plot_grid_by_idx(grid_path='',
                     idx: list = [],
                     avg: bool = False,
                     offset_x: float = 0,
                     offset_y: float = 0,
                     xlim: list = [],
                     ylim: list = [],
                     ratio_xy: float = 1 / 2,
                     cmap: str = 'viridis',
                     output: str = ''):
    if isinstance(grid_path, str):
        grid = loader(grid_path)
    else:
        grid = grid_path
    fig, ax = plt.subplots(figsize=(3.375, 3.375 / ratio_xy))
    if not avg:
        for i in range(len(idx)):
            cm = plt.get_cmap(cmap)
            spec = (grid.data[idx[i]][1] + grid.data[idx[i]][4]) / 2
            if np.abs(grid.Parameters[i][0] - grid.Parameters[i][1]) >= 1:
                bias = np.linspace(grid.Parameters[i][0],
                                   grid.Parameters[i][1],
                                   grid.header['Points']) - offset_x
                ax.set_xlabel('Bias [V]', fontsize=16)
            else:
                bias = (
                    np.linspace(grid.Parameters[i][0], grid.Parameters[i][1],
                                grid.header['Points']) - offset_x) * 1e3
                ax.set_xlabel('Bias [mV]', fontsize=16)
            ax.plot(bias,
                    spec + offset_y * i,
                    'v',
                    ms=2,
                    color=cm(i / len(idx)))
            ax.axhline(
                y=offset_y * i,
                xmin=(0 - (bias.max() - bias.min()) * 0.02 - bias.min()) /
                (bias.max() - bias.min()),
                xmax=(0 + (bias.max() - bias.min()) * 0.02 - bias.min()) /
                (bias.max() - bias.min()),
                ls='-',
                lw=2,
                color=cm(i / len(idx)))
    else:
        spec = np.zeros(grid.header['Points'])
        if idx:
            for i in idx:
                spec += (grid.data[i][1] + grid.data[i][4]) / 2
            spec /= len(idx)
        else:
            for i in range(len(grid.Parameters)):
                spec += (grid.data[i][1] + grid.data[i][4]) / 2
            spec /= (i + 1)
        if np.abs(grid.Parameters[0][0] - grid.Parameters[0][1]) >= 1:
            bias = np.linspace(grid.Parameters[0][0], grid.Parameters[0][1],
                               grid.header['Points']) - offset_x
            ax.set_xlabel('Bias [V]', fontsize=16)
        else:
            bias = (np.linspace(grid.Parameters[0][0], grid.Parameters[0][1],
                                grid.header['Points']) - offset_x) * 1e3
            ax.set_xlabel('Bias [mV]', fontsize=16)
        ax.plot(bias, spec, 'v', ms=2, color='r')

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim((bias.max() + bias.min()) / 2 -
                    (bias.max() - bias.min()) / 0.9 / 2,
                    (bias.max() + bias.min()) / 2 +
                    (bias.max() - bias.min()) / 0.9 / 2)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        ax.set_ylim(0, )
    ax.set_ylabel('dI/dV [arb. units]', fontsize=16)
    fig.tight_layout(w_pad=0, h_pad=0)
    fig.savefig(output, dpi=100)
