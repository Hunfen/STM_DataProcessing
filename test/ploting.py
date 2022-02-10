# -*- coding: utf-8 -*-
__all__ = ['plot_topo']

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def plot_topo(input: str = '', v_min: 'list[float]' = []) -> None:
    """Plotting topographic images and saving the figure in the same directory.

    Args:
        input (str, optional): path to the .txt topographic image file.
        v_min (list[float], optional): Range of topographic image.
    """
    topo: np.ndarray = np.loadtxt(input)
    ratio = topo.shape[0] / topo.shape[1]
    if ratio <= 1:
        size = (3.375 / ratio, 3.375)
    else:
        size = (3.375, 3.375 * ratio)
    fig, ax = plt.subplots(figsize=size)

    if v_min:
        ax.imshow(topo, cmap=gwyddion, vmin=v_min)
    else:
        ax.imshow(topo, cmap=gwyddion)
    ax.axis('off')

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    fig.savefig(input.replace(input[-3:], 'png'), dpi=300)
