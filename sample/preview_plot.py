# -*- coding: utf-8 -*-

__all__ = ['np', 'pd']

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from . import np, pd


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
