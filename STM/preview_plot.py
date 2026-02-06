# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# color map
# cdict_gwyddion: dict = {
#     'red': [(0.0, 0.0, 0.0), (0.34, 168 / 256, 168 / 256),
#             (0.67, 243 / 256, 243 / 256), (1.0, 1.0, 1.0)],
#     'green': [(0.0, 0.0, 0.0), (0.34, 40 / 256, 40 / 256),
#               (0.67, 194 / 256, 194 / 256), (1.0, 1.0, 1.0)],
#     'blue': [(0.0, 0.0, 0.0), (0.34, 15 / 256, 15 / 256),
#              (0.67, 93 / 256, 93 / 256), (1.0, 1.0, 1.0)]
# }
# gwyddion = LinearSegmentedColormap('gwyddion',
#                                    segmentdata=cdict_gwyddion,
#                                    N=256)
cdict_gwyddion: dict = {
    "red": [
        (0.0, 0.0, 0.0),  # Position 0: R=0
        (0.344671, 0.658824, 0.658824),
        (0.687075, 0.953506, 0.953506),
        (1.0, 1.0, 1.0),  # Position 1.0: R=1.0
    ],
    "green": [
        (0.0, 0.0, 0.0),  # G=0
        (0.344671, 0.156863, 0.156863),  # G=0.156863
        (0.687075, 0.759686, 0.759686),  # G=0.759686
        (1.0, 1.0, 1.0),  # G=1.0
    ],
    "blue": [
        (0.0, 0.0, 0.0),  # B=0
        (0.344671, 0.0588235, 0.0588235),  # B=0.0588235
        (0.687075, 0.363821, 0.363821),  # B=0.363821
        (1.0, 1.0, 1.0),  # B=1.0
    ],
}

# Create color map
gwyddion = LinearSegmentedColormap("gwyddion", segmentdata=cdict_gwyddion, N=4096)


def plot_topo(input_file, output, v_min, sigma, color_map=gwyddion) -> None:
    """Plot topography data from a file or NumPy array.

    Args:
        input_file (str or np.ndarray): Path to the input file or a NumPy
            array containing the topography data.
        output (str, optional): Path to save the output image.
            If None, attempts to save with the input filename
            (if input is a str) or prompts the user.
        v_min (float, optional): Minimum value for the color scale.
            If provided, overrides sigma-based scaling.
        sigma (float, optional): Number of standard deviations to
            set the color scale range around the median.
            Ignored if v_min is provided.
        color_map (str or Colormap, optional): Colormap for the plot.
            Defaults to 'gwyddion'.

    Raises:
        ValueError:
            If input_file is neither a string nor a NumPy array.
        FileNotFoundError:
            If input_file is a string but the file does not exist.
    """
    # Load or validate input data
    if isinstance(input_file, str):
        try:
            topo = np.loadtxt(input_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Input file not found: {input_file}") from exc
    elif isinstance(input_file, np.ndarray):
        topo = input_file
    else:
        raise ValueError(
            "Input must be either a file path (str) or a NumPy array (np.ndarray)"
        )
    # Set figure size based on aspect ratio
    ratio = topo.shape[0] / topo.shape[1]
    size = (3.375 / ratio, 3.375) if ratio <= 1 else (3.375, 3.375 * ratio)
    # Calculate statistics for color scaling
    topo_median, topo_std = float(np.median(topo)), float(np.std(topo))

    # Create plot
    fig, ax = plt.subplots(figsize=size)

    if v_min:
        ax.imshow(topo, cmap=color_map, vmin=v_min)
    elif sigma is not None:
        ax.imshow(
            topo,
            cmap=color_map,
            vmin=topo_median - sigma * topo_std,
            vmax=topo_median + sigma * topo_std,
        )
    else:
        ax.imshow(topo, cmap=color_map)
    ax.axis("off")

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)

    # Determine output path
    if output is None:
        if isinstance(input_file, str):
            save_path = input_file.rsplit(".", 1)[0] + ".png"
        else:
            save_path = input("Please specify output file path: ")
    else:
        save_path = output
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
