from matplotlib.colors import LinearSegmentedColormap

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
