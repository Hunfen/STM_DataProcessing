import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nanonispy as nap
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

np.float = float
np.int = int

logger = logging.getLogger(__name__)


def get_divider(file_path: Path) -> int:
    """Return a scaling factor based on whether the path contains 'd1', 'd10', or 'd100'."""
    name = str(file_path)
    if "d100" in name:
        return 100
    if "d10" in name:
        return 10
    return 1


def angle_def(angle):
    """Convert angle to a positive value in the range 0-360."""
    if angle >= 0:
        return angle
    return 360 + angle


def img_rotate(data, angle, range_x, range_y):
    """Rotate the image (centered, taking scan range change into account)."""
    image = data.astype(np.float32)
    rows, cols = image.shape
    center = ((cols - 1) / 2.0, (rows - 1) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    cols_new = int(rows * sin_theta + cols * cos_theta)
    rows_new = int(rows * cos_theta + cols * sin_theta)

    rotation_matrix[0, 2] += (cols_new - cols) / 2
    rotation_matrix[1, 2] += (rows_new - rows) / 2

    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (cols_new, rows_new), flags=cv2.INTER_NEAREST
    )
    range_x_new = range_y * sin_theta + range_x * cos_theta
    range_y_new = range_y * cos_theta + range_x * sin_theta
    return rotated_image, range_x_new, range_y_new


def img_rotate_for_box(data, degree=90, zoom_pan=1):
    """Rotate the image only (size unchanged), used for overview box."""
    img = data
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degree, zoom_pan)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def subtractMeanPlane(matrix):
    """Subtract the best-fit plane from the matrix (NaN-safe).

    Interrupted Nanonis scans leave unmeasured regions as NaN. Only finite
    pixels participate in the least-squares plane fit; the fitted plane is
    then evaluated over the whole image and subtracted, so NaN pixels keep
    their NaN values and the finite region keeps the usual plane-subtracted
    result. If fewer than 3 finite points are available the original array
    is returned unchanged (with a warning) instead of raising or returning
    an all-NaN image.
    """
    xdim, ydim = matrix.shape
    y, x = np.meshgrid(np.arange(ydim), np.arange(xdim))
    data = np.asarray(matrix)
    finite = np.isfinite(data)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < 3:
        logger.warning(
            "subtractMeanPlane: only %d finite point(s); cannot fit a plane, "
            "returning the input unchanged.",
            n_finite,
        )
        return matrix.copy()
    A = np.column_stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())])
    coeffs, _, _, _ = np.linalg.lstsq(
        A[finite.ravel()], data.ravel()[finite.ravel()], rcond=None
    )
    plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    return data - plane


def finite_range(arr):
    """Return (vmin, vmax) of the finite values, or (None, None) if none exist.

    NaN-safe replacement for arr.min() / arr.max() in display-range
    computations; used so interrupted scans (NaN regions) do not poison the
    color scale.
    """
    vals = np.asarray(arr)[np.isfinite(arr)]
    if vals.size == 0:
        return None, None
    return float(vals.min()), float(vals.max())


def topo_colormap(name: str = "Blues_r"):
    """Return a copy of the named colormap with a distinct NaN (bad) color.

    Unmeasured regions of interrupted scans stay NaN; rendering them with a
    dedicated color makes them identifiable instead of blending into the data
    range (matplotlib renders NaN pixels with the colormap bad color).
    """
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="#d9d9d9")
    return cmap


_SEGMENT_BIAS_HEADER = (
    "Segment Start (V), Segment End (V), Settling (s), Integration (s), "
    "Steps (xn), Lockin, Init. Settling (s)"
)


def _bias_from_segments(seg_bias, divider, dedup):
    """Build bias labels (mV) from the per-segment ramp header entries."""
    labels = []
    for i, seg in enumerate(seg_bias):
        p = seg.split(",")
        ramp = np.linspace(float(p[0]), float(p[1]), int(p[4])) * 1000 / divider
        labels.extend(ramp if i == 0 or not dedup else ramp[1:])
    return np.asarray(labels, dtype=float)


def build_bias_labels(raw_data, divider=1):
    """Return one bias label (mV) per data frame, aligned with the sweep axis.

    Nanonis 3DS files store the bias ramp of every sweep segment in the
    header. Consecutive segments may share their boundary point, and the
    on-disk sweep axis may keep or drop that duplicate, so the nominal
    per-segment list is rebuilt (with or without boundary duplicates) until
    its length matches the actual frame count. A final length assertion
    guarantees the labels can be indexed by frame index; a mismatch raises a
    descriptive ValueError instead of a later IndexError.
    """
    sweep_signal = np.asarray(raw_data.signals["sweep_signal"], dtype=float)
    n_frames = len(sweep_signal)
    try:
        seg_bias = raw_data.header[_SEGMENT_BIAS_HEADER]
        bias = _bias_from_segments(seg_bias, divider, dedup=True)
        if len(bias) != n_frames:
            # The sweep axis keeps the segment-boundary repeats.
            bias = _bias_from_segments(seg_bias, divider, dedup=False)
    except Exception:
        # Header missing or unparseable: fall back to the sweep signal itself.
        bias = sweep_signal * 1000 / divider
    if len(bias) != n_frames:
        raise ValueError(
            f"Bias label length ({len(bias)}) does not match the number of "
            f"data frames ({n_frames}); cannot annotate the map frames."
        )
    return bias


# ---------- Single image plotting functions ----------


def plot_sxm_topo(topopath: Path, output_path: Path) -> None:
    """
    Plot a simple topography image (no rotation, no annotation) and save to output_path.
    """
    raw_data = nap.read.Scan(str(topopath))
    topo = raw_data.signals["Z"]["forward"]
    size = raw_data.header["scan_range"]
    direction = raw_data.header["scan_dir"]

    fig, ax = plt.subplots(figsize=(2.55, 2.55))
    if direction == "up":
        ax.imshow(
            topo,
            origin="lower",
            cmap=topo_colormap("Blues_r"),
            extent=(0, size[0] * 1e9, 0, size[1] * 1e9),
        )
    else:
        ax.imshow(
            topo,
            cmap=topo_colormap("Blues_r"),
            extent=(0, size[0] * 1e9, 0, size[1] * 1e9),
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(
        output_path,
        format="tif",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig)


def plot_map_bias(mappath: Path, n: int, output_dir: Path) -> Path:
    """
    Plot a map image for the specified bias index n,
    save to output_dir/f"temp_map_{n}.tif".
    Returns the saved path.
    """
    divider = get_divider(mappath)
    raw_data = nap.read.Grid(str(mappath))
    # Extract bias list
    bias = build_bias_labels(raw_data, divider)

    try:
        data = raw_data.signals["LI Demod 1 Y (A)"][:, :, n]
    except Exception:
        data = raw_data.signals["LI Demod 1 Y [AVG] (A)"][:, :, n]

    scan_range = raw_data.header["size_xy"]

    fig, ax = plt.subplots(figsize=(2.55, 2.55))
    ax.imshow(
        data,
        origin="lower",
        cmap=topo_colormap("rainbow"),
        extent=(0, scan_range[0] * 1e9, 0, scan_range[1] * 1e9),
    )
    ax.text(
        0.04,
        0.90,
        f"{bias[n]:.2f} mV",
        transform=ax.transAxes,
        fontdict={"family": "Arial", "size": "13", "color": "black", "weight": "bold"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_path = output_dir / f"temp_map_{n}.tif"
    fig.savefig(
        out_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig)
    return out_path


def plot_qpi_bias(mappath: Path, n: int, output_dir: Path) -> Path:
    """
    Plot a QPI image, save to output_dir/f"temp_QPI_{n}.tif".
    """
    divider = get_divider(mappath)
    raw_data = nap.read.Grid(str(mappath))

    bias = build_bias_labels(raw_data, divider)

    try:
        data = raw_data.signals["LI Demod 1 Y (A)"][:, :, n]
    except Exception:
        data = raw_data.signals["LI Demod 1 Y [AVG] (A)"][:, :, n]

    scan_range = raw_data.header["size_xy"]

    fft2 = np.fft.fft2(data)
    shift2center = np.fft.fftshift(fft2)
    qpi2 = np.log(1 + np.abs(shift2center))
    if np.any(np.isfinite(qpi2)):
        mean_val = float(np.nanmean(qpi2))
        std_dev = float(np.nanstd(qpi2))
        vmin = float(np.nanmin(qpi2))
        vmax = mean_val + 1.5 * std_dev
    else:
        vmin, vmax = None, None

    range_qx = 2 * np.pi / (scan_range[0] * 1e9 / data.shape[1])
    range_qy = 2 * np.pi / (scan_range[1] * 1e9 / data.shape[0])

    fig, ax = plt.subplots(figsize=(2.55, 2.55))
    ax.imshow(
        qpi2,
        origin="lower",
        cmap=topo_colormap("gray_r"),
        vmin=vmin,
        vmax=vmax,
        extent=(-range_qx / 2, range_qx / 2, -range_qy / 2, range_qy / 2),
    )
    ax.text(
        0.04,
        0.90,
        f"{bias[n]:.2f} mV",
        transform=ax.transAxes,
        fontdict={"family": "Arial", "size": "13", "color": "red", "weight": "bold"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_path = output_dir / f"temp_QPI_{n}.tif"
    fig.savefig(
        out_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig)
    return out_path


def plot_map_current_bias(
    mappath: Path, n: int, output_dir: Path, smooth: bool = False
) -> Path:
    """
    Plot a current map image, save to output_dir/f"temp_mapI_{n}.tif".
    If smooth=True, apply Gaussian filter (sigma=1) to the image.
    """
    divider = get_divider(mappath)
    raw_data = nap.read.Grid(str(mappath))

    bias = build_bias_labels(raw_data, divider)

    data = raw_data.signals["Current (A)"][:, :, n]
    if smooth:
        data = gaussian_filter(data, sigma=1)
    scan_range = raw_data.header["size_xy"]

    fig, ax = plt.subplots(figsize=(2.55, 2.55))
    ax.imshow(
        data,
        origin="lower",
        cmap=topo_colormap("rainbow"),
        extent=(0, scan_range[0] * 1e9, 0, scan_range[1] * 1e9),
    )
    ax.text(
        0.04,
        0.90,
        f"{bias[n]:.2f} mV",
        transform=ax.transAxes,
        fontdict={"family": "Arial", "size": "13", "color": "black", "weight": "bold"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    out_path = output_dir / f"temp_mapI_{n}.tif"
    fig.savefig(
        out_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig)
    return out_path


def plot_sts(stspath: Path, topopath: Path, output_dir: Path, smooth: bool = False):
    """
    Plot a single spectrum and the corresponding topography with marker.
    topopath can be None. If smooth=True, apply Gaussian filter (sigma=1).
    Returns (sts_path, topo_marked_path); the second is None if topopath is None.
    """
    divider = get_divider(stspath)
    raw_sts = nap.read.Spec(str(stspath))
    bias = raw_sts.signals["Bias calc (V)"] * 1000 / divider
    try:
        didv = raw_sts.signals["LI Demod 1 Y [AVG] (A)"]
    except Exception:
        didv = raw_sts.signals["LI Demod 1 Y (A)"]
    if smooth:
        didv = gaussian_filter1d(didv, sigma=1)

    X, Y = float(raw_sts.header["X (m)"]) * 1e9, float(raw_sts.header["Y (m)"]) * 1e9

    # Save the single spectrum
    fig_sts, ax_sts = plt.subplots(figsize=(2.15, 2.15))
    ax_sts.plot(bias, didv, "r-")
    ax_sts.tick_params(axis="both", which="major", pad=1)
    plt.rcParams["font.size"] = 6
    ax_sts.set_xlabel(
        "Bias (mV)", fontdict={"family": "Arial", "size": 6}, labelpad=0.1
    )
    ax_sts.set_ylabel(
        r"d$\it{I}$/d$\it{V}$ (a.u.)",
        fontdict={"family": "Arial", "size": 6},
        labelpad=0.1,
    )
    sts_path = output_dir / "temp_sts.tif"
    fig_sts.savefig(
        sts_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig_sts)

    if topopath is None:
        return sts_path, None

    # Plot the topography with marker
    raw_topo = nap.read.Scan(str(topopath))
    topo = raw_topo.signals["Z"]["forward"]
    scan_range = raw_topo.header["scan_range"]
    scan_offset = raw_topo.header["scan_offset"]
    angle = float(raw_topo.header["scan_angle"])
    direction = raw_topo.header["scan_dir"]

    topo2, range_x_new, range_y_new = img_rotate(
        topo, angle if direction == "up" else -angle, scan_range[0], scan_range[1]
    )
    extent = (
        (scan_offset[0] - range_x_new / 2) * 1e9,
        (scan_offset[0] + range_x_new / 2) * 1e9,
        (scan_offset[1] - range_y_new / 2) * 1e9,
        (scan_offset[1] + range_y_new / 2) * 1e9,
    )
    fig_topo, ax_topo = plt.subplots(figsize=(2.55, 2.55))
    ax_topo.imshow(
        topo2,
        origin="lower" if direction == "up" else "upper",
        cmap=topo_colormap("Blues_r"),
        vmin=finite_range(topo)[0],
        vmax=finite_range(topo)[1],
        extent=extent,
    )
    ax_topo.plot(X, Y, "ro", markersize=3)
    ax_topo.set_xticks([])
    ax_topo.set_yticks([])
    ax_topo.axis("off")
    topo_marked_path = output_dir / "temp_ststopo.tif"
    fig_topo.savefig(
        topo_marked_path,
        format="tif",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig_topo)
    return sts_path, topo_marked_path


def plot_linecut(lcpath: Path, topopath: Path, output_dir: Path, smooth: bool = False):
    """
    Plot three figures for a linecut: waterfall plot, overlap plot,
    and topography with marker.
    topopath can be None. If smooth=True, apply Gaussian filter (sigma=1)
    to each spectrum along the bias axis.
    Returns (lc_path, ol_path, topo_marked_path); third is None if topopath is None.
    """
    divider = get_divider(lcpath)
    raw_lc = nap.read.Grid(str(lcpath))
    L = raw_lc.header["size_xy"][0] * 1e9
    bias = raw_lc.signals["sweep_signal"][:] * 1000 / divider
    try:
        lcdata = raw_lc.signals["LI Demod 1 Y [AVG] (A)"][0, :, :]
    except Exception:
        lcdata = raw_lc.signals["LI Demod 1 Y (A)"][0, :, :]

    if smooth:
        lcdata = gaussian_filter1d(lcdata, sigma=1, axis=1)

    # Waterfall plot + stacked spectra + topography line
    fig_wf, (ax0, ax1, ax2) = plt.subplots(
        1, 3, figsize=(5.8, 3), gridspec_kw={"width_ratios": [2, 2, 1]}
    )
    ax1.imshow(
        lcdata,
        extent=(bias[0], bias[-1], 0, L),
        aspect="auto",
        origin="lower",
        cmap=topo_colormap("rainbow"),
        interpolation="none",
    )
    ax1.tick_params(axis="both", which="major", pad=1)
    ax1.set_xlabel("Bias (mV)", fontdict={"family": "Arial", "size": 8}, labelpad=0.3)
    ax1.set_ylabel("Length(nm)", fontdict={"family": "Arial", "size": 8}, labelpad=0.3)

    offset = 0.25 * np.nanmean(lcdata[0])
    cmap = plt.get_cmap("brg")
    colors = [cmap(i) for i in np.linspace(0, 1, len(lcdata))]
    for i, spec in enumerate(lcdata):
        ax0.plot(bias, spec + i * offset, linewidth=0.2, color=colors[i])
    ax0.axhline(y=0, xmin=0.3, xmax=0.7, c="k", lw=0.1, ls="-")
    ax0.text(bias[-1], spec[-1] + offset * (len(lcdata) - 1), str(len(lcdata)))
    ax0.text(bias[-1], lcdata[0][-1], "0")
    ax0.tick_params(axis="both", which="major", pad=1)
    ax0.set_yticks([])
    ax0.set_xlabel("Bias (mV)", fontdict={"family": "Arial", "size": 8}, labelpad=0.3)
    ax0.set_ylabel(
        r"d$\it{I}$/d$\it{V}$ (a.u.)",
        fontdict={"family": "Arial", "size": 8},
        labelpad=0.1,
    )

    height = raw_lc.signals["topo"][0][:] * 1e12
    ax2.plot(height - min(height), np.linspace(0, L, len(height)))
    ax2.set_ylim(0, L)
    ax2.set_xlabel("Height (pm)", fontdict={"family": "Arial", "size": 8}, labelpad=0.3)
    ax2.tick_params(axis="both", which="major", pad=1)

    fig_wf.subplots_adjust(wspace=0.25)
    lc_path = output_dir / "temp_lc.tif"
    fig_wf.savefig(
        lc_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig_wf)

    # Overlap plot
    fig_ol, ax_ol = plt.subplots(figsize=(2.8, 2.8))
    for i, spec in enumerate(lcdata):
        ax_ol.plot(bias, spec, linewidth=0.5, color=colors[i])
    ax_ol.axhline(y=0, xmin=0.3, xmax=0.7, c="k", lw=0.1, ls="-")
    ax_ol.tick_params(axis="both", which="major", pad=1)
    ax_ol.set_xlabel("Bias (mV)", fontdict={"family": "Arial", "size": 8}, labelpad=0.3)
    ax_ol.set_ylabel(
        r"d$\it{I}$/d$\it{V}$ (a.u.)",
        fontdict={"family": "Arial", "size": 8},
        labelpad=0.1,
    )
    ol_path = output_dir / "temp_ol.tif"
    fig_ol.savefig(
        ol_path, format="tif", bbox_inches="tight", transparent=True, pad_inches=0
    )
    plt.close(fig_ol)

    if topopath is None:
        return lc_path, ol_path, None

    # Topography with marker
    raw_topo = nap.read.Scan(str(topopath))
    topo = raw_topo.signals["Z"]["forward"]
    scan_range = raw_topo.header["scan_range"]
    scan_offset = raw_topo.header["scan_offset"]
    angle = float(raw_topo.header["scan_angle"])
    direction = raw_topo.header["scan_dir"]

    lc_center = raw_lc.header["pos_xy"]
    lc_size = raw_lc.header["size_xy"][0]
    lc_angle = raw_lc.header["angle"] * np.pi / 180
    X = (lc_center[0] - lc_size / 2 * np.cos(lc_angle)) * 1e9
    Y = (lc_center[1] + lc_size / 2 * np.sin(lc_angle)) * 1e9
    XX = 2 * lc_center[0] * 1e9 - X
    YY = 2 * lc_center[1] * 1e9 - Y

    topo2, range_x_new, range_y_new = img_rotate(
        topo, angle if direction == "up" else -angle, scan_range[0], scan_range[1]
    )
    extent = (
        (scan_offset[0] - range_x_new / 2) * 1e9,
        (scan_offset[0] + range_x_new / 2) * 1e9,
        (scan_offset[1] - range_y_new / 2) * 1e9,
        (scan_offset[1] + range_y_new / 2) * 1e9,
    )
    fig_mark, ax_mark = plt.subplots(figsize=(2.8, 2.8))
    ax_mark.imshow(
        topo2,
        origin="lower" if direction == "up" else "upper",
        cmap=topo_colormap("Blues_r"),
        vmin=finite_range(topo)[0],
        vmax=finite_range(topo)[1],
        extent=extent,
    )
    ax_mark.plot(X, Y, "ko", fillstyle="none", markersize=5)
    ax_mark.plot([X, XX], [Y, YY], "k--")
    ax_mark.set_xticks([])
    ax_mark.set_yticks([])
    ax_mark.axis("off")
    topo_marked_path = output_dir / "temp_lctopo.tif"
    fig_mark.savefig(
        topo_marked_path,
        format="tif",
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig_mark)
    return lc_path, ol_path, topo_marked_path


def update_frame_from_dir(frame_dir: Path, n, ax):
    """Read temp_map_{n}.tif from the specified directory and display it on ax."""
    ax.cla()
    ax.set_axis_off()
    img_path = frame_dir / f"temp_map_{n}.tif"
    img = plt.imread(str(img_path))
    range_x, range_y = img.shape[1], img.shape[0]
    ax.imshow(img, extent=[0, range_x, 0, range_y])
