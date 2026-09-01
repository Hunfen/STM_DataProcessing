"""Regression checks for core/misc bug fixes (M1, M10, M11, M12, M13).

Run from the repository root:
    .venv/bin/python scripts/regression/check_core_misc.py

Covers:
  - TmatQPI: minimal mock MLWFHamiltonian instantiation and CPU call
    (no TypeError from the missing ``self``);
  - load_wout_file: row-aligned spreads/O_D with and without an Initial
    State row;
  - BackendArray: requesting gpu without usable CuPy stays self-consistent
    (xp is numpy, backend corrected to cpu) and emits a warning;
  - LatticeLoader.create_lattice with a (2, 2) bvecs array succeeds and
    preserves the in-plane reciprocal vectors.
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

import stm_data_processing.config as config_mod
import stm_data_processing.dft.wannier90.mlwf_gk as mlwf_gk_mod
import stm_data_processing.stm.qpi_tmat as qpi_tmat_mod
import stm_data_processing.utils.monitor as monitor_mod
from stm_data_processing.config import BackendArray
from stm_data_processing.io.lattice_loader import LatticeLoader


class MockMLWFHamiltonian:
    """Minimal stand-in exposing the MLWFHamiltonian interface used by TmatQPI."""

    def __init__(self, num_wann: int = 2) -> None:
        self.num_wann = num_wann
        self.bvecs = None
        self._rng = np.random.default_rng(42)

    def hk(self, k_points: np.ndarray) -> np.ndarray:
        """Return a batch of random Hermitian H(k) matrices."""
        n = len(k_points)
        nw = self.num_wann
        h = self._rng.normal(size=(n, nw, nw)) + 1j * self._rng.normal(
            size=(n, nw, nw)
        )
        return (h + np.conj(np.swapaxes(h, 1, 2))) / 2


def test_tmat_qpi_instantiate_and_call() -> None:
    """TmatQPI must instantiate and run on CPU without TypeError."""
    ham = MockMLWFHamiltonian(num_wann=2)
    tq = qpi_tmat_mod.TmatQPI(ham, nk=8, eta=0.01, V=0.1)

    # Force numpy everywhere so the CPU path is deterministic even on a
    # machine where CuPy happens to be installed.
    original_cfg_get_xp = config_mod.get_xp
    original_gk_get_xp = mlwf_gk_mod.get_xp
    original_backend = qpi_tmat_mod.BACKEND

    def _np_xp() -> type:
        return np

    config_mod.get_xp = _np_xp
    mlwf_gk_mod.get_xp = _np_xp
    qpi_tmat_mod.BACKEND = "cpu"
    try:
        qpi_map = tq._compute_tmat(0.1)
        assert qpi_map.shape == (8, 8)
        assert np.all(np.isfinite(qpi_map))
        result = tq.calculate(energy_range=[0.1, 0.2])
    finally:
        config_mod.get_xp = original_cfg_get_xp
        mlwf_gk_mod.get_xp = original_gk_get_xp
        qpi_tmat_mod.BACKEND = original_backend

    assert result["qpi_layers"].shape == (2, 8, 8)
    assert result["metadata"]["module_type"] == "tmat"
    assert result["qx_grid"] is None and result["qy_grid"] is None

    # The GPU placeholder must raise a clear NotImplementedError, not a
    # TypeError, when dispatched on the gpu backend.
    qpi_tmat_mod.BACKEND = "gpu"
    try:
        try:
            tq.calculate(energy_range=[0.1])
        except NotImplementedError:
            pass
        else:
            raise AssertionError("expected NotImplementedError on gpu backend")
    finally:
        qpi_tmat_mod.BACKEND = original_backend


def _spread_lines(base: float) -> list[str]:
    """Two "WF centre and spread" lines starting from ``base``."""
    return [
        f" WF centre and spread    1  (  0.100000,  0.200000,  0.300000 )     {base + 1.0:.8f}",
        f" WF centre and spread    2  (  0.400000,  0.500000,  0.600000 )     {base + 2.0:.8f}",
    ]


def _od_line(value: float) -> str:
    """One Wannierise iteration line carrying O_D and the DLTA marker."""
    return (
        "    1     0.10000000     0.20000000       0.300E+00      0.40E+00 "
        f"   O_D=  {value:.3E} <-- DLTA"
    )


def _write_synthetic_wout(
    with_initial_state: bool,
    initial_state_od: bool = False,
    n_cycles: int = 2,
) -> Path:
    """Write a minimal synthetic .wout file and return its path.

    Parameters
    ----------
    with_initial_state : bool
        Whether to include an "Initial State" section.
    initial_state_od : bool
        Whether the Initial State section prints an O_D line. The standard
        Wannier90 format does not; this mirrors a legacy variant.
    n_cycles : int
        Number of "Cycle:" sections after the Initial State.
    """
    fd, name = tempfile.mkstemp(suffix=".wout", prefix="monitor_reg_")
    os.close(fd)
    path = Path(name)
    lines: list[str] = [
        "  Number of Wannier Functions              :         2",
        "",
        "*------------------------------- WANNIERISE ---------------------------------*",
        "|  Convergence tolerence                     :         1.000E-08             |",
        "*----------------------------------------------------------------------------*",
        "*------------------------------- WANNIERISE ---------------------------------*",
    ]
    if with_initial_state:
        lines.append("Initial State")
        lines.extend(_spread_lines(1.0))
        if initial_state_od:
            lines.append(_od_line(3.0))
    for cycle in range(1, n_cycles + 1):
        base = 4.0 * cycle
        lines.append(f"Cycle: {cycle}")
        lines.extend(_spread_lines(base))
        lines.append(_od_line(base + 2.0))
    lines.append("All done")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_wout_file_row_alignment() -> None:
    """spreads and O_D must stay row-aligned with/without an Initial State."""
    for with_initial in (True, False):
        path = _write_synthetic_wout(with_initial)
        try:
            data = monitor_mod.load_wout_file(str(path))
        finally:
            path.unlink(missing_ok=True)
        spreads = data["wannierise_spreads"]
        od = data["wannierise_od"]
        assert spreads.shape == (2, 2, 1), f"spreads shape {spreads.shape}"
        assert od.shape == (2, 1), f"od shape {od.shape}"
        assert spreads.shape[0] == od.shape[0], "spreads/O_D rows mismatch"
        # First actual cycle must be cycle 1 (Initial State dropped if any).
        assert np.allclose(spreads[0, :, 0], [5.0, 6.0])
        assert np.isclose(od[0, 0], 6.0)


def test_load_wout_file_initial_state_without_od_real_format() -> None:
    """Standard Wannier90 format: Initial State has no O_D row.

    Both arrays must contain exactly num_cycle rows, aligned, and the parser
    must not raise the row-mismatch ValueError.
    """
    n_cycles = 3
    path = _write_synthetic_wout(with_initial_state=True, n_cycles=n_cycles)
    try:
        data = monitor_mod.load_wout_file(str(path))
    finally:
        path.unlink(missing_ok=True)
    spreads = data["wannierise_spreads"]
    od = data["wannierise_od"]
    assert spreads.shape == (n_cycles, 2, 1), f"spreads shape {spreads.shape}"
    assert od.shape == (n_cycles, 1), f"od shape {od.shape}"
    assert spreads.shape[0] == od.shape[0] == n_cycles
    # Initial State spreads (2.0, 3.0) must be gone; cycle 1 is first.
    assert np.allclose(spreads[0, :, 0], [5.0, 6.0])
    assert np.isclose(od[0, 0], 6.0)


def test_load_wout_file_initial_state_with_od_legacy() -> None:
    """Legacy variant: Initial State prints an O_D row.

    The Initial State O_D must be dropped together with the spreads row so
    both arrays stay aligned.
    """
    path = _write_synthetic_wout(with_initial_state=True, initial_state_od=True)
    try:
        data = monitor_mod.load_wout_file(str(path))
    finally:
        path.unlink(missing_ok=True)
    spreads = data["wannierise_spreads"]
    od = data["wannierise_od"]
    assert spreads.shape == (2, 2, 1)
    assert od.shape == (2, 1)
    assert spreads.shape[0] == od.shape[0]
    assert np.allclose(spreads[0, :, 0], [5.0, 6.0])
    assert np.isclose(od[0, 0], 6.0)  # cycle 1 O_D, not the Initial State 3.0


def test_backend_array_self_consistent() -> None:
    """BackendArray must always keep backend and xp consistent."""
    ba = BackendArray(backend="gpu")
    if ba.backend == "gpu":
        assert ba.xp.__name__ == "cupy"
    else:
        assert ba.backend == "cpu"
        assert ba.xp is np


def test_backend_array_gpu_unavailable_fallback() -> None:
    """Requesting gpu without usable CuPy must warn and fall back to cpu."""
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    def _cupy_unavailable() -> bool:
        return False

    handler = _Capture()
    config_logger = logging.getLogger("stm_data_processing.config")
    original = config_mod._cupy_usable
    config_mod._cupy_usable = _cupy_unavailable
    config_logger.addHandler(handler)
    try:
        ba = BackendArray(backend="gpu")
    finally:
        config_mod._cupy_usable = original
        config_logger.removeHandler(handler)

    assert ba.backend == "cpu"
    assert ba.xp is np
    assert any("Falling back to CPU" in msg for msg in records), (
        "expected a fallback warning when GPU is unavailable"
    )


def test_lattice_loader_2d_bvecs() -> None:
    """A (2, 2) bvecs array must produce a usable, consistent lattice."""
    bvecs_2d = np.array([[2.0, 0.0], [0.0, 2.0]])
    lattice = LatticeLoader.create_lattice(bvecs_array=bvecs_2d)
    assert lattice.bvecs.shape == (3, 3)
    assert np.allclose(lattice.bvecs[:2, :2], bvecs_2d)
    assert lattice.bvecs[2, 2] == 1.0
    assert lattice.avecs is not None
    assert lattice.verify_consistency(atol=1e-10)

    # A non-square 2D cell must work as well.
    bvecs_2d_oblique = np.array([[1.5, 0.7], [0.3, 2.2]])
    lattice2 = LatticeLoader.create_lattice(bvecs_array=bvecs_2d_oblique)
    assert np.allclose(lattice2.bvecs[:2, :2], bvecs_2d_oblique)
    assert lattice2.verify_consistency(atol=1e-10)


def main() -> None:
    tests = [
        test_tmat_qpi_instantiate_and_call,
        test_load_wout_file_row_alignment,
        test_load_wout_file_initial_state_without_od_real_format,
        test_load_wout_file_initial_state_with_od_legacy,
        test_backend_array_self_consistent,
        test_backend_array_gpu_unavailable_fallback,
        test_lattice_loader_2d_bvecs,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\nAll {len(tests)} core/misc regression checks passed.")


if __name__ == "__main__":
    main()
