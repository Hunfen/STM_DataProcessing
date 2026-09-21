"""Static real part of the Lindhard susceptibility from Wannier90 ``hr.dat``.

Reference
---------
``docs/stm_data_processing/dft/wannier90/Lindhard_Re_chi_from_Wannier90_hr.md``.
The section numbers quoted below refer to that document.

Convention (doc section 7.1)
----------------------------
The implemented quantity is the document convention

.. math::

   \\operatorname{Re}\\chi_0(\\mathbf q, 0)
   = -\\frac{1}{N_k}\\sum_{\\mathbf k,n,m}
   |M_{mn}|^2\\left(f_{n\\mathbf k}-f_{m,\\mathbf{k+q}}\\right)
   \\frac{\\Delta\\varepsilon}{\\Delta\\varepsilon^2+\\eta^2},
   \\qquad
   \\Delta\\varepsilon=\\varepsilon_{n\\mathbf k}-\\varepsilon_{m,\\mathbf{k+q}},

i.e. the overall minus sign is prefixed and the retarded ``+i eta``
prescription is used.  It differs from the standard/textbook Lindhard
function (written without the minus sign, static real part <= 0) by an
overall factor -1 in every component:

    chi0_doc = -chi0_std,   Re chi0_doc >= 0.

Every term of the doc-convention sum above is non-negative on its own,
because ``f`` is monotone in its own energy argument, so the static real
part is non-negative for any q.  In the long-wavelength limit the intraband
channel reduces to the positive compressibility/DOS term
``Re chi0_doc(q -> 0, 0) = (1/N_k) sum_{k,n} (-df/dE) >= 0`` (doc sections
7.1 and 8.2).

Approximations and numerical details
------------------------------------
* Local-density (point-like Wannier) approximation for the density vertex,
  so that ``M_mn(k, q) = sum_a U*_{a m}(k+q) U_{a n}(k)`` (doc 6.1, 10.3).
* The intraband 0/0 ratio is replaced by ``df/dE`` when ``n == m`` and
  ``|Delta eps| <= degeneracy_tolerance`` (doc 8.2).  The substitution is
  applied *before* the overall minus sign.  ``df/dE`` is taken from the
  finite-temperature Fermi function, so the substitution only contributes
  for ``temperature > 0``; at ``temperature = 0`` the tabulated step
  function has zero derivative and the intraband channel collapses to zero
  (the q -> 0 Drude/compressibility weight is a delta at the Fermi level
  that a finite k-mesh cannot represent).
* The k mesh is ``linspace(-0.5, 0.5, nk, endpoint=False)`` and k+q is
  folded back into the first Brillouin zone by index wrapping (doc 10.1).
  The returned q grids are the discrete FFT frequency grid
  ``fftshift(fftfreq(nk))``, which labels the fftshifted data exactly for
  both even and odd ``nk``.
* NumPy only: the module diagonalizes through ``MLWFHamiltonian._hk_cpu``
  so that it never builds GPU arrays and does not depend on the package-wide
  backend setting.  No GPU/cupy path and no dependency beyond NumPy are used.

Vectorized CPU engine
---------------------
* The q-sum is vectorized over the band indices: for one q point the working
  arrays are ``(k, m, n)`` tensors, ``Delta = eps_m(k+q) - eps_n(k)`` and the
  regularized ratio is built elementwise, while the orbital overlap
  ``M_mn = sum_a U*_{a m}(k+q) U_{a n}(k)`` is one batched ``numpy.matmul``
  (contraction over the selected orbitals).  ``einsum`` is used only for the
  final contraction of the overlap with the ratio.  Only q, the band-m block and
  the k-row chunk remain as Python loops.
* The cyclic k -> k+q shift is applied through ``_shifted_ranges``
  (destination/source slice pairs), so the eigenvector array is never copied
  the way ``np.roll`` would copy it for every q point.
* Memory blocking: band-m columns are processed in ``band_block``-wide blocks
  and the k rows of one block are chunked so that the working tensors stay near
  1 GB, which keeps nk=256 inside a 16 GB machine.  Both widths come from the
  constructor (``band_block`` / ``block_entries``), which fall back to the
  module constants :data:`_MAX_BAND_BLOCK` / :data:`_MAX_BLOCK_ENTRIES` when
  ``None``; they are pure performance knobs that only reorder the summation.
  ``H(k)`` is evaluated in ``_HK_ROW_BLOCK``-row blocks, which is what keeps the
  75-orbital model (``num_wann**2 = 5625`` columns) away from the Apple
  Accelerate zgemm segfault that an unblocked 1024-row build triggers.
* The eigenvector array is materialized only for the selected orbitals and only
  when ``include_matrix_elements=True``: the scalar vertex needs no
  eigenvectors at all.

Observability
-------------
The module logs through ``logging.getLogger(__name__)`` and configures no
handler itself, so a caller (or a regression check) can attach its own handler
and read the records:

* one configuration echo per :meth:`RealLindhardCalculator.calculate` call,
* a BLAS thread advisory (WARNING when a thread variable is unset or > 1),
* one ``stage=...`` timing record per stage (``diagonalize``, ``occupations``,
  ``q_sum``, ``fftshift``, ``extend``, ``h5_write``), failures logged as ERROR
  with the traceback,
* ``progress`` records throttled by ``progress_interval_s``, one per completed
  q1 row and carrying the machine-readable ``lindhard_progress`` extra,
* a closing summary with the ``sum``/``max``/``min`` digest of the three
  returned arrays, the ``max|data-(intra+inter)|`` residual and the peak RSS
  (:func:`peak_rss_bytes`, standard library only).

q row slices
------------
``calculate(..., q_index_range=(start, stop))`` evaluates only the first q index
range ``iq1 in [start, stop)`` and returns the three arrays in raw FFT order
(no fftshift, no HDF5 output, no ``q_range`` extension).  A single q pixel is a
pure function of ``(iq1, iq2)``, ``nk``, ``nw`` and the blocking parameters,
which do not depend on the row range, so a slice reproduces the corresponding
rows of the full-mesh result bit for bit; this is the invariant the
multi-process driver in :mod:`lindhard_re_chi_parallel` assembles its result
from.

Optional outputs
----------------
``calculate(..., output_path=...)`` writes the fftshifted primitive-BZ
result to an HDF5 file (before any ``q_range`` cropping), and
``calculate(..., q_range=(qmin, qmax))`` periodically extends and crops the
result to ``[qmin, qmax)`` through :func:`extend_qpi` (doc section 9).
"""

from __future__ import annotations

import logging
import operator
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property, wraps
from pathlib import Path
from typing import Any

import numpy as np

from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.io.susceptibility_io import save_susceptibility_to_h5
from stm_data_processing.utils.miscellaneous import extend_qpi, fermi, frac_to_real_2d

logger = logging.getLogger(__name__)

_BOLTZMANN_EV_PER_K = 8.617333262145e-5

#: ``module_type`` tag written into the HDF5 attributes and the metadata dict.
#: It records the sign convention used for the stored response.
_MODULE_TYPE = "real_Lindhard"

#: Number of k points whose H(k) is built and diagonalized in one call.
#: Apple Accelerate's complex GEMM (zgemm) segfaults for the 75-orbital model
#: (``num_wann**2 = 5625`` columns) as soon as the H(k) build is handed 1024 or
#: more rows, which already happens at nk=32.  Splitting the build into
#: ``_HK_ROW_BLOCK``-row blocks is bitwise identical (the phase arithmetic is
#: elementwise and ``numpy.linalg.eigh`` diagonalizes one matrix per block) and
#: stays inside the range Accelerate handles.
_HK_ROW_BLOCK = 512

#: Band-m block width (columns of the k+q band index) of the vectorized q-sum.
_MAX_BAND_BLOCK = 16

#: Upper bound for the number of ``(k, m, n)`` entries of the largest live
#: working array of the q-sum.  Roughly 115 bytes stay live per entry (the
#: float64 ratio stack plus the complex overlap and its squared modulus), so
#: 8e6 entries keep the transients near 1 GB for every mesh size; the k rows of
#: one (q, band-m) block are chunked to satisfy it.
_MAX_BLOCK_ENTRIES = 8_000_000

#: Environment variables that steer the BLAS thread pools of the numerical
#: libraries NumPy may be linked against.  They are only advisory: the pools are
#: created while NumPy is imported, so a value assigned after that import no
#: longer has any effect.
_BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def peak_rss_bytes() -> int:
    """Return the peak resident set size of this process in bytes.

    Standard library only.  Linux reads the resident page count from
    ``/proc/self/statm`` (multiplied by the page size); other platforms use the
    kernel high-water mark of :func:`resource.getrusage`, which is reported in
    bytes on macOS and in kilobytes elsewhere (so it is scaled accordingly).
    The caller is expected to keep the maximum across the samples it takes,
    which turns the Linux current-RSS reading into a peak measurement.
    """
    if sys.platform.startswith("linux"):
        try:
            statm = Path("/proc/self/statm").read_text(encoding="ascii")
            resident_pages = int(statm.split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            pass
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows only
        return 0
    high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return high_water
    return high_water * 1024


def format_bytes(num_bytes: float) -> str:
    """Human readable binary size, e.g. ``3.62GB``."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def format_hms(seconds: float) -> str:
    """Format a duration as ``HH:MM:SS`` (``--:--:--`` when unknown)."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0.0:
        return "--:--:--"
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole // 60) % 60:02d}:{whole % 60:02d}"


def array_digest(values: np.ndarray) -> dict[str, float]:
    """``sum``/``max``/``min`` digest of a real array, for logs and sidecars."""
    return {
        "sum": float(np.sum(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def blas_thread_advisory() -> tuple[str, list[str]]:
    """Describe the BLAS thread environment and return the offending variables.

    Returns ``(message, offenders)``: the message echoes every variable of
    :data:`_BLAS_THREAD_ENV_VARS` with its current value, and ``offenders``
    lists the variables that are unset or larger than one, i.e. the settings
    that let a multi-process run oversubscribe the machine.
    """
    values = {name: os.environ.get(name) for name in _BLAS_THREAD_ENV_VARS}
    offenders = [
        name
        for name, value in values.items()
        if value is None or value.strip() not in ("", "1")
    ]
    message = "threads " + " ".join(
        f"{name}={values[name] if values[name] is not None else 'unset'}"
        for name in _BLAS_THREAD_ENV_VARS
    )
    return message, offenders


def _log_progress(
    rows_done: int,
    rows_total: int,
    columns: int,
    started: float,
    rss_peak: int,
) -> None:
    """Emit one progress record for the completed q rows.

    The record carries the machine-readable tuple ``lindhard_progress =
    (rows_done, rows_total, pixels_done)`` as a ``logging`` extra, which is how
    the multi-process driver relays per-worker progress to its parent without
    parsing the message text.
    """
    elapsed = time.perf_counter() - started
    rate = rows_done / elapsed if elapsed > 0 else 0.0
    remaining = (rows_total - rows_done) / rate if rate > 0 else float("inf")
    pixels = rows_done * columns
    logger.info(
        "[RealLindhardCalculator] progress rows=%d/%d (%.1f%%) rate=%.2f row/s "
        "px/s=%.0f elapsed=%.1fs eta=%s rss_peak=%s",
        rows_done,
        rows_total,
        100.0 * rows_done / rows_total,
        rate,
        pixels / elapsed if elapsed > 0 else float("inf"),
        elapsed,
        format_hms(remaining),
        format_bytes(rss_peak),
        extra={"lindhard_progress": (rows_done, rows_total, pixels)},
    )


@contextmanager
def _stage(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log one ``stage=<name> <field>=<value> s=<elapsed>`` INFO record.

    The yielded dict can be extended inside the ``with`` body with fields that
    are only known afterwards (a byte count, a throughput).  On failure the
    stage is logged as ERROR together with the traceback and the exception is
    re-raised unchanged.
    """
    started = time.perf_counter()
    try:
        yield fields
    except Exception:
        logger.error(
            "[RealLindhardCalculator] stage=%s failed s=%.3f",
            name,
            time.perf_counter() - started,
            exc_info=True,
        )
        raise
    elapsed = time.perf_counter() - started
    rendered = "".join(f" {key}={value}" for key, value in fields.items())
    logger.info(
        "[RealLindhardCalculator] stage=%s%s s=%.3f", name, rendered, elapsed
    )


def _log_failures(func):
    """Log any exception escaping the wrapped method as ERROR with traceback."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(
                "[RealLindhardCalculator] %s failed", func.__qualname__, exc_info=True
            )
            raise

    return wrapper


def _shifted_ranges(nk: int, shift: int) -> list[tuple[slice, slice]]:
    """Destination/source slice pairs of the cyclic shift ``i -> (i + shift) % nk``.

    The k+q mesh is reached by cyclic shifts of the band index rank.  Splitting
    a shift into ``(destination, source)`` slice pairs keeps it a *view*
    operation, so the large eigenvector array is never copied per q point (a
    plain ``np.roll`` would copy it nk**2 times).
    """
    shift %= nk
    if shift == 0:
        return [(slice(0, nk), slice(0, nk))]
    return [
        (slice(0, nk - shift), slice(shift, nk)),
        (slice(nk - shift, nk), slice(0, shift)),
    ]


class RealLindhardCalculator:
    """Calculate :math:`\\operatorname{Re}\\chi_0(q, 0)` on a 2D mesh.

    The implemented convention follows ``Lindhard_Re_chi_from_Wannier90_hr``:

    .. math::

       \\operatorname{Re}\\chi_0 = -\\frac{1}{N_k}\\sum_{k,n,m}
       |M_{mn}|^2 (f_{n,k} - f_{m,k+q})
       \\frac{\\Delta\\epsilon}{\\Delta\\epsilon^2 + \\eta^2},

    with :math:`\\Delta\\epsilon = \\epsilon_{n,k} - \\epsilon_{m,k+q}`.
    The overall minus sign makes the static response non-negative; the
    standard textbook convention (no minus sign) is related by
    :math:`\\chi_0^{\\text{doc}} = -\\chi_0^{\\text{std}}`.
    """

    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 256,
        eta: float = 5e-3,
        band_block: int | None = None,
        block_entries: int | None = None,
    ):
        """Configure the engine.

        Parameters
        ----------
        hamiltonian : MLWFHamiltonian
            Tight-binding model whose ``_hk_cpu`` is diagonalized.
        nk : int, default 256
            Number of k points per reciprocal direction.
        eta : float, default 5e-3
            Lorentzian broadening in eV (doc 7.1/8.3).
        band_block : int or None, optional
            Band-m block width of the vectorized q-sum.  ``None`` (default) uses
            the module constant :data:`_MAX_BAND_BLOCK` *at call time*, so an
            explicit value or a patched module constant both take effect.  The
            value is a pure performance knob: it only changes the order in which
            the same terms are accumulated.
        block_entries : int or None, optional
            Upper bound for the number of ``(k, m, n)`` entries of the largest
            live working array; ``None`` uses :data:`_MAX_BLOCK_ENTRIES`.  Also
            performance/memory only.
        """
        if not isinstance(hamiltonian, MLWFHamiltonian):
            raise TypeError(
                f"Expected MLWFHamiltonian, got {type(hamiltonian).__name__}"
            )
        if nk <= 0:
            raise ValueError(f"nk must be positive, got {nk}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        for name, value in (("band_block", band_block), ("block_entries", block_entries)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.ham = hamiltonian
        self.nk = nk
        self.eta = eta
        self.band_block = None if band_block is None else int(band_block)
        self.block_entries = None if block_entries is None else int(block_entries)

    def _validate_q_index_range(self, q_index_range: Any) -> tuple[int, int]:
        """Validate ``q_index_range`` and return it as a pair of plain ints.

        Exactly two integer entries ``(start, stop)`` are accepted: a sequence
        of any other length is rejected instead of silently dropping the extra
        entries (a stray ``(start, stop, step)`` is a caller mistake, not a
        request to ignore the step).
        """
        try:
            length = len(q_index_range)
        except TypeError as exc:
            raise ValueError(
                "q_index_range must be a (start, stop) pair of integers, "
                f"got {q_index_range!r}"
            ) from exc
        if length != 2:
            raise ValueError(
                "q_index_range must contain exactly 2 entries (start, stop), "
                f"got {length}: {q_index_range!r}"
            )
        try:
            start = operator.index(q_index_range[0])
            stop = operator.index(q_index_range[1])
        except (TypeError, IndexError, KeyError) as exc:
            raise ValueError(
                "q_index_range must be a (start, stop) pair of integers, "
                f"got {q_index_range!r}"
            ) from exc
        if not 0 <= start < stop <= self.nk:
            raise ValueError(
                "q_index_range must satisfy 0 <= start < stop <= nk "
                f"({self.nk}), got ({start}, {stop})"
            )
        return start, stop

    @cached_property
    def eigen(self) -> tuple[np.ndarray, np.ndarray]:
        """Band energies and eigenvectors on the primitive 2D k mesh.

        ``_hk_cpu`` is called intentionally so this calculator stays CPU-only
        even if the package-wide backend has been configured as ``"gpu"``.
        The full eigenvector matrix is kept here for backward compatibility;
        :meth:`calculate` only materializes the orbital slice it needs.
        """
        evals, evecs_orb = self._diagonalize(np.arange(self.ham.num_wann))
        return evals, evecs_orb

    def _diagonalize(
        self, orbitals: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Diagonalize H(k) on the k mesh, a block of k points at a time.

        H(k) is evaluated with at most ``_HK_ROW_BLOCK`` k points per call (see
        that constant: it is what keeps the 75-orbital model away from the
        Accelerate zgemm segfault) and every block is diagonalized on its own,
        which is bitwise equivalent to one large batched ``eigh``.

        Parameters
        ----------
        orbitals : numpy.ndarray or None
            Wannier orbital indices (the ``alpha`` index of the local density
            vertex) kept in the eigenvector slice.  ``None`` stores no
            eigenvectors at all, which is all the scalar vertex needs.

        Returns
        -------
        (evals, evecs_orb)
            ``evals`` has shape ``(nk, nk, nw)``.  ``evecs_orb`` has shape
            ``(nk, nk, len(orbitals), nw)`` in the orbital-major convention
            ``U[alpha, band]``, or is ``None`` when ``orbitals`` is ``None``.
        """
        nk = self.nk
        nw = self.ham.num_wann
        n_rows = nk * nk
        k_values = np.linspace(-0.5, 0.5, nk, endpoint=False)
        k1, k2 = np.meshgrid(k_values, k_values, indexing="ij")
        k_points = np.column_stack((k1.ravel(), k2.ravel(), np.zeros(n_rows)))

        n_orb = nw if orbitals is None else len(orbitals)
        evals = np.empty((n_rows, nw), dtype=float)
        evecs = (
            None
            if orbitals is None
            else np.empty((n_rows, n_orb, nw), dtype=np.complex128)
        )
        for start in range(0, n_rows, _HK_ROW_BLOCK):
            stop = min(start + _HK_ROW_BLOCK, n_rows)
            block_evals, block_evecs = np.linalg.eigh(
                self.ham._hk_cpu(k_points[start:stop])
            )
            evals[start:stop] = block_evals
            if evecs is not None:
                evecs[start:stop] = block_evecs[:, orbitals, :]

        if evecs is None:
            return evals.reshape(nk, nk, nw), None
        return evals.reshape(nk, nk, nw), evecs.reshape(nk, nk, n_orb, nw)

    @staticmethod
    def _fermi_derivative(
        energies: np.ndarray, mu: float, temperature: float
    ) -> np.ndarray:
        """Return ``df/dE`` for the finite-temperature Fermi distribution."""
        if temperature <= 1e-12:
            return np.zeros_like(energies, dtype=float)
        occupation = fermi(energies, mu=mu, T=temperature)
        return -occupation * (1.0 - occupation) / (_BOLTZMANN_EV_PER_K * temperature)

    @_log_failures
    def calculate(
        self,
        chemical_potential: float = 0.0,
        temperature: float = 4.2,
        orbital_select: list[int] | np.ndarray | None = None,
        include_matrix_elements: bool = True,
        q_chunk_size: int = 8,
        degeneracy_tolerance: float = 1e-12,
        q_range: tuple[float, float] | None = None,
        output_path: str | None = None,
        q_index_range: tuple[int, int] | None = None,
        progress_interval_s: float = 30.0,
    ) -> dict[str, Any]:
        """Return total, intraband, and interband static real susceptibility.

        Parameters
        ----------
        chemical_potential : float, default 0.0
            Fermi level in eV.
        temperature : float, default 4.2
            Temperature in K used for the Fermi-Dirac occupation and for the
            ``df/dE`` intraband substitution.
        orbital_select : sequence of int, optional
            Wannier orbitals kept in the local density vertex (the projection
            acts on the ``alpha`` index of ``M_mn``).  Defaults to all
            orbitals.
        include_matrix_elements : bool, default True
            If False, every band-pair vertex gets unit weight
            (``|M_mn|^2 -> 1``) and the scalar Lindhard result is returned.
        q_chunk_size : int, default 8
            Retained for API compatibility.  The vectorized engine fixes its
            own (band-m, k-row) blocking; the result is independent of this
            value.
        degeneracy_tolerance : float, default 1e-12
            ``|Delta eps|`` below which the intraband ratio is replaced by
            ``df/dE`` (doc 8.2).
        q_range : tuple of float, optional
            ``(qmin, qmax)`` fractional window in units of the reciprocal
            lattice vectors.  If given, the fftshifted primitive-BZ result is
            periodically extended and cropped to ``[qmin, qmax)``.  The
            default (None) returns the unwrapped primitive BZ.  ``metadata``
            keeps the primitive mesh size (``nk``/``nq``) in either case.
        output_path : str, optional
            If given, the fftshifted primitive-BZ result is written to this
            HDF5 file *before* any ``q_range`` cropping.
        q_index_range : tuple of int, optional
            Half-open range ``(start, stop)`` of the *first* q index (the
            ``iq1`` rows of the raw FFT-order array, ``q = iq/nk``).  Exactly
            two integer entries are required; any other length raises
            :class:`ValueError`.  If given,
            only ``iq1 in [start, stop)`` is evaluated and the function returns
            the ``(stop - start, nk)`` slabs of the three arrays in raw FFT
            order: no fftshift, no HDF5 output and no ``q_range`` extension are
            performed (combining the argument with ``output_path`` or
            ``q_range`` raises :class:`ValueError`).  ``q1_grid`` carries the
            raw ``fftfreq(nk)[start:stop]`` labels, ``q2_grid`` the full raw
            ``fftfreq(nk)`` labels, and ``metadata`` gains
            ``q_index_range=(start, stop)`` and ``fft_order=True``.

            Bitwise consistency with the full run: the value of one q pixel is a
            pure function of ``(iq1, iq2)``, ``nk``, ``nw`` and the blocking
            parameters ``m_block``/``row_block``, which follow from ``nk``/``nw``
            alone and therefore do not change with the row range.  A slice only
            selects which ``iq1`` values are iterated, so every returned pixel is
            produced by exactly the same floating-point operations as in the
            full run and the slabs equal the corresponding raw rows of the
            full-mesh result bit for bit.
        progress_interval_s : float, default 30.0
            Throttling period of the ``progress`` log record (one record per
            completed ``iq1`` row, at most one per interval; a final record is
            always emitted for the completed rows).  Values ``<= 0`` switch the
            per-row progress records off.

        Returns
        -------
        dict
            ``'data'``: total Re chi0(q, 0) with q=(0,0) at index
            ``(nk//2, nk//2)``; ``'intraband'`` / ``'interband'``: the n == m
            and n != m parts of the same sum (``data == intraband +
            interband``); ``'q1_grid'``, ``'q2_grid'``: fractional grids
            matching the data; ``'qx_grid'``, ``'qy_grid'``: reciprocal
            coordinates in 1/Angstrom (None when ``bvecs`` is None);
            ``'metadata'``: calculation parameters.  All data arrays are real
            ``float64`` in the doc-convention sign (Re chi0 >= 0).  With
            ``q_index_range`` set the three data arrays and the two fractional
            grids have shape ``(stop - start, nk)`` and stay in raw FFT order.

        Notes
        -----
        The run is observable through the module logger (no handler is
        configured here): one configuration echo, a BLAS thread advisory, one
        ``stage=...`` timing record per stage, throttled ``progress`` records
        and a closing summary with the three array digests and the
        ``max|data-(intra+inter)|`` residual.  Peak RSS is sampled with the
        standard library only (:func:`peak_rss_bytes`).
        """
        calculation_started = time.perf_counter()
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if q_chunk_size <= 0:
            raise ValueError(f"q_chunk_size must be positive, got {q_chunk_size}")
        if degeneracy_tolerance < 0:
            raise ValueError("degeneracy_tolerance must be non-negative")

        nw = self.ham.num_wann
        if orbital_select is None:
            orbitals = np.arange(nw)
        else:
            orbitals = np.asarray(orbital_select, dtype=int)
            if orbitals.ndim != 1 or len(orbitals) == 0:
                raise ValueError(
                    "orbital_select must be a non-empty one-dimensional sequence"
                )
            if np.any((orbitals < 0) | (orbitals >= nw)):
                raise ValueError(f"orbital_select entries must be in [0, {nw})")

        # Resolve the engine blocking parameters before the heavy stages so the
        # configuration echo can report the values the run will actually use.
        # ``None`` falls back to the module constant at call time, which keeps
        # both an explicit constructor value and a patched module constant
        # effective.  Both parameters are performance knobs: they select how the
        # same terms are blocked, so they only perturb the summation order.
        band_block = _MAX_BAND_BLOCK if self.band_block is None else self.band_block
        block_entries = (
            _MAX_BLOCK_ENTRIES if self.block_entries is None else self.block_entries
        )
        m_block = min(band_block, nw)
        row_block = max(1, block_entries // (m_block * nw))

        slice_mode = q_index_range is not None
        if slice_mode:
            if output_path is not None:
                raise ValueError(
                    "q_index_range cannot be combined with output_path: a slice "
                    "is returned in raw FFT order and is never written to HDF5"
                )
            if q_range is not None:
                raise ValueError(
                    "q_index_range cannot be combined with q_range: a slice is "
                    "not fftshifted and cannot be periodically extended"
                )
            row_start, row_stop = self._validate_q_index_range(q_index_range)
        else:
            row_start, row_stop = 0, self.nk
        rows_total = row_stop - row_start
        pixels_total = rows_total * self.nk

        if len(orbitals) <= 8:
            orbital_text = f"n={len(orbitals)} list={[int(o) for o in orbitals]}"
        else:
            orbital_text = (
                f"n={len(orbitals)} first={int(orbitals[0])} "
                f"last={int(orbitals[-1])}"
            )
        logger.info(
            "[RealLindhardCalculator] config nk=%d nw=%d orbitals=%s eta=%.6e "
            "temperature=%s chemical_potential=%s include_matrix_elements=%s "
            "degeneracy_tolerance=%.3e band_block=%d block_entries=%d row_chunk=%d "
            "q_index_range=%s q_pixels=%d pid=%d cpu_count=%s",
            self.nk,
            nw,
            orbital_text,
            self.eta,
            temperature,
            chemical_potential,
            include_matrix_elements,
            degeneracy_tolerance,
            m_block,
            block_entries,
            row_block,
            f"({row_start}, {row_stop})" if slice_mode else "full",
            pixels_total,
            os.getpid(),
            os.cpu_count(),
        )
        thread_message, thread_offenders = blas_thread_advisory()
        logger.info("[RealLindhardCalculator] %s", thread_message)
        if thread_offenders:
            logger.warning(
                "[RealLindhardCalculator] BLAS thread advice: %s is unset or > 1 "
                "(current values above). The BLAS thread pool is created while "
                "NumPy is imported, so it can no longer be changed from here; a "
                "multi-process run then accumulates these threads per worker and "
                "the workers compete for the same cores. Export "
                "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS="
                "VECLIB_MAXIMUM_THREADS=1 before importing NumPy.",
                ", ".join(thread_offenders),
            )

        if include_matrix_elements:
            with _stage(
                "diagonalize",
                rows=self.nk * self.nk,
                blocks=-(-self.nk * self.nk // _HK_ROW_BLOCK),
            ):
                evals, evecs_orb = self._diagonalize(orbitals)
        else:
            # The scalar vertex is orbital independent, so the eigen stage must
            # not store the (large) eigenvector array at all.
            with _stage(
                "diagonalize",
                rows=self.nk * self.nk,
                blocks=-(-self.nk * self.nk // _HK_ROW_BLOCK),
            ):
                evals, _ = self._diagonalize(None)
            evecs_orb = None

        nk1, nk2 = evals.shape[0], evals.shape[1]
        num_kpoints = nk1 * nk2

        logger.info(
            "[RealLindhardCalculator] Re chi0 on %dx%d k-mesh, %d bands, "
            "%d selected orbitals, eta=%.3e eV, T=%.1f K",
            nk1,
            nk2,
            nw,
            len(orbitals),
            self.eta,
            temperature,
        )

        with _stage("occupations"):
            occupations = fermi(evals, mu=chemical_potential, T=temperature)
            derivative = self._fermi_derivative(
                evals, chemical_potential, temperature
            )
        total = np.zeros((rows_total, nk2), dtype=float)
        intraband = np.zeros_like(total)
        interband = np.zeros_like(total)
        band_index = np.arange(nw)

        # Blocking of the vectorized q-sum: band-m columns are handled in
        # band_block-wide blocks and the k rows of one (q, band) block are
        # chunked so that the (rows, m, n) working tensors stay near 1 GB.  For
        # nk<=64 a single row chunk covers the whole mesh.  Both parameters come
        # from the constructor, with the module constants as fallback.
        progress_enabled = progress_interval_s > 0
        rows_done = 0
        peak_rss = peak_rss_bytes()
        q_sum_started = time.perf_counter()
        # The first record is due one interval after the loops start, so a run
        # shorter than the interval carries the closing record only.
        last_progress_time = q_sum_started
        last_progress_rows = 0

        for iq1 in range(row_start, row_stop):
            irow = iq1 - row_start
            for iq2 in range(nk2):
                q_total = 0.0
                q_intra = 0.0
                # Cyclic k -> k+q shift, kept as destination/source views.
                for dest1, src1 in _shifted_ranges(nk1, iq1):
                    for dest2, src2 in _shifted_ranges(nk2, iq2):
                        width2 = dest2.stop - dest2.start
                        step = max(1, row_block // width2)
                        for r0 in range(0, dest1.stop - dest1.start, step):
                            r1 = min(r0 + step, dest1.stop - dest1.start)
                            d1 = slice(dest1.start + r0, dest1.start + r1)
                            s1 = slice(src1.start + r0, src1.start + r1)
                            eps_n = evals[d1, dest2]
                            eps_m = evals[s1, src2]
                            f_n = occupations[d1, dest2]
                            f_m = occupations[s1, src2]
                            d_n = derivative[d1, dest2]

                            for m0 in range(0, nw, m_block):
                                m1 = min(m0 + m_block, nw)
                                # Diagonal entries of this band block, i.e. the
                                # (m, n) pairs with n == m (doc 8.2).
                                picks = band_index[m0:m1][None, None, :, None]
                                # Delta[k, m, n] = eps_m(k+q) - eps_n(k).  The
                                # textbook ratio is invariant under swapping
                                # (m, n) and flipping both signs, so this
                                # transposed layout carries the same sum.
                                delta = eps_m[..., m0:m1, None] - eps_n[..., None, :]
                                f_difference = (
                                    f_m[..., m0:m1, None] - f_n[..., None, :]
                                )
                                ratio = f_difference * delta / (
                                    delta * delta + self.eta**2
                                )
                                # Doc 8.2: replace the 0/0 intraband ratio by
                                # df/dE before applying the overall minus sign.
                                diag_delta = np.take_along_axis(
                                    delta, picks, axis=3
                                )[..., 0]
                                replaced = np.abs(diag_delta) <= degeneracy_tolerance
                                diag_ratio = np.take_along_axis(
                                    ratio, picks, axis=3
                                )[..., 0]
                                if np.any(replaced):
                                    diag_ratio = np.where(
                                        replaced, d_n[..., m0:m1], diag_ratio
                                    )
                                    np.put_along_axis(
                                        ratio, picks, diag_ratio[..., None], axis=3
                                    )

                                if evecs_orb is not None:
                                    # Doc 10.3: M_mn = sum_a U*_{a m}(k+q) U_{a n}(k).
                                    # The batched matmul contracts the selected
                                    # orbitals and yields [k, m, n].
                                    u_n = evecs_orb[d1, dest2]
                                    u_m = evecs_orb[s1, src2][..., :, m0:m1]
                                    overlap = np.matmul(
                                        u_m.conj().swapaxes(-2, -1), u_n
                                    )
                                    weight = overlap.real**2 + overlap.imag**2
                                    q_total += float(
                                        np.einsum("kcmn,kcmn->", weight, ratio)
                                    )
                                    diag_weight = np.take_along_axis(
                                        weight, picks, axis=3
                                    )[..., 0]
                                    q_intra += float(
                                        np.einsum("kcm,kcm->", diag_weight, diag_ratio)
                                    )
                                else:
                                    q_total += float(np.einsum("kcmn->", ratio))
                                    q_intra += float(diag_ratio.sum())

                # Doc 7.1: the overall minus sign is applied to the accumulated
                # k-sum of this q pixel.
                total[irow, iq2] = -q_total / num_kpoints
                intraband[irow, iq2] = -q_intra / num_kpoints
                interband[irow, iq2] = -(q_total - q_intra) / num_kpoints

            # One progress record per completed iq1 row, throttled by wall time.
            # Emitting it here (and not inside the iq2 loop) is what keeps the
            # record count independent of the mesh size.
            rows_done = irow + 1
            now = time.perf_counter()
            if progress_enabled and now - last_progress_time >= progress_interval_s:
                peak_rss = max(peak_rss, peak_rss_bytes())
                _log_progress(rows_done, rows_total, nk2, q_sum_started, peak_rss)
                last_progress_time = now
                last_progress_rows = rows_done

        q_sum_elapsed = time.perf_counter() - q_sum_started
        peak_rss = max(peak_rss, peak_rss_bytes())
        logger.info(
            "[RealLindhardCalculator] stage=q_sum pixels=%d s=%.3f px_per_s=%.1f",
            pixels_total,
            q_sum_elapsed,
            pixels_total / max(q_sum_elapsed, 1e-9),
        )
        if progress_enabled and rows_done > last_progress_rows:
            # Always report the completed rows, so even a run much shorter than
            # progress_interval_s carries one machine-readable progress record.
            _log_progress(rows_done, rows_total, nk2, q_sum_started, peak_rss)

        if slice_mode:
            # A slice is returned in raw FFT order: no fftshift here.
            logger.info(
                "[RealLindhardCalculator] stage=fftshift skipped=True "
                "reason=q_index_range_returns_raw_fft_order"
            )
            q_values = np.fft.fftfreq(self.nk)
            q1_grid, q2_grid = np.meshgrid(
                q_values[row_start:row_stop], q_values, indexing="ij"
            )
        else:
            # The raw array is in FFT order (integer index iq carries the
            # transfer q = iq/nk mod 1), so q=(0,0) moves to index
            # (nk//2, nk//2) here.
            with _stage("fftshift"):
                total = np.fft.fftshift(total, axes=(0, 1))
                intraband = np.fft.fftshift(intraband, axes=(0, 1))
                interband = np.fft.fftshift(interband, axes=(0, 1))
            q_values = np.fft.fftshift(np.fft.fftfreq(self.nk))
            q1_grid, q2_grid = np.meshgrid(q_values, q_values, indexing="ij")

        # Save the primitive-BZ result before any q_range cropping.
        if output_path is not None:
            with _stage("h5_write", path=str(output_path)) as h5_fields:
                save_susceptibility_to_h5(
                    susceptibility=total,
                    output_path=output_path,
                    module_type=_MODULE_TYPE,
                    bvecs=self.ham.bvecs,
                    eta=self.eta,
                    nq=self.nk,
                    chemical_potential=chemical_potential,
                    temperature=temperature,
                )
                h5_fields["bytes"] = Path(output_path).stat().st_size

        # Optionally extend/crop to the requested q-window (doc 9).
        if q_range is not None:
            with _stage("extend", q_range=tuple(q_range)):
                logger.info(
                    "Extending Re chi0 to the fractional q-range [%s, %s)",
                    q_range[0],
                    q_range[1],
                )
                base_q1_grid, base_q2_grid = q1_grid, q2_grid
                total, q1_grid, q2_grid = extend_qpi(
                    total, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
                )
                # The intraband/interband split must follow the same crop so that
                # data == intraband + interband still holds afterwards.
                intraband, _, _ = extend_qpi(
                    intraband, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
                )
                interband, _, _ = extend_qpi(
                    interband, base_q1_grid, base_q2_grid, q_range[0], q_range[1]
                )

        qx_grid, qy_grid = frac_to_real_2d(q1_grid, q2_grid, self.ham.bvecs)

        metadata: dict[str, Any] = {
            "module_type": _MODULE_TYPE,
            "eta": self.eta,
            # "nk" is kept for backward compatibility with the pre-merge module;
            # "nq" matches the sibling susceptibility modules and the HDF5 file.
            "nk": self.nk,
            "nq": self.nk,
            "bvecs": self.ham.bvecs,
            "chemical_potential": chemical_potential,
            "temperature": temperature,
            "include_matrix_elements": include_matrix_elements,
            "orbital_select": orbitals,
            "degeneracy_tolerance": degeneracy_tolerance,
            "note": (
                "Re chi0 in the document convention (doc 7.1), fftshifted so "
                "that q=(0,0) sits at index (nk//2, nk//2)"
            ),
        }
        if slice_mode:
            metadata["q_index_range"] = (int(row_start), int(row_stop))
            metadata["fft_order"] = True
            metadata["note"] = (
                "Re chi0 in the document convention (doc 7.1), raw FFT-order "
                "rows [q_index_range) of the primitive mesh (no fftshift)"
            )

        residual = float(np.max(np.abs(total - (intraband + interband))))
        logger.info(
            "[RealLindhardCalculator] summary rows=%d pixels=%d bands=%d "
            "total_s=%.3f px_per_s=%.1f rss_peak=%s max|data-(intra+inter)|=%.3e "
            "q_index_range=%s",
            rows_total,
            pixels_total,
            nw,
            time.perf_counter() - calculation_started,
            pixels_total / max(time.perf_counter() - calculation_started, 1e-9),
            format_bytes(peak_rss),
            residual,
            f"({row_start}, {row_stop})" if slice_mode else "full",
        )
        for digest_name, digest_array in (
            ("data", total),
            ("intraband", intraband),
            ("interband", interband),
        ):
            digest = array_digest(digest_array)
            logger.info(
                "[RealLindhardCalculator] digest %s sum=%.12e max=%.12e min=%.12e",
                digest_name,
                digest["sum"],
                digest["max"],
                digest["min"],
            )

        logger.info("[RealLindhardCalculator] Re chi0 calculation completed.")

        return {
            "data": total,
            "intraband": intraband,
            "interband": interband,
            "q1_grid": q1_grid,
            "q2_grid": q2_grid,
            "qx_grid": qx_grid,
            "qy_grid": qy_grid,
            "metadata": metadata,
        }
