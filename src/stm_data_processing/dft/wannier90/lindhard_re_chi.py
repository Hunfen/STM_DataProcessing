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
* Memory blocking: band-m columns are processed in ``_MAX_BAND_BLOCK``-wide
  blocks and the k rows of one block are chunked so that the working tensors
  stay near 1 GB, which keeps nk=256 inside a 16 GB machine.  ``H(k)`` is
  evaluated in ``_HK_ROW_BLOCK``-row blocks, which is what keeps the
  75-orbital model (``num_wann**2 = 5625`` columns) away from the Apple
  Accelerate zgemm segfault that an unblocked 1024-row build triggers.
* The eigenvector array is materialized only for the selected orbitals and only
  when ``include_matrix_elements=True``: the scalar vertex needs no
  eigenvectors at all.

Optional outputs
----------------
``calculate(..., output_path=...)`` writes the fftshifted primitive-BZ
result to an HDF5 file (before any ``q_range`` cropping), and
``calculate(..., q_range=(qmin, qmax))`` periodically extends and crops the
result to ``[qmin, qmax)`` through :func:`extend_qpi` (doc section 9).
"""

from __future__ import annotations

import logging
from functools import cached_property
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

    def __init__(self, hamiltonian: MLWFHamiltonian, nk: int = 256, eta: float = 5e-3):
        if not isinstance(hamiltonian, MLWFHamiltonian):
            raise TypeError(
                f"Expected MLWFHamiltonian, got {type(hamiltonian).__name__}"
            )
        if nk <= 0:
            raise ValueError(f"nk must be positive, got {nk}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")

        self.ham = hamiltonian
        self.nk = nk
        self.eta = eta

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
            ``float64`` in the doc-convention sign (Re chi0 >= 0).
        """
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

        if include_matrix_elements:
            evals, evecs_orb = self._diagonalize(orbitals)
        else:
            # The scalar vertex is orbital independent, so the eigen stage must
            # not store the (large) eigenvector array at all.
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

        occupations = fermi(evals, mu=chemical_potential, T=temperature)
        derivative = self._fermi_derivative(evals, chemical_potential, temperature)
        total = np.zeros((nk1, nk2), dtype=float)
        intraband = np.zeros_like(total)
        interband = np.zeros_like(total)
        band_index = np.arange(nw)

        # Blocking of the vectorized q-sum: band-m columns are handled in
        # _MAX_BAND_BLOCK-wide blocks and the k rows of one (q, band) block are
        # chunked so that the (rows, m, n) working tensors stay near 1 GB.  For
        # nk<=64 a single row chunk covers the whole mesh.
        m_block = min(_MAX_BAND_BLOCK, nw)
        row_block = max(1, _MAX_BLOCK_ENTRIES // (m_block * nw))

        for iq1 in range(nk1):
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
                total[iq1, iq2] = -q_total / num_kpoints
                intraband[iq1, iq2] = -q_intra / num_kpoints
                interband[iq1, iq2] = -(q_total - q_intra) / num_kpoints

        # The raw array is in FFT order (integer index iq carries the transfer
        # q = iq/nk mod 1), so q=(0,0) moves to index (nk//2, nk//2) here.
        total = np.fft.fftshift(total, axes=(0, 1))
        intraband = np.fft.fftshift(intraband, axes=(0, 1))
        interband = np.fft.fftshift(interband, axes=(0, 1))
        q_values = np.fft.fftshift(np.fft.fftfreq(self.nk))
        q1_grid, q2_grid = np.meshgrid(q_values, q_values, indexing="ij")

        # Save the primitive-BZ result before any q_range cropping.
        if output_path is not None:
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

        # Optionally extend/crop to the requested q-window (doc 9).
        if q_range is not None:
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
