# mlwf_susceptibility.py API Documentation

This document provides detailed information about the `stm_data_processing.dft.wannier90.mlwf_susceptibility` module, including its API usage, data structures, and the underlying physical formulas. This module is primarily used for calculating static Lindhard susceptibility ($\mathrm{Im}[\chi(\mathbf{q, \omega})]$) based on tight-binding Hamiltonians.

## 1. Module Overview

The `SusceptibilityCalculator_wang2012` class uses the Green's function method to calculate magnetic susceptibility in real or reciprocal space by integrating the single-particle spectral function. It supports both CPU (NumPy/pyFFTW) and GPU (CuPy) backends for acceleration.

**Reference:**

DOI: <https://doi.org/10.1103/PhysRevB.85.224529>

**Dependencies:**

- `mlwf_hamiltonian.MLWFHamiltonian`: Provides $H(\mathbf{k})$ calculation.
- `mlwf_gk.GreenFunction`: Provides $G(\mathbf{k}, \omega)$ calculation.

## 2. Core Class: SusceptibilityCalculator_wang2012

### 2.1 Initialization

```python src/STM_DataProcessing/src/stm_data_processing/dft/wannier90/mlwf_susceptibility.py
class SusceptibilityCalculator_wang2012:
    def __init__(
        self,
        hamiltonian: MLWFHamiltonian,
        nk: int = 256,
        eta: float = 5e-3,
        minit: np.ndarray | None = None,
        mfin: np.ndarray | None = None,
    ):
        # ...
```

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `hamiltonian` | `MLWFHamiltonian` | Required | Initialized Wannier Hamiltonian object. |
| `nk` | `int` | `256` | k-space grid density (nk x nk). Grid range is [-0.5, 0.5). |
| `eta` | `float` | `5e-3` | Broadening factor (eV) in Green's function, i.e., imaginary energy part $i\eta$. |
| `minit` | `np.ndarray` | `None` | Initial state orbital selection matrix. If `None`, uses identity matrix. |
| `mfin` | `np.ndarray` | `None` | Final state orbital selection matrix. If `None`, uses identity matrix. |

**Properties:**

- `num_wann`: Number of Wannier functions.
- `hk_grid`: Cached $H(\mathbf{k})$ grid data (lazy-loaded).
- `gf`: `GreenFunction` instance (lazy-loaded).
- `k_points`: Generated fractional k-point grid coordinates `(N, 3)`.
- `k1_grid`, `k2_grid`: 2D k-space grid arrays, shape `(nk, nk)`.
- `q1_grid`, `q2_grid`: 2D q-space grid arrays, shape `(nk, nk)`.
- `minit`, `mfin`: Orbital selection matrices, shape `(num_wann, num_wann)`.
- `xp`: Backend array module (`numpy` or `cupy`).

### 2.2 Main Method: calculate

This is the primary entry point for users, executing the complete susceptibility calculation workflow.

```python src/STM_DataProcessing/src/stm_data_processing/dft/wannier90/mlwf_susceptibility.py
    def calculate(
        self,
        omega_limit: float,
        resolution: float,
        q_range: tuple[float, float] | None = (-0.5, 0.5),
        output_path: str | None = None,
    ) -> dict[str, Any]:
        # ...
```

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `omega_limit` | `float` | Required | Energy integration limit (eV). Integration range is $[-\vert\text{limit}\vert, 0]$. |
| `resolution` | `float` | Required | Energy integration step size (eV). |
| `q_range` | `tuple` | `(-0.5, 0.5)` | Cropping range for output q-space. If `None`, no cropping is applied. |
| `output_path` | `str` | `None` | Optional. If provided, saves results to an HDF5 file. |

**Returns:** `dict[str, Any]`

A dictionary containing calculation results and grid information.

| Key | Type | Description |
| :--- | :--- | :--- |
| `data` | `ndarray` | Calculated $\chi_0(\mathbf{q})$ data, shape `(nq, nq)`. |
| `q1_grid`, `q2_grid` | `ndarray` | Fractional q-grids, shape `(nq, nq)`. |
| `qx_grid`, `qy_grid` | `ndarray` | Real-space reciprocal lattice q-grids (if `bvecs` available). |
| `metadata` | `dict` | Contains `eta`, `omega_limit`, `bvecs`, and other calculation parameters. |

### 2.3 Additional Methods

| Method | Description |
| :--- | :--- |
| `set_orbital_selection(minit, mfin)` | Update orbital selection matrices after initialization. |
| `clear_cache()` | Clear cached data (`hk_grid`, `gf`, `k_points`, etc.). Call after switching backend. |

## 3. Formulas <-> Coding

The calculation process consists of four main steps, utilizing Fast Fourier Transform (FFT) to accelerate k-space accumulation.

### 3.1 Hamiltonian Construction $H(\mathbf{k})$

First, construct the Wannier Hamiltonian on a uniform k-grid.

$$H(\mathbf{k}) = \sum_{\mathbf{R}} \frac{e^{i 2\pi \mathbf{R} \cdot \mathbf{k}}}{N_{\text{degen}}(\mathbf{R})} H(\mathbf{R})$$

- **Corresponding Code**: `self.ham.hk(self.k_points)`
- **Data Structure**: Complex array of shape `(N_k, num_wann, num_wann)`.

### 3.2 Retarded Green's Function Calculation $G^R(\mathbf{k}, \omega)$

For each energy point $\omega$, compute the retarded Green's function:
$$G^R(\mathbf{k}, \omega) = (\omega + i\eta - H(\mathbf{k}))^{-1}$$

- **Corresponding Code**: `self.gf.compute_green(hk_grid, omega)`
- **Formula Details**: Obtained by solving the linear system $ (\omega + i\eta - H) X = I $ for $G^R$.

### 3.3 Single-Particle Spectral Function $A(\mathbf{k}, \omega)$

Calculate the spectral function, i.e., the imaginary part of the Green's function:
$$ A(\mathbf{k}, \omega) = -\frac{1}{\pi} \text{Im}[G^R(\mathbf{k}, \omega)] $$

- **Corresponding Code**: `_compute_single_particle_spectra`
- **Data Structure**: Real array of shape `(N_k, num_wann, num_wann)` (before trace operation).
- **Note**: The code retains the matrix-form spectral function for subsequent Wannier index contraction.

### 3.4 Imaginary Part of Lindhard Function $\mathrm{Im}[\chi^L(\mathbf{q},\omega)]$

#### 3.4.1 Simplification of Finite-temperature Lindhard Function

$$\begin{aligned}
\chi^L(\mathbf{q},\omega) = -\frac{1}{2\pi i} \int\frac{d^3k}{(2\pi)^3} \int d\epsilon f(\epsilon) \text{Tr} \big[
& M_{init} G^R(\mathbf{k},\epsilon) M_{fin} G^R(\mathbf{k}+\mathbf{q}, \epsilon+\omega) \\
- & M_{init} G^A(\mathbf{k},\epsilon) M_{fin} G^A(\mathbf{k}+\mathbf{q}, \epsilon+\omega) \big]
\end{aligned}$$

$$\begin{aligned}
\chi^L(\mathbf{q},\omega) = \int\frac{d^3k}{(2\pi)^3} \int d\epsilon f(\epsilon) \text{Tr} \big[
& M_{init} G^R(\mathbf{k},\epsilon) M_{fin} A(\mathbf{k}+\mathbf{q}, \epsilon+\omega) \\
- & M_{init} A(\mathbf{k},\epsilon) M_{fin} G^A(\mathbf{k}+\mathbf{q}, \epsilon+\omega) \big]
\end{aligned}$$

$$ \mathrm{Im}\big[\chi^L(\mathbf{q},\omega)\big] = \pi \int\frac{d^3k}{(2\pi)^3}\int d\epsilon [f(\epsilon)-f(\epsilon+\omega)] \text{Tr}\big[M_{init}A(\mathbf{k}, \epsilon)M_{fin}A(\mathbf{k}+\mathbf{q}, \epsilon+\omega)\big] $$

#### 3.4.2 Zero-temperature Lindhard Function

$$\mathrm{Im}[\chi^L(\mathbf{q},\omega)] = \int_{-\omega}^{0} d\epsilon \int\frac{d^3k}{(2\pi)^3}M_{init}A(\mathbf{k}, \epsilon)M_{fin}A(\mathbf{k}+\mathbf{q}, \epsilon+\omega) $$The code uses FFT to accelerate the integral process (Convolution Theorem):$$\int d^3k f(\mathbf{k})g(\mathbf{k}+\mathbf{q})=\int d^3r \tilde{f}(\mathbf{r})\tilde{g}(-\mathbf{r})e^{-i\mathbf{q\cdot r}}$$

1. **Energy Grid Generation**: Discretize the integration range $[-\omega, 0]$ into `n_eps` points.

   $$ \epsilon_{\text{occ}} \in [-\omega, 0], \quad \epsilon_{\text{unocc}} = \epsilon_{\text{occ}} + \omega $$

   ```python
   n_eps = int(np.round(np.abs(omega_limit) / resolution)) + 1
   eps_occ = np.linspace(-np.abs(omega_limit), 0.0, n_eps)
   eps_unocc = eps_occ + omega_limit
   ```

2. **Spectral Function Calculation**: Compute $A(\mathbf{k}, \epsilon)$ for occupied and unoccupied states.

   $$ A_{\text{occ}}(\mathbf{k}, \epsilon) = A(\mathbf{k}, \epsilon_{\text{occ}}), \quad A_{\text{unocc}}(\mathbf{k}, \epsilon+\omega) = A(\mathbf{k}, \epsilon_{\text{unocc}}) $$

   ```python
   spectra_occ = self._compute_single_particle_spectra(eps_occ[i])  # A(k, ε)
   spectra_unocc = self._compute_single_particle_spectra(eps_unocc[i])  # A(k, ε+ω)
   ```

3. **Orbital Selection**: Apply $M_{init}$ and $M_{fin}$ matrices via Einstein summation.

   $$ A_{\text{occ}} \leftarrow M_{init} \cdot A_{\text{occ}}, \quad A_{\text{unocc}} \leftarrow M_{fin} \cdot A_{\text{unocc}} $$

   ```python
   spectra_occ = np.einsum("ac,ijcb->ijab", self._minit, spectra_occ)
   spectra_unocc = np.einsum("ac,ijcb->ijab", self._mfin, spectra_unocc)
   ```

4. **Fourier Transform**: Apply FFT to $A(\mathbf{k})$ over k-space (axes 0,1).

   $$\tilde{A}_{init}(\mathbf{r}, \epsilon) = \mathcal{F}_{\mathbf{k} \to \mathbf{r}} [M_{init}A(\mathbf{k}, \epsilon) ] = \int\frac{d^3k}{(2\pi)^3} M_{init}A(\mathbf{k}, \epsilon)e^{i\mathbf{k\cdot r}} $$

   $$\tilde{A}_{fin}(\mathbf{r}, \epsilon+\omega) = \mathcal{F}_{\mathbf{k} \to \mathbf{r}} [M_{fin}A(\mathbf{k}, \epsilon+\omega)] = \int\frac{d^3k}{(2\pi)^3} M_{fin}A(\mathbf{k}, \epsilon+\omega)e^{i\mathbf{k\cdot r}}$$

   ```python
   b_occ = np.fft.fftn(spectra_occ, axes=(0, 1))  # F[k→r]
   b_occ_shifted = np.fft.fftshift(b_occ, axes=(0, 1))
   b_unocc = np.fft.fftn(spectra_unocc, axes=(0, 1))
   b_unocc_shifted = np.fft.fftshift(b_unocc, axes=(0, 1))
   ```

5. **Real-space Product**: Contract Wannier indices and multiply in real space.

   $$ P(\mathbf{r}, \epsilon, \omega) = \tilde{A}_{init}(\mathbf{r}, \epsilon)\tilde{A}_{fin}(-\mathbf{r}, \epsilon + \omega)$$

   ```python
   b_prod = np.einsum("ijab,ijba->ij", b_occ_shifted, b_unocc_shifted)  # Tr[M_init·A·M_fin·A]
   ```

6. **Inverse Fourier Transform**: Transform back to q-space.

   $$ f(\mathbf{q}, \epsilon, \omega) = \mathcal{F}^{-1}_{\mathbf{r} \to \mathbf{q}} [ P(\mathbf{r}, \epsilon, \omega) ] = \int d^3r \tilde{A}_{init}(\mathbf{r}, \epsilon)\tilde{A}_{fin}(-\mathbf{r}, \epsilon + \omega)e^{-i\mathbf{q\cdot r}} $$

   ```python
   conv_q = np.fft.ifftn(
       np.fft.ifftshift(b_prod, axes=(0, 1)), axes=(0, 1)
   ).real  # F⁻¹[r→q]
   ```

7. **Energy Integration**: Accumulate contributions from all energy points.

   $$ \mathrm{Im}[\chi^L(\mathbf{q},\omega)] = \int_{-\omega}^0 d\epsilon f(\mathbf{q}, \epsilon, \omega) $$

   ```python
   chi_q_accum += conv_q  # Σ_ε f(q, ε, ω)
   ```

8. **Final Scaling**: Apply prefactor and shift q-grid to centered coordinates.

   $$ \chi(\mathbf{q}) = -\frac{\Delta\epsilon}{2\pi} \cdot \text{fftshift}(\chi_{\text{accum}}) $$

   ```python
   chi_q = np.fft.fftshift(chi_q_accum)
   chi_q = -np.abs(resolution) / (2 * np.pi) * chi_q  # Δε/(2π) prefactor
   ```

- **Corresponding Code**: `_compute_imag_chi` (CPU) or `_compute_imag_chi_cuda` (GPU).
- **GPU Optimization**: The GPU version accumulates energy points one by one to save VRAM, avoiding storage of all energy slices at once.

## 4. Data Structure Details

### 4.1 Input Object: MLWFHamiltonian

`SusceptibilityCalculator_wang2012` strongly depends on the `MLWFHamiltonian` instance.

- **Required Attributes**:
  - `num_wann`: int
  - `bvecs`: `(3, 3)` reciprocal lattice vector matrix (for converting to real-space q-grids).
  - `hk(k_points)`: Method returning `(N, num_wann, num_wann)` array.

### 4.2 Output Grid Coordinates

The calculation results include two sets of coordinate grids:

1. **Fractional Grids (`q1_grid`, `q2_grid`)**:
   - Range: Default $[-0.5, 0.5)$ or cropped according to `q_range`.
   - Units: Fractional reciprocal lattice units (multiples of reciprocal lattice vectors).
   - Shape: `(nq, nq)`.

2. **Real-space Grids (`qx_grid`, `qy_grid`)**:
   - Formula: $\mathbf{q}_{\text{real}} = q_1 \mathbf{b}_1 + q_2 \mathbf{b}_2$.
   - Units: $\text{\AA}^{-1}$ (depends on `bvecs` units).
   - Condition: Only calculated if `hamiltonian.bvecs` is not `None`.

### 4.3 Memory Layout (GPU vs CPU)

- **CPU**: Uses `numpy.ndarray`. Supports pyFFTW for accelerated FFT operations.
  - **pyFFTW Optimization**: When available, uses multi-threaded FFT plans with wisdom caching for repeated calculations.
  - Memory usage is proportional to `nk^2 * num_wann^2 * n_energy`.
- **GPU**: Uses `cupy.ndarray`.
  - **VRAM Optimization**: The GPU implementation does not store spectral functions for all energy points. Instead, it calculates each $\omega$, transforms it immediately, accumulates to `chi_q_accum`, and then releases VRAM (`mem_pool.free_all_blocks()`).
  - Memory pool limit is set to 75% of available GPU memory (based on 24GB reference).
  - Suitable for scenarios with large `nk` but limited VRAM.

### 4.4 Orbital Selection Matrices

The `minit` and `mfin` matrices allow selective orbital contributions to the susceptibility:

- **Default**: Identity matrices (all orbitals contribute equally).
- **Custom**: Can be used to project onto specific orbital subspaces.
- **Shape**: Must be `(num_wann, num_wann)`.
- **Application**: Applied via `np.einsum("ac,ijcb->ijab", matrix, spectra)` before FFT.

## 5. Usage Example

```python src/STM_DataProcessing/examples/calculate_susceptibility.py
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.dft.wannier90.mlwf_susceptibility import SusceptibilityCalculator_wang2012
import numpy as np

# 1. Load Hamiltonian
ham = MLWFHamiltonian.from_seedname(folder="./wannier", seedname="material")

# 2. Initialize calculator
calculator = SusceptibilityCalculator_wang2012(
    hamiltonian=ham,
    nk=256,          # k-grid density
    eta=5e-3,        # broadening
    # Optional: orbital selection matrices
    # minit=np.eye(ham.num_wann),
    # mfin=np.eye(ham.num_wann),
)

# 3. Execute calculation
results = calculator.calculate(
    omega_limit=1.0,      # integration limit 1.0 eV
    resolution=0.01,      # energy step 0.01 eV
    q_range=(-0.5, 0.5),  # output q range
    output_path="./output/susceptibility.h5"
)

# 4. Access results
chi_data = results["data"]
qx = results["qx_grid"]
qy = results["qy_grid"]

print(f"Susceptibility shape: {chi_data.shape}")

# 5. Update orbital selection (optional)
# calculator.set_orbital_selection(minit=new_minit, mfin=new_mfin)

# 6. Clear cache when switching backend
# calculator.clear_cache()
```

## 6. Notes

1. **Backend Configuration**: Controlled via `stm_data_processing.config.BACKEND`. If set to `"gpu"` and `cupy` is installed, CUDA acceleration is automatically enabled. Call `clear_cache()` after switching backend.
2. **Energy Integration Range**: The code defaults to integration range `[-omega_limit, 0]` (below Fermi level).
   - **Physical Basis**: At $T=0$, susceptibility arises from electron-hole excitations. Transitions occur from occupied states ($\epsilon < 0$) to unoccupied states ($\epsilon + \omega > 0$). This constrains the initial energy to $-\omega < \epsilon < 0$.
   - **Implementation**: This ensures only valid transitions across the Fermi level are counted. To adjust, modify the `eps` generation logic in `_compute_imag_chi`.
3. **HDF5 Saving**: If `output_path` is provided, the original un-cropped data is saved to an HDF5 file for subsequent re-cropping or analysis.
4. **Periodic Boundary Conditions**: Calculation is based on periodic boundary conditions; q-space results are periodic. The `extend_qpi` function handles q-space repetition/expansion.
5. **Class Naming**: The class is named `SusceptibilityCalculator_wang2012` to indicate the method follows the approach from Wang et al. (2012).
6. **pyFFTW Wisdom**: FFTW plans are cached in `fftw_wisdom/` directory for faster initialization on subsequent runs with the same `nk` and `num_wann`.
7. **GPU Memory Management**: GPU implementation includes automatic memory pool management with periodic cleanup every 20 spectral function computations.
