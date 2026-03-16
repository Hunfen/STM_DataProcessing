# MLWF EK2D Module Interface Documentation

## Module Overview

`mlwf_ek2d.py` provides 2D band structure (E-k) calculation functionality based on the Wannier90 MLWF Hamiltonian, supporting both CPU and GPU backends.

**Backend Selection**: Inherited from the `mlwf_hamiltonian` module, automatically determined via the `BACKEND` constant without additional configuration.

**k-grid**: Fixed to the range `[-0.5, 0.5)`, with `nkx = nky = nk`.

**Data Saving Strategy**:

- HDF5 saving uses the original grid `[-0.5, 0.5)` (saves storage space).
- `calculate()` returns data expanded according to the `k_range` parameter (convenient for subsequent use).
- The return structure is fully consistent with `EK2DIO.load_ek2d()`.

---

## Core Class: `EK2DCalculator`

### Class Definition

```python
class EK2DCalculator:
    """A class for calculating 2D band structure contour maps from Wannier90 Hamiltonian data."""
```

### Constructor

```python
def __init__(
    self,
    hamiltonian: MLWFHamiltonian,
    nk: int = 256,
)
```

| Parameter | Type | Description |
|------|------|------|
| `hamiltonian` | `MLWFHamiltonian` | Initialized Hamiltonian instance |
| `nk` | `int` | k-point grid size (nkx = nky = nk), default 256 |

**Internal Attributes**:

| Attribute | Type | Description |
|------|------|------|
| `ham` | `MLWFHamiltonian` | Associated Hamiltonian instance |
| `num_wann` | `int` | Number of Wannier functions |
| `nk` | `int` | k-grid size |
| `bvecs` | `np.ndarray \| None` | Reciprocal lattice vectors |
| `k1_grid`, `k2_grid` | `np.ndarray` | k-point grids `(nk, nk)` |
| `k_points` | `np.ndarray` | Flattened k-point list `(nk*nk, 3)` |

**⚠️ Initialization Checks**:

- `hamiltonian.num_wann` must be initialized and positive.
- Otherwise, raises `ValueError`.

---

### Instance Methods

#### `calculate(k_range, save_to_file)`

Unified band calculator, automatically selects CPU/GPU backend.

```python
def calculate(
    self,
    k_range: tuple[float, float] | None = None,
    save_to_file: str | None = None,
) -> dict[str, np.ndarray]
```

| Parameter | Type | Description |
|------|------|------|
| `k_range` | `tuple[float, float]` or `None` | k-range `[k_min, k_max]`, default `None` (calculates only original BZ) |
| `save_to_file` | `str` or `None` | HDF5 output path (optional), automatically adds `.h5` extension |

**Returns**: `dict[str, np.ndarray]` containing the following keys:

| Key | Type | Shape | Description |
|----|------|------|------|
| `energies` | `np.ndarray` | `(num_wann, Nk, Nk)` | Band energies (eV) |
| `kx`, `ky` | `np.ndarray` | `(Nk, Nk)` | Real-space k-coordinates (1/Å) |
| `k1_grid`, `k2_grid` | `np.ndarray` | `(Nk, Nk)` | Fractional coordinate k-grids |
| `bvecs` | `np.ndarray \| None` | `(3, 3)` | Reciprocal lattice vectors (1/Å) |

**⚠️ Important Notes**:

- Return structure is fully consistent with `EK2DIO.load_ek2d()`.
- HDF5 files save the **original grid** `[-0.5, 0.5)` data.
- `calculate()` returns the **expanded** grid (based on the `k_range` parameter).
- This design saves storage space while remaining convenient for users.

**Example**:

```python
# Calculate only original BZ [-0.5, 0.5)
result = calc.calculate()
energies = result["energies"]  # shape: (num_wann, 256, 256)

# Expand to larger k-range
result = calc.calculate(k_range=(-1.0, 1.0))
energies = result["energies"]  # shape: (num_wann, 512, 512)

# Save results (saves original grid, returns expanded grid)
result = calc.calculate(
    k_range=(-1.0, 1.0),
    save_to_file="./ek2d.h5",
)

# Load from file (structure identical)
from stm_data_processing.io.ek2d_io import EK2DIO
loaded = EK2DIO.load_ek2d("./ek2d.h5", k_range=(-1.0, 1.0))
# loaded is fully consistent with result
```

---

## Backend Detection and Checks

### Backend Inheritance Logic

The module inherits backend status from `mlwf_hamiltonian` upon import:

```python
from ..dft.wannier90.mlwf_hamiltonian import BACKEND, cp

if BACKEND == "gpu":
    # Use GPU acceleration
else:
    # Use CPU calculation
```

### Public Constant: `BACKEND`

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND

print(BACKEND)  # 'cpu' or 'gpu'
```

| Constant | Type | Description |
|------|------|------|
| `BACKEND` | `Literal["cpu", "gpu"]` | Currently active calculation backend (imported from `mlwf_hamiltonian`) |

**⚠️ Note**:

- `BACKEND` is determined at module import and does not change at runtime.
- In GPU mode, internal use of `cp.ndarray`, but ultimately returns `np.ndarray`.
- No environment variable setup required; backend is automatically detected by `mlwf_hamiltonian`.

---

## Internal Methods (Not Recommended for Direct Call)

| Method | Description |
|------|------|
| `_compute_ek2d(hamiltonian)` | CPU backend calculation, returns `(num_wann, nk, nk)` |
| `_compute_ek2d_cuda(hamiltonian)` | GPU backend calculation, returns `(num_wann, nk, nk)` cp.ndarray |

---

## Mathematical Formulas

### Band Calculation

1. Calculate k-point Hamiltonian: `H(k) = Σ_R exp(2π i R·k) / ndegen(R) * H(R)`
2. Diagonalization: `H(k) * ψ_n(k) = ε_n(k) * ψ_n(k)`
3. Get Eigenvalues: `ε_n(k)` is the energy of the nth band at k-point k.

### k-space Expansion

Extends the original BZ `[-0.5, 0.5)` to the target range `[k_min, k_max)` via periodic extension:

```
k_extended = k_primitive + G
```

Where `G` is an integer multiple of reciprocal lattice vectors.

---

## Usage Examples

### Basic Usage

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator
import numpy as np

# Load Hamiltonian
ham = MLWFHamiltonian.from_seedname("./wannier", "silicon")

# Create band calculator
calc = EK2DCalculator(ham, nk=256)

# Calculate original BZ bands
result = calc.calculate()
energies = result["energies"]  # shape: (num_wann, 256, 256)
kx = result["kx"]              # shape: (256, 256)
ky = result["ky"]              # shape: (256, 256)

# Access specific band
band_0 = energies[0]  # First band
```

### Determine Calculation Strategy Based on BACKEND

```python
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import BACKEND
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator

calc = EK2DCalculator(ham, nk=256)

if BACKEND == "gpu":
    # GPU backend: Can handle larger grids
    calc = EK2DCalculator(ham, nk=512)
    result = calc.calculate(k_range=(-1.0, 1.0))
else:
    # CPU backend: Use smaller grid to avoid memory pressure
    calc = EK2DCalculator(ham, nk=256)
    result = calc.calculate(k_range=(-0.5, 0.5))
```

### Save Results to HDF5

```python
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator

calc = EK2DCalculator(ham, nk=256)

result = calc.calculate(
    k_range=(-1.0, 1.0),
    save_to_file="./output/ek2d_data.h5",
)

# Saved HDF5 contains original grid [-0.5, 0.5)
# result returns data expanded according to k_range
```

### Custom k-range Expansion

```python
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator

calc = EK2DCalculator(ham, nk=256)

# Expand to [-1.0, 1.0] range
result = calc.calculate(k_range=(-1.0, 1.0))

# No expansion (only original BZ)
result = calc.calculate(k_range=None)

# Access coordinate information
print(f"kx range: {result['kx'].min():.4f} -> {result['kx'].max():.4f} 1/Å")
print(f"ky range: {result['ky'].min():.4f} -> {result['ky'].max():.4f} 1/Å")
```

### Alignment with EK2DIO Interface

```python
from stm_data_processing.dft.wannier90.mlwf_ek2d import EK2DCalculator
from stm_data_processing.io.ek2d_io import EK2DIO

# Calculate and save
calc = EK2DCalculator(ham, nk=256)
result_calc = calc.calculate(
    k_range=(-1.0, 1.0),
    save_to_file="./ek2d.h5",
)

# Load from file (using same k_range)
result_load = EK2DIO.load_ek2d("./ek2d.h5", k_range=(-1.0, 1.0))

# Both structures are fully consistent
assert result_calc.keys() == result_load.keys()
assert result_calc["energies"].shape == result_load["energies"].shape
```

---

## GPU Batch Optimization

Automatic memory management and batch diagonalization in GPU mode:

| Parameter | Description |
|------|------|
| `batch_size` | Number of k-points diagonalized per batch (1024) |
| Memory Release | GPU memory automatically released after each batch |

**Automatic Adjustment Logic**:

- Large grids (≥1024) processed in batches to avoid OOM.
- Progress information printed after each batch.
- GPU memory pool unified release after calculation completes.

---

## Dependencies

| Dependency | Required | Description |
|------|------|------|
| `numpy` | Yes | Core calculation |
| `cupy` | No | GPU acceleration (optional, detected by `mlwf_hamiltonian`) |
| `MLWFHamiltonian` | Yes | Hamiltonian calculation |
| `EK2DIO` | Yes | HDF5 file save/load |
| `frac_to_real_2d` | Yes | Fractional to real-space coordinate conversion |

---

## Error Handling

| Exception | Trigger Condition |
|------|----------|
| `ValueError` | `hamiltonian.num_wann` uninitialized or non-positive |
| `RuntimeError` | GPU backend active but CuPy unavailable |

---

## Interface Alignment Checklist

When generating code calling this module, ensure:

- [ ] **`hamiltonian` must be an initialized `MLWFHamiltonian` instance**
- [ ] **`hamiltonian.num_wann` must be positive**
- [ ] **Check current backend via `BACKEND` constant to determine calculation strategy**
- [ ] **`nk` parameter is a positive integer, typical values 256~512**
- [ ] **Return results are always `np.ndarray` (converted after GPU internal calculation)**
- [ ] **No expansion when `k_range=None`**
- [ ] **Note: `BACKEND` is determined at module import, does not change at runtime**
- [ ] **HDF5 saving requires `save_to_file` parameter**
- [ ] **Return structure fully consistent with `EK2DIO.load_ek2d()`**
- [ ] **GPU calculation automatically releases memory pool**

---

## Interface Alignment with mlwf_hamiltonian

| Check Item | Status | Description |
|-------|------|------|
| Use `BACKEND` constant to judge backend | ✅ | Consistent with `mlwf_hamiltonian` |
| `cp` conditionally imported based on `BACKEND` | ✅ | Consistent with `mlwf_hamiltonian` |
| No instance backend state caching | ✅ | Directly uses module-level `BACKEND` |
| Return type unified to `np.ndarray` | ✅ | Converted after GPU internal calculation |
| k-point input shape is `(N, 3)` | ✅ | Via `ham.hk()` call |

---

## Interface Alignment with EK2DIO

| Check Item | Status | Description |
|-------|------|------|
| `calculate()` return structure consistent with `load_ek2d()` | ✅ | Both include energies, kx, ky, k1_grid, k2_grid, bvecs |
| HDF5 saves original grid, returns expanded grid | ✅ | Saves storage space, convenient for use |
| `k_range` handling logic consistent | ✅ | Both use `_extend_ek2d_static` for expansion |
| `frac_to_real_2d` conversion consistent | ✅ | Both convert from fractional to real-space coordinates |

---

## Version Information

- Module Path: `src/STM_DataProcessing/src/stm_data_processing/dft/wannier90/mlwf_ek2d.py`
- Backend Detection: Inherited from `mlwf_hamiltonian` (automatically completed at import)
- Log Level: `INFO` for calculation progress notifications
- **No environment variable configuration, backend automatically inherited**
- **k-grid range: Fixed to `[-0.5, 0.5)`**
- **Output Format: Always returns `np.ndarray`**
- **Return Structure: Fully aligned with `EK2DIO.load_ek2d()`**
- **Saving Strategy: HDF5 saves original grid, returns expanded grid**

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        calculate()                              │
├─────────────────────────────────────────────────────────────────┤
│  1. Calculate E(k) (Original grid [-0.5, 0.5))                  │
│  2. Save HDF5 (Original grid, saves space)                      │
│  3. Expand according to k_range                                 │
│  4. Return dict {energies, kx, ky, k1_grid, k2_grid, bvecs}    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EK2DIO.load_ek2d()                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Load HDF5 (Original grid)                                   │
│  2. Expand according to k_range                                 │
│  3. Return dict {energies, kx, ky, k1_grid, k2_grid, bvecs}    │
└─────────────────────────────────────────────────────────────────┘

✅ Both return structures are fully consistent and interchangeable
```
