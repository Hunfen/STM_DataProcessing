# Bare Lindhard Susceptibility Calculation Algorithm

## Algorithm Overview

This algorithm calculates the **Bare Lindhard Susceptibility** on a 2D k-grid, with optional orbital selection support. It is commonly used in condensed matter physics to study the response properties of electronic systems.

---

## Mathematical Formula

The basic expression for Bare Lindhard susceptibility is:

$$\chi_0(\mathbf{q}) = \frac{1}{N_k} \sum_{\mathbf{k}} \sum_{m,n} \frac{f(\varepsilon_{m\mathbf{k}}) - f(\varepsilon_{n\mathbf{k+q}})}{\varepsilon_{n\mathbf{k+q}} - \varepsilon_{m\mathbf{k}} + i\eta} \cdot |M_{mn}(\mathbf{k}, \mathbf{q})|^2$$

where the orbital weight factor is:

$$M_{mn}(\mathbf{k}, \mathbf{q}) = \sum_{a \in \text{orb\_sel}} \psi_{m\mathbf{k}}^{(a)*} \cdot \psi_{n\mathbf{k+q}}^{(a)}$$

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ef` | `float` | `0.0` | Chemical potential (eV) |
| `temperature` | `float` | `4.2` | Temperature (K) |
| `orb_sel` | `np.ndarray[int]` | `None` | Array of selected orbital indices |
| `eta` | `float` | - | Broadening factor (imaginary part) |

---

## Algorithm Flow

```mermaid
flowchart TD
    A[Start] --> B[Get eigenvalues evals and eigenvectors evecs]
    B --> C[Initialize chi_q as zero matrix nk1×nk2]
    C --> D{Iterate q1 ∈ [0, nk1)}
    D --> E{Iterate q2 ∈ [0, nk2)}
    E --> F{Iterate k1 ∈ [0, nk1)}
    F --> G{Iterate k2 ∈ [0, nk2)}
    G --> H[Calculate k+q point: k1q, k2q using periodic boundary]
    H --> I{Iterate band m}
    I --> J[Calculate fmk = Fermi distribution εmk]
    J --> K{Iterate band n}
    K --> L[Calculate fnkq = Fermi distribution εnk+q]
    L --> M[Calculate numerator: fmk - fnkq]
    M --> N[Calculate denominator: εnk+q - εmk + iη]
    N --> O{Iterate selected orbitals a}
    O --> P[Accumulate overlap: ψ*mk · ψnk+q]
    P --> Q[Calculate weight: |overlap|²]
    Q --> R[Accumulate to total_q]
    R --> K
    K --> I
    I --> G
    G --> F
    F --> E
    E --> D
    D --> S[chi_q[q1,q2] = total_q / num_kpts]
    S --> T[Return chi_q]
    T --> U[End]
```

---

## Core Calculation Steps

### 1. Fermi Distribution Function

```python
def fermi(energy: float, mu: float, T: float) -> float:
    """Fermi-Dirac distribution"""
    kB = 8.617333262e-5  # eV/K
    if T == 0:
        return 1.0 if energy < mu else 0.0
    return 1.0 / (np.exp((energy - mu) / (kB * T)) + 1.0)
```

### 2. Periodic Boundary Handling

```python
k1q = (k1 + q1) % nk1  # First direction
k2q = (k2 + q2) % nk2  # Second direction
```

### 3. Orbital Overlap Calculation

```python
overlap = 0.0 + 0.0j
for a in orb_sel:
    overlap += evecs[k1, k2, a, m] * np.conj(evecs[k1q, k2q, a, n])
weight = abs(overlap) ** 2
```

### 4. Susceptibility Accumulation

```python
numerator = fmk - fnkq
denominator = (enkq - emk) + 1j * eta
total_q += weight * numerator / denominator
chi_q[q1, q2] = total_q / num_kpts
```

---

## Computational Complexity

| Loop Level | Complexity |
|------------|------------|
| q-space iteration | O(nk1 × nk2) |
| k-space iteration | O(nk1 × nk2) |
| Band iteration | O(num_wann²) |
| Orbital iteration | O(len(orb_sel)) |
| **Total** | **O(nk1² × nk2² × num_wann² × len(orb_sel))** |

---

## Notes

1. **Periodic Boundary Conditions**: k+q must be folded back to the first Brillouin zone when exceeding the Brillouin zone boundary
2. **Numerical Stability**: Add imaginary part `iη` to denominator to avoid singularities
3. **Memory Optimization**: Consider using vectorized operations to replace some loops
4. **Parallelization**: q-points are independent, suitable for parallel computation

---

## Output Format

```python
chi_q: np.ndarray
# shape: (nk1, nk2)
# dtype: np.complex128
# Meaning: Bare Lindhard susceptibility at each q-point
```

---