from stm_data_processing.dft.wannier90.mlwf_gk import GreenFunction
from stm_data_processing.dft.wannier90.mlwf_hamiltonian import MLWFHamiltonian


class TmatQPI:
    def __init__(
        self, hamiltonian: MLWFHamiltonian, nk: int = 128, eta: float = 0.001
    ) -> None:
        self.ham: MLWFHamiltonian = hamiltonian
        # Check MLWFHamiltonian initialization status
        if not hasattr(hamiltonian, "num_wann") or hamiltonian.num_wann is None:
            raise ValueError("Invalid MLWFHamiltonian: num_wann is not initialized.")
        if hamiltonian.num_wann <= 0:
            raise ValueError(
                f"Invalid MLWFHamiltonian: num_wann must be positive, "
                f"got {hamiltonian.num_wann}."
            )
        self.num_wann: int | None = hamiltonian.num_wann
        self.nk: int = int(nk)
        self.eta: float = float(eta)
        self.gf: GreenFunction = GreenFunction(hamiltonian, eta=eta)

    # ============================================================
    # Compute Core math (CPU/GPU)
    # ============================================================
    def _compute_tmat():
        return None

    # ============================================================
    # Public API
    # ============================================================
    def calculate():
        return None
