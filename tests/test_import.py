"""Test basic imports."""
import stm_data_processing

def test_import():
    """Test that the package can be imported."""
    assert stm_data_processing is not None
    
def test_submodules():
    """Test that submodules can be imported."""
    from stm_data_processing.STM import LATTICE2D
    from stm_data_processing.dft_codes.openmx import OpenMX
    assert LATTICE2D is not None
    assert OpenMX is not None

if __name__ == "__main__":
    test_import()
    test_submodules()
    print("All tests passed!")
