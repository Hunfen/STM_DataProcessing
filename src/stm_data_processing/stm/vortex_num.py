def vortex_num(field: float = 0, area: float = 0):
    """Calculate the vortex number using the given field and area.

    Args:
        field (float): Magnetic field in tesla (T).
        area (float): Area in square meters (m^2).

    Returns:
        float: The calculated vortex number.
    """
    phi_0 = 2.067833848e-15  # Flux quantum in Wb (1 Wb = 1 T*m^2)
    return field * area / phi_0
