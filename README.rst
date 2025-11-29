Nanonis Data File Loader
========================

.. image:: https://img.shields.io/badge/Python-3.7%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

A Python toolkit for parsing Nanonis scanning probe microscopy data files (.sxm/.dat/.3ds). Provides unified interfaces to access scan parameters, channel data, and metadata.

Features
--------
- **Auto-detection** of file types (SXM/DAT/3DS)
- **Structured extraction** of scan parameters (bias, range, setpoint)
- **Multi-channel data** support with directional scan correction
- **Header parsing** with module-specific organization
- **Data reformatting** for backward/upward scan correction
- **Topography visualization** with custom colormaps

Installation
------------
.. code-block:: bash

   pip install numpy pandas matplotlib  # Required dependencies
   git clone https://github.com/your-repo/nanonis-loader.git
   cd nanonis-loader
   pip install -e .

Usage
-----
Basic Example:
.. code-block:: python

   from sample.nanonis_loader import NanonisFileLoader
   from sample.preview_plot import plot_topo
   
   # Load Nanonis file
   loader = NanonisFileLoader("scan.sxm")
   print(loader.header["SCAN_PIXELS"])  # Get pixel dimensions
   print(loader.pixels)  # Get formatted pixel dimensions
   
   # Access scan parameters
   print(f"Scan range: {loader.range}")
   print(f"Bias: {loader.bias} V")
   print(f"Channels: {loader.channels}")
   
   # Plot topography data
   plot_topo(loader.data[0], "topography.png", sigma=2)

Supported File Types
-------------------
+--------+--------------------------+----------------------------------+
| Format | Key Features             | Example Use Case                 |
+========+==========================+==================================+
| .sxm   | Topography images        | Atomic resolution surface scans  |
+--------+--------------------------+----------------------------------+
| .dat   | Spectroscopy curves      | Point spectroscopy measurements  |
+--------+--------------------------+----------------------------------+
| .3ds   | Grid spectroscopy        | Magnetic field mapping           |
+--------+--------------------------+----------------------------------+

API Reference
-------------
Main Class:
- ``NanonisFileLoader(f_path: str)``
  - Properties:
    - ``.header``: Dictionary of parsed metadata
    - ``.data``: Numpy array or DataFrame of measurement data
    - ``.pixels``: Tuple of (width, height) dimensions
    - ``.parameters``: DataFrame of measurement parameters (3DS files)
    - ``.range``: Scan range tuple
    - ``.bias``: Bias voltage
    - ``.setpoint``: Z-controller setpoint
    - ``.channels``: List of data channel names

Utility Functions:
- ``vortex_num(field: float, area: float)``: Calculates superconducting flux quanta
- ``plot_topo(input_file, output, v_min, sigma, color_map)``: Plot topography data

Visualization Features
----------------------
- **Custom colormap**: Gwyddion-style colormap optimized for topography
- **Flexible scaling**: Sigma-based or fixed value color scaling
- **High-resolution output**: 300 DPI image export
- **Aspect ratio preservation**: Automatic figure sizing

Dependencies
------------
- Python 3.7+
- Required Packages:
  - ``numpy`` (array operations)
  - ``pandas`` (tabular data handling)
  - ``matplotlib`` (visualization)

Module Structure
----------------
::

   sample/
   ├── __init__.py      # Package initialization
   ├── nanonis_loader.py # Main file loader class
   ├── preview_plot.py   # Visualization utilities
   └── utility.py        # Physics calculation utilities

.. Examples
.. --------
.. See the ``examples/`` directory for:
.. - Basic file loading scripts
.. - Data visualization samples
.. - Advanced parameter extraction

License
-------
MIT License - Free for academic and commercial use