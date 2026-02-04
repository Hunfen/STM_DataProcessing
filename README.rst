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

Installation
------------
.. code-block:: bash

   pip install numpy pandas  # Required dependencies
   git clone https://github.com/your-repo/nanonis-loader.git
   cd nanonis-loader
   pip install -e .

Usage
-----
Basic Example:
.. code-block:: python

   from sample import NanonisFileLoader
   loader = NanonisFileLoader("scan.3ds")
   print(loader.header["Grid dim"])  # Get grid dimensions
   spectrum = loader.data  # Get 3D spectroscopy data

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

Utility Functions:
- ``vortex_num(field: float, area: float)``: Calculates superconducting flux quanta

Dependencies
------------
- Python 3.7+
- Required Packages:
  - ``numpy`` (array operations)
  - ``pandas`` (tabular data handling)
  - ``matplotlib`` (preview plotting, optional)

.. Examples
.. --------
.. See the ``examples/`` directory for:
.. - Basic file loading scripts
.. - Data visualization samples
.. - Advanced parameter extraction

License
-------
MIT License - Free for academic and commercial use