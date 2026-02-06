---

Nanonis Data File Loader
========================

.. image:: https://img.shields.io/badge/Python-3.7%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

A lightweight Python module for reading and parsing Nanonis scanning probe microscopy (SPM) data files (``.sxm``, ``.dat``, ``.3ds``). Provides lazy-loading, structured metadata access, and automatic data reshaping with scan-direction correction.
Features
--------
- **Unified interface** for `.sxm` (images), `.dat` (spectroscopy), and `.3ds` (grid spectroscopy)
- **Lazy parsing**: Header and data parsed only on first access
- **Automatic scan direction correction** (upward/downward) for `.sxm`
- **Structured header parsing** with module-aware organization (e.g., "Z-CONTROLLER", "Bias Spectroscopy")
- **Data reformatting**:
  - `.sxm`: Reshaped to `(2 × channels, height, width)` with line-flip correction
  - `.dat`: Loaded as `pandas.DataFrame` with numeric coercion
  - `.3ds`: Separated into parameters (`DataFrame`) and grid (`ndarray`)
- **Convenient properties** for common SPM parameters (bias, setpoint, scan range, etc.)
Installation
------------
.. code-block:: bash

   pip install numpy pandas
   # Optional for plotting: matplotlib

   # Install in development mode
   git clone https://github.com/Hunfen/STM_DataProcessing.git
   cd STM
   pip install -e .

Usage
-----
.. code-block:: python

   from STM.nanonis_loader import NanonisFileLoader

   # Load any supported file
   loader = NanonisFileLoader("data.sxm")

   # Access parsed metadata
   print(loader.pixels)        # (width, height)
   print(loader.range)         # (x_range, y_range) in nm
   print(loader.bias)          # Bias voltage (V)
   print(loader.setpoint)      # Z-controller setpoint (typically current in nA)
   print(loader.channels)      # List of channel names

   # Get full structured header
   header = loader.header      # dict with modules as nested dicts/DataFrames

   # Access measurement data
   data = loader.data          # np.ndarray (.sxm/.3ds) or pd.DataFrame (.dat)

   # For .3ds files: also get per-pixel parameters
   if loader.file_type == "3ds":
       params = loader.parameters  # pd.DataFrame
Supported File Types
-------------------
+--------+--------------------------+----------------------------------+
| Format | Content                  | Data Structure                   |
+========+==========================+==================================+
| .sxm   | Topography / channel maps| ``np.ndarray`` of shape ``(2*C, H, W)``<br>Even indices: forward scan<br>Odd indices: backward scan (flipped) |
+--------+--------------------------+----------------------------------+
| .dat   | Point spectroscopy       | ``pd.DataFrame`` with columns = channels |
+--------+--------------------------+----------------------------------+
| .3ds   | Grid spectroscopy (dI/dV)| ``parameters``: per-pixel metadata (DataFrame)<br>``data``: grid array ``(N_pixels, C, N_points)`` |
+--------+--------------------------+----------------------------------+

API Reference
-------------
Class: ``NanonisFileLoader(f_path: str)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Properties:
^^^^^^^^^^^

- ``.file_type`` → ``str``: One of ``"sxm"``, ``"dat"``, ``"3ds"``
- ``.header`` → ``dict``: Fully parsed and reformatted metadata
- ``.data`` → ``np.ndarray | pd.DataFrame``: Main measurement data
- ``.parameters`` → ``pd.DataFrame | None``: Only for `.3ds` files
- ``.pixels`` → ``tuple[int, int]``: ``(width, height)`` in pixels
- ``.channels`` → ``list[str]``: Names of recorded channels

SXM-Specific Properties (raise ``AttributeError`` for other types):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``.range`` → ``tuple[float, float]``: Scan range in nm (from ``SCAN_RANGE``)
- ``.center`` → ``tuple[float, float]``: Scan center offset (from ``SCAN_OFFSET``)
- ``.frame_angle`` → ``float``: Scan angle in degrees (from ``SCAN_ANGLE``)
- ``.dir`` → ``bool``: ``True`` if upward scan, ``False`` if downward (from ``SCAN_DIR``)
- ``.bias`` → ``float``: Bias voltage in volts (from ``BIAS``)
- ``.setpoint`` → ``float``: Z-controller setpoint (parsed from ``Z-CONTROLLER`` table)
Dependencies
------------
- **Required**: ``numpy >=1.20``, ``pandas >=1.3``
- **Optional**: ``matplotlib`` (for visualization examples)
Module Structure
----------------
::

   STM/
   ├── nanonis_loader.py   # Core loader class (this module)
   └── ...                 # Other utilities (e.g., plotting) not part of core loader

Notes
-----
- All binary data is assumed to be **big-endian** (``>f``), consistent with Nanonis.
- UTF-8 decoding uses ``errors="replace"`` for robustness against malformed headers.
- The loader does **not** modify original data; corrections (e.g., flipping lines) are applied during parsing.
License
-------
MIT License

---

This version now reflects all the improvements you proposed — including a clearer feature list, better formatting, more accurate descriptions, and enhanced structure. Let me know if you'd like this exported as a README.rst or further modified!
