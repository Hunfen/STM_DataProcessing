"""Simplified logging for stm_data_processing.

A thin wrapper around the stdlib `logging` module that removes the usual
boilerplate (``basicConfig`` / handler / formatter setup) while keeping every
existing ``logging.getLogger(__name__)`` call in the package working:

- one-call logger factory:      ``logger = get_logger()``
- level control:                ``set_level("debug")``
- log file output:              ``enable_file("run.log")``
- format customization:         ``set_format(...)``
- one-shot configuration:       ``setup(level="debug", file="run.log")``

All handlers live on the root logger, so loggers created anywhere in the
package (or in user scripts) inherit the same formatting and file output
automatically.  ``get_logger()`` defaults to the calling module's ``__name__``,
so every module keeps its own ``%(name)s`` tag with zero boilerplate:

    from stm_data_processing.logger import get_logger

    logger = get_logger()          # name = current module's __name__
    logger.info("loading data")
    logger.debug("verbose detail")
    logger.warning("something odd")
    logger.error("something failed")
"""

from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"

_PACKAGE_LOGGER = "stm_data_processing"

_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET,
}

_state = {
    "level": DEFAULT_LEVEL,
    "format": DEFAULT_FORMAT,
    "datefmt": DEFAULT_DATE_FORMAT,
}

_stream_handler: logging.StreamHandler | None = None
_file_handler: logging.FileHandler | None = None

# Loggers created via get_logger(), so set_level() can update them live.
_managed: dict[str, logging.Logger] = {}
# Logger names that were given an explicit level (immune to set_level()).
_fixed: set[str] = set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_level(level: str | int) -> int:
    """Convert a level name ("info") or number (logging.INFO) to an int."""
    if isinstance(level, str):
        try:
            return _LEVELS[level.lower()]
        except KeyError:
            raise ValueError(
                f"unknown log level {level!r}; use one of {sorted(_LEVELS)}"
            ) from None
    return int(level)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_state["format"], datefmt=_state["datefmt"])


def _ensure_configured() -> None:
    """Attach a default console handler once; keep the package visible."""
    global _stream_handler
    root = logging.getLogger()
    # Respect a user-provided basicConfig; only add our own when unconfigured.
    if not root.handlers:
        _stream_handler = logging.StreamHandler(sys.stderr)
        _stream_handler.setFormatter(_make_formatter())
        root.addHandler(_stream_handler)
    # Make package logs visible without raising third-party verbosity.
    logging.getLogger(_PACKAGE_LOGGER).setLevel(_resolve_level(_state["level"]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_logger(
    name: str | None = None, level: str | int | None = None
) -> logging.Logger:
    """
    Return a fully-configured logger (levels, formatting, file output).

    Parameters
    ----------
    name : str, optional
        Logger name.  Defaults to the calling module's ``__name__``.
    level : str or int, optional
        Level for this logger ("debug", "info", ... or logging constants).
        Defaults to the globally configured level.

    Examples
    --------
    >>> from stm_data_processing.logger import get_logger
    >>> logger = get_logger()
    >>> logger.info("hello")
    """
    if name is None:
        name = inspect.currentframe().f_back.f_globals.get("__name__", _PACKAGE_LOGGER)
    _ensure_configured()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(_resolve_level(level))
        _fixed.add(name)
    else:
        # Latest call wins: a logger created earlier without a fixed level
        # follows the global default, and vice versa.
        _fixed.discard(name)
        logger.setLevel(_resolve_level(_state["level"]))
    _managed[name] = logger
    return logger


def set_level(level: str | int) -> None:
    """
    Set the default level used by :func:`get_logger` and package loggers.

    >>> from stm_data_processing.logger import set_level
    >>> set_level("debug")
    """
    lvl = _resolve_level(level)
    _state["level"] = lvl
    _ensure_configured()
    logging.getLogger(_PACKAGE_LOGGER).setLevel(lvl)
    for name, managed in _managed.items():
        if name not in _fixed:
            managed.setLevel(lvl)


def set_format(fmt: str | None = None, datefmt: str | None = None) -> None:
    """
    Customize the message format (and optional date format) of our handlers.

    Parameters
    ----------
    fmt : str, optional
        logging format string, e.g. ``"%(levelname)s %(message)s"``.
    datefmt : str, optional
        datetime format string, e.g. ``"%H:%M:%S"``.

    >>> from stm_data_processing.logger import set_format
    >>> set_format("%(levelname)s | %(message)s", "%H:%M:%S")
    """
    if fmt is not None:
        _state["format"] = fmt
    if datefmt is not None:
        _state["datefmt"] = datefmt
    formatter = _make_formatter()
    if _stream_handler is not None:
        _stream_handler.setFormatter(formatter)
    if _file_handler is not None:
        _file_handler.setFormatter(formatter)


def enable_file(
    path: str | Path, level: str | int | None = None
) -> logging.FileHandler:
    """
    Route log output to a file (in addition to the console).

    Calling this again with a new path replaces the previous log file.

    Parameters
    ----------
    path : str or Path
        Log file path (parent directories are created automatically).
    level : str or int, optional
        Level threshold for the file handler.

    Returns
    -------
    logging.FileHandler
        The newly attached file handler.

    >>> from stm_data_processing.logger import enable_file
    >>> enable_file("logs/run.log")
    """
    global _file_handler
    _ensure_configured()
    root = logging.getLogger()
    if _file_handler is not None:
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setFormatter(_make_formatter())
    if level is not None:
        handler.setLevel(_resolve_level(level))
    root.addHandler(handler)
    _file_handler = handler
    return handler


def disable_file() -> None:
    """Detach and close the log file handler added by :func:`enable_file`."""
    global _file_handler
    if _file_handler is not None:
        root = logging.getLogger()
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None


def setup(
    level: str | int = DEFAULT_LEVEL,
    file: str | Path | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
) -> logging.Logger:
    """
    One-shot configuration, then return a logger for the calling module.

    Parameters
    ----------
    level : str or int
        Default log level.
    file : str or Path, optional
        Also write log output to this file.
    fmt : str, optional
        Message format string.
    datefmt : str, optional
        Datetime format string.

    Returns
    -------
    logging.Logger
        A configured logger named after the calling module.

    Examples
    --------
    >>> from stm_data_processing.logger import setup
    >>> logger = setup(level="debug", file="run.log")
    >>> logger.info("ready")
    """
    if fmt is not None or datefmt is not None:
        set_format(fmt, datefmt)
    set_level(level)
    if file is not None:
        enable_file(file)
    # Resolve the caller here: get_logger() would otherwise pick up setup().
    name = inspect.currentframe().f_back.f_globals.get("__name__", _PACKAGE_LOGGER)
    return get_logger(name)
