import sys
from pathlib import Path

from loguru import logger

from eigenforge.common.config import config

logger.remove()
_json = config.log_format == "json"
logger.add(
    sys.stdout,
    colorize=not _json,
    serialize=_json,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=config.log_level,
    diagnose=False,
    backtrace=False,
)

# File sink is opt-out and never fatal at import in read-only environments, and a
# malformed LOG_FILE must not be fatal either: loguru formats the path with
# str.format_map, so a value containing a brace field raises KeyError. A bad log
# path degrades file logging and warns rather than breaking import.
_log_file = (config.log_file or "").strip()
if _log_file:
    try:
        Path(_log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            _log_file,
            rotation="10 MB",
            retention="14 days",
            compression="zip",
            level=config.log_level,
            serialize=_json,
            diagnose=False,
            backtrace=False,
            delay=True,
        )
    except KeyError as exc:  # pragma: no cover - env-dependent
        logger.warning(
            "File logging disabled ({}): loguru treats '{{...}}' in LOG_FILE as format fields (missing key {})",
            _log_file,
            exc,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - env-dependent
        logger.warning("File logging disabled ({}): {}", _log_file, exc)

__all__ = ["logger"]
