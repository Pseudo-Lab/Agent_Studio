"""
Centralized logging configuration for Kiosk Agent.

Usage:
    from kiosk_agent.utils import get_logger
    
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Log format
DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
SIMPLE_FORMAT = "[%(levelname)s] %(name)s: %(message)s"
DEBUG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"

# Module-specific log levels (can be overridden via env)
MODULE_LOG_LEVELS = {
    "kiosk_agent.core": "INFO",
    "kiosk_agent.llm": "INFO",
    "kiosk_agent.voice": "INFO",
    "kiosk_agent.frameworks": "DEBUG",
    "kiosk_agent.api": "INFO",
}

_initialized = False


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    format_style: str = "default",
) -> None:
    """
    Setup logging configuration for the entire application.
    
    Args:
        level: Root log level (default: from LOG_LEVEL env or INFO)
        log_file: Optional file to write logs to
        format_style: 'default', 'simple', or 'debug'
    """
    global _initialized
    
    if _initialized:
        return
    
    # Determine log level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Select format
    formats = {
        "default": DEFAULT_FORMAT,
        "simple": SIMPLE_FORMAT,
        "debug": DEBUG_FORMAT,
    }
    log_format = formats.get(format_style, DEFAULT_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(DEBUG_FORMAT))
        root_logger.addHandler(file_handler)
    
    # Apply module-specific levels
    for module, mod_level in MODULE_LOG_LEVELS.items():
        env_key = f"LOG_LEVEL_{module.upper().replace('.', '_')}"
        actual_level = os.getenv(env_key, mod_level)
        logging.getLogger(module).setLevel(getattr(logging, actual_level, logging.INFO))
    
    # Suppress noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "PIL", "google"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    _initialized = True
    logging.getLogger(__name__).debug(f"Logging initialized: level={level}, file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    # Auto-initialize if not done
    if not _initialized:
        setup_logging()
    
    return logging.getLogger(name)


# Convenience loggers for common modules
def core_logger() -> logging.Logger:
    """Get logger for core module."""
    return get_logger("kiosk_agent.core")


def llm_logger() -> logging.Logger:
    """Get logger for LLM module."""
    return get_logger("kiosk_agent.llm")


def voice_logger() -> logging.Logger:
    """Get logger for voice module."""
    return get_logger("kiosk_agent.voice")


def api_logger() -> logging.Logger:
    """Get logger for API module."""
    return get_logger("kiosk_agent.api")


def agent_logger() -> logging.Logger:
    """Get logger for agent/frameworks module."""
    return get_logger("kiosk_agent.frameworks")
