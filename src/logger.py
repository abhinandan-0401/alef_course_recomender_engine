import logging
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "logs/app.log"


def _get_console_handler():
    """Sets the console handler with formatter"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def _get_file_handler():
    """Sets the file handler with formatter.

    Timed rotating file handler will ensure to keep rotating the logs.
    """
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name: str):
    """Initializes the logger with a name and adds console and file handlers.

    Args:
        logger_name: Name pf the logging activity/process

    Returns:
        logger: logger object
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_get_console_handler())
    logger.addHandler(_get_file_handler())
    logger.propagate = False
    return logger
