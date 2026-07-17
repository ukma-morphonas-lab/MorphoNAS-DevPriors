import logging
import os
import sys
from logging.handlers import RotatingFileHandler, QueueHandler


from typing import Any


class SafeQueueHandler(QueueHandler):
    """A QueueHandler that ignores errors during emit (e.g. BrokenPipe when shutting down)."""

    def emit(self, record):
        try:
            super().emit(record)
        except (BrokenPipeError, EOFError, ConnectionResetError, Exception):
            # Silently ignore errors during shutdown
            pass


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "morphonas.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    queue: Any = None,
):
    """
    Sets up logging. If queue is provided, uses SafeQueueHandler (for workers).
    Otherwise, sets up StreamHandler and RotatingFileHandler.
    """
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG)

    if queue is not None:
        # Worker mode: just send everything to the queue safely
        qh = SafeQueueHandler(queue)
        root_logger.addHandler(qh)
        return

    # Main process mode:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(processName)s - %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logging.debug(
        f"Logging system initialized. Console: {logging.getLevelName(console_level)}, File: {log_path}"
    )
