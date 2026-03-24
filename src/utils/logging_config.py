import logging
import os


def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure and return a logger for the given module name."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", os.path.join("logs", "app.log"))
    log_to_file = os.getenv("LOG_TO_FILE", "").lower() in {"1", "true", "yes"}
    is_hosted = bool(os.getenv("VERCEL") or os.getenv("RENDER"))
    if log_to_file and not is_hosted:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    if not root.handlers:
        root.setLevel(log_level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        if log_to_file and not is_hosted:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
    else:
        root.setLevel(log_level)
    return logging.getLogger(name)
