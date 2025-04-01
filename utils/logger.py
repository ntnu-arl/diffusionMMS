import logging

FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATETIME_FORMAT = "%m-%d %H:%M"

logging.basicConfig(
    filename=None,
    encoding="utf-8",
    level=logging.INFO,
    format=FORMAT,
    datefmt=DATETIME_FORMAT,
)
logger_initialized = {}


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[38;5;78m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=DATETIME_FORMAT)
        return formatter.format(record)


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(CustomFormatter())
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name="root", log_file=log_file, log_level=log_level)
    return logger
