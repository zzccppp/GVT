import logging

logger = logging.getLogger("default")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_logger():
    return logger


def info(*args):
    # concatenate all arguments into a single string
    logger.info(" ".join(map(str, args)))


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def error(*args, **kwargs):
    logger.error(*args, **kwargs)


def warning(*args, **kwargs):
    logger.warning(*args, **kwargs)
