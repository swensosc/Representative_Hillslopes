import logging
import os

logger = logging.getLogger(__name__)
printFlush = True


def config_logger(logfile):
    logging.basicConfig(
        filename=logfile,
        encoding="utf-8",
        filemode="w",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def set_logger_level(checkSinglePoint, args_debug, args_printFlush):
    level = logging.DEBUG if checkSinglePoint or args_debug else logging.INFO
    print(f"Setting logger level to {level}")
    logger.setLevel(level)
    printFlush = args_printFlush

def logger_level_debug():
    return logger.level <= logging.DEBUG

def concatenate_like_print(list_in):
    msg = " ".join([x if isinstance(x, str) else str(x) for x in list_in])
    msg = msg.replace("  ", " ")
    return msg

def _print_and_log(level, logger_fn, *args):
    if logger.level > level:
        return
    msg = concatenate_like_print(args)
    print(msg, flush=printFlush)
    logger_fn(msg)


def info(*args):
    _print_and_log(logging.INFO, logger.info, *args)

def warning(*args):
    _print_and_log(logging.WARNING, logger.warning, *args)

def error(*args):
    _print_and_log(logging.ERROR, logger.error, *args)

def debug(*args):
    _print_and_log(logging.DEBUG, logger.debug, *args)
