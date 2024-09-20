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

def info(*args):
    msg = concatenate_like_print(args)
    if logger.level <= logging.INFO:
        print(msg, flush=printFlush)
    logger.info(msg)


def warning(*args):
    msg = concatenate_like_print(args)
    if logger.level <= logging.WARNING:
        print(msg, flush=printFlush)
    logger.warning(msg)


def error(*args):
    msg = concatenate_like_print(args)
    if logger.level <= logging.ERROR:
        print(msg, flush=printFlush)
    logger.error(msg)


def debug(*args):
    msg = concatenate_like_print(args)
    if logger.level <= logging.DEBUG:
        print(msg, flush=printFlush)
    logger.debug(msg)
