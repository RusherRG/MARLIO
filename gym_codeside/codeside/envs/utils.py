import math
import colorlog
import logging

LOG_LEVEL = logging.DEBUG


def get_logger(name):
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq}'
        '%(log_color)s'
        '%(asctime)s | %(name)s/%(funcName)s | '
        '%(levelname)s:%(reset)s %(message)s'
    )
    colorlog.basicConfig(format=colorlog_format,
                         level=logging.DEBUG, datefmt='%d/%m/%Y %H:%M:%S')

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger


def class_to_dict(state):
    if isinstance(state, list):
        return [class_to_dict(s) for s in state]
    if isinstance(state, dict):
        if "__objclass__" in state:
            return {
                "name": state["_name_"],
                "value": state["_value_"]
            }
        for prop in state:
            state[prop] = class_to_dict(state[prop])
        return state
    try:
        return class_to_dict(state.__dict__)
    except Exception:
        return state


def calc_distance(pos1, pos2):
    return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
