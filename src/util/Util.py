import gc
import json
import os
import logging
from logging.handlers import RotatingFileHandler
import torch

logger = None


def readAllConfig():
    """
    Read config file
    :return: config
    """
    if os.path.exists("../../config.json"):  # Reading config in editor
        config_path = "../../config.json"
    elif os.path.exists("config.json"):  # Reading config in shell
        config_path = "config.json"
    else:
        logging.error("Config not found")
        exit(0)

    logging.info(f"Config found at: {config_path}")

    with open(config_path) as config_file:
        config = json.load(config_file)

    createLogger(config['path'] + "Logs/")  # Create logger first
    return config


def createLogger(path):
    global logger

    logger = logging.getLogger("Cross-lingual-claim-detection")
    log_format = '%(asctime)s [%(levelname)-s] - %(message)s'
    date_format = '%d-%b-%y %H:%M:%S'
    log_file = path + 'claim-detection.log'
    formatter = logging.Formatter(log_format, date_format)
    file_handler = RotatingFileHandler(log_file, maxBytes=100000, backupCount=10)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)
    # console.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(console)
    logger.setLevel(logging.DEBUG)

    logger.info(f"Logger created. Logs location - {path}")


def getDevice():
    """
    Get torch device
    :return: torch device
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def clearMemory():
    gc.collect()
    torch.cuda.empty_cache()
