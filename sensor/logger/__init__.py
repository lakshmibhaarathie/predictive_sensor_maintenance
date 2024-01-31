import os
import logging
import pandas as pd
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%d%m%Y_%H%M%S')}"

LOG_DIR = os.path.join("logs",LOG_FILE_NAME)

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, f"{LOG_FILE_NAME}.log")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w",
    format='[%(asctime)s]~%(levelname)s~line no:%(lineno)s~filename:%(filename)s~%(funcName)s()~%(message)s', level=logging.INFO
)


