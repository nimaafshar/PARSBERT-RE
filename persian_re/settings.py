import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = Path(os.environ.get('BASE_PATH'))
MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-zwnj-base'
MAX_LEN = 64
