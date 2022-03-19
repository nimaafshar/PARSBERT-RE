import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = Path(os.environ.get('BASE_PATH'))
MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-zwnj-base'
OUTPUT_PATH = BASE_PATH / 'output' / 'model.bin'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
EEVERY_EPOCH = 1000
INITIAL_LEARNING_RATE = 2e-5
CLIP = 0.0
