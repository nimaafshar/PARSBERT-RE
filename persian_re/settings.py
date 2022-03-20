from pathlib import Path


class Config:
    BASE_PATH = Path("persian_relation_extraction")
    MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-zwnj-base'
    OUTPUT_PATH = BASE_PATH / 'output' / 'model.bin'
    MAX_LEN = 64
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 8
    CALLBACK = 1000
    INITIAL_LEARNING_RATE = 2e-5
    CLIP = 0.0
