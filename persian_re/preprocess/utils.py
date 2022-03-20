from ..settings import Config
import pandas as pd
from typing import Tuple, List, Dict
import pickle
import numpy as np


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loading data from BASE_PATH
    :return: (train_df, test_df)
    """
    train_df = pd.read_csv(Config.BASE_PATH / 'PERLEX' / 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(Config.BASE_PATH / 'PERLEX' / 'test.csv', encoding='utf-8')
    return train_df, test_df


def remove_re_type(df: pd.DataFrame, re_type: str) -> pd.DataFrame:
    result: pd.DataFrame = df.copy()
    result.drop(result[result['re_type'] == re_type].index, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


class PerlexData:
    _instance: 'PerlexData' = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = PerlexData()
        return cls._instance

    def __init__(self):
        with open(Config.BASE_PATH / 'PERLEX' / 'transformed_data.bin', 'rb') as binary_file:
            data = pickle.load(binary_file)

        self._x_train: List[str] = data['train'][0]
        self._y_train: List[str] = data['train'][1]

        self._x_valid: List[str] = data['valid'][0]
        self._y_valid: List[str] = data['valid'][1]

        self._x_test: List[str] = data['test'][0]
        self._y_test: List[str] = data['test'][1]

        self._label2ids: Dict[str, int] = data['labels']['label2id']
        self._id2labels: Dict[int, str] = data['labels']['id2label']

        self._class_weights: np.ndarray = data['class_weights']

    @property
    def labels(self) -> List[str]:
        return list(self._label2ids.keys())

    @property
    def x_train(self) -> List[str]:
        return self._x_train

    @property
    def y_train(self) -> List[str]:
        return self._y_train

    @property
    def x_valid(self) -> List[str]:
        return self._x_valid

    @property
    def y_valid(self) -> List[str]:
        return self._y_valid

    @property
    def x_test(self) -> List[str]:
        return self._x_test

    @property
    def y_test(self) -> List[str]:
        return self._y_test

    @property
    def label2ids(self) -> Dict[str, int]:
        return self._label2ids

    @property
    def id2labels(self) -> Dict[int, str]:
        return self._id2labels

    @property
    def class_weights(self) -> np.ndarray:
        return self._class_weights
