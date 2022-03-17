from ..settings import BASE_PATH
import pandas as pd
from typing import Tuple


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loading data from BASE_PATH
    :return: (train_df, test_df)
    """
    train_df = pd.read_csv(BASE_PATH / 'PERLEX' / 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(BASE_PATH / 'PERLEX' / 'test.csv', encoding='utf-8')
    return train_df, test_df


def remove_re_type(df: pd.DataFrame, re_type: str) -> pd.DataFrame:
    result = df.copy()
    result.drop(result[result['re_type'] == re_type].index, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result

# train_df = train_df[train_df['re_type'] != 'Entity-Destination(e2,e1)']
    # test_df = test_df[test_df['re_type'] != 'Entity-Destination(e2,e1)']
#remove re_type train_df , test_df


def