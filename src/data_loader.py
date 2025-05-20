import pandas as pd
import numpy as np


def load_yield_data(file_path='data/EU_yield_curves_combined.csv') -> pd.DataFrame:
    """
    Load and preprocess the EU yield curve data.
    """
    yield_data = pd.read_csv(file_path)

    expected_columns = ['EU_1Y', 'EU_5Y', 'EU_10Y', 'EU_20Y', 'EU_30Y']

    if not all(col in yield_data.columns for col in expected_columns):
        raise ValueError('The expected columns do not exist in the dataframe')

    yield_data['DATE'] = pd.to_datetime(yield_data['DATE'])
    yield_data.set_index('DATE', inplace=True)

    return yield_data


def check_data_quality(yield_data: pd.DataFrame) -> dict:
    """
    Perform basic quality checks on the yield data.
    """
    results = {}

    # Check for missing values
    results['missing_values'] = yield_data.isna().sum().to_dict()

    # Check for zero variance
    results['variance'] = yield_data.var().to_dict()

    # Check condition number
    results['condition_number'] = np.linalg.cond(yield_data)

    return results