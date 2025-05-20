import pandas as pd
import numpy as np
import scipy.interpolate
from config import TENORS


def calculate_duration(yields, tenors=TENORS) -> pd.DataFrame:
    """
    Calculate duration for each tenor based on yield data.
    """
    durations = pd.DataFrame(index=yields.index, columns=[f'DUR_{t}Y' for t in tenors])

    for tenor in tenors:
        col = f'EU_{tenor}Y'
        durations[f'DUR_{tenor}Y'] = tenor / (1 + yields[col] / 100)

    return durations


def calculate_dv01(yields, durations, face_value: int = 100) -> pd.DataFrame:
    """
    Calculate DV01 (dollar value of 01) for each tenor.
    """
    dv01 = pd.DataFrame(index=yields.index, columns=[f'DV01_{t}Y' for t in TENORS])

    for tenor in TENORS:
        bond_price = face_value
        dur_col = f'DUR_{tenor}Y'
        dv01_col = f'DV01_{tenor}Y'
        dv01[dv01_col] = durations[dur_col] * bond_price * 0.0001

    return dv01


def calculate_carry_and_rolldown(
        yield_df: pd.DataFrame,
        funding_rates: pd.DataFrame = None,
        repo_spread: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calculate carry and rolldown for each tenor.
    """
    tenors = TENORS
    carry_rolldown = pd.DataFrame(index=yield_df.index)

    # If funding rates are not provided, approximate using shortest tenor
    if funding_rates is None:
        funding_rates = pd.DataFrame(
            yield_df['EU_1Y'].values,
            index=yield_df.index,
            columns=['Funding_Rate']
        )

    # If repo spread not provided use reasonable defaults
    if repo_spread is None:
        repo_spread = pd.DataFrame(
            index=yield_df.index,
            columns=[f'Repo_Spread_{t}Y' for t in tenors]
        )

        repo_spread['Repo_Spread_1Y'] = 0.05
        repo_spread['Repo_Spread_5Y'] = 0.10
        repo_spread['Repo_Spread_10Y'] = 0.15
        repo_spread['Repo_Spread_20Y'] = 0.20
        repo_spread['Repo_Spread_30Y'] = 0.25

    for date in yield_df.index:
        curve = yield_df.loc[date].values

        # Calculate financing adjusted carry
        carry = np.zeros_like(curve)
        for i, tenor in enumerate(tenors):
            financing_cost = funding_rates.loc[date, 'Funding_Rate'] + repo_spread.loc[date, f'Repo_Spread_{tenor}Y']
            carry[i] = curve[i] - financing_cost

        # Calculate rolldown using cubic spline
        curve_spline = scipy.interpolate.CubicSpline(tenors, curve)
        rolldown = np.zeros_like(curve)
        for i, tenor in enumerate(tenors):
            if tenor > 1:
                rolldown[i] = curve[i] - curve_spline(tenor - 1)
            else:
                rolldown[i] = 0  # No rolldown for shorter tenor

        carry_rolldown.loc[date, [f'Carry_{t}Y' for t in tenors]] = carry
        carry_rolldown.loc[date, [f'Rolldown_{t}Y' for t in tenors]] = rolldown

    for tenor in tenors:
        carry_rolldown[f'Total_{tenor}Y'] = carry_rolldown[f'Carry_{tenor}Y'] + carry_rolldown[f'Rolldown_{tenor}Y']

    return carry_rolldown