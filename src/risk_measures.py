import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
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


def calculate_dv01(
        yields,
        durations,
        face_value: int = 100
) -> pd.DataFrame:
    """
    Calculate DV01 (dollar value of 01) for each tenor.
    """
    dv01 = pd.DataFrame(index=yields.index, columns=[f'DV01_{t}Y' for t in TENORS])

    for tenor in TENORS:
        col = f'EU_{tenor}Y'
        dur_col = f'DUR_{tenor}Y'
        dv01_col = f'DV01_{tenor}Y'

        # Calculate approximate bond price (zero-coupon assumption)
        bond_price = face_value / ((1 + yields[col] / 100) ** tenor)

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

    # Make repo spreads dynamic
    if repo_spread is None:
        repo_spread = pd.DataFrame(
            index=yield_df.index,
            columns=[f'Repo_Spread_{t}Y' for t in tenors]
        )

        base_spreads = {
            1: 0.05,
            5: 0.10,
            10: 0.15,
            20: 0.20,
            30: 0.25,
        }

        # Adjust spreads based on yield volatility
        yield_vol = yield_df.rolling(21).std().mean(axis=1)
        vol_adjustment = yield_vol / yield_vol.mean()

        for tenor in tenors:
            repo_spread[f'Repo_Spread_{tenor}Y'] = base_spreads[tenor] * vol_adjustment

    for date in yield_df.index:

        if date == yield_df.index[0]:
            continue

        current_curve = []
        valid_tenors = []

        for tenor in tenors:
            col = f'EU_{tenor}Y'
            if col in yield_df.columns and not pd.isna(yield_df.loc[date, col]):
                current_curve.append(yield_df.loc[date, col])
                valid_tenors.append(tenor)

        if len(valid_tenors) < 3: # Need at least 3 for spline
            continue

        curve_spline = CubicSpline(valid_tenors, current_curve, extrapolate=False)

        for i, tenor in enumerate(valid_tenors):
            col = f'EU_{tenor}Y'
            financing_cost = funding_rates.loc[date, 'Funding_Rate'] + repo_spread.loc[date, f'Repo_Spread_{tenor}Y']
            carry_rolldown.loc[date, f'Carry_{tenor}Y'] = yield_df.loc[date, col] - financing_cost

        rolldown_period = 0.25 # 3 months

        for i, tenor in enumerate(valid_tenors):
            if tenor > rolldown_period:
                try:
                    current_yield = yield_df.loc[date, f'EU_{tenor}Y']
                    rolled_tenor = tenor - rolldown_period
                    rolled_yield = curve_spline(rolled_tenor)

                    if not np.isnan(rolled_yield):
                        rolldown = (current_yield - rolled_yield) * rolldown_period
                        carry_rolldown.loc[date, f'Rolldown_{tenor}Y'] = rolldown
                    else:
                        carry_rolldown.loc[date, f'Rolldown_{tenor}Y'] = 0
                except:
                    carry_rolldown.loc[date, f'Rolldown_{tenor}Y'] = 0
            else:
                carry_rolldown.loc[date, f'Rolldown_{tenor}Y'] = 0

    for tenor in tenors:
        carry_col = f'Carry_{tenor}Y'
        roll_col = f'Rolldown_{tenor}Y'
        if carry_col in carry_rolldown.columns and roll_col in carry_rolldown.columns:
            carry_rolldown[f'Total_{tenor}Y'] = (
                carry_rolldown[carry_col].fillna(0) +
                carry_rolldown[roll_col].fillna(0)
            )

    return carry_rolldown