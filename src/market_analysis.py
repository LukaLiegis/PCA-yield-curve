import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from config import REGIME_WINDOW


def detect_regime(yield_df: pd.DataFrame, window: int = REGIME_WINDOW) -> pd.DataFrame:
    """
    Detect market regimes based on yield curve behavior.
    """
    df = pd.DataFrame(index=yield_df.index)

    # Slope measures
    df['2s10s_Slope'] = yield_df['EU_10Y'] - yield_df['EU_1Y']

    df['EU_10Y'] = yield_df['EU_10Y']

    # Level
    df['Level'] = yield_df['EU_10Y']

    df['Yield_Volatility'] = yield_df['EU_10Y'].rolling(window=21).std() * 100

    regimes = pd.DataFrame(index=yield_df.index, columns=['Regime'])

    bull_steepening = (df['2s10s_Slope'].rolling(window=window).mean() > 0) & (
                df['EU_10Y'].rolling(window=window).mean().diff(window) < 0)
    bull_flattening = (df['2s10s_Slope'].rolling(window=window).mean() < 0) & (
                df['EU_10Y'].rolling(window=window).mean().diff(window) < 0)
    bear_steepening = (df['2s10s_Slope'].rolling(window=window).mean() > 0) & (
                df['EU_10Y'].rolling(window=window).mean().diff(window) > 0)
    bear_flattening = (df['2s10s_Slope'].rolling(window=window).mean() < 0) & (
                df['EU_10Y'].rolling(window=window).mean().diff(window) > 0)
    # TODO: flattening_twist
    # TODO: steepening_twist

    regimes.loc[bull_steepening, 'Regime'] = 'Bull_Steepening'
    regimes.loc[bull_flattening, 'Regime'] = 'Bull_Flattening'
    regimes.loc[bear_steepening, 'Regime'] = 'Bear_Steepening'
    regimes.loc[bear_flattening, 'Regime'] = 'Bear_Flattening'
    # TODO: flattening_twist
    # TODO: steepening_twist

    regimes['Regime'].fillna('Normal', inplace=True)

    return regimes


def mean_reversion_tests(deviations, window: int = 126) -> pd.DataFrame:
    """
    Perform statistical tests for mean reversion on yield curve deviations.
    """
    results = pd.DataFrame(index=deviations.columns,
                           columns=['ADF_Statistic', 'p-value', 'Mean_Reversion_Score'])

    for col in deviations.columns:
        series = deviations[col].dropna()
        if len(series) > window:
            adf_result = adfuller(series.values,
                                  maxlag=int(np.ceil(np.power(len(series) / 100, 0.25))))
            results.loc[col, 'ADF_statistic'] = adf_result[0]
            results.loc[col, 'p-value'] = adf_result[1]
            results.loc[col, 'Mean_Reversion_Score'] = 1 - adf_result[1]

    return results


def estimate_half_life(
        deviations,
        window: int = 126
) -> pd.DataFrame:
    """
    Estimate the half life of mean reversion for each tenor.
    """
    half_lives = pd.DataFrame(index=deviations.index, columns=deviations.columns)

    for col in deviations.columns:
        for i in range(window, len(deviations)):
            # Use AR(1) model to estimate mean reversion speed.
            y = deviations[col].iloc[i - window:i]
            y = y.dropna()

            if len(y) > window // 2:
                y_lag = y.shift(1).dropna()
                y = y.iloc[1:]  # Align with lagged values

                model = sm.OLS(y, sm.add_constant(y_lag))
                try:
                    result = model.fit()
                    phi = result.params[1]

                    if 0 < phi < 1:
                        half_life = -np.log(2) / np.log(phi)
                        half_lives.loc[deviations.index[i], col] = half_life
                    else:
                        half_lives.loc[deviations.index[i], col] = np.nan
                except:
                    half_lives.loc[deviations.index[i], col] = np.nan

    return half_lives