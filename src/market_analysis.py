import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from config import REGIME_WINDOW


def detect_regime(
        yield_df: pd.DataFrame,
        window: int = REGIME_WINDOW
) -> pd.DataFrame:
    """
    Detect market regimes based on yield curve behavior.
    """
    df = pd.DataFrame(index=yield_df.index)

    df['2s10s_Slope'] = yield_df['EU_10Y'] - yield_df['EU_1Y']
    df['EU_10Y'] = yield_df['EU_10Y']

    df['Level'] = yield_df[['EU_1Y', 'EU_5Y', 'EU_10Y', 'EU_20Y', 'EU_30Y']].mean(axis=1)

    df['Yield_Volatility'] = yield_df['EU_10Y'].rolling(window=21).std() * np.sqrt(252)

    regimes = pd.DataFrame(index=yield_df.index, columns=['Regime'])
    regimes['Regime'] = 'Normal'

    level_change = df['Level'].rolling(window=window).mean().diff(window)
    slope_change = df['2s10s_Slope'].rolling(window=window).mean().diff(window)

    bull_steepening = (level_change < -0.05) & (slope_change > 0.02)
    bear_steepening = (level_change > 0.05) & (slope_change > 0.02)
    bull_flattening = (level_change < -0.05) & (slope_change < -0.02)
    bear_flattening = (level_change > 0.05) & (slope_change < -0.02)

    vol_90th = df['Yield_Volatility'].rolling(window=252).quantile(0.90)
    high_vol = df['Yield_Volatility'] > vol_90th

    vol_25th = df['Yield_Volatility'].rolling(window=252).quantile(0.25)
    range_bound = (df['Yield_Volatility'] < vol_25th) & (abs(level_change) < 0.02)

    regimes.loc[bull_steepening, 'Regime'] = 'Bull_Steepening'
    regimes.loc[bull_flattening, 'Regime'] = 'Bull_Flattening'
    regimes.loc[bear_steepening, 'Regime'] = 'Bear_Steepening'
    regimes.loc[bear_flattening, 'Regime'] = 'Bear_Flattening'
    regimes.loc[range_bound, 'Regime'] = 'Range_Bound'
    regimes.loc[high_vol, 'Regime'] = 'High_Volatility'


    regimes['Regime'] = regimes['Regime'].fillna('Normal')

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
            # Convert to numeric to ensure clean data
            series = pd.to_numeric(series, errors='coerce').dropna()
            if len(series) > window:  # Check again after cleaning
                try:
                    adf_result = adfuller(series.values,
                                          maxlag=int(np.ceil(np.power(len(series) / 100, 0.25))))
                    results.loc[col, 'ADF_statistic'] = adf_result[0]
                    results.loc[col, 'p-value'] = adf_result[1]
                    results.loc[col, 'Mean_Reversion_Score'] = 1 - adf_result[1]
                except Exception as e:
                    print(f"Error in ADF test for {col}: {str(e)}")
                    # Provide default values
                    results.loc[col, 'ADF_statistic'] = 0
                    results.loc[col, 'p-value'] = 0.5
                    results.loc[col, 'Mean_Reversion_Score'] = 0.5

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
            try:
                # Extract the data for the current window
                series_slice = deviations[col].iloc[i - window:i]

                # Explicitly convert to numeric values, coercing errors to NaN
                series_clean = pd.to_numeric(series_slice, errors='coerce')

                # Drop NaN values
                series_clean = series_clean.dropna()

                # Skip if not enough data
                if len(series_clean) <= window // 2:
                    continue

                # Create lagged series (shifted by 1)
                series_lag = series_clean.shift(1)

                # Drop the first value (which will be NaN due to the shift)
                series_lag = series_lag.iloc[1:]
                series_clean = series_clean.iloc[1:]

                # Make sure both series are the same length
                if len(series_clean) != len(series_lag):
                    # This shouldn't happen, but just in case
                    min_len = min(len(series_clean), len(series_lag))
                    series_clean = series_clean.iloc[:min_len]
                    series_lag = series_lag.iloc[:min_len]

                # Skip if we don't have enough data points after cleaning
                if len(series_clean) < 10:
                    continue

                # Fit the AR(1) model
                X = sm.add_constant(series_lag)
                model = sm.OLS(series_clean, X)
                result = model.fit()
                phi = result.params[1]

                # Calculate half-life only if the AR coefficient is between 0 and 1
                if 0 < phi < 1:
                    half_life = -np.log(2) / np.log(phi)
                    half_lives.loc[deviations.index[i], col] = half_life
                else:
                    half_lives.loc[deviations.index[i], col] = np.nan

            except Exception as e:
                print(f"Error estimating half-life for {col} at index {i}: {str(e)}")
                continue

    return half_lives