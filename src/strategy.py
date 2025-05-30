import pandas as pd
import numpy as np
from config import MIN_SIGNAL_CONFIDENCE, MAX_POSITION_SIZE, STOP_LOSS_THRESHOLD


def calculate_position_size(
        deviation,
        confidence,
        half_life,
        carry_rolldown,
        dv01,
        current_regime = 'Normal',
        portfolio_value = 1_000_000,
        max_position_pct: float = MAX_POSITION_SIZE
):
    """
    Calculate position size based on kelly and risk limits.
    """
    signal = -deviation

    if half_life > 0 and half_life < 252:
        expected_holding_period = min(half_life * 2, 60)
        mean_reversion_return = abs(deviation) * (1 - 0.5 ** (expected_holding_period / half_life))
    else:
        mean_reversion_return = abs(deviation) * 0.5

    carry_daily = carry_rolldown / 252 if carry_rolldown > 0 else 0
    total_expected_return = mean_reversion_return + carry_daily * expected_holding_period

    volatility = 0.5
    # TODO: make this value dependant on past data

    kelly_fraction = total_expected_return / (volatility  ** 2)

    regime_multipliers = {
        'High_Volatility': 0.5,
        'Range_Bound': 1.2,
        'Normal': 1.0,
        'Bull_Steepening': 0.8,
        'Bear_Steepening': 0.8,
        'Bull_Flattening': 0.9,
        'Bear_Flattening': 0.9
    }

    regime_mult = regime_multipliers.get(current_regime, 1.0)

    safe_kelly = 0.25 * kelly_fraction * regime_mult * confidence

    position_value = safe_kelly * portfolio_value
    position_dv01_units = position_value / (dv01 * 10000) if dv01 > 0 else 0

    max_position_value = max_position_pct * portfolio_value
    max_position_dv01 = max_position_value / (dv01 * 10000) if dv01 > 0 else MAX_POSITION_SIZE

    final_size = np.sign(signal) * min(abs(position_dv01_units), max_position_dv01)

    return final_size


def simulate_enhanced_trading_strategy(
        yield_data,
        z_scores,
        mean_reversion_stats,
        half_lives,
        carry_rolldown,
        dv01,
        threshold=1.5,
        min_confidence=MIN_SIGNAL_CONFIDENCE,
        stop_loss_threshold=STOP_LOSS_THRESHOLD,
        regime_data = None
):
    """
    Simulate trading strategy based on PCA yield curve deviations.
    """
    pd.set_option('future.no_silent_downcasting', True)

    positions = pd.DataFrame(0, index=yield_data.index, columns=yield_data.columns)
    trades = pd.DataFrame(index=yield_data.index, columns=yield_data.columns)
    unrealized_pnl = pd.DataFrame(0, index=yield_data.index, columns=yield_data.columns)
    realized_pnl = pd.DataFrame(0, index=yield_data.index, columns=yield_data.columns)

    active_trades = {}
    for col in yield_data.columns:
        active_trades[col] = {
            'position': 0,
            'entry_date': None,
            'entry_price': 0,
            'stop_loss': 0,
            'target': 0,
            'half_life': 0
        }

    portfolio_metrics = pd.DataFrame(
        index = yield_data.index,
        columns = ['total_risk', 'var_95', 'positions_count'],
    )

    # Simulate day by day
    for i, date in enumerate(yield_data.index[252:], 252):  # Start after warmup period
        prev_date = yield_data.index[i - 1]

        current_regime = regime_data.loc[date, 'Regime'] if regime_data is not None else 'Normal'

        for col in yield_data.columns:
            tenor = col.split('_')[1]

            if pd.isna(z_scores.loc[date, col]) or pd.isna(dv01.loc[date, f'DV01_{tenor}']):
                continue

            deviation = z_scores.loc[date, col]

            confidence = mean_reversion_stats.loc[
                col, 'Mean_Reversion_Score'] if col in mean_reversion_stats.index else 0

            hl = half_lives.loc[date, col] if not pd.isna(half_lives.loc[date, col]) else 0

            cr = carry_rolldown.loc[date, f'Total_{tenor}'] if not pd.isna(
                carry_rolldown.loc[date, f'Total_{tenor}']) else 0

            current_dv01 = dv01.loc[date, f'DV01_{tenor}']

            # 1. Check if we need to close existing position due to stop loss or target reached
            current_trade = active_trades[col]
            if current_trade['position'] != 0:

                current_yield = yield_data.loc[date, col]
                entry_yield = current_trade['entry_price']
                pnl_bps = -current_trade['position'] * (current_yield - entry_yield) * 100

                unrealized_pnl.loc[date, col] = pnl_bps

                position_vol = abs(current_trade['position']) * 0.5
                dynamic_stop = -2 * position_vol * 100

                if pnl_bps <= dynamic_stop:

                    trades.loc[date, col] = -current_trade['position']
                    realized_pnl.loc[date, col] = pnl_bps
                    active_trades[col] = {
                        'position': 0,
                        'entry_date': None,
                        'entry_price': 0,
                        'stop_loss': 0,
                        'target': 0,
                        'half_life': 0
                    }
                    continue

                if (current_trade['position'] > 0 and deviation < 0.5) or \
                    (current_trade['position'] < 0 and deviation > -0.5):

                    trades.loc[date, col] = -current_trade['position']
                    realized_pnl.loc[date, col] = pnl_bps
                    active_trades[col] = {
                        'position': 0,
                        'entry_date': None,
                        'entry_price': 0,
                        'stop_loss': 0,
                        'target': 0,
                        'half_life': 0
                    }
                    continue

                entry_date = current_trade['entry_date']
                days_held = (yield_data.index.get_loc(date) - yield_data.index.get_loc(entry_date))

                max_holding = current_trade['half_life'] * 2 if current_trade['half_life'] > 0 else 60

                if days_held > max_holding:

                    trades.loc[date, col] = -current_trade['position']
                    realized_pnl.loc[date, col] = pnl_bps
                    active_trades[col] = {
                        'position': 0,
                        'entry_date': None,
                        'entry_price': 0,
                        'stop_loss': 0,
                        'target': 0,
                        'half_life': 0
                    }
                    continue

            # 2. Check for new entry signal if no position
            if active_trades[col]['position'] == 0:

                regime_thresholds = {
                    'High_Volatility': threshold * 1.5,
                    'Range_Bound': threshold * 0.75,
                    'Normal': threshold,
                }
                current_threshold = regime_thresholds.get(current_regime, threshold)

                if (abs(deviation) > current_threshold) and (confidence > min_confidence):
                    # Calculate position size
                    size = calculate_position_size(
                        deviation=deviation,
                        confidence=confidence,
                        half_life=hl if hl > 0 else 20,  # Default to 20 days if unknown
                        carry_rolldown=cr,
                        dv01=current_dv01,
                        current_regime=current_regime,
                    )

                    if abs(size) > 0.1:  # Meaningful position size
                        # Enter new position
                        trades.loc[date, col] = size

                        active_trades[col] = {
                            'position': size,
                            'entry_date': date,
                            'entry_price': yield_data.loc[date, col],
                            'stop_loss': yield_data.loc[date, col] + (stop_loss_threshold / size),
                            'target': yield_data.loc[date, col] - (deviation / 2),
                            'half_life': hl if hl > 0 else 20
                        }

        if i > 0:
            trades_slice = trades.loc[date]
            trades_slice = pd.Series(
                np.where(trades_slice.isna(), 0, trades_slice),
                index=trades_slice.index
            )
            positions.loc[date] = positions.loc[prev_date] + trades_slice

    return {
        'positions': positions,
        'trades': trades,
        'unrealized_pnl': unrealized_pnl,
        'realized_pnl': realized_pnl,
        'portfolio_metrics': portfolio_metrics,
    }


def calculate_transaction_costs(
        positions,
        dv01,
        market_volatility = None,
) -> pd.DataFrame:
    """
    Calculate transaction costs based on position changes.
    """
    bid_ask_bips = {
        'EU_1Y': 0.25,
        'EU_5Y': 0.50,
        'EU_10Y': 0.75,
        'EU_20Y': 1.25,
        'EU_30Y': 1.5,
    }

    transaction_costs = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    for col in positions.columns:
        position_changes = positions[col].diff()
        position_changes = pd.Series(
            np.where(position_changes.isna(), 0, position_changes),
            index=position_changes.index
        ).abs()

        if market_volatility is not None:
            vol_adjustment = np.clip(market_volatility / market_volatility.mean(), 0.5, 2.0)
        else:
            vol_adjustment = 1.0

        adjustment_spread = bid_ask_bips[col] * vol_adjustment

        dv01_col = f'DV01_{col.split("_")[1]}'
        transaction_costs[col] = position_changes * (adjustment_spread / 10000 / 2) * dv01[dv01_col] * 10000

    return transaction_costs


def calculate_financing_costs(
        positions,
        yield_data,
        funding_rates: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Calculate financing costs for positions.
    """
    if funding_rates is None:
        funding_rates = pd.DataFrame(
            yield_data['EU_1Y'].values * 0.9,  # Assume funding at 90% of 1Y rate
            index=yield_data.index,
            columns=['Funding_Rate']
        )

    financing_spreads = {
        'EU_1Y': 0.05,
        'EU_5Y': 0.10,
        'EU_10Y': 0.15,
        'EU_20Y': 0.20,
        'EU_30Y': 0.25,
    }

    financing_costs = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    for col in positions.columns:
        abs_position = positions[col].abs()
        daily_cost = abs_position * (funding_rates['Funding_Rate'] + financing_spreads[col]) / 25200
        financing_costs[col] = daily_cost

    return financing_costs