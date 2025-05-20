import pandas as pd
import numpy as np
from config import TENORS


def calculate_total_pnl(
        yield_data,
        strategy_results,
        dv01,
        carry_rolldown
):
    """
    Calculate total P&L including all components.
    """
    positions = strategy_results['positions']
    trades = strategy_results['trades']

    # Initialize P&L components
    mtm_pnl = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Import from strategy module to avoid circular imports
    from src.strategy import calculate_transaction_costs, calculate_financing_costs

    transaction_cost = calculate_transaction_costs(positions, dv01)
    financing_cost = calculate_financing_costs(positions, yield_data)
    carry_benefit = pd.DataFrame(0, index=positions.index, columns=positions.columns)

    # Calculate mark-to-market P&L (from yield changes)
    for col in positions.columns:
        tenor = col.split('_')[1]
        dv01_col = f"DV01_{tenor}"

        # Calculate daily yield changes
        yield_change = yield_data[col].diff()

        # P&L = -position * yield_change * DV01
        # Negative position (short) profits from yield increases
        mtm_pnl[col] = -positions[col].shift(1) * yield_change * dv01[dv01_col].shift(1)

        # Calculate carry benefit
        carry_col = f"Total_{tenor}"
        # Daily carry = position * carry_rolldown / 252
        carry_benefit[col] = positions[col] * carry_rolldown[carry_col] / 252

    # Combine all P&L components
    total_pnl = pd.DataFrame(index=positions.index)
    total_pnl['MTM_PnL'] = mtm_pnl.sum(axis=1)
    total_pnl['Transaction_Costs'] = -transaction_cost.sum(axis=1)
    total_pnl['Financing_Costs'] = -financing_cost.sum(axis=1)
    total_pnl['Carry_Benefit'] = carry_benefit.sum(axis=1)
    total_pnl['Total_PnL'] = total_pnl['MTM_PnL'] + total_pnl['Transaction_Costs'] + \
                             total_pnl['Financing_Costs'] + total_pnl['Carry_Benefit']

    # Cumulative P&L
    total_pnl['Cumulative_PnL'] = total_pnl['Total_PnL'].cumsum()

    return total_pnl


def calculate_performance_metrics(pnl_data, benchmark_returns=None):
    """
    Calculate performance metrics for strategy.
    """
    daily_returns = pnl_data.dropna()

    # Basic return metrics
    total_return = daily_returns.sum()
    avg_daily_return = daily_returns.mean()
    annualized_return = avg_daily_return * 252

    # Risk metrics
    volatility = daily_returns.std()
    annualized_vol = volatility * np.sqrt(252)
    downside_vol = daily_returns[daily_returns < 0].std()
    annualized_downside_vol = downside_vol * np.sqrt(252) if not np.isnan(downside_vol) else np.nan

    # Ratios
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    sortino_ratio = annualized_return / annualized_downside_vol if not np.isnan(
        annualized_downside_vol) and annualized_downside_vol != 0 else np.nan

    # Drawdown analysis
    cum_returns = daily_returns.cumsum()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max)
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Win/loss metrics
    win_days = (daily_returns > 0).sum()
    loss_days = (daily_returns < 0).sum()
    win_rate = win_days / (win_days + loss_days) if (win_days + loss_days) > 0 else np.nan

    gross_wins = daily_returns[daily_returns > 0].sum()
    gross_losses = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses != 0 else np.nan

    info_ratio = np.nan
    if benchmark_returns is not None:

        aligned_benchmark = benchmark_returns[daily_returns.index]

        tracking_error = (daily_returns - aligned_benchmark).std() * np.sqrt(252)

        benchmark_return = aligned_benchmark.mean() * 252
        info_ratio = (annualized_return - benchmark_return) / tracking_error if tracking_error != 0 else np.nan

    metrics = {
        'Total_Return': total_return,
        'Annualized_Return': annualized_return,
        'Annualized_Volatility': annualized_vol,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Max_Drawdown': max_drawdown,
        'Calmar_Ratio': calmar_ratio,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Information_Ratio': info_ratio
    }

    return metrics


def perform_pnl_attribution(strategy_results, pnl_components):
    """
    Perform P&L attribution analysis.
    """
    # Attribution by tenor
    tenor_attribution = pd.DataFrame(index=strategy_results['positions'].columns)

    for col in strategy_results['positions'].columns:
        # Calculate contribution from each tenor
        tenor_attribution.loc[col, 'Total_PnL'] = strategy_results['realized_pnl'][col].sum()
        tenor_attribution.loc[col, 'Contribution'] = tenor_attribution.loc[col, 'Total_PnL'] / tenor_attribution[
            'Total_PnL'].sum()

    # Attribution by P&L component
    component_attribution = pd.DataFrame(index=pnl_components.columns[:-1])  # Exclude cumulative column

    for col in pnl_components.columns[:-1]:
        component_attribution.loc[col, 'Total'] = pnl_components[col].sum()
        component_attribution.loc[col, 'Contribution'] = component_attribution.loc[col, 'Total'] / pnl_components[
            'Total_PnL'].sum()

    # Time-based rolling attribution
    rolling_contribution = pnl_components.rolling(63).sum()

    return {
        'by_tenor': tenor_attribution,
        'by_component': component_attribution,
        'rolling': rolling_contribution
    }


def perform_stress_test(strategy, yield_data, regime_data, shock_scenarios=None):
    """
    Perform stress testing by simulating market shocks.
    """

    if shock_scenarios is None:
        # Define default shock scenarios
        shock_scenarios = {
            'Parallel_Up_50bp': {
                'type': 'parallel',
                'magnitude': 0.5  # 50 bps
            },
            'Parallel_Up_100bp': {
                'type': 'parallel',
                'magnitude': 1.0  # 100 bps
            },
            'Parallel_Down_50bp': {
                'type': 'parallel',
                'magnitude': -0.5  # -50 bps
            },
            'Steepening_50bp': {
                'type': 'steepening',
                'magnitude': 0.5  # 50 bps
            },
            'Flattening_50bp': {
                'type': 'flattening',
                'magnitude': 0.5  # 50 bps
            },
            'Volatility_Double': {
                'type': 'volatility',
                'magnitude': 2.0  # Double volatility
            }
        }

    # Add historical stress periods if available
    if 'High_Volatility' in regime_data['Regime'].values:
        # Find historical stress periods
        stress_periods = regime_data[regime_data['Regime'] == 'High_Volatility'].index
        if len(stress_periods) > 0:
            shock_scenarios['Historical_Stress'] = {
                'type': 'historical',
                'periods': stress_periods
            }

    # Initialize results
    stress_results = {}

    # Get current positions
    current_positions = strategy['positions'].iloc[-1]

    # Create shocked yield curves and calculate P&L impact
    for scenario_name, scenario in shock_scenarios.items():
        if scenario['type'] == 'parallel':
            # Parallel shift
            shocked_yields = yield_data.iloc[-1] + scenario['magnitude']

            # Calculate P&L impact (simplified)
            pnl_impact = -current_positions * scenario['magnitude']

        elif scenario['type'] == 'steepening':
            # Steepening: short end unchanged, long end up
            shocked_yields = yield_data.iloc[-1].copy()

            for i, tenor in enumerate(TENORS):
                # Linear increase in shock from short to long end
                tenor_shock = scenario['magnitude'] * i / (len(TENORS) - 1)
                shocked_yields[f'EU_{tenor}Y'] += tenor_shock

            # Calculate P&L impact
            pnl_impact = pd.Series(0, index=current_positions.index)
            for i, col in enumerate(current_positions.index):
                tenor = col.split('_')[1]
                tenor_idx = TENORS.index(int(tenor[:-1]))
                tenor_shock = scenario['magnitude'] * tenor_idx / (len(TENORS) - 1)
                pnl_impact[col] = -current_positions[col] * tenor_shock

        elif scenario['type'] == 'flattening':
            # Flattening: short end up, long end unchanged
            shocked_yields = yield_data.iloc[-1].copy()

            for i, tenor in enumerate(TENORS):
                # Linear decrease in shock from short to long end
                tenor_shock = scenario['magnitude'] * (len(TENORS) - 1 - i) / (len(TENORS) - 1)
                shocked_yields[f'EU_{tenor}Y'] += tenor_shock

            # Calculate P&L impact
            pnl_impact = pd.Series(0, index=current_positions.index)
            for i, col in enumerate(current_positions.index):
                tenor = col.split('_')[1]
                tenor_idx = TENORS.index(int(tenor[:-1]))
                tenor_shock = scenario['magnitude'] * (len(TENORS) - 1 - tenor_idx) / (len(TENORS) - 1)
                pnl_impact[col] = -current_positions[col] * tenor_shock

        elif scenario['type'] == 'volatility':
            # Assume positions remain, but vol increases, widening bid-ask
            # This impacts transaction costs on exit

            # Calculate impact on transaction costs
            pnl_impact = pd.Series(0, index=current_positions.index)
            for col in current_positions.index:
                tenor = col.split('_')[1]
                # Base bid-ask in bps, scaled by volatility factor
                base_bid_ask = {'1Y': 0.5, '5Y': 1.0, '10Y': 1.5, '20Y': 2.5, '30Y': 3.0}
                shocked_bid_ask = base_bid_ask[tenor] * scenario['magnitude']

                # Impact = position * increased half spread
                pnl_impact[col] = -abs(current_positions[col]) * ((shocked_bid_ask - base_bid_ask[tenor]) / 10000 / 2)

        elif scenario['type'] == 'historical':
            # Use historical stress period returns
            pnl_impact = pd.Series(0, index=current_positions.index)

            # Find worst day in stress period
            worst_changes = {}
            for col in current_positions.index:
                period_returns = yield_data.loc[scenario['periods'], col].diff()
                worst_change = period_returns.abs().max() * np.sign(period_returns.abs().idxmax())
                worst_changes[col] = worst_change

                # Calculate impact
                pnl_impact[col] = -current_positions[col] * worst_change

        # Store results
        stress_results[scenario_name] = {
            'shocked_yields': shocked_yields if 'shocked_yields' in locals() else None,
            'pnl_impact': pnl_impact,
            'total_impact': pnl_impact.sum()
        }

    return stress_results