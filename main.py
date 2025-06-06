import os

import pandas as pd

from src.data_loader import load_yield_data, check_data_quality
from src.risk_measures import calculate_duration, calculate_dv01, calculate_carry_and_rolldown
from src.market_analysis import detect_regime, mean_reversion_tests, estimate_half_life
from src.pca import perform_yield_pca, calculate_deviations
from src.strategy import simulate_enhanced_trading_strategy
from src.performance import calculate_total_pnl, calculate_performance_metrics, perform_pnl_attribution, \
    perform_stress_test
from src.visualization import visualize_strategy_results, visualize_pca_components, visualize_stress_test_results


def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('output/images', exist_ok=True)


def run_enhanced_strategy():
    """
    Main execution function to run the enhanced trading strategy.
    """
    print("Loading data...")
    yield_data = load_yield_data()

    print("Checking data quality...")
    quality_checks = check_data_quality(yield_data)
    print(f"Missing values: {quality_checks['missing_values']}")
    print(f"Condition number: {quality_checks['condition_number']:.2f}")

    print("Calculating risk measures...")
    durations = calculate_duration(yield_data)
    dv01 = calculate_dv01(yield_data, durations)

    print("Computing carry and rolldown...")
    carry_rolldown = calculate_carry_and_rolldown(yield_data)

    print("Detecting market regimes...")
    regime_data = detect_regime(yield_data)

    print(f"Regime distribution:\n{regime_data['Regime'].value_counts()}")

    print("Performing PCA analysis...")
    cutoff_date = yield_data.index[int(len(yield_data) * 0.7)]
    in_sample = yield_data.loc[:cutoff_date]
    out_sample = yield_data.loc[cutoff_date:]

    pca_results = perform_yield_pca(
        in_sample,
        regime_data.loc[:cutoff_date],
        use_changes=True
    )

    print("Calculating yield curve deviations...")
    deviation_results = calculate_deviations(yield_data, pca_results, regime_data)
    z_scores = deviation_results['z_scores']

    print("Testing for mean reversion...")
    mr_test = mean_reversion_tests(deviation_results['raw_deviations'])
    half_lives = estimate_half_life(deviation_results['raw_deviations'])

    avg_half_lives = half_lives.mean()
    print("Average half-lives by tenor:")
    for col in avg_half_lives.index:
        if not pd.isna(avg_half_lives[col]):
            print(f"{col}: {avg_half_lives[col]:.1f} days")

    print("Simulating trading strategy...")
    strategy_results = simulate_enhanced_trading_strategy(
        yield_data,
        z_scores,
        mr_test,
        half_lives,
        carry_rolldown,
        dv01,
        regime_data,
    )

    print("Calculating performance metrics...")
    total_pnl = calculate_total_pnl(
        yield_data,
        strategy_results,
        dv01,
        carry_rolldown
    )

    in_sample_pnl = total_pnl.loc[:cutoff_date, 'Total_PnL']
    out_sample_pnl = total_pnl.loc[cutoff_date:, 'Total_PnL']

    in_sample_metrics = calculate_performance_metrics(in_sample_pnl)
    out_sample_metrics = calculate_performance_metrics(out_sample_pnl)

    print("Performing P&L attribution...")
    attribution = perform_pnl_attribution(strategy_results, total_pnl)

    print("Running stress tests...")
    stress_results = perform_stress_test(strategy_results, yield_data, regime_data)

    turnover = strategy_results['trades'].abs().sum(axis=1).sum()
    print(f"Average daily turnover: {turnover:.2f} DV01 units")

    avg_positions = (strategy_results['positions'] != 0).sum(axis=1).sum()
    print(f"Average daily positions: {avg_positions:.2f} DV01 units")

    position_concentration = strategy_results['positions'].abs().sum(axis=1)
    max_concentration = position_concentration.max()
    print(f"Maximum position concentration: {max_concentration:.1f} DV01 units")

    results = {
        'yield_data': yield_data,
        'regime_data': regime_data,
        'pca_results': pca_results,
        'deviation_results': deviation_results,
        'mean_reversion_stats': mr_test,
        'half_lives': half_lives,
        'strategy_results': strategy_results,
        'total_pnl': total_pnl,
        'in_sample_metrics': in_sample_metrics,
        'out_sample_metrics': out_sample_metrics,
        'attribution': attribution,
        'stress_results': stress_results,
        'turnover': turnover,
        'avg_positions': avg_positions,
        'max_concentration': max_concentration,
    }

    print("Creating visualizations...")
    visualize_strategy_results(results)
    visualize_pca_components(pca_results)
    visualize_stress_test_results(stress_results)

    return results


def print_performance_summary(results):
    """Print a summary of strategy performance."""
    print("\nIn-Sample Performance:")
    for metric, value in results['in_sample_metrics'].items():
        if 'Ratio' in metric or 'Rate' in metric:
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value:.4f}")

    print("\nOut-of-Sample Performance:")
    for metric, value in results['out_sample_metrics'].items():
        if 'Ratio' in metric or 'Rate' in metric:
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value:.4f}")

    print("\nPerformance Degradation (IS vs OOS):")
    sharpe_degradation = (
            results['in_sample_metrics']['Sharpe_Ratio'] -
            results['out_sample_metrics']['Sharpe_Ratio']
    )
    print(f"Sharpe Ratio degradation: {sharpe_degradation:.3f}")

    print("\nStress Test Results:")
    for scenario, result in results['stress_results'].items():
        print(f"{scenario}: {result['total_impact']:.4f}")

    print("\nRisk Metrics:")
    print(f"Average daily turnover: {results['turnover']:.2f} DV01 units")
    print(f"Average positions: {results['avg_positions']:.1f}")
    print(f"Max concentration: {results['max_concentration']:.1f} DV01 units")

    print("\nP&L Attribution by Tenor:")
    print(results['attribution']['by_tenor'].sort_values('Total_PnL', ascending=False))

    print("\nP&L Attribution by Component:")
    print(results['attribution']['by_component'])


if __name__ == "__main__":
    ensure_directories()

    results = run_enhanced_strategy()

    print_performance_summary(results)