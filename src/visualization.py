import numpy as np
import matplotlib.pyplot as plt

from config import TENORS


def visualize_strategy_results(results, output_dir='output/images/'):
    """
    Create visualizations for the strategy results.
    """
    # PnL Visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(results['total_pnl']['Cumulative_PnL'])
    plt.title('Cumulative P&L')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(results['total_pnl']['MTM_PnL'].rolling(21).sum(), label='MTM P&L')
    plt.plot(results['total_pnl']['Carry_Benefit'].rolling(21).sum(), label='Carry')
    plt.plot(results['total_pnl']['Transaction_Costs'].rolling(21).sum(), label='Transaction Costs')
    plt.plot(results['total_pnl']['Financing_Costs'].rolling(21).sum(), label='Financing')
    plt.title('21-Day Rolling P&L Components')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    # Calculate drawdown
    cum_returns = results['total_pnl']['Total_PnL'].cumsum()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max * 100  # in percentage
    plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    plt.title('Drawdown (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}enhanced_strategy_pnl.png')

    visualize_positions_and_deviations(results, output_dir)

    visualize_regimes(results, output_dir)

    return "Visualizations saved to images directory."


def visualize_positions_and_deviations(results, output_dir='output/images/'):
    """
    Visualize positions and deviations for each tenor.
    """
    from config import TENORS

    plt.figure(figsize=(15, 12))

    for i, tenor in enumerate(TENORS):
        col = f'EU_{tenor}Y'
        plt.subplot(len(TENORS), 1, i + 1)

        # Plot z-score
        plt.plot(results['deviation_results']['z_scores'][col], label='Z-score', color='blue', alpha=0.5)

        # Plot position
        ax2 = plt.twinx()
        ax2.plot(results['strategy_results']['positions'][col], label='Position', color='green')

        plt.title(f'{tenor}Y Deviation and Position')
        plt.grid(True)
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}enhanced_deviations_positions.png')


def visualize_regimes(results, output_dir='output/images/'):
    """
    Visualize market regimes and yield curves.
    """
    plt.figure(figsize=(15, 8))

    # Create a numeric mapping for regimes
    regime_map = {regime: i for i, regime in enumerate(results['regime_data']['Regime'].unique())}
    numeric_regime = results['regime_data']['Regime'].map(regime_map)

    plt.subplot(2, 1, 1)
    plt.plot(results['yield_data']['EU_10Y'], label='10Y Yield')
    plt.title('10Y Yield and Regimes')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    cmap = plt.cm.get_cmap('tab10', len(regime_map))
    for regime, i in regime_map.items():
        mask = results['regime_data']['Regime'] == regime
        plt.scatter(
            results['regime_data'].index[mask],
            [i] * mask.sum(),
            label=regime,
            color=cmap(i),
            s=10
        )
    plt.yticks(list(regime_map.values()), list(regime_map.keys()))
    plt.title('Market Regimes')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}enhanced_regimes.png')


def visualize_pca_components(pca_results, output_dir='output/images/'):
    """
    Visualize PCA components and their loadings.
    """
    pca_model = pca_results['General']['pca_model']
    explained_variance = pca_results['General']['explained_variance']

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    from config import TENOR_COLS

    for i in range(len(explained_variance)):
        plt.plot(TENOR_COLS, pca_model.components_[i], marker='o', label=f'PC{i + 1}')

    plt.xlabel('Tenor')
    plt.ylabel('Loading')
    plt.title('PCA Component Loadings')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}pca_components.png')


def visualize_stress_test_results(stress_results, output_dir='output/images/'):
    """
    Visualize stress test results.
    """
    plt.figure(figsize=(12, 8))

    scenarios = list(stress_results.keys())
    impacts = [result['total_impact'] for result in stress_results.values()]

    sorted_indices = np.argsort(np.abs(impacts))[::-1]
    scenarios = [scenarios[i] for i in sorted_indices]
    impacts = [impacts[i] for i in sorted_indices]

    plt.barh(scenarios, impacts)
    plt.xlabel('P&L Impact')
    plt.title('Stress Test Results')

    plt.grid(True, linestyle='--', alpha=0.7)

    for i, impact in enumerate(impacts):
        plt.text(impact + np.sign(impact) * 0.1, i, f'{impact:.2f}',
                 va='center', ha='left' if impact > 0 else 'right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}stress_test_results.png')


def visualize_pca_reconstruction(results, output_dir='output/images/'):

    actual = results['yield_data']
    reconstructed = results['deviation_results']['reconstructed_yield']

    sample_dates = actual.index[::len(actual) // 5][:5]

    plt.figure(figsize=(15, 10))

    for i, date in enumerate(sample_dates):
        plt.subplot(3, 2, i + 1)

        tenors = TENORS
        actual_yields = [actual.loc[date, f'EU_{t}Y'] for t in tenors]
        recon_yields = [reconstructed.loc[date, f'EU_{t}Y'] for t in tenors]

        plt.plot(tenors, actual_yields, 'bo--', label='Actual')
        plt.plot(tenors, recon_yields, 'r^--', label='Reconstructed')

        plt.xlabel('Tenor')
        plt.ylabel('Yield')
        plt.title(f'Yield Curve on {date.strftime("%Y-%m,-%d")}')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}pca_reconstruction.png')
    plt.close()


def visualize_regime_performance(results, output_dir='output/images/'):
    """
    Visualize strategy performance by regime.
    """
    pnl = results['total_pnl']['Total_PnL']
    regimes = results['regime_data']['Regime']

    common_dates = pnl.index.intersection(regimes.index)
    pnl_aligned = pnl.loc[common_dates]
    regimes_aligned = regimes.loc[common_dates]

    regime_stats = {}
    for regime in regimes_aligned.unique():
        regime_pnl = pnl_aligned[regimes_aligned == regime]
        if len(regime_pnl) > 0:
            regime_stats[regime] = {
                'count': len(regime_pnl),
                'total_pnl': regime_pnl.sum(),
                'avg_daily': regime_pnl.mean(),
                'volatility': regime_pnl.std(),
                'sharpe': regime_pnl.mean() / regime_pnl.std() * np.sqrt(252) if regime_pnl.std() > 0 else 0
            }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    regimes_list = list(regime_stats.keys())
    total_pnls = [regime_stats[r]['total_pnl'] for r in regimes_list]
    colors = ['green' if p > 0 else 'red' for p in total_pnls]
    ax.bar(regimes_list, total_pnls, color=colors)
    ax.set_title('Total P&L by Regime')
    ax.set_ylabel('P&L ($)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax = axes[0, 1]
    sharpes = [regime_stats[r]['sharpe'] for r in regimes_list]
    ax.bar(regimes_list, sharpes)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Sharpe Ratio by Regime')
    ax.set_ylabel('Sharpe Ratio')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax = axes[1, 0]
    counts = [regime_stats[r]['count'] for r in regimes_list]
    ax.pie(counts, labels=regimes_list, autopct='%1.1f%%')
    ax.set_title('Time Spent in Each Regime')

    ax = axes[1, 1]
    avg_returns = [regime_stats[r]['avg_daily'] for r in regimes_list]
    volatilities = [regime_stats[r]['volatility'] for r in regimes_list]
    ax.scatter(volatilities, avg_returns, s=100)
    for i, regime in enumerate(regimes_list):
        ax.annotate(regime, (volatilities[i], avg_returns[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Daily Volatility')
    ax.set_ylabel('Average Daily Return')
    ax.set_title('Risk-Return by Regime')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}regime_performance.png')
    plt.close()