# EU Yield Curve PCA Trading Strategy

A Python implementation of a fixed income trading strategy based on Principal Component Analysis (PCA) of European yield curves.

## Project Overview

This project implements a relative value trading strategy for European government bond yields using PCA to identify temporary dislocations in the yield curve. The strategy:

1. Decomposes the yield curve using PCA to extract principal components (level, slope, curvature)
2. Calculates deviations between actual yields and PCA-reconstructed yields
3. Trades on the assumption that large deviations will mean-revert
4. Incorporates carry, rolldown, and regime detection to enhance the strategy

## Project Structure

```
eu_yield_pca_strategy/
├── data/
│   └── EU_yield_curves_combined.csv
├── output/
│   └── images/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── risk_measures.py   # Duration, DV01, carry and rolldown
│   ├── market_analysis.py # Regime detection and mean reversion tests
│   ├── pca.py             # PCA implementation for yield curves
│   ├── strategy.py        # Trading strategy implementation
│   ├── performance.py     # Performance metrics and stress testing
│   ├── visualization.py   # Plotting functions
├── main.py                # Main entry point
├── config.py              # Configuration parameters
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Requirements

* Python 3.8+
* Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the dependencies:
```
pip install -r requirements.txt
```

3. Place your yield curve data in the `data/` directory

## Usage

Run the strategy with:

```
python main.py
```

This will:
1. Load the yield curve data
2. Calculate risk measures (duration, DV01)
3. Detect market regimes
4. Perform PCA analysis
5. Simulate the trading strategy
6. Calculate performance metrics
7. Perform stress tests
8. Generate visualizations in the `output/images/` directory

## Data Format

The input data should be a CSV file with the following structure:
- Column 'DATE': Date in a format parseable by pandas
- Columns 'EU_1Y', 'EU_5Y', 'EU_10Y', 'EU_20Y', 'EU_30Y': Yield values for each tenor

## Features

- **Regime Detection**: Identifies different market regimes like bull steepening, bear flattening, etc.
- **Mean Reversion Testing**: Tests for stationarity and estimates half-life of mean reversion
- **PCA-based Yield Curve Modeling**: Decomposes the yield curve into principal components
- **Enhanced Strategy**: Incorporates carry, rolldown, and trade sizing based on confidence scores
- **Performance Analysis**: Calculates key metrics including Sharpe ratio, drawdowns, win rate
- **Stress Testing**: Simulates yield curve shocks to assess portfolio resilience
- **P&L Attribution**: Breaks down performance by tenor and P&L component

## License

[MIT License](LICENSE)