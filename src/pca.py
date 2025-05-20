import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import TRAINING_WINDOW


def perform_yield_pca(
        yield_df: pd.DataFrame,
        regime_data: pd.DataFrame = None,
        n_components: int = 3,
        window: int = TRAINING_WINDOW,
):
    """
    Perform PCA analysis on yield curve data.
    """
    pca_results = {}

    if regime_data is not None and 'Regime' in regime_data.columns:
        # Only proceed with regime-specific PCA if the Regime column exists
        regimes = regime_data['Regime'].unique()

        for regime in regimes:
            regime_idx = regime_data[regime_data['Regime'] == regime].index
            if len(regime_idx) > window // 2:
                regime_yield_data = yield_df.loc[regime_idx]

                pca_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components)),
                ])

                components = pca_pipeline.fit_transform(regime_yield_data)

                pca_df = pd.DataFrame(components,
                                      index=regime_yield_data.index,
                                      columns=[f'PC{i + 1}' for i in range(n_components)]
                                      )

                pca_model = pca_pipeline.named_steps['pca']
                explained_variance = pca_model.explained_variance_ratio_

                pca_results[regime] = {
                    'pca_df': pca_df,
                    'explained_variance': explained_variance,
                    'pca_model': pca_model,
                    'pca_pipeline': pca_pipeline,
                }

    # Always perform general PCA regardless of regime data
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
    ])

    components = pca_pipeline.fit_transform(yield_df)

    pca_df = pd.DataFrame(components,
                          index=yield_df.index,
                          columns=[f'PC{i + 1}' for i in range(n_components)]
                          )

    pca_model = pca_pipeline.named_steps['pca']
    explained_variance = pca_model.explained_variance_ratio_

    # Store general model
    pca_results['General'] = {
        'pca_df': pca_df,
        'explained_variance': explained_variance,
        'pca_model': pca_model,
        'pca_pipeline': pca_pipeline,
    }

    return pca_results


def get_appropriate_pca_model(date, pca_results, regime_data: pd.DataFrame = None):
    """
    Select the appropriate PCA model based on the current market regime.
    """
    if regime_data is not None:
        try:
            current_regime = regime_data.loc[date, 'Regime']
            if current_regime in pca_results:
                return pca_results[current_regime]
        except (KeyError, TypeError):
            pass

    return pca_results['General']


def reconstruct_yield_curve(
        date,
        yield_data,
        pca_results,
        regime_data: pd.DataFrame = None,
):
    """
    Reconstruct yield curve using PCA model.
    """
    actual_yield = yield_data.loc[date]

    model_info = get_appropriate_pca_model(date, pca_results, regime_data)
    pca_pipeline = model_info['pca_pipeline']

    actual_yield_df = pd.DataFrame([actual_yield.values], columns = yield_data.columns)

    transformed = pca_pipeline.transform([actual_yield_df])
    reconstructed = pca_pipeline.inverse_transform(transformed)

    return pd.Series(reconstructed[0], index=yield_data.columns)


def calculate_deviations(
        yield_data,
        pca_results,
        regime_data: pd.DataFrame = None,
        window: int = 252
):
    """
    Calculate deviations between actual and reconstructed yield curves.
    """
    reconstructed_yields = pd.DataFrame(index=yield_data.index, columns=yield_data.columns)

    for date in yield_data.index:
        try:
            reconstructed_yields.loc[date] = reconstruct_yield_curve(
                date, yield_data, pca_results, regime_data
            )
        except:
            continue

    raw_deviations = yield_data - reconstructed_yields

    tenor_vols = raw_deviations.rolling(window=window).std()

    z_scores = raw_deviations / tenor_vols

    return {
        'reconstructed_yield': reconstructed_yields,
        'raw_deviations': raw_deviations,
        'tenor_vols': tenor_vols,
        'z_scores': z_scores,
    }