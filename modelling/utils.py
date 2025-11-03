"""
Utility Functions for Modelling Module

This module provides helper functions for model validation, ensemble creation,
and recursive least squares (RLS) operations.

Key Functions:
--------------
- apply_rls_on_holdout: Apply RLS with warmup period before holdout testing
- validate_rls_data_splits: Validate that RLS data splits are correct and non-overlapping
- safe_mape: Calculate MAPE safely with automatic protection against small values
- build_weighted_ensemble_model: Build weighted ensemble by averaging coefficients
- create_ensemble_model_from_results: Create ensemble models from CV results

Constants:
----------
- DEFAULT_RLS_LAMBDA_GRID: Default grid of RLS forgetting factors
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models import RecursiveLeastSquares


# ═══════════════════════════════════════════════════════════════════════════
# RLS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def apply_rls_on_holdout(trained_model, X_train, y_train,
                         X_warmup, y_warmup,
                         X_holdout, y_holdout,
                         feature_names, forgetting_factor,
                         nonnegative_features=None, nonpositive_features=None,
                         initial_beta=None, initial_intercept=None):
    """
    Apply RLS with warmup period before holdout testing

    Parameters:
    -----------
    trained_model : model object or None
        If provided, extract initial betas from this model
    initial_beta : array-like, optional
        Initial beta coefficients (overrides trained_model if provided)
    initial_intercept : float, optional
        Initial intercept (overrides trained_model if provided)
    """
    # Create RLS model
    rls_model = RecursiveLeastSquares(
        forgetting_factor=forgetting_factor,
        initial_P=1.0,  # REDUCED for stability with pre-trained models
        non_negative_features=nonnegative_features,
        non_positive_features=nonpositive_features
    )

    # Get initial beta: use provided params, or extract from trained model
    if initial_beta is None or initial_intercept is None:
        if trained_model is not None:
            if hasattr(trained_model, 'coef_'):
                initial_beta = trained_model.coef_
                initial_intercept = trained_model.intercept_ if hasattr(trained_model, 'intercept_') else 0.0
            elif hasattr(trained_model, 'W'):
                initial_beta = trained_model.W
                initial_intercept = trained_model.b if hasattr(trained_model, 'b') else 0.0

    # Initialize RLS with betas from week 44 model (or ensemble betas)
    rls_model.fit(X_train, y_train, feature_names=feature_names,
                  initial_beta=initial_beta, initial_intercept=initial_intercept)

    # NOTE: P matrix is already initialized in fit() method, no need to reinitialize

    # Store initial state (after training on 1-44, before warmup)
    initial_beta_state = rls_model.beta.copy()

    # ═══════════════════════════════════════════════════════════════
    # WARMUP PHASE: Predict AND update on weeks 45-48
    # ═══════════════════════════════════════════════════════════════
    warmup_predictions = []
    warmup_beta_snapshots = [initial_beta_state.copy()]  # Start with week 44 betas

    if len(X_warmup) > 0:
        for i in range(len(X_warmup)):
            # Predict BEFORE update
            y_pred_warmup = rls_model.predict(X_warmup[i:i+1])[0]
            warmup_predictions.append(y_pred_warmup)

            # Update with actual
            rls_model.update(X_warmup[i:i+1], y_warmup[i:i+1])

            # Store beta after this warmup update
            warmup_beta_snapshots.append(rls_model.beta.copy())

    # Store warmed-up state (after week 48, before testing on 49-52)
    warmed_beta_state = rls_model.beta.copy()

    # ═══════════════════════════════════════════════════════════════
    # HOLDOUT PHASE: Predict with FROZEN betas (NO updates) on weeks 49-52
    # ═══════════════════════════════════════════════════════════════
    holdout_predictions = []
    holdout_beta_snapshots = [warmed_beta_state.copy()]  # Start with warmed-up betas

    # CRITICAL VERIFICATION: Store beta before holdout to ensure no changes
    beta_before_holdout = rls_model.beta.copy()

    for i in range(len(X_holdout)):
        # Predict with frozen betas
        y_pred = rls_model.predict(X_holdout[i:i+1])[0]
        holdout_predictions.append(y_pred)

        # NO UPDATE during holdout - betas must remain frozen
        # This is critical for proper out-of-sample testing

        # Verify betas haven't changed (safety check)
        if not np.allclose(rls_model.beta, beta_before_holdout):
            raise ValueError("CRITICAL ERROR: Beta coefficients changed during holdout period!")

        # Store beta (same as warmed state, verified no change)
        holdout_beta_snapshots.append(rls_model.beta.copy())

    # Calculate metrics on holdout period only
    actuals_holdout = y_holdout
    preds_holdout = np.array(holdout_predictions)

    # Calculate metrics on warmup period (for tracking)
    actuals_warmup = y_warmup if len(warmup_predictions) > 0 else np.array([])
    preds_warmup = np.array(warmup_predictions) if len(warmup_predictions) > 0 else np.array([])

    return {
        # Holdout predictions and metrics
        'predictions': preds_holdout,
        'R2_Holdout': r2_score(actuals_holdout, preds_holdout),
        'MAE_Holdout': mean_absolute_error(actuals_holdout, preds_holdout),
        'MSE_Holdout': mean_squared_error(actuals_holdout, preds_holdout),
        'RMSE_Holdout': np.sqrt(mean_squared_error(actuals_holdout, preds_holdout)),
        'MAPE_Holdout': safe_mape(actuals_holdout, preds_holdout),

        # Warmup predictions (NEW)
        'warmup_predictions': preds_warmup,
        'warmup_actuals': actuals_warmup,

        # Beta history (ENHANCED)
        'beta_history': {
            'feature_names': ['Intercept'] + list(feature_names),
            'warmup_beta_snapshots': warmup_beta_snapshots,  # Betas during warmup
            'holdout_beta_snapshots': holdout_beta_snapshots,  # Betas during holdout
            'n_warmup': len(X_warmup),
            'n_holdout': len(X_holdout)
        },

        'final_betas': rls_model.coef_,
        'final_intercept': rls_model.intercept_
    }


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def validate_rls_data_splits(train_df, warmup_df, holdout_df, total_weeks=52):
    """
    Validate that RLS data splits are correct and non-overlapping
    """
    n_train = len(train_df)
    n_warmup = len(warmup_df)
    n_holdout = len(holdout_df)
    n_total = n_train + n_warmup + n_holdout

    # Check for expected weeks
    if total_weeks is not None and n_total != total_weeks:
        print(f"WARNING: Expected {total_weeks} weeks, got {n_total}")

    # Check for non-overlapping indices
    if not train_df.index.intersection(warmup_df.index).empty:
        raise ValueError("Data leakage: Training and warmup sets overlap!")
    if not train_df.index.intersection(holdout_df.index).empty:
        raise ValueError("Data leakage: Training and holdout sets overlap!")
    if not warmup_df.index.intersection(holdout_df.index).empty:
        raise ValueError("Data leakage: Warmup and holdout sets overlap!")

    return True

def safe_mape(y_true, y_pred):
    """
    Calculate MAPE safely with automatic protection against small values
    - Excludes values where |y_true| < 1.0 (handles count data with 0s)
    - Caps individual errors at 500% to prevent outlier explosion
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Only include values where |y_true| >= 1.0
    valid_mask = (np.abs(y_true) >= 1.0)

    if not valid_mask.any():
        return float("nan")

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Calculate percentage errors
    percent_errors = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100

    # Cap extreme percentage errors at 500% to prevent small values from dominating
    percent_errors = np.minimum(percent_errors, 500.0)

    return np.mean(percent_errors)

# Default grid of RLS forgetting factors for automatic tuning
# FIXED: Use higher values (less forgetting) for more stable updates
DEFAULT_RLS_LAMBDA_GRID = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_weighted_ensemble_model(models_df, weight_metric='MAPE Test', grouping_keys=None):
    """
    Build a weighted ensemble model by averaging coefficients across models.
    Uses MAPE-based exponential weighting (lower MAPE = higher weight).

    Parameters:
    -----------
    models_df : pd.DataFrame
        DataFrame with model results (must have Fold='Avg' rows)
    weight_metric : str
        Column name to use for weighting (default: 'MAPE Test')
    grouping_keys : list
        Keys that define unique combinations

    Returns:
    --------
    dict with ensemble coefficients and metadata
    """
    if weight_metric not in models_df.columns:
        st.warning(f"Metric '{weight_metric}' not found. Using equal weights.")
        weights = np.ones(len(models_df))
    else:
        metric_values = pd.to_numeric(models_df[weight_metric], errors='coerce')

        if metric_values.isna().all():
            weights = np.ones(len(models_df))
        else:
            # Exponential weighting: lower metric = higher weight
            best_value = metric_values.min()
            weights = np.exp(-0.5 * (metric_values - best_value))
            weights = np.nan_to_num(weights, nan=0.0)

    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum == 0 or np.isnan(weight_sum):
        weights = np.ones(len(models_df))
        weight_sum = weights.sum()

    weights = weights / weight_sum

    # Extract beta columns
    beta_cols = [c for c in models_df.columns if c.startswith('Beta_')]

    # Weighted average of betas
    ensemble_betas = {}
    for beta_col in beta_cols:
        if beta_col in models_df.columns:
            values = pd.to_numeric(models_df[beta_col], errors='coerce').fillna(0)
            ensemble_betas[beta_col] = np.average(values, weights=weights)

    # Weighted average of intercept
    ensemble_intercept = 0.0
    if 'B0 (Original)' in models_df.columns:
        b0_values = pd.to_numeric(models_df['B0 (Original)'], errors='coerce').fillna(0)
        ensemble_intercept = np.average(b0_values, weights=weights)

    # Metadata
    result = {
        'ensemble_betas': ensemble_betas,
        'ensemble_intercept': ensemble_intercept,
        'num_models': len(models_df),
        'weights': weights,
        'model_names': models_df['Model'].tolist() if 'Model' in models_df.columns else [],
        'best_model_idx': int(np.argmax(weights)),
        'weight_concentration': float(weights.max())
    }

    # Add weighted metrics
    for metric_col in ['R2 Test', 'R2 Train', 'MAPE Test', 'MAPE Train', 'MAE Test', 'MAE Train']:
        if metric_col in models_df.columns:
            metric_values = pd.to_numeric(models_df[metric_col], errors='coerce').fillna(0)
            result[f'ensemble_{metric_col.lower().replace(" ", "_")}'] = np.average(metric_values, weights=weights)

    return result


def create_ensemble_model_from_results(results_df, grouping_keys, feature_names,
                                       weight_metric='MAPE Test',
                                       filter_r2_min=None, filter_mape_max=None, filter_mae_max=None,
                                       filter_positive_features=None, filter_negative_features=None):
    """
    Create ensemble models for each unique combination from CV results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from cross-validation (must include Fold='Avg' rows)
    grouping_keys : list
        Keys that define unique combinations
    feature_names : list
        List of feature/predictor names
    weight_metric : str
        Metric to use for weighting ('MAPE Test', 'MAE Test', etc.)
    filter_r2_min : float, optional
        Minimum R² Test threshold to include a model
    filter_mape_max : float, optional
        Maximum MAPE Test threshold to include a model
    filter_mae_max : float, optional
        Maximum MAE Test threshold to include a model
    filter_positive_features : list, optional
        Features that must have positive coefficients (≥0) in models to be included
    filter_negative_features : list, optional
        Features that must have negative coefficients (≤0) in models to be included

    Returns:
    --------
    dict: {combination_key: ensemble_model_dict}
    """
    # Filter to only averaged results
    avg_results = results_df[results_df['Fold'] == 'Avg'].copy()

    if avg_results.empty:
        return {}

    # Apply optional filters
    if filter_r2_min is not None and 'R2 Test' in avg_results.columns:
        r2_values = pd.to_numeric(avg_results['R2 Test'], errors='coerce')
        avg_results = avg_results[r2_values >= filter_r2_min]

    if filter_mape_max is not None and 'MAPE Test' in avg_results.columns:
        mape_values = pd.to_numeric(avg_results['MAPE Test'], errors='coerce')
        avg_results = avg_results[mape_values <= filter_mape_max]

    if filter_mae_max is not None and 'MAE Test' in avg_results.columns:
        mae_values = pd.to_numeric(avg_results['MAE Test'], errors='coerce')
        avg_results = avg_results[mae_values <= filter_mae_max]

    # Apply sign-based filtering
    if filter_positive_features or filter_negative_features:
        initial_count = len(avg_results)
        mask = pd.Series([True] * len(avg_results), index=avg_results.index)

        # Check positive constraints
        if filter_positive_features:
            for feature in filter_positive_features:
                beta_col = f"Beta_{feature}"
                if beta_col in avg_results.columns:
                    beta_values = pd.to_numeric(avg_results[beta_col], errors='coerce')
                    # Keep only models where this coefficient is >= 0 (with small tolerance for numerical errors)
                    mask &= (beta_values >= -1e-6)

        # Check negative constraints
        if filter_negative_features:
            for feature in filter_negative_features:
                beta_col = f"Beta_{feature}"
                if beta_col in avg_results.columns:
                    beta_values = pd.to_numeric(avg_results[beta_col], errors='coerce')
                    # Keep only models where this coefficient is <= 0 (with small tolerance for numerical errors)
                    mask &= (beta_values <= 1e-6)

        avg_results = avg_results[mask]
        filtered_count = initial_count - len(avg_results)

        if filtered_count > 0:
            st.info(f"🔍 Sign filtering: Excluded {filtered_count} model(s) with incorrect coefficient signs")

    if avg_results.empty:
        st.warning("⚠️ No models passed the ensemble filters. Relax filter thresholds.")
        return {}

    # Build ensemble for each unique combination
    ensembles = {}

    if not grouping_keys:
        # Single group case
        ensemble = build_weighted_ensemble_model(avg_results, weight_metric, grouping_keys)
        ensembles['ALL'] = ensemble
    else:
        # Group by combination keys
        for combo_vals, group_df in avg_results.groupby(grouping_keys, dropna=False):
            if len(group_df) == 0:
                continue

            combo_key = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, combo_vals)]) if isinstance(combo_vals, tuple) else f"{grouping_keys[0]}={combo_vals}"
            ensemble = build_weighted_ensemble_model(group_df, weight_metric, grouping_keys)
            ensembles[combo_key] = ensemble

    return ensembles
