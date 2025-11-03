"""
Main Modeling Pipeline

This module contains the core modeling pipeline function that orchestrates the entire
cross-validation and model training process. It supports various model types including
regular models, stacked interaction models, and ensemble models with optional RLS
(Recursive Least Squares) adaptation.

Key features:
- K-fold cross-validation with adaptive fold selection
- Support for multiple model types (Ridge, Lasso, ElasticNet, etc.)
- Stacked interaction models with group-specific coefficients
- RLS adaptive modeling with warmup and holdout periods
- Weighted ensemble model creation from CV results
- Comprehensive metrics tracking (R2, MAPE, MAE, MSE, RMSE)
- Beta history tracking for RLS models
"""

import numpy as np
import pandas as pd
import streamlit as st
import time

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models import (
    CustomConstrainedRidge,
    ConstrainedLinearRegression,
    StackedInteractionModel,
    StatsMixedEffectsModel,
    RecursiveLeastSquares
)

from utils import (
    validate_rls_data_splits,
    safe_mape,
    apply_rls_on_holdout,
    DEFAULT_RLS_LAMBDA_GRID,
    create_ensemble_model_from_results
)


def run_model_pipeline(
    df,
    grouping_keys,
    X_columns,
    target_col,
    k_folds,
    std_cols,
    models_dict,
    use_stacked=False,
    stacking_keys=None,
    filter_keys_for_stacking=None,
    log_transform_y=False,
    min_y_share_pct=1.0,
    # RLS parameters:
    enable_rls=False,
    holdout_weeks=4,
    warmup_weeks=4,
    positive_constraints=None,
    negative_constraints=None,
    rls_lambda_candidates=None,
    # Ensemble parameters:
    enable_ensemble=False,
    ensemble_weight_metric='MAPE Test',
    ensemble_filter_r2_min=None,
    ensemble_filter_mape_max=None,
    ensemble_filter_mae_max=None,
    ensemble_filter_positive_features=None,
    ensemble_filter_negative_features=None
):
    """
    Run modeling pipeline
    Returns aggregated results (one row per group-model) and predictions
    """
    rows = []
    preds_records = []
    optimized_lambda_records = []

    # Separate stacked and non-stacked models
    stacked_models = {k: v for k, v in models_dict.items() if isinstance(v, StackedInteractionModel)}
    regular_models = {k: v for k, v in models_dict.items() if not isinstance(v, StackedInteractionModel)}

    # For regular models: use ALL grouping_keys
    if grouping_keys and regular_models:
        grouped_regular = df.groupby(grouping_keys)
        group_list_regular = list(grouped_regular)
    else:
        group_list_regular = [((None,), df)] if regular_models else []

    # For stacked models: use ONLY filter_keys_for_stacking
    if filter_keys_for_stacking and stacked_models:
        grouped_stacked = df.groupby(filter_keys_for_stacking)
        group_list_stacked = list(grouped_stacked)
    else:
        group_list_stacked = [((None,), df)] if stacked_models else []

    # ═════════════════════════════════════════════════════════════════════════
    # FILTER GROUPS BY Y VARIABLE SHARE (min_y_share_pct)
    # ═════════════════════════════════════════════════════════════════════════
    total_y = df[target_col].sum()

    if min_y_share_pct > 0 and total_y > 0:
        # Filter regular models groups
        filtered_regular = []
        skipped_regular = []
        for gvals, gdf in group_list_regular:
            group_y_sum = gdf[target_col].sum()
            group_y_share = (group_y_sum / total_y) * 100
            if group_y_share >= min_y_share_pct:
                filtered_regular.append((gvals, gdf))
            else:
                gvals_tuple = (gvals,) if not isinstance(gvals, tuple) else gvals
                group_name = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, gvals_tuple)]) if grouping_keys else "All"
                skipped_regular.append(f"{group_name} ({group_y_share:.2f}%)")

        # Filter stacked models groups
        filtered_stacked = []
        skipped_stacked = []
        for gvals, gdf in group_list_stacked:
            group_y_sum = gdf[target_col].sum()
            group_y_share = (group_y_sum / total_y) * 100
            if group_y_share >= min_y_share_pct:
                filtered_stacked.append((gvals, gdf))
            else:
                gvals_tuple = (gvals,) if not isinstance(gvals, tuple) else gvals
                group_name = " | ".join([f"{k}={v}" for k, v in zip(filter_keys_for_stacking if filter_keys_for_stacking else [], gvals_tuple)]) if filter_keys_for_stacking else "All"
                skipped_stacked.append(f"{group_name} ({group_y_share:.2f}%)")

        # Update group lists
        group_list_regular = filtered_regular
        group_list_stacked = filtered_stacked

        # Show info about filtered groups
        if skipped_regular:
            st.info(f"🔍 Filtered {len(skipped_regular)} regular model group(s) with <{min_y_share_pct}% Y share:\n" +
                   "\n".join([f"• {name}" for name in skipped_regular[:10]]) +
                   (f"\n• ... and {len(skipped_regular) - 10} more" if len(skipped_regular) > 10 else ""))

        if skipped_stacked:
            st.info(f"🔍 Filtered {len(skipped_stacked)} stacked model group(s) with <{min_y_share_pct}% Y share:\n" +
                   "\n".join([f"• {name}" for name in skipped_stacked[:10]]) +
                   (f"\n• ... and {len(skipped_stacked) - 10} more" if len(skipped_stacked) > 10 else ""))

    n_regular_ops = len(group_list_regular) * len(regular_models) * k_folds
    n_stacked_ops = len(group_list_stacked) * len(stacked_models) * k_folds
    total_operations = n_regular_ops + n_stacked_ops

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()
    operation_count = 0

    def update_progress(group_name, model_name, fold_num):
        nonlocal operation_count
        operation_count += 1
        progress = operation_count / total_operations if total_operations > 0 else 1
        progress_bar.progress(progress)

        elapsed_time = time.time() - start_time
        if operation_count > 0:
            avg_time_per_op = elapsed_time / operation_count
            remaining_ops = total_operations - operation_count
            estimated_remaining = avg_time_per_op * remaining_ops

            status_text.text(f"Processing: {model_name} | {group_name} | Fold {fold_num}/{k_folds} | ~{estimated_remaining:.0f}s remaining")

    # ═════════════════════════════════════════════════════════════════════════
    # PROCESS REGULAR MODELS (grouped by ALL keys)
    # ═════════════════════════════════════════════════════════════════════════
    for gvals, gdf in group_list_regular:
        gvals = (gvals,) if not isinstance(gvals, tuple) else gvals
        group_display_name = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, gvals)]) if grouping_keys else "All"

        present_cols = [c for c in X_columns if c in gdf.columns]
        if len(present_cols) < len(X_columns):
            for mname in regular_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        X_full = gdf[present_cols].fillna(0).copy()
        y_full = gdf[target_col].copy()

        n_samples = len(X_full)
        n_features = len(present_cols)

        # CRITICAL FIX: Check for overfitting conditions
        min_samples_needed = max(k_folds, n_features * 2)  # Need at least 2x features for stable modeling

        if n_samples < min_samples_needed:
            st.warning(f"⚠️ Skipping {group_display_name}: Only {n_samples} samples but need {min_samples_needed} (have {n_features} features)")
            for mname in regular_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        # ADAPTIVE K-FOLD: Reduce folds for smaller groups
        adaptive_k = k_folds
        if n_samples < 20:
            adaptive_k = 2  # Use 2-fold for very small groups
        elif n_samples < 50:
            adaptive_k = min(3, k_folds)  # Use 3-fold for small groups

        kf = KFold(n_splits=adaptive_k, shuffle=True, random_state=42)

        if adaptive_k != k_folds:
            st.info(f"ℹ️ {group_display_name}: Using {adaptive_k}-fold CV (only {n_samples} samples)")

        for mname, mdl in regular_models.items():
            fold_results = []

            # Split data into train/warmup/holdout if RLS is enabled
            if enable_rls and holdout_weeks > 0:
                gdf_sorted = gdf.sort_index()
                n_total = len(gdf_sorted)

                # NEW: Three-way split
                n_train = n_total - holdout_weeks - warmup_weeks  # weeks 1-44
                n_warmup_end = n_total - holdout_weeks             # up to week 48

                # Check if enough data for warmup approach
                min_train_needed = 10
                if n_train < min_train_needed:
                    st.warning(f"⚠️ {group_display_name}: Not enough data for {warmup_weeks}-week warmup + {holdout_weeks}-week holdout. Using full data.")
                    train_df = gdf_sorted
                    warmup_df = None
                    holdout_df = None
                    use_holdout_for_group = False
                else:
                    train_df = gdf_sorted.iloc[:n_train]                # weeks 1-44
                    warmup_df = gdf_sorted.iloc[n_train:n_warmup_end]   # weeks 45-48
                    holdout_df = gdf_sorted.iloc[n_warmup_end:]         # weeks 49-52
                    use_holdout_for_group = True

                    # Validate data splits to ensure no overlap
                    try:
                        validate_rls_data_splits(train_df, warmup_df, holdout_df)
                    except ValueError as e:
                        st.error(f"Data split validation failed: {e}")
                        use_holdout_for_group = False
            else:
                train_df = gdf
                warmup_df = None
                holdout_df = None
                use_holdout_for_group = False


            # Use TRAIN data for CV (not full data)
            X_full = train_df[present_cols].fillna(0).copy()
            y_full = train_df[target_col].copy()

            display_name = f"{mname} + RLS" if use_holdout_for_group else mname


            # Regular model processing
            for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
                update_progress(group_display_name, mname, fold_id)

                X_tr, X_te = X_full.iloc[tr_idx].copy(), X_full.iloc[te_idx].copy()
                y_tr, y_te = y_full.iloc[tr_idx].copy(), y_full.iloc[te_idx].copy()

                # Store original y values for metrics calculation
                y_tr_original = y_tr.copy()
                y_te_original = y_te.copy()

                # Apply log transformation if requested
                if log_transform_y:
                    y_tr = np.log1p(y_tr)  # log(1 + y)
                    y_te = np.log1p(y_te)

                # Standardization
                scaler = {}
                # Only check if model itself is RLS (not wrapper)
                if isinstance(mdl, RecursiveLeastSquares):
                    cols_to_scale = list(X_tr.columns)
                elif std_cols:
                    cols_to_scale = [c for c in std_cols if c in X_tr.columns]
                else:
                    cols_to_scale = []

                if cols_to_scale:
                    sc = StandardScaler().fit(X_tr[cols_to_scale])
                    X_tr[cols_to_scale] = sc.transform(X_tr[cols_to_scale])
                    X_te[cols_to_scale] = sc.transform(X_te[cols_to_scale])
                    scaler = {c: (m, s) for c, m, s in zip(cols_to_scale, sc.mean_, sc.scale_)}

                # Train model (no RLS wrapper handling)
                model_copy = clone(mdl)

                # Fit based on model type
                if isinstance(model_copy, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                    model_copy.fit(X_tr.values, y_tr.values, X_tr.columns.tolist())
                    y_tr_pred = model_copy.predict(X_tr.values)
                    y_te_pred = model_copy.predict(X_te.values)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_
                elif isinstance(model_copy, StatsMixedEffectsModel):
                    tr_orig_idx = X_tr.index
                    te_orig_idx = X_te.index

                    grp_col = model_copy.group_col

                    if '_' in grp_col and grp_col not in gdf.columns:
                        component_keys = grp_col.split('_')
                        if all(k in gdf.columns for k in component_keys):
                            groups_tr_values = gdf.loc[tr_orig_idx, component_keys].astype(str).apply(
                                lambda row: "_".join(row), axis=1
                            )
                            groups_te_values = gdf.loc[te_orig_idx, component_keys].astype(str).apply(
                                lambda row: "_".join(row), axis=1
                            )
                        else:
                            groups_tr_values = gdf.loc[tr_orig_idx, grouping_keys[0]]
                            groups_te_values = gdf.loc[te_orig_idx, grouping_keys[0]]
                    else:
                        if grp_col in gdf.columns:
                            groups_tr_values = gdf.loc[tr_orig_idx, grp_col]
                            groups_te_values = gdf.loc[te_orig_idx, grp_col]
                        else:
                            groups_tr_values = gdf.loc[tr_orig_idx, grouping_keys[0]]
                            groups_te_values = gdf.loc[te_orig_idx, grouping_keys[0]]

                    model_copy.fit(X_tr, y_tr, groups_tr_values)
                    y_tr_pred = model_copy.predict(X_tr, groups_tr_values)
                    y_te_pred = model_copy.predict(X_te, groups_te_values)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_
                else:
                    model_copy.fit(X_tr, y_tr)
                    y_tr_pred = model_copy.predict(X_tr)
                    y_te_pred = model_copy.predict(X_te)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_


                # Reverse transform predictions if log was applied
                if log_transform_y:
                    y_tr_pred = np.expm1(y_tr_pred)  # exp(pred) - 1
                    y_te_pred = np.expm1(y_te_pred)
                    # Ensure non-negative predictions
                    y_tr_pred = np.maximum(y_tr_pred, 0)
                    y_te_pred = np.maximum(y_te_pred, 0)

                # Metrics (calculated on original scale)
                r2_tr = r2_score(y_tr_original, y_tr_pred)
                r2_te = r2_score(y_te_original, y_te_pred)
                mape_tr = safe_mape(y_tr_original, y_tr_pred)
                mape_te = safe_mape(y_te_original, y_te_pred)
                mae_tr = np.mean(np.abs(y_tr_original - y_tr_pred))
                mae_te = np.mean(np.abs(y_te_original - y_te_pred))
                mse_tr = np.mean((y_tr_original - y_tr_pred)**2)
                mse_te = np.mean((y_te_original - y_te_pred)**2)
                rmse_tr = np.sqrt(mse_tr)
                rmse_te = np.sqrt(mse_te)

                # Reverse standardization
                raw_int, raw_coefs = B0_std, B1_std.copy()
                for i, col in enumerate(present_cols):
                    if col in scaler:
                        mu, sd = scaler[col]
                        raw_coefs[i] = raw_coefs[i] / sd
                        raw_int -= raw_coefs[i] * mu

                # FIX: Mean X - calculate ONLY on TRAINING data to avoid leakage
                mean_x = X_tr.mean(numeric_only=True).to_dict()

                # Create fold result
                d = {k: v for k, v in zip(grouping_keys, gvals)}
                d.update({
                    "Model": display_name,
                    "Fold": fold_id,
                    "B0 (Original)": raw_int,
                    "R2 Train": r2_tr,
                    "R2 Test": r2_te,
                    "MAPE Train": mape_tr,
                    "MAPE Test": mape_te,
                    "MAE Train": mae_tr,
                    "MAE Test": mae_te,
                    "MSE Train": mse_tr,
                    "MSE Test": mse_te,
                    "RMSE Train": rmse_tr,
                    "RMSE Test": rmse_te,
                })

                # Add mean X
                for c, v in mean_x.items():
                    d[c] = v

                # Add betas
                for i, c in enumerate(present_cols):
                    d[f"Beta_{c}"] = raw_coefs[i]

                fold_results.append(d)

                # Predictions
                pr = gdf.loc[X_te.index].copy()
                pr["Actual"] = y_te.values
                pr["Predicted"] = y_te_pred
                pr["Model"] = display_name
                pr["Fold"] = fold_id
                preds_records.append(pr)

            # Report fold results: individual folds + aggregated
            if fold_results:
                fold_df = pd.DataFrame(fold_results)

                # Add individual fold rows
                rows.append(fold_df)

                # Create aggregated row (average across folds)
                # Identify keys
                key_cols = [col for col in fold_df.columns if col in grouping_keys + list(getattr(mdl, 'group_keys', [])) or col == 'Model']

                # Numeric columns to average
                numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'Fold']

                # String columns to take first
                string_cols = [col for col in fold_df.columns if col not in numeric_cols and col not in key_cols and col != 'Fold']

                # Aggregate
                agg_dict = {}
                for col in numeric_cols:
                    agg_dict[col] = 'mean'
                for col in string_cols:
                    agg_dict[col] = 'first'

                aggregated = fold_df.groupby(key_cols).agg(agg_dict).reset_index()
                aggregated['Fold'] = 'Avg'  # Mark as average row
                rows.append(aggregated)

            # ═══════════════════════════════════════════════════════════════
            # RLS HOLDOUT LOGIC: Train on 1-44, warmup 45-48, test 49-52
            # CRITICAL: Skip per-model RLS when ensemble is enabled
            # When ensemble is enabled, RLS runs ONLY on the ensemble (after CV)
            # ═══════════════════════════════════════════════════════════════
            if use_holdout_for_group and holdout_df is not None and not enable_ensemble:
                # Train model on weeks 1-44 ONLY (for RLS initialization)
                X_train_final = train_df[present_cols].fillna(0).copy()
                y_train_final = train_df[target_col].copy()
                y_train_final_original = y_train_final.copy()

                # Log data split sizes for verification
                debug_msg = f"RLS Data Split for {display_name}: Train={len(train_df)}, Warmup={len(warmup_df)}, Holdout={len(holdout_df)}"
                # st.caption(debug_msg)  # Uncomment for debugging

                if log_transform_y:
                    y_train_final = np.log1p(y_train_final)

                # Standardize
                scaler_final = {}
                if isinstance(mdl, RecursiveLeastSquares):
                    cols_to_scale_final = list(X_train_final.columns)
                elif std_cols:
                    cols_to_scale_final = [c for c in std_cols if c in X_train_final.columns]
                else:
                    cols_to_scale_final = []

                if cols_to_scale_final:
                    sc_final = StandardScaler().fit(X_train_final[cols_to_scale_final])
                    X_train_final[cols_to_scale_final] = sc_final.transform(X_train_final[cols_to_scale_final])
                    scaler_final = {c: (m, s) for c, m, s in zip(cols_to_scale_final, sc_final.mean_, sc_final.scale_)}

                # Train model on weeks 1-44 for RLS initialization
                rls_init_model = clone(mdl)

                if isinstance(rls_init_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                    rls_init_model.fit(X_train_final.values, y_train_final.values, X_train_final.columns.tolist())
                elif isinstance(rls_init_model, StatsMixedEffectsModel):
                    grp_col = rls_init_model.group_col
                    if grp_col in gdf.columns:
                        groups_values = train_df[grp_col]
                    else:
                        groups_values = train_df[grouping_keys[0]]
                    rls_init_model.fit(X_train_final, y_train_final, groups_values)
                else:
                    rls_init_model.fit(X_train_final, y_train_final)

                # ═══════════════════════════════════════════════════════════════
                # Prepare WARMUP data (weeks 45-48)
                # ═══════════════════════════════════════════════════════════════
                X_warmup = warmup_df[present_cols].fillna(0).copy()
                y_warmup = warmup_df[target_col].copy()
                y_warmup_original = y_warmup.copy()

                if log_transform_y:
                    y_warmup = np.log1p(y_warmup)

                if cols_to_scale_final:
                    X_warmup[cols_to_scale_final] = sc_final.transform(X_warmup[cols_to_scale_final])

                # ═══════════════════════════════════════════════════════════════
                # Prepare HOLDOUT data (weeks 49-52)
                # ═══════════════════════════════════════════════════════════════
                X_holdout = holdout_df[present_cols].fillna(0).copy()
                y_holdout = holdout_df[target_col].copy()
                y_holdout_original = y_holdout.copy()

                if log_transform_y:
                    y_holdout = np.log1p(y_holdout)

                if cols_to_scale_final:
                    X_holdout[cols_to_scale_final] = sc_final.transform(X_holdout[cols_to_scale_final])

                # ═══════════════════════════════════════════════════════════════
                # Apply RLS with warmup using lambda grid search
                # ═══════════════════════════════════════════════════════════════
                lambda_grid = rls_lambda_candidates if rls_lambda_candidates is not None else DEFAULT_RLS_LAMBDA_GRID
                best_lambda = None
                best_holdout_preds = None
                best_rls_results = None
                best_warmup_mae = None

                for candidate_lambda in lambda_grid:
                    candidate_results = apply_rls_on_holdout(
                        trained_model=rls_init_model,
                        X_train=X_train_final.values,
                        y_train=y_train_final.values,
                        X_warmup=X_warmup.values,
                        y_warmup=y_warmup.values,
                        X_holdout=X_holdout.values,
                        y_holdout=y_holdout.values,
                        feature_names=present_cols,
                        forgetting_factor=candidate_lambda,
                        nonnegative_features=positive_constraints,
                        nonpositive_features=negative_constraints
                    )

                    # CRITICAL FIX: Use WARMUP predictions for lambda selection to avoid data leakage
                    warmup_preds = candidate_results.get('warmup_predictions', np.array([]))

                    if len(warmup_preds) > 0:
                        if log_transform_y:
                            warmup_preds_transformed = np.expm1(warmup_preds)
                            warmup_preds_transformed = np.maximum(warmup_preds_transformed, 0)
                        else:
                            warmup_preds_transformed = warmup_preds

                        # Calculate MAE on warmup period for hyperparameter selection
                        candidate_warmup_mae = mean_absolute_error(y_warmup_original, warmup_preds_transformed)
                    else:
                        candidate_warmup_mae = float('inf')

                    # Still get holdout predictions for final evaluation
                    candidate_preds = candidate_results['predictions']
                    if log_transform_y:
                        candidate_preds = np.expm1(candidate_preds)
                        candidate_preds = np.maximum(candidate_preds, 0)

                    if np.isnan(candidate_warmup_mae):
                        continue

                    if best_warmup_mae is None or candidate_warmup_mae < best_warmup_mae:
                        best_warmup_mae = candidate_warmup_mae
                        best_lambda = candidate_lambda
                        best_holdout_preds = candidate_preds
                        best_rls_results = candidate_results

                if best_rls_results is None:
                    fallback_lambda = lambda_grid[0]
                    best_lambda = fallback_lambda
                    best_rls_results = apply_rls_on_holdout(
                        trained_model=rls_init_model,
                        X_train=X_train_final.values,
                        y_train=y_train_final.values,
                        X_warmup=X_warmup.values,
                        y_warmup=y_warmup.values,
                        X_holdout=X_holdout.values,
                        y_holdout=y_holdout.values,
                        feature_names=present_cols,
                        forgetting_factor=fallback_lambda,
                        nonnegative_features=positive_constraints,
                        nonpositive_features=negative_constraints
                    )
                    best_holdout_preds = best_rls_results['predictions']
                    if log_transform_y:
                        best_holdout_preds = np.expm1(best_holdout_preds)
                        best_holdout_preds = np.maximum(best_holdout_preds, 0)

                rls_results = best_rls_results
                holdout_preds = best_holdout_preds

                # Get RLS metrics (retain MAE for comparisons, keep MAPE for result table)
                r2_holdout = r2_score(y_holdout_original, holdout_preds)
                mae_holdout = mean_absolute_error(y_holdout_original, holdout_preds)
                mse_holdout = mean_squared_error(y_holdout_original, holdout_preds)
                rmse_holdout = np.sqrt(mse_holdout)
                mape_holdout = safe_mape(y_holdout_original, holdout_preds)

                lambda_record = {k: v for k, v in zip(grouping_keys, gvals)}
                lambda_record.update({
                    "Model": display_name,
                    "Best Lambda": best_lambda,
                    "Holdout MAE": mae_holdout
                })
                optimized_lambda_records.append(lambda_record)

                # ═══════════════════════════════════════════════════════════════
                # STATIC BASELINE: Train on weeks 1-48, predict 49-52
                # ═══════════════════════════════════════════════════════════════
                # Combine train + warmup for static model (weeks 1-48)
                train_warmup_df = pd.concat([train_df, warmup_df], axis=0)
                X_train_static = train_warmup_df[present_cols].fillna(0).copy()
                y_train_static = train_warmup_df[target_col].copy()

                if log_transform_y:
                    y_train_static = np.log1p(y_train_static)

                # Use same scaler (fitted on 1-44, but apply to 1-48)
                if cols_to_scale_final:
                    X_train_static[cols_to_scale_final] = sc_final.transform(X_train_static[cols_to_scale_final])

                # Train static model on full 48 weeks
                static_model = clone(mdl)

                if isinstance(static_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                    static_model.fit(X_train_static.values, y_train_static.values, X_train_static.columns.tolist())
                elif isinstance(static_model, StatsMixedEffectsModel):
                    grp_col = static_model.group_col
                    if grp_col in gdf.columns:
                        groups_static = train_warmup_df[grp_col]
                    else:
                        groups_static = train_warmup_df[grouping_keys[0]]
                    static_model.fit(X_train_static, y_train_static, groups_static)
                else:
                    static_model.fit(X_train_static, y_train_static)

                # Predict with frozen betas
                baseline_static_preds = static_model.predict(
                    X_holdout.values if hasattr(X_holdout, 'values') else X_holdout
                )

                if log_transform_y:
                    baseline_static_preds = np.expm1(baseline_static_preds)
                    baseline_static_preds = np.maximum(baseline_static_preds, 0)

                # Calculate static metrics
                r2_static = r2_score(y_holdout_original, baseline_static_preds)
                mae_static = mean_absolute_error(y_holdout_original, baseline_static_preds)
                rmse_static = np.sqrt(mean_squared_error(y_holdout_original, baseline_static_preds))
                mape_static = safe_mape(y_holdout_original, baseline_static_preds)

                # ═══════════════════════════════════════════════════════════════
                # STORE WARMUP PREDICTIONS (NEW)
                # ═══════════════════════════════════════════════════════════════
                warmup_preds = rls_results.get('warmup_predictions', np.array([]))
                warmup_actuals = rls_results.get('warmup_actuals', np.array([]))

                if len(warmup_preds) > 0:
                    # Reverse log transform on warmup predictions
                    if log_transform_y:
                        warmup_preds = np.expm1(warmup_preds)
                        warmup_preds = np.maximum(warmup_preds, 0)

                    # Store warmup predictions
                    pr_warmup = warmup_df.copy()
                    pr_warmup["Actual"] = warmup_actuals if not log_transform_y else np.expm1(warmup_actuals)
                    pr_warmup["Predicted"] = warmup_preds
                    pr_warmup["Model"] = display_name
                    pr_warmup["Fold"] = "Warmup"
                    preds_records.append(pr_warmup)

                # ═══════════════════════════════════════════════════════════════
                # STORE FOR COMPARISON VISUALIZATION
                # ═══════════════════════════════════════════════════════════════
                if 'rls_comparison_store' not in st.session_state:
                    st.session_state.rls_comparison_store = []

                st.session_state.rls_comparison_store.append({
                    'Group': group_display_name,
                    'Model': mname,
                    'Dates': [f"Week {i+1}" for i in range(len(y_holdout_original))],
                    'Actuals': y_holdout_original.values,
                    'Predictions_Static': baseline_static_preds,
                    'Predictions_RLS': holdout_preds,
                    'R2_Static': r2_static,
                    'R2_RLS': r2_holdout,
                    'MAE_Static': mae_static,
                    'MAE_RLS': mae_holdout,
                    'RMSE_Static': rmse_static,
                    'RMSE_RLS': rmse_holdout
                })

                # Store beta history
                if 'beta_history' in rls_results:
                    group_key = f"{group_display_name} | {mname} + RLS"

                    if 'beta_history_store' not in st.session_state:
                        st.session_state.beta_history_store = {}

                    st.session_state.beta_history_store[group_key] = rls_results['beta_history']

                # Create holdout row
                if fold_results:
                    cv_avg = aggregated[aggregated['Fold'] == 'Avg'].iloc[0].to_dict() if len(aggregated) > 0 else {}

                    d = {k: v for k, v in zip(grouping_keys, gvals)}
                    d.update({
                        "Model": display_name,
                        "Fold": "Holdout",
                        "B0 (Original)": cv_avg.get("B0 (Original)", np.nan),
                        "R2 Train": cv_avg.get("R2 Train", np.nan),
                        "R2 Test": cv_avg.get("R2 Test", np.nan),
                        "R2 Holdout": r2_holdout,
                        "MAPE Train": cv_avg.get("MAPE Train", np.nan),
                        "MAPE Test": cv_avg.get("MAPE Test", np.nan),
                        "MAPE Holdout": mape_holdout,
                        "MAE Train": cv_avg.get("MAE Train", np.nan),
                        "MAE Test": cv_avg.get("MAE Test", np.nan),
                        "MAE Holdout": mae_holdout,
                        "MSE Train": cv_avg.get("MSE Train", np.nan),
                        "MSE Test": cv_avg.get("MSE Test", np.nan),
                        "MSE Holdout": mse_holdout,
                        "RMSE Train": cv_avg.get("RMSE Train", np.nan),
                        "RMSE Test": cv_avg.get("RMSE Test", np.nan),
                        "RMSE Holdout": rmse_holdout,
                    })

                    # Add mean X from training
                    mean_x = train_df[present_cols].mean(numeric_only=True).to_dict()
                    for c, v in mean_x.items():
                        d[c] = v

                    # Add final betas (FIXED - use holdout_beta_snapshots)
                    if 'beta_history' in rls_results:
                        holdout_snapshots = rls_results['beta_history'].get('holdout_beta_snapshots', [])
                        if len(holdout_snapshots) > 0:
                            final_beta = holdout_snapshots[-1]

                            for i, col in enumerate(present_cols):
                                beta_val = final_beta[i+1] if i+1 < len(final_beta) else 0
                                if col in scaler_final:
                                    mu, sd = scaler_final[col]
                                    beta_val = beta_val / sd
                                d[f"Beta_{col}"] = beta_val

                    rows.append(pd.DataFrame([d]))

                    # Store holdout predictions
                    pr = holdout_df.copy()
                    pr["Actual"] = y_holdout_original.values
                    pr["Predicted"] = holdout_preds
                    pr["Model"] = display_name
                    pr["Fold"] = "Holdout"
                    preds_records.append(pr)


    # ═════════════════════════════════════════════════════════════════════════
    # PROCESS STACKED MODELS (grouped by FILTER keys only, interaction on STACKING keys)
    # ═════════════════════════════════════════════════════════════════════════
    for gvals, gdf in group_list_stacked:
        gvals = (gvals,) if not isinstance(gvals, tuple) else gvals
        group_display_name = " | ".join([f"{k}={v}" for k, v in zip(filter_keys_for_stacking, gvals)]) if filter_keys_for_stacking else "All"

        present_cols = [c for c in X_columns if c in gdf.columns]
        if len(present_cols) < len(X_columns):
            for mname in stacked_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        X_full = gdf[present_cols].fillna(0).copy()
        y_full = gdf[target_col].copy()

        n_samples = len(X_full)
        n_features = len(present_cols)

        # Count number of unique groups in stacking keys
        if stacked_models:
            first_stacked_model = list(stacked_models.values())[0]
            n_groups = gdf[first_stacked_model.group_keys].drop_duplicates().shape[0]
            # Each group needs enough samples
            min_samples_per_group = max(3, n_features // 2)
            min_samples_needed = max(k_folds, n_groups * min_samples_per_group)
        else:
            min_samples_needed = max(k_folds, n_features * 2)

        if n_samples < min_samples_needed:
            st.warning(f"⚠️ Skipping stacked model for {group_display_name}: Only {n_samples} samples but need {min_samples_needed}")
            for mname in stacked_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        # ADAPTIVE K-FOLD: Reduce folds for smaller groups
        adaptive_k = k_folds
        if n_samples < 30:
            adaptive_k = 2  # Use 2-fold for very small groups (stacked needs more samples)
        elif n_samples < 60:
            adaptive_k = min(3, k_folds)  # Use 3-fold for small groups

        kf = KFold(n_splits=adaptive_k, shuffle=True, random_state=42)

        if adaptive_k != k_folds:
            st.info(f"ℹ️ {group_display_name} (stacked): Using {adaptive_k}-fold CV (only {n_samples} samples)")

        for mname, mdl in stacked_models.items():
            fold_results = []

            for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
                update_progress(group_display_name, mname, fold_id)

                # CRITICAL FIX: Use .iloc to get by position, keep track of original indices
                # tr_idx and te_idx are POSITIONS in X_full
                X_tr_orig = X_full.iloc[tr_idx].copy()
                X_te_orig = X_full.iloc[te_idx].copy()
                y_tr_orig = y_full.iloc[tr_idx].copy()
                y_te_orig = y_full.iloc[te_idx].copy()

                # Store ORIGINAL y values for metrics calculation (before any transformation)
                y_tr_original_scale = y_tr_orig.copy()
                y_te_original_scale = y_te_orig.copy()

                # Apply log transformation if requested
                if log_transform_y:
                    y_tr_orig = np.log1p(y_tr_orig)  # log(1 + y)
                    y_te_orig = np.log1p(y_te_orig)

                # Store original indices for later use
                tr_index_map = X_tr_orig.index.tolist()
                te_index_map = X_te_orig.index.tolist()

                # Get groups using the SAME POSITIONAL indices
                # This ensures perfect alignment: row i of groups corresponds to row i of X
                groups_tr = gdf.iloc[tr_idx][mdl.group_keys].copy()
                groups_te = gdf.iloc[te_idx][mdl.group_keys].copy()

                # Standardization BEFORE resetting indices
                scaler = {}
                # Check if base model of stacked model is RLS
                is_rls_base = isinstance(mdl.base_model, RecursiveLeastSquares) if hasattr(mdl, 'base_model') else False

                # RLS ALWAYS needs ALL features standardized
                if is_rls_base:
                    cols_to_scale = list(X_tr_orig.columns)
                elif std_cols:
                    cols_to_scale = [c for c in std_cols if c in X_tr_orig.columns]
                else:
                    cols_to_scale = []

                if cols_to_scale:
                        sc = StandardScaler().fit(X_tr_orig[cols_to_scale])
                        X_tr_orig[cols_to_scale] = sc.transform(X_tr_orig[cols_to_scale])
                        X_te_orig[cols_to_scale] = sc.transform(X_te_orig[cols_to_scale])
                        scaler = {c: (m, s) for c, m, s in zip(cols_to_scale, sc.mean_, sc.scale_)}

                # NOW reset all indices in perfect synchronization
                X_tr_reset = X_tr_orig.reset_index(drop=True)
                X_te_reset = X_te_orig.reset_index(drop=True)
                y_tr_reset = y_tr_orig.reset_index(drop=True)
                y_te_reset = y_te_orig.reset_index(drop=True)
                y_tr_original_scale_reset = y_tr_original_scale.reset_index(drop=True)
                y_te_original_scale_reset = y_te_original_scale.reset_index(drop=True)
                groups_tr = groups_tr.reset_index(drop=True)
                groups_te = groups_te.reset_index(drop=True)

                model_copy = clone(mdl)
                model_copy.fit(X_tr_reset, y_tr_reset, feature_names=present_cols, groups_df=groups_tr)

                y_tr_pred = model_copy.predict(X_tr_reset, groups_df=groups_tr)
                y_te_pred = model_copy.predict(X_te_reset, groups_df=groups_te)

                # Reverse transform predictions if log was applied
                if log_transform_y:
                    y_tr_pred = np.expm1(y_tr_pred)  # exp(pred) - 1
                    y_te_pred = np.expm1(y_te_pred)
                    # Ensure non-negative predictions
                    y_tr_pred = np.maximum(y_tr_pred, 0)
                    y_te_pred = np.maximum(y_te_pred, 0)

                # Get group coefficients
                group_coefs = model_copy.get_group_coefficients()

                # Create result rows for each stacking group
                if len(mdl.group_keys) == 1:
                    test_groups = groups_te[mdl.group_keys[0]].astype(str)
                    train_groups = groups_tr[mdl.group_keys[0]].astype(str)
                else:
                    test_groups = groups_te[mdl.group_keys].astype(str).apply(lambda row: "_".join(row), axis=1)
                    train_groups = groups_tr[mdl.group_keys].astype(str).apply(lambda row: "_".join(row), axis=1)

                unique_test_groups = test_groups.unique()

                for group in unique_test_groups:
                    group_mask_te = (test_groups == group).values

                    if not group_mask_te.any():
                        continue

                    # Get original scale y values for this group
                    y_te_group_original = y_te_original_scale_reset[group_mask_te]
                    y_pred_te_group = y_te_pred[group_mask_te]

                    # Metrics on original scale
                    r2_te = r2_score(y_te_group_original, y_pred_te_group) if len(y_te_group_original) > 1 else np.nan
                    mape_te = safe_mape(y_te_group_original, y_pred_te_group)
                    mae_te = np.mean(np.abs(y_te_group_original - y_pred_te_group))
                    mse_te = np.mean((y_te_group_original - y_pred_te_group)**2)
                    rmse_te = np.sqrt(mse_te)

                    group_mask_tr = (train_groups == group).values

                    if group_mask_tr.any():
                        y_tr_group_original = y_tr_original_scale_reset[group_mask_tr]
                        y_pred_tr_group = y_tr_pred[group_mask_tr]
                        r2_tr = r2_score(y_tr_group_original, y_pred_tr_group) if len(y_tr_group_original) > 1 else np.nan
                        mape_tr = safe_mape(y_tr_group_original, y_pred_tr_group)
                        mae_tr = np.mean(np.abs(y_tr_group_original - y_pred_tr_group))
                        mse_tr = np.mean((y_tr_group_original - y_pred_tr_group)**2)
                        rmse_tr = np.sqrt(mse_tr)
                    else:
                        r2_tr = mape_tr = mae_tr = mse_tr = rmse_tr = np.nan

                    if group in group_coefs:
                        raw_int = group_coefs[group]['intercept']
                        raw_coefs_dict = group_coefs[group]['coefficients']

                        # Reverse standardization
                        if scaler and std_cols:
                            for col in std_cols:
                                if col in raw_coefs_dict and col in scaler:
                                    mu, sd = scaler[col]
                                    raw_coefs_dict[col] = raw_coefs_dict[col] / sd
                                    raw_int -= raw_coefs_dict[col] * mu

                        # FIX: Calculate mean X ONLY from TRAINING data to avoid leakage
                        if group_mask_tr.any():
                            train_indices_for_group = np.where(group_mask_tr)[0]
                            train_original_indices = [tr_index_map[i] for i in train_indices_for_group]
                            group_train_data = gdf.loc[train_original_indices]
                            mean_x = group_train_data[present_cols].mean(numeric_only=True).to_dict()
                        else:
                            # Fallback if group not in training (shouldn't happen in CV)
                            mean_x = {c: np.nan for c in present_cols}

                        # Create fold result
                        group_parts = group.split('_')
                        d = {}

                        # Add filter grouping keys
                        for idx, key in enumerate(filter_keys_for_stacking):
                            d[key] = gvals[idx]

                        # Add stacking keys
                        for idx, key in enumerate(mdl.group_keys):
                            d[key] = group_parts[idx] if idx < len(group_parts) else ''

                        d.update({
                            "Model": mname,
                            "Fold": fold_id,
                            "B0 (Original)": raw_int,
                            "R2 Train": r2_tr,
                            "R2 Test": r2_te,
                            "MAPE Train": mape_tr,
                            "MAPE Test": mape_te,
                            "MAE Train": mae_tr,
                            "MAE Test": mae_te,
                            "MSE Train": mse_tr,
                            "MSE Test": mse_te,
                            "RMSE Train": rmse_tr,
                            "RMSE Test": rmse_te,
                        })

                        # Add mean X
                        for c, v in mean_x.items():
                            d[c] = v

                        # Add betas
                        for feat_name in present_cols:
                            d[f"Beta_{feat_name}"] = raw_coefs_dict.get(feat_name, 0)

                        fold_results.append(d)

                        # FIX: Store predictions with consistent indices (original scale)
                        test_indices_for_group = np.where(group_mask_te)[0]
                        test_original_indices = [te_index_map[i] for i in test_indices_for_group]
                        pr = gdf.loc[test_original_indices].copy()
                        pr["Actual"] = y_te_group_original.values
                        pr["Predicted"] = y_pred_te_group
                        pr["Model"] = mname
                        pr["Fold"] = fold_id
                        preds_records.append(pr)

            # Report fold results: individual folds + aggregated
            if fold_results:
                fold_df = pd.DataFrame(fold_results)

                # Add individual fold rows
                rows.append(fold_df)

                # Create aggregated row (average across folds)
                # Identify keys
                key_cols = [col for col in fold_df.columns if col in (filter_keys_for_stacking + mdl.group_keys) or col == 'Model']

                # Numeric columns to average
                numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'Fold']

                # String columns to take first
                string_cols = [col for col in fold_df.columns if col not in numeric_cols and col not in key_cols and col != 'Fold']

                # Aggregate
                agg_dict = {}
                for col in numeric_cols:
                    agg_dict[col] = 'mean'
                for col in string_cols:
                    agg_dict[col] = 'first'

                aggregated = fold_df.groupby(key_cols).agg(agg_dict).reset_index()
                aggregated['Fold'] = 'Avg'  # Mark as average row
                rows.append(aggregated)

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    total_time = time.time() - start_time
    st.success(f"✅ Completed in {total_time:.1f} seconds")

    if not rows:
        return None, None, None

    # Combine results
    results_df = pd.concat(rows, ignore_index=True)

    # Order columns (include Fold right after Model)
    front = grouping_keys + ["Model", "Fold"]
    metric_block = ["B0 (Original)",
                    "R2 Train", "R2 Test", "R2 Holdout",
                    "MAPE Train", "MAPE Test", "MAPE Holdout",
                    "MAE Train", "MAE Test", "MAE Holdout",
                    "MSE Train", "MSE Test", "MSE Holdout",
                    "RMSE Train", "RMSE Test", "RMSE Holdout"]

    mean_x_cols = [c for c in results_df.columns if c not in front + metric_block and not c.startswith("Beta_") and not c.startswith("Mean_")]
    beta_cols = [c for c in results_df.columns if c.startswith("Beta_")]
    mean_cols = [c for c in results_df.columns if c.startswith("Mean_")]

    existing_cols = []
    for col_group in [front, metric_block, mean_cols, beta_cols, mean_x_cols]:
        existing_cols.extend([c for c in col_group if c in results_df.columns])

    # Ensure we include any remaining columns
    for col in results_df.columns:
        if col not in existing_cols:
            existing_cols.append(col)

    results_df = results_df[existing_cols]

    preds_df = pd.concat(preds_records, ignore_index=True) if preds_records else None
    optimized_lambda_df = pd.DataFrame(optimized_lambda_records) if optimized_lambda_records else None

    # ═══════════════════════════════════════════════════════════════════════════
    # ENSEMBLE MODEL CREATION (if enabled)
    # ═══════════════════════════════════════════════════════════════════════════
    ensemble_df = None
    if enable_ensemble:
        st.info("🔄 Creating ensemble models from CV results (averaging coefficients across all models)...")

        ensembles = create_ensemble_model_from_results(
            results_df,
            grouping_keys,
            X_columns,
            weight_metric=ensemble_weight_metric,
            filter_r2_min=ensemble_filter_r2_min,
            filter_mape_max=ensemble_filter_mape_max,
            filter_mae_max=ensemble_filter_mae_max,
            filter_positive_features=ensemble_filter_positive_features,
            filter_negative_features=ensemble_filter_negative_features
        )

        if ensembles:
            # Convert ensemble dict to DataFrame
            ensemble_rows = []
            for combo_key, ensemble_data in ensembles.items():
                row = {}

                # Parse combination key back to individual keys
                if grouping_keys and " | " in combo_key:
                    parts = combo_key.split(" | ")
                    for part in parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            row[k] = v
                elif grouping_keys and len(grouping_keys) == 1:
                    row[grouping_keys[0]] = combo_key.split("=")[1] if "=" in combo_key else combo_key

                row['Model'] = 'Weighted Ensemble'
                row['Fold'] = 'Ensemble'
                row['B0 (Original)'] = ensemble_data['ensemble_intercept']

                # Add ensemble betas
                for beta_name, beta_value in ensemble_data['ensemble_betas'].items():
                    row[beta_name] = beta_value

                # Add ensemble metrics
                for metric_key, metric_value in ensemble_data.items():
                    if metric_key.startswith('ensemble_'):
                        # Convert ensemble_r2_test -> R2 Test format
                        metric_name = metric_key.replace('ensemble_', '').replace('_', ' ').title()
                        row[metric_name] = metric_value

                # Add metadata
                row['Num_Models'] = ensemble_data['num_models']
                row['Best_Model'] = ensemble_data['model_names'][ensemble_data['best_model_idx']] if ensemble_data['model_names'] else ''
                row['Weight_Concentration'] = ensemble_data['weight_concentration']

                ensemble_rows.append(row)

            ensemble_df = pd.DataFrame(ensemble_rows)

            # Append ensemble results to main results
            results_df = pd.concat([results_df, ensemble_df], ignore_index=True)

            st.success(f"✅ Created {len(ensembles)} ensemble models")

            # ═══════════════════════════════════════════════════════════════════════════
            # RLS ON ENSEMBLE MODELS (if both ensemble and RLS are enabled)
            # Uses ensemble betas from weeks 1-44 → warmup 45-48 → test 49-52
            # ═══════════════════════════════════════════════════════════════════════════
            if enable_rls and holdout_weeks > 0:
                st.info(f"🔄 Applying RLS to ensemble models: Train weeks 1-{44 if warmup_weeks == 4 else 'N-H-W'} → Warmup → Holdout test...")

                ensemble_rls_rows = []

                # Create expander for debug messages
                debug_expander = st.expander(f"📋 RLS Processing Details ({len(ensembles)} ensemble(s))", expanded=False)

                for combo_key, ensemble_data in ensembles.items():
                    debug_expander.caption(f"🔄 Processing: {combo_key}")
                    # Get the original data for this combination
                    if not grouping_keys or combo_key == 'ALL':
                        gdf = df.copy()
                        group_display_name = "All"
                    else:
                        # Parse combo_key to filter data
                        if " | " in combo_key:
                            parts = combo_key.split(" | ")
                            filters = {}
                            for part in parts:
                                if "=" in part:
                                    k, v = part.split("=", 1)
                                    filters[k] = v

                            gdf = df.copy()
                            for k, v in filters.items():
                                if k in gdf.columns:
                                    gdf = gdf[gdf[k].astype(str) == str(v)]
                            group_display_name = combo_key
                        elif "=" in combo_key:
                            # Single key case: "Brand=A"
                            k, v = combo_key.split("=", 1)
                            gdf = df.copy()
                            if k in gdf.columns:
                                gdf = gdf[gdf[k].astype(str) == str(v)]
                            group_display_name = combo_key
                        else:
                            # Fallback: use entire dataset
                            gdf = df.copy()
                            group_display_name = combo_key

                    if len(gdf) == 0:
                        continue

                    # Sort by index (assuming time order)
                    gdf_sorted = gdf.sort_index()
                    n_total = len(gdf_sorted)

                    # Split: train (1-44), warmup (45-48), holdout (49-52)
                    n_train = n_total - holdout_weeks - warmup_weeks
                    n_warmup_end = n_total - holdout_weeks

                    debug_expander.caption(f"  → Total={n_total}, Train={n_train}, Warmup={n_warmup_end-n_train}, Holdout={holdout_weeks}")

                    if n_train < 10:
                        debug_expander.warning(f"  ⚠️ Skipped {combo_key}: Not enough training data (need ≥10, have {n_train})")
                        continue

                    train_df = gdf_sorted.iloc[:n_train]
                    warmup_df = gdf_sorted.iloc[n_train:n_warmup_end]
                    holdout_df = gdf_sorted.iloc[n_warmup_end:]

                    # Prepare data
                    present_cols = [c for c in X_columns if c in gdf_sorted.columns]
                    if len(present_cols) < len(X_columns):
                        continue

                    X_train = train_df[present_cols].fillna(0).values
                    y_train = train_df[target_col].values
                    y_train_original = y_train.copy()

                    X_warmup = warmup_df[present_cols].fillna(0).values
                    y_warmup = warmup_df[target_col].values
                    y_warmup_original = y_warmup.copy()

                    X_holdout = holdout_df[present_cols].fillna(0).values
                    y_holdout = holdout_df[target_col].values
                    y_holdout_original = y_holdout.copy()

                    if log_transform_y:
                        y_train = np.log1p(y_train)
                        y_warmup = np.log1p(y_warmup)
                        y_holdout = np.log1p(y_holdout)

                    # Standardization (RLS needs ALL features standardized)
                    sc = StandardScaler().fit(X_train)
                    X_train = sc.transform(X_train)
                    X_warmup = sc.transform(X_warmup)
                    X_holdout = sc.transform(X_holdout)

                    # Extract ensemble betas (already in original feature names format: Beta_<feature>)
                    ensemble_betas_dict = ensemble_data['ensemble_betas']
                    ensemble_intercept = ensemble_data['ensemble_intercept']

                    # Convert to array in correct order
                    ensemble_beta_array = np.array([
                        ensemble_betas_dict.get(f"Beta_{feat}", 0.0)
                        for feat in present_cols
                    ])

                    # CRITICAL FIX: Transform ensemble betas from original scale to standardized scale
                    # Since X data is standardized, we need to adjust the betas accordingly
                    # For standardized data: beta_std = beta_orig / sd
                    # And intercept_std = intercept_orig - sum(beta_orig * mean / sd)
                    standardized_beta_array = ensemble_beta_array / sc.scale_
                    standardized_intercept = ensemble_intercept - np.sum(ensemble_beta_array * sc.mean_ / sc.scale_)

                    # Apply RLS with ensemble betas as initialization
                    lambda_grid = rls_lambda_candidates if rls_lambda_candidates is not None else DEFAULT_RLS_LAMBDA_GRID
                    best_lambda = None
                    best_holdout_preds = None
                    best_mae = None
                    best_rls_results = None

                    for candidate_lambda in lambda_grid:
                        rls_results = apply_rls_on_holdout(
                            trained_model=None,  # Will use initial_beta/initial_intercept instead
                            X_train=X_train,
                            y_train=y_train,
                            X_warmup=X_warmup,
                            y_warmup=y_warmup,
                            X_holdout=X_holdout,
                            y_holdout=y_holdout,
                            feature_names=present_cols,
                            forgetting_factor=candidate_lambda,
                            nonnegative_features=positive_constraints,
                            nonpositive_features=negative_constraints,
                            initial_beta=standardized_beta_array,  # Use standardized ensemble betas!
                            initial_intercept=standardized_intercept  # Use standardized ensemble intercept!
                        )

                        candidate_preds = rls_results['predictions']
                        if log_transform_y:
                            candidate_preds = np.expm1(candidate_preds)
                            # Check for negative predictions before clipping
                            n_negative = np.sum(candidate_preds < 0)
                            if n_negative > 0:
                                debug_expander.warning(f"  ⚠️ {n_negative}/{len(candidate_preds)} RLS predictions were negative (clipped to 0)")
                            candidate_preds = np.maximum(candidate_preds, 0)

                        # CRITICAL FIX: Use WARMUP MAE for lambda selection to avoid data leakage
                        warmup_preds = rls_results.get('warmup_predictions', np.array([]))
                        if len(warmup_preds) > 0:
                            if log_transform_y:
                                warmup_preds_transformed = np.expm1(warmup_preds)
                                warmup_preds_transformed = np.maximum(warmup_preds_transformed, 0)
                            else:
                                warmup_preds_transformed = warmup_preds
                            candidate_warmup_mae = mean_absolute_error(y_warmup_original, warmup_preds_transformed)
                        else:
                            candidate_warmup_mae = float('inf')

                        if np.isnan(candidate_warmup_mae):
                            continue

                        if best_mae is None or candidate_warmup_mae < best_mae:
                            best_mae = candidate_warmup_mae
                            best_lambda = candidate_lambda
                            best_holdout_preds = candidate_preds
                            best_rls_results = rls_results

                    if best_rls_results is None:
                        debug_expander.warning(f"  ⚠️ RLS failed for {combo_key}: No valid lambda found")
                        continue

                    debug_expander.caption(f"  ✅ RLS successful! Best lambda={best_lambda:.3f}")

                    # Calculate metrics
                    r2_holdout = r2_score(y_holdout_original, best_holdout_preds)
                    mape_holdout = safe_mape(y_holdout_original, best_holdout_preds)
                    rmse_holdout = np.sqrt(mean_squared_error(y_holdout_original, best_holdout_preds))

                    # Create ensemble + RLS row
                    row = {}
                    if grouping_keys and " | " in combo_key:
                        parts = combo_key.split(" | ")
                        for part in parts:
                            if "=" in part:
                                k, v = part.split("=", 1)
                                row[k] = v

                    row['Model'] = 'Weighted Ensemble + RLS'
                    row['Fold'] = 'Holdout'
                    row['R2 Holdout'] = r2_holdout
                    row['MAPE Holdout'] = mape_holdout
                    row['MAE Holdout'] = best_mae
                    row['RMSE Holdout'] = rmse_holdout
                    row['Best_Lambda'] = best_lambda

                    # Add final betas from RLS
                    if 'beta_history' in best_rls_results:
                        holdout_snapshots = best_rls_results['beta_history'].get('holdout_beta_snapshots', [])
                        if len(holdout_snapshots) > 0:
                            final_beta = holdout_snapshots[-1]
                            row['B0 (Original)'] = final_beta[0]
                            for i, feat in enumerate(present_cols):
                                # Reverse standardization
                                beta_val = final_beta[i+1]
                                beta_val = beta_val / sc.scale_[i]
                                row[f"Beta_{feat}"] = beta_val

                    ensemble_rls_rows.append(row)

                    # ═══════════════════════════════════════════════════════════════
                    # STORE WARMUP PREDICTIONS
                    # ═══════════════════════════════════════════════════════════════
                    warmup_preds = best_rls_results.get('warmup_predictions', np.array([]))
                    warmup_actuals = best_rls_results.get('warmup_actuals', np.array([]))

                    if len(warmup_preds) > 0:
                        # Reverse log transform on warmup predictions
                        if log_transform_y:
                            warmup_preds = np.expm1(warmup_preds)
                            warmup_preds = np.maximum(warmup_preds, 0)
                            warmup_actuals = np.expm1(warmup_actuals)

                        # Store warmup predictions
                        pr_warmup = warmup_df.copy()
                        pr_warmup["Actual"] = warmup_actuals
                        pr_warmup["Predicted"] = warmup_preds
                        pr_warmup["Model"] = 'Weighted Ensemble + RLS'
                        pr_warmup["Fold"] = "Warmup"
                        preds_records.append(pr_warmup)

                    # Store holdout predictions
                    pr = holdout_df.copy()
                    pr["Actual"] = y_holdout_original
                    pr["Predicted"] = best_holdout_preds
                    pr["Model"] = 'Weighted Ensemble + RLS'
                    pr["Fold"] = "Holdout"
                    preds_records.append(pr)

                    # Store lambda
                    lambda_record = {}
                    if grouping_keys and " | " in combo_key:
                        parts = combo_key.split(" | ")
                        for part in parts:
                            if "=" in part:
                                k, v = part.split("=", 1)
                                lambda_record[k] = v
                    lambda_record['Model'] = 'Weighted Ensemble + RLS'
                    lambda_record['Best Lambda'] = best_lambda
                    lambda_record['Holdout MAE'] = best_mae
                    optimized_lambda_records.append(lambda_record)

                    # ═══════════════════════════════════════════════════════════════
                    # CREATE STATIC BASELINE FOR COMPARISON (ensemble without RLS)
                    # ═══════════════════════════════════════════════════════════════
                    # Create static predictions using ensemble betas on train+warmup (weeks 1-48)
                    train_warmup_data = pd.concat([train_df, warmup_df])
                    X_train_warmup = train_warmup_data[present_cols].fillna(0).values
                    y_train_warmup = train_warmup_data[target_col].values

                    if log_transform_y:
                        y_train_warmup = np.log1p(y_train_warmup)

                    # Standardize with same scaler
                    sc_trainwarmup = StandardScaler().fit(X_train_warmup)
                    X_train_warmup_scaled = sc_trainwarmup.transform(X_train_warmup)
                    X_holdout_scaled_trainwarmup = sc_trainwarmup.transform(X_holdout)

                    # Static predictions using ensemble betas (no RLS adaptation)
                    # Convert ensemble betas to standardized space
                    ensemble_beta_array_trainwarmup = np.array([
                        ensemble_betas_dict.get(f"Beta_{feat}", 0.0) * sc_trainwarmup.scale_[i]
                        for i, feat in enumerate(present_cols)
                    ])

                    baseline_static_preds = (
                        X_holdout_scaled_trainwarmup @ ensemble_beta_array_trainwarmup +
                        ensemble_intercept
                    )

                    if log_transform_y:
                        baseline_static_preds = np.expm1(baseline_static_preds)
                        # Check for negative predictions before clipping
                        n_negative_static = np.sum(baseline_static_preds < 0)
                        if n_negative_static > 0:
                            debug_expander.warning(f"  ⚠️ {n_negative_static}/{len(baseline_static_preds)} Static predictions were negative (clipped to 0)")
                        baseline_static_preds = np.maximum(baseline_static_preds, 0)

                    # Calculate static metrics
                    r2_static = r2_score(y_holdout_original, baseline_static_preds)
                    mae_static = mean_absolute_error(y_holdout_original, baseline_static_preds)
                    rmse_static = np.sqrt(mean_squared_error(y_holdout_original, baseline_static_preds))

                    # ═══════════════════════════════════════════════════════════════
                    # STORE FOR COMPARISON VISUALIZATION
                    # ═══════════════════════════════════════════════════════════════
                    debug_expander.caption(f"  💾 Storing comparison data for {group_display_name}")

                    if 'rls_comparison_store' not in st.session_state:
                        st.session_state.rls_comparison_store = []

                    st.session_state.rls_comparison_store.append({
                        'Group': group_display_name,
                        'Model': 'Weighted Ensemble',
                        'Dates': [f"Week {i+1}" for i in range(len(y_holdout_original))],
                        'Actuals': y_holdout_original,
                        'Predictions_Static': baseline_static_preds,
                        'Predictions_RLS': best_holdout_preds,
                        'R2_Static': r2_static,
                        'R2_RLS': r2_holdout,
                        'MAE_Static': mae_static,
                        'MAE_RLS': best_mae,
                        'RMSE_Static': rmse_static,
                        'RMSE_RLS': rmse_holdout
                    })

                    # Store beta history
                    if 'beta_history' in best_rls_results:
                        group_key = f"{group_display_name} | Weighted Ensemble + RLS"

                        if 'beta_history_store' not in st.session_state:
                            st.session_state.beta_history_store = {}

                        st.session_state.beta_history_store[group_key] = best_rls_results['beta_history']

                if ensemble_rls_rows:
                    ensemble_rls_df = pd.DataFrame(ensemble_rls_rows)
                    results_df = pd.concat([results_df, ensemble_rls_df], ignore_index=True)
                    st.success(f"✅ Applied RLS to {len(ensemble_rls_rows)} ensemble models")

    return results_df, preds_df, optimized_lambda_df, ensemble_df
