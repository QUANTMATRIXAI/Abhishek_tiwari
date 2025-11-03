"""
Streamlit UI Application for Regression Modeling

This module provides the user interface for the regression modeling application.
It allows users to upload data, configure models, apply constraints, and visualize results.

The application supports:
- Multiple regression models (Linear, Ridge, Lasso, ElasticNet, Bayesian, Custom Constrained)
- Group-specific modeling with stacking
- Coefficient constraints (positive/negative)
- Recursive Least Squares (RLS) for adaptive modeling
- Model ensembling with weighted averaging
- Auto-residualization for handling multicollinearity
- Time series visualization and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import r2_score

from models import (
    CustomConstrainedRidge,
    ConstrainedLinearRegression,
    StackedInteractionModel,
    StatsMixedEffectsModel,
    RecursiveLeastSquares
)
from pipeline import run_model_pipeline
from utils import DEFAULT_RLS_LAMBDA_GRID, safe_mape

warnings.filterwarnings('ignore')


def main():
    st.set_page_config(page_title="Modeling App", layout="wide")

    st.title("🎯 Regression Modeling App")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.data = df
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

        if st.session_state.data is not None:
            st.markdown("---")
            st.metric("Total Rows", len(st.session_state.data))
            st.metric("Total Columns", len(st.session_state.data.columns))

    if st.session_state.data is None:
        st.info("👆 Upload a file to begin")
        return

    st.markdown("---")

    # Configuration
    st.header("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Grouping Keys")
        available_cols = list(df.columns)
        selected_grouping_keys = st.multiselect(
            "Select grouping columns:",
            options=available_cols,
            default=[],
            help="Each unique combination gets its own model"
        )

        if selected_grouping_keys:
            combo_counts = df.groupby(selected_grouping_keys).size().reset_index(name='Count')
            st.caption(f"📌 {len(combo_counts)} unique combinations")

    with col2:
        st.subheader("2️⃣ Target & Predictors")
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        target_col = st.selectbox(
            "🎯 Target Variable:",
            options=numeric_cols,
            help="What to predict"
        )

        available_predictors = [c for c in numeric_cols if c != target_col and c not in selected_grouping_keys]
        default_predictors = st.session_state.get(
            'selected_predictors',
            available_predictors[:min(5, len(available_predictors))]
        )
        # Ensure defaults exist in current options
        default_predictors = [p for p in default_predictors if p in available_predictors]
        selected_predictors = st.multiselect(
            "📊 Predictors:",
            options=available_predictors,
            default=default_predictors,
            help="Features for the model"
        )
        st.session_state['selected_predictors'] = selected_predictors

    # ═══════════════════════════════════════════════════════════════════════════
    # RESIDUALIZATION FEATURE (Automatic per product/brand)
    # ═══════════════════════════════════════════════════════════════════════════
    residualization_mapping = {}
    with st.expander("🔧 Advanced: Auto-Residualization (Remove Multicollinearity)", expanded=False):
        st.markdown("""
        **Automatically remove correlation** by finding product-specific primary variables.

        **How it works:**
        - For each product (e.g., "paw diamond necklace"), finds its specific column (e.g., "paw diamond necklace_meta_impression")
        - Uses that as primary variable for that product
        - Residualizes all other predictors against it
        - Falls back to a general column (e.g., "impressions") if product-specific column doesn't exist
        """)

        enable_auto_residualization = st.checkbox(
            "Enable Auto-Residualization",
            value=False,
            help="Automatically detect primary variable per product/brand and residualize others"
        )

        # Clear residualisation state if checkbox is unchecked
        if not enable_auto_residualization and st.session_state.get('residualization_applied', False):
            st.session_state['residualization_applied'] = False
            st.session_state.pop('df_residualized', None)
            st.session_state.pop('selected_predictors_residualized', None)
            st.session_state.pop('residualization_mapping', None)
            st.info("🔄 Residualisation disabled - using original data")

        if enable_auto_residualization and selected_grouping_keys and len(selected_predictors) > 1:
            # Identify fallback primary variable - prefer general "impressions" column if present
            default_fallback = next(
                (
                    col for col in numeric_cols
                    if col.lower().strip() == "impressions"
                ),
                None
            )
            fallback_primary = st.selectbox(
                "📌 Fallback Primary Variable (if no product-specific column found):",
                options=numeric_cols,
                index=numeric_cols.index(default_fallback) if default_fallback and default_fallback in numeric_cols else 0,
                help="Column to use when no product-specific match exists (defaults to 'impressions' when available)"
            )

            # Auto-detect and show mapping
            st.markdown("**🔍 Detected Primary Variables:**")

            # Get unique values from first grouping key
            if selected_grouping_keys:
                first_group_key = selected_grouping_keys[0]
                unique_groups = df[first_group_key].unique()

                detected_mappings = {}

                # Candidate columns for impressions
                def _normalize(text: str) -> str:
                    text = text.lower().strip()
                    text = text.replace('-', ' ')
                    text = text.replace('/', ' ')
                    text = text.replace('&', ' and ')
                    text = text.replace("'", '')
                    text = "".join(ch if ch.isalnum() or ch == ' ' else ' ' for ch in text)
                    return "_".join(text.split())

                impression_cols = [
                    col for col in numeric_cols
                    if '_meta_impression' in col.lower()
                ]
                st.caption(f"**Matched {len(impression_cols)} impression column candidates**")

                for group_val in unique_groups:
                    if pd.notna(group_val):
                        group_key = str(group_val)
                        group_norm = _normalize(group_key)

                        # Try to find product-specific column
                        primary_found = None
                        best_match_score = 0

                        for col in impression_cols:
                            col_norm = _normalize(col.replace('_meta_impression', ''))
                            if not col_norm:
                                continue
                            if col_norm == group_norm:
                                primary_found = col
                                best_match_score = len(group_norm)
                                break
                            if group_norm in col_norm or col_norm in group_norm:
                                score = min(len(group_norm), len(col_norm))
                                if score > best_match_score:
                                    primary_found = col
                                    best_match_score = score

                        # Use fallback if not found (e.g., "impressions")
                        if primary_found is None:
                            primary_found = fallback_primary

                        detected_mappings[group_key] = primary_found

                # Show detected mappings in table format
                matched_count = sum(1 for v in detected_mappings.values() if v != fallback_primary)
                st.caption(f"**Matched {matched_count} of {len(detected_mappings)} products to specific impression columns**")

                # Display mappings in a dataframe table instead of list
                mapping_data = []
                for group, primary in detected_mappings.items():
                    status = "✅ Matched" if primary != fallback_primary else "⚠️ Fallback"
                    mapping_data.append({
                        'Product': group,
                        'Primary Variable': primary,
                        'Status': status
                    })

                mapping_df = pd.DataFrame(mapping_data)
                st.dataframe(mapping_df, use_container_width=True, height=400)

                # Apply residualization per group
                if st.button("✅ Apply Auto-Residualization", key="apply_residualization"):
                    df_residual = df.copy()
                    residualization_stats = []

                    for group_val, primary_var in detected_mappings.items():
                        # Filter to this group
                        group_mask = df_residual[first_group_key].astype(str) == str(group_val)
                        group_df = df_residual[group_mask]

                        if len(group_df) < 10:
                            continue

                        # Variables to residualize (all except primary and target)
                        vars_to_residualize = [
                            p for p in selected_predictors
                            if p != primary_var and p != target_col
                        ]

                        for var in vars_to_residualize:
                            if var in group_df.columns and primary_var in group_df.columns:
                                valid_mask = group_df[[primary_var, var]].notna().all(axis=1)

                                if valid_mask.sum() > 5:
                                    X_primary = group_df.loc[valid_mask, primary_var].values.reshape(-1, 1)
                                    y_secondary = group_df.loc[valid_mask, var].values

                                    # Fit regression
                                    lr = LinearRegression()
                                    lr.fit(X_primary, y_secondary)

                                    # Calculate residuals for this group
                                    X_all = group_df[primary_var].fillna(0).values.reshape(-1, 1)
                                    predicted = lr.predict(X_all)
                                    residuals = group_df[var].fillna(0).values - predicted

                                    # Store in original dataframe with product-specific name
                                    residual_col_name = f"{var}_residual"
                                    df_residual.loc[group_mask, residual_col_name] = residuals

                                    residualization_stats.append({
                                        'Group': group_val,
                                        'Primary': primary_var,
                                        'Residualized': var,
                                        'New Column': residual_col_name,
                                        'R²': lr.score(X_primary, y_secondary)
                                    })

                    if residualization_stats:
                        # Store the residualised dataframe in session state
                        st.session_state['df_residualized'] = df_residual

                        # Update predictor list - replace original with residual versions
                        residualized_var_names = set([s['Residualized'] for s in residualization_stats])

                        # Remove original variables and add their residual versions
                        new_selected_predictors = [p for p in selected_predictors
                                            if p not in residualized_var_names]

                        # Add unique residual columns
                        residual_cols = list(set([s['New Column'] for s in residualization_stats]))
                        new_selected_predictors.extend(residual_cols)

                        # Store both the new predictors and a flag in session state
                        st.session_state['selected_predictors_residualized'] = new_selected_predictors
                        st.session_state['residualization_applied'] = True
                        st.session_state['residualization_mapping'] = residualization_mapping

                        st.success(f"✅ Applied residualization to {len(residualized_var_names)} variable(s) across {len(detected_mappings)} groups")

                        # Show detailed stats table
                        stats_df = pd.DataFrame(residualization_stats)
                        st.dataframe(stats_df, use_container_width=True, height=300)

                        # Show residual counts per group
                        group_counts = (
                            stats_df.groupby('Group')['Residualized']
                            .count()
                            .reset_index()
                            .rename(columns={'Residualized': 'Residualized Features'})
                            .sort_values('Residualized Features', ascending=False)
                        )
                        st.caption("**Residualized features per product**")
                        st.dataframe(group_counts, use_container_width=True, height=250)

                        st.info(f"""
                        **Next Steps:**
                        - Created {len(residual_cols)} residual column(s)
                        - Original variables replaced with their residualized versions
                        - Constraints and standardization will now use residualized columns
                        - Products with specific columns: Use their primary + residuals
                        - Products using fallback ({fallback_primary}): All use same primary + residuals
                        """)

    # Check if residualization was applied and update data and predictors accordingly
    if st.session_state.get('residualization_applied', False):
        df_working = st.session_state.get('df_residualized', df)
        selected_predictors_working = st.session_state.get('selected_predictors_residualized', selected_predictors)

        # Show a notice that we're using residualized data
        st.success("📊 Using residualized data for all subsequent operations")
    else:
        df_working = df
        selected_predictors_working = selected_predictors

    st.markdown("---")





    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3️⃣ Constraints")
        positive_constraints = st.multiselect(
            "Force ≥ 0:",
            options=selected_predictors_working,
            default=[],
            help="Variables that must have positive coefficients"
        )

        available_for_negative = [p for p in selected_predictors_working if p not in positive_constraints]
        negative_constraints = st.multiselect(
            "Force ≤ 0:",
            options=available_for_negative,
            default=[],
            help="Variables that must have negative coefficients"
        )

    with col4:
        st.subheader("4️⃣ Model Settings")
        k_folds = st.number_input(
            "CV Folds:",
            min_value=2,
            max_value=20,
            value=5
        )

        standardize_cols = st.multiselect(
            "Standardize:",
            options=selected_predictors_working,
            default=[]
        )

        log_transform_y = st.checkbox(
            "🔄 Log Transform Y Variable: log(y+1)",
            value=False,
            help="Apply log(y+1) transformation to target variable. Helps with:\n• Zero values in count data\n• Reducing impact of outliers\n• Stabilizing variance\n• Improving model fit for small discrete values"
        )

        min_y_share_pct = st.number_input(
            "Min Y Share % (Filter Groups):",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Only train models on groups where Y variable sum is at least this % of total Y.\n"
                 "• Focuses on significant segments\n"
                 "• Improves model quality by excluding tiny groups\n"
                 "• Speeds up computation\n"
                 "Example: 1% means group must contribute ≥1% of total Y"
        )

    st.markdown("---")

    # Stacking configuration
    use_stacked = st.checkbox(
        "Enable Stacking (Group-Specific Coefficients)",
        value=False,
        help="Creates models with interaction terms for selected keys"
    )

    if use_stacked and selected_grouping_keys:
        st.info("📌 **Stacking Strategy:** Select which keys to filter by vs which to use for interactions")

        col_stack1, col_stack2 = st.columns(2)

        with col_stack1:
            filter_keys_for_stacking = st.multiselect(
                "🔍 Filter By (separate models):",
                options=selected_grouping_keys,
                default=[selected_grouping_keys[0]] if selected_grouping_keys else [],
                help="Create separate models for each unique value of these keys"
            )

        with col_stack2:
            remaining_keys = [k for k in selected_grouping_keys if k not in filter_keys_for_stacking]
            if remaining_keys:
                st.multiselect(
                    "🔄 Interaction Keys (within models):",
                    options=remaining_keys,
                    default=remaining_keys,
                    disabled=True,
                    help="These keys will create interaction terms within each filtered model"
                )
                stacking_keys = remaining_keys
            else:
                st.warning("⚠️ No keys left for interactions. Select fewer filter keys.")
                stacking_keys = []
    else:
        filter_keys_for_stacking = []
        stacking_keys = []

    st.markdown("---")

    # RLS Settings
    with st.expander("🔄 Recursive Least Squares (RLS) Settings", expanded=False):
        st.caption("RLS continuously updates model parameters as new data arrives")

        col_rls1, col_rls2 = st.columns(2)

        with col_rls1:
            enable_rls = st.checkbox(
                "Enable RLS",
                value=False,
                help="Recursive Least Squares with automated forgetting factor tuning"
            )

        with col_rls2:
            warmup_weeks = st.number_input(
                "Warmup Weeks:",
                min_value=0,
                max_value=20,
                value=4,
                step=1,
                help="Weeks for RLS to adapt before holdout testing. RLS updates on these weeks without scoring."
            )

        holdout_weeks = st.number_input(
            "Holdout Weeks for Testing:",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            help="Number of recent weeks to hold out for testing RLS vs static predictions"
        )

        st.caption("📋 Forgetting factor is selected automatically per model to minimize holdout MAE.")
        st.caption(f"📋 Split: Train on weeks 1-{f'N-{holdout_weeks}-{warmup_weeks}' if warmup_weeks > 0 else f'N-{holdout_weeks}'}, "
                f"warmup on weeks {f'N-{holdout_weeks}-{warmup_weeks}+1 to N-{holdout_weeks}' if warmup_weeks > 0 else 'none'}, "
                f"test on last {holdout_weeks} weeks")


    st.markdown("---")

    # Model selection
    st.subheader("5️⃣ Select Models")

    base_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Bayesian Ridge": BayesianRidge(),
        "Custom Constrained Ridge": CustomConstrainedRidge(
            l2_penalty=0.1,
            learning_rate=0.001,
            iterations=10000,
            non_negative_features=positive_constraints,
            non_positive_features=negative_constraints
        ),
        "Constrained Linear Regression": ConstrainedLinearRegression(
            learning_rate=0.001,
            iterations=10000,
            non_negative_features=positive_constraints,
            non_positive_features=negative_constraints
        )
    }

    col_models = st.columns(4)
    selected_models = []

    for idx, model_name in enumerate(base_models.keys()):
        with col_models[idx % 4]:
            if st.checkbox(model_name, value=(idx < 3), key=f"model_{idx}"):
                selected_models.append(model_name)

    # Apply RLS to selected models if enabled
    models_to_run = {}
    for model_name in selected_models:
        # Just use base model
        models_to_run[model_name] = base_models[model_name]

    # Add stacked versions
    if use_stacked and stacking_keys:
        stacked_models = {}
        for name, model in models_to_run.items():
            # Add stacked versions for non-stacked models
            if not isinstance(model, StackedInteractionModel):
                is_constrained = isinstance(model, (CustomConstrainedRidge, ConstrainedLinearRegression))
                stacked_models[f"Stacked {name}"] = StackedInteractionModel(
                    base_model=model,
                    group_keys=stacking_keys,
                    enforce_combined_constraints=is_constrained
                )

        # Update models_to_run with stacked versions
        models_to_run.update(stacked_models)


    # Display summary
    base_count = len(selected_models)
    stacked_count = sum(1 for k in models_to_run.keys() if k.startswith('Stacked'))

    if enable_rls:
        st.info(f"📊 Will run **{len(models_to_run)}** model variants: "
                f"{base_count} base models (with RLS updates)" +
                (f" + {stacked_count} stacked" if stacked_count > 0 else ""))
    else:
        st.info(f"📊 Will run **{len(models_to_run)}** model variants: "
                f"{base_count} base models" +
                (f" + {stacked_count} stacked" if stacked_count > 0 else ""))

    st.markdown("---")

    # Ensemble Settings
    with st.expander("🎯 Ensemble Settings (Model Averaging)", expanded=False):
        st.caption("Combine multiple models per combination using MAPE-weighted averaging")

        col_ens1, col_ens2 = st.columns(2)

        with col_ens1:
            enable_ensemble = st.checkbox(
                "Enable Ensemble",
                value=False,
                help="Create weighted ensemble models by averaging coefficients across all models"
            )

            ensemble_weight_metric = st.selectbox(
                "Weighting Metric:",
                options=['MAPE Test', 'MAE Test'],
                index=0,
                help="Metric used for weighting models (lower MAPE/MAE = higher weight)"
            )

        with col_ens2:
            st.caption("Optional: Filter models before ensemble")

            use_r2_filter = st.checkbox("Filter by R² Test ≥", value=False)
            ensemble_r2_min = st.number_input(
                "Min R² Test:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not use_r2_filter
            ) if use_r2_filter else None

            use_mape_filter = st.checkbox("Filter by MAPE Test ≤", value=False)
            ensemble_mape_max = st.number_input(
                "Max MAPE Test (%):",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                disabled=not use_mape_filter
            ) if use_mape_filter else None

            use_mae_filter = st.checkbox("Filter by MAE Test ≤", value=False)
            ensemble_mae_max = st.number_input(
                "Max MAE Test:",
                min_value=0.0,
                value=100.0,
                step=10.0,
                disabled=not use_mae_filter
            ) if use_mae_filter else None

        # Sign-based filtering for ensemble
        st.markdown("---")
        st.caption("**🔍 Sign-Based Filtering**: Only include models where coefficients match expected signs")

        use_sign_filter = st.checkbox(
            "Enable Sign Filtering",
            value=False,
            help="Filter out models where coefficients don't match the expected positive/negative constraints"
        )

        if use_sign_filter:
            col_sign1, col_sign2 = st.columns(2)

            with col_sign1:
                ensemble_positive_features = st.multiselect(
                    "Must be Positive (≥ 0):",
                    options=selected_predictors_working,
                    default=positive_constraints if positive_constraints else [],
                    help="Only include models where these features have positive coefficients"
                )

            with col_sign2:
                available_for_negative_ensemble = [p for p in selected_predictors_working if p not in ensemble_positive_features]
                ensemble_negative_features = st.multiselect(
                    "Must be Negative (≤ 0):",
                    options=available_for_negative_ensemble,
                    default=[c for c in negative_constraints if c in available_for_negative_ensemble] if negative_constraints else [],
                    help="Only include models where these features have negative coefficients"
                )

            if ensemble_positive_features or ensemble_negative_features:
                st.info(f"📌 Will filter models: {len(ensemble_positive_features)} features must be ≥0, {len(ensemble_negative_features)} features must be ≤0")
        else:
            ensemble_positive_features = None
            ensemble_negative_features = None

        if enable_ensemble:
            st.info("📌 Ensemble will create one weighted model per combination by averaging all individual model coefficients")

    st.markdown("---")

    # Run button
    if not selected_predictors_working:
        st.error("❌ Please select at least one predictor")
        return

    if st.button("▶️ RUN MODELS", type="primary", use_container_width=True):
        # ═══════════════════════════════════════════════════════════════
        # Initialize stores (clear old data from previous runs)
        # ═══════════════════════════════════════════════════════════════
        st.session_state.rls_comparison_store = []
        st.session_state.beta_history_store = {}

        # Store flags in session state so they're available in display section
        st.session_state.enable_rls_flag = enable_rls
        st.session_state.enable_ensemble_flag = enable_ensemble

        # Use the working dataframe (either residualized or original)
        df_to_use = df_working
        predictors_to_use = selected_predictors_working

        # Show workflow info
        if enable_ensemble and enable_rls:
            st.info("🔄 **Workflow**: CV on all models → Create ensemble → Apply RLS to ensemble only (train → warmup → holdout)")
        elif enable_ensemble:
            st.info("🔄 **Workflow**: CV on all models → Create weighted ensemble")
        elif enable_rls:
            st.info("🔄 **Workflow**: CV on all models → Apply RLS to each model individually (train → warmup → holdout)")

        with st.spinner("Running models..."):
            st.session_state.optimized_lambdas = None
            results_df, predictions_df, lambda_df, ensemble_df = run_model_pipeline(
                df=df_to_use,
                grouping_keys=selected_grouping_keys,
                X_columns=predictors_to_use,
                target_col=target_col,
                k_folds=k_folds,
                std_cols=standardize_cols,
                models_dict=models_to_run,
                use_stacked=use_stacked,
                stacking_keys=stacking_keys,
                filter_keys_for_stacking=filter_keys_for_stacking,
                log_transform_y=log_transform_y,
                min_y_share_pct=min_y_share_pct,
                enable_rls=enable_rls,
                holdout_weeks=holdout_weeks,
                warmup_weeks=warmup_weeks,
                positive_constraints=positive_constraints,
                negative_constraints=negative_constraints,
                rls_lambda_candidates=DEFAULT_RLS_LAMBDA_GRID if enable_rls else None,
                enable_ensemble=enable_ensemble,
                ensemble_weight_metric=ensemble_weight_metric,
                ensemble_filter_r2_min=ensemble_r2_min if use_r2_filter else None,
                ensemble_filter_mape_max=ensemble_mape_max if use_mape_filter else None,
                ensemble_filter_mae_max=ensemble_mae_max if use_mae_filter else None,
                ensemble_filter_positive_features=ensemble_positive_features if use_sign_filter else None,
                ensemble_filter_negative_features=ensemble_negative_features if use_sign_filter else None
            )

            if results_df is not None:
                st.session_state.results = results_df
                st.session_state.predictions = predictions_df
                st.session_state.optimized_lambdas = lambda_df
                st.session_state.ensemble_results = ensemble_df

                # Show success message with store counts
                if enable_rls:
                    n_comparisons = len(st.session_state.get('rls_comparison_store', []))
                    n_beta_histories = len(st.session_state.get('beta_history_store', {}))
                    if n_comparisons > 0 or n_beta_histories > 0:
                        st.success(f"✅ RLS Analysis Complete: {n_comparisons} comparisons, {n_beta_histories} beta histories stored")


    # Display results
    if st.session_state.results is not None:
        st.markdown("---")
        st.header("📈 Results")

        results_df = st.session_state.results

        st.subheader("📋 Detailed Results (Folds & Aggregates)")
        st.caption("Includes per-fold rows, holdout metrics, coefficients, and feature means")
        st.dataframe(results_df, use_container_width=True, height=min(600, 100 + len(results_df) * 20))

        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_results,
            file_name="modeling_results.csv",
            mime="text/csv",
            key='download_model_results'
        )

        optimized_df = st.session_state.get('optimized_lambdas')
        if optimized_df is not None and not optimized_df.empty:
            with st.expander("🔧 Optimized RLS forgetting factors", expanded=False):
                st.caption("Best lambda chosen per model/group combination based on holdout MAE")
                st.dataframe(optimized_df, use_container_width=True, height=min(400, 100 + len(optimized_df) * 22))
                csv_lambdas = optimized_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Lambda Table",
                    data=csv_lambdas,
                    file_name="optimized_rls_lambdas.csv",
                    mime="text/csv",
                    key='download_lambda_results'
                )

        # Display ensemble results
        ensemble_df = st.session_state.get('ensemble_results')
        if ensemble_df is not None and not ensemble_df.empty:
            with st.expander("🎯 Ensemble Models (Weighted Average)", expanded=True):
                st.caption("Weighted ensemble models created by averaging coefficients across individual models")

                # Show key metrics
                st.markdown("### 📊 Ensemble Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Combinations", len(ensemble_df))

                with col2:
                    avg_models = ensemble_df['Num_Models'].mean() if 'Num_Models' in ensemble_df.columns else 0
                    st.metric("Avg Models per Combination", f"{avg_models:.1f}")

                with col3:
                    avg_concentration = ensemble_df['Weight_Concentration'].mean() if 'Weight_Concentration' in ensemble_df.columns else 0
                    st.metric("Avg Weight Concentration", f"{avg_concentration:.2%}")

                st.markdown("---")

                # Display full ensemble data
                st.dataframe(ensemble_df, use_container_width=True, height=min(400, 100 + len(ensemble_df) * 22))

                csv_ensemble = ensemble_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Ensemble Results",
                    data=csv_ensemble,
                    file_name="ensemble_models.csv",
                    mime="text/csv",
                    key='download_ensemble_results'
                )

        # Time series visualization of predictions
        if st.session_state.predictions is not None:
            st.subheader("📈 Predicted vs Actual Over Time")

            predictions_df = st.session_state.predictions.copy()

            # Check if date column exists
            date_columns = [col for col in predictions_df.columns if 'date' in col.lower() or 'time' in col.lower() or col.lower() in ['week', 'month', 'year', 'period']]

            if date_columns:
                col1, col2 = st.columns(2)

                with col1:
                    # Date column selection
                    selected_date_col = st.selectbox(
                        "Select Date Column:",
                        options=date_columns,
                        index=0,
                        key='date_col_selector'
                    )

                with col2:
                    # Get grouping columns (exclude prediction-specific columns and numeric/data columns)
                    grouping_cols = [col for col in predictions_df.columns
                                if col not in ['Actual', 'Predicted', 'Model', 'Fold', selected_date_col]]

                    # Filter to keep only product/category identifier columns (not numeric data)
                    product_cols = []
                    for col in grouping_cols:
                        # Keep only columns that look like identifiers (not pure numeric data)
                        if predictions_df[col].dtype == 'object' or predictions_df[col].nunique() < 100:
                            product_cols.append(col)

                    # If we have product columns, use them; otherwise use first grouping column
                    if product_cols:
                        # Use only the first column (usually product name)
                        predictions_df['_group_id'] = predictions_df[product_cols[0]].astype(str)
                        unique_groups = sorted(predictions_df['_group_id'].unique())

                        selected_group = st.selectbox(
                            "Select Product:",
                            options=unique_groups,
                            index=0,
                            key='group_selector'
                        )
                    elif grouping_cols:
                        predictions_df['_group_id'] = predictions_df[grouping_cols[0]].astype(str)
                        unique_groups = sorted(predictions_df['_group_id'].unique())

                        selected_group = st.selectbox(
                            "Select Product:",
                            options=unique_groups,
                            index=0,
                            key='group_selector'
                        )
                    else:
                        selected_group = None

                # Convert date column
                try:
                    predictions_df[selected_date_col] = pd.to_datetime(predictions_df[selected_date_col])
                except:
                    pass  # Keep as is if conversion fails

                # Filter by selected group
                if selected_group is not None:
                    group_data = predictions_df[predictions_df['_group_id'] == selected_group].copy()
                else:
                    group_data = predictions_df.copy()

                # Aggregate predictions by date and model (average across folds)
                agg_data = group_data.groupby([selected_date_col, 'Model']).agg({
                    'Actual': 'mean',
                    'Predicted': 'mean'
                }).reset_index()

                agg_data = agg_data.sort_values(selected_date_col)

                # Get unique models
                unique_models = sorted(agg_data['Model'].unique())

                # Create single plot with all models
                fig = go.Figure()

                # Color palette for models
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                # Add actual values (only once - all models have same actuals)
                actual_data = agg_data.drop_duplicates(subset=[selected_date_col])[
                    [selected_date_col, 'Actual']
                ].sort_values(selected_date_col)

                fig.add_trace(go.Scatter(
                    x=actual_data[selected_date_col],
                    y=actual_data['Actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=3),
                    marker=dict(size=8, symbol='circle')
                ))

                # Add predicted values for each model
                metrics_list = []
                for idx, model in enumerate(unique_models):
                    model_data = agg_data[agg_data['Model'] == model].copy()
                    model_data = model_data.sort_values(selected_date_col)

                    # Calculate metrics
                    r2 = r2_score(model_data['Actual'], model_data['Predicted'])
                    mae = np.mean(np.abs(model_data['Actual'] - model_data['Predicted']))
                    rmse = np.sqrt(np.mean((model_data['Actual'] - model_data['Predicted'])**2))
                    mape = safe_mape(model_data['Actual'], model_data['Predicted'])

                    metrics_list.append({
                        'Model': model,
                        'R²': r2,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })

                    # Get color
                    color = colors[idx % len(colors)]

                    # Check if this is an RLS model with holdout predictions
                    is_rls_model = '+ RLS' in model

                    if is_rls_model and 'Fold' in predictions_df.columns:
                        # Get holdout dates
                        model_preds = predictions_df[predictions_df['Model'] == model].copy()
                        holdout_dates = model_preds[model_preds['Fold'] == 'Holdout'][selected_date_col].unique()

                        if len(holdout_dates) > 0:
                            # Create marker properties: circles for training, diamonds for holdout
                            model_data['marker_symbol'] = model_data[selected_date_col].apply(
                                lambda x: 'diamond' if x in holdout_dates else 'circle'
                            )
                            model_data['marker_size'] = model_data[selected_date_col].apply(
                                lambda x: 12 if x in holdout_dates else 6
                            )

                            # Single continuous trace with variable markers
                            fig.add_trace(go.Scatter(
                                x=model_data[selected_date_col],
                                y=model_data['Predicted'],
                                mode='lines+markers',
                                name=f'{model} (R²={r2:.3f})',
                                line=dict(color=color, width=2, dash='dash'),
                                marker=dict(
                                    size=model_data['marker_size'].tolist(),
                                    symbol=model_data['marker_symbol'].tolist(),
                                    color=color,
                                    line=dict(width=1, color='white')
                                ),
                                hovertemplate=f'<b>{model}</b><br>Date: %{{x}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                            ))
                        else:
                            # No holdout data found, use regular display
                            fig.add_trace(go.Scatter(
                                x=model_data[selected_date_col],
                                y=model_data['Predicted'],
                                mode='lines+markers',
                                name=f'{model} (R²={r2:.3f})',
                                line=dict(color=color, width=2, dash='dash'),
                                marker=dict(size=6, symbol='circle'),
                                hovertemplate=f'<b>{model}</b><br>Date: %{{x}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                            ))
                    else:
                        # Non-RLS models: regular circles
                        fig.add_trace(go.Scatter(
                            x=model_data[selected_date_col],
                            y=model_data['Predicted'],
                            mode='lines+markers',
                            name=f'{model} (R²={r2:.3f})',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=6, symbol='circle'),
                            hovertemplate=f'<b>{model}</b><br>Date: %{{x}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                        ))


                # Update layout
                title_text = f"<b>{selected_group if selected_group else 'All Data'}</b>"

                fig.update_layout(
                    title=title_text,
                    xaxis_title=selected_date_col,
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(255,255,255,0.9)"
                    ),
                    margin=dict(t=80, b=60, l=60, r=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show summary table
                st.subheader("📊 Model Performance Summary")
                if metrics_list:
                    summary_df = pd.DataFrame(metrics_list)
                    summary_df['R²'] = summary_df['R²'].apply(lambda x: f"{x:.4f}")
                    summary_df['MAE'] = summary_df['MAE'].apply(lambda x: f"{x:.2f}")
                    summary_df['RMSE'] = summary_df['RMSE'].apply(lambda x: f"{x:.2f}")
                    summary_df['MAPE'] = summary_df['MAPE'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            else:
                st.info("💡 No date/time column found in predictions. Please ensure your data has a date column for time series visualization.")

        # ═══════════════════════════════════════════════════════════════════════════
        # RLS ANALYSIS DASHBOARD (independent section, always shown when RLS enabled)
        # ═══════════════════════════════════════════════════════════════════════════
        rls_enabled = st.session_state.get('enable_rls_flag', False)
        ensemble_enabled = st.session_state.get('enable_ensemble_flag', False)
        has_comparison = bool(st.session_state.get('rls_comparison_store'))
        has_beta_history = bool(st.session_state.get('beta_history_store'))

        # Debug info (you can remove this later)
        if rls_enabled:
            st.caption(f"🔍 Debug: RLS={rls_enabled}, Ensemble={ensemble_enabled}, Comparisons={len(st.session_state.get('rls_comparison_store', []))}, BetaHistories={len(st.session_state.get('beta_history_store', {}))}")

        if rls_enabled and (has_comparison or has_beta_history):
            st.markdown("---")
            st.header("🔬 RLS Analysis Dashboard")

            # Show info about what's being compared
            if ensemble_enabled:
                st.info("📌 **Ensemble + RLS Mode**: Comparing Static Ensemble (weeks 1-48) vs Ensemble + RLS (train 1-44 → warmup 45-48 → test 49-52)")
            else:
                st.info("📌 **Per-Model RLS Mode**: Comparing each model's static vs RLS performance on holdout period")

            # Create tabs
            tab1, tab2, tab3 = st.tabs([
                "📊 Static vs RLS Comparison",
                "📉 Prediction Visualization",
                "📈 Beta Evolution Over Time"
            ])

            # TAB 1: STATIC VS RLS COMPARISON
            with tab1:
                if st.session_state.get('rls_comparison_store'):
                    comparison_data = st.session_state.rls_comparison_store

                    st.subheader("📊 Performance Comparison Table")
                    if ensemble_enabled:
                        st.caption("📌 **Ensemble Comparison**: Static Ensemble (trained on weeks 1-48) vs Ensemble + RLS (trained 1-44, warmed 45-48, tested 49-52)")
                    else:
                        st.caption("📌 **Per-Model Comparison**: Each model's static vs RLS performance on holdout period")

                    comparison_df = pd.DataFrame([{
                        'Group': rec['Group'],
                        'Model': rec['Model'],
                        'R² Static': f"{rec['R2_Static']:.4f}",
                        'R² RLS': f"{rec['R2_RLS']:.4f}",
                        'MAE Static': f"{rec['MAE_Static']:.2f}",
                        'MAE RLS': f"{rec['MAE_RLS']:.2f}",
                        'MAE Improve %': f"{((rec['MAE_Static'] - rec['MAE_RLS']) / (rec['MAE_Static'] + 1e-6) * 100):.1f}%",
                        'RMSE Static': f"{rec['RMSE_Static']:.2f}",
                        'RMSE RLS': f"{rec['RMSE_RLS']:.2f}"
                    } for rec in comparison_data])

                    st.dataframe(comparison_df, use_container_width=True)

                    # Download button
                    csv_comparison = comparison_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Comparison Table",
                        csv_comparison,
                        "rls_vs_static.csv",
                        "text/csv",
                        key='download_comparison'
                    )

                    # Summary metrics
                    st.markdown("---")
                    st.subheader("📈 Overall Summary")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        avg_mae_static = np.mean([rec['MAE_Static'] for rec in comparison_data])
                        st.metric("Avg MAE Static", f"{avg_mae_static:.2f}")

                    with col2:
                        avg_mae_rls = np.mean([rec['MAE_RLS'] for rec in comparison_data])
                        improvement = ((avg_mae_static - avg_mae_rls) / avg_mae_static * 100) if avg_mae_static != 0 else 0
                        st.metric("Avg MAE RLS", f"{avg_mae_rls:.2f}", delta=f"{-improvement:.1f}%")

                    with col3:
                        wins = sum(1 for rec in comparison_data if rec['MAE_RLS'] < rec['MAE_Static'])
                        st.metric("RLS Wins", f"{wins}/{len(comparison_data)}")
                else:
                    st.info("No comparison data available. Run models with RLS enabled.")

            # TAB 2: PREDICTION VISUALIZATION
            with tab2:
                if st.session_state.get('rls_comparison_store'):
                    comparison_data = st.session_state.rls_comparison_store

                    st.subheader("📉 Predictions: Static vs RLS vs Actual")
                    st.caption("Visualize how well each approach predicted holdout weeks")

                    # Selector
                    comparison_options = [f"{rec['Group']} | {rec['Model']}" for rec in comparison_data]
                    selected_comparison = st.selectbox(
                        "Select Product/Brand + Model:",
                        options=comparison_options,
                        key='prediction_viz_selector'
                    )

                    selected_idx = comparison_options.index(selected_comparison)
                    selected_rec = comparison_data[selected_idx]

                    # Metrics cards
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("MAE Static", f"{selected_rec['MAE_Static']:.2f}")
                        st.metric("R² Static", f"{selected_rec['R2_Static']:.4f}")

                    with col2:
                        mae_delta = selected_rec['MAE_Static'] - selected_rec['MAE_RLS']
                        st.metric("MAE RLS", f"{selected_rec['MAE_RLS']:.2f}", delta=f"{-mae_delta:.2f}")
                        r2_delta = selected_rec['R2_RLS'] - selected_rec['R2_Static']
                        st.metric("R² RLS", f"{selected_rec['R2_RLS']:.4f}", delta=f"{r2_delta:.4f}")

                    with col3:
                        improvement = ((selected_rec['MAE_Static'] - selected_rec['MAE_RLS']) / (selected_rec['MAE_Static'] + 1e-6) * 100)
                        st.metric("MAE Improvement", f"{improvement:.1f}%")
                        winner = "🏆 RLS" if selected_rec['MAE_RLS'] < selected_rec['MAE_Static'] else "📊 Static"
                        st.metric("Winner", winner)

                    # Chart
                    fig_comp = go.Figure()

                    fig_comp.add_trace(go.Scatter(
                        x=selected_rec['Dates'],
                        y=selected_rec['Actuals'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='black', width=3),
                        marker=dict(size=12, symbol='circle')
                    ))

                    fig_comp.add_trace(go.Scatter(
                        x=selected_rec['Dates'],
                        y=selected_rec['Predictions_Static'],
                        mode='lines+markers',
                        name='Static (Frozen Betas)',
                        line=dict(color='blue', width=2, dash='dash'),
                        marker=dict(size=10, symbol='square')
                    ))

                    fig_comp.add_trace(go.Scatter(
                        x=selected_rec['Dates'],
                        y=selected_rec['Predictions_RLS'],
                        mode='lines+markers',
                        name='RLS (Adaptive Betas)',
                        line=dict(color='green', width=2),
                        marker=dict(size=10, symbol='diamond')
                    ))

                    fig_comp.update_layout(
                        title=f"Prediction Comparison: {selected_comparison}",
                        xaxis_title="Holdout Week",
                        yaxis_title="Target Value",
                        height=500,
                        hovermode='x unified',
                        showlegend=True
                    )

                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.info("No prediction data available.")

            # TAB 3: BETA EVOLUTION
            with tab3:
                if st.session_state.get('beta_history_store'):
                    beta_history_store = st.session_state.beta_history_store

                    if len(beta_history_store) > 0:
                        st.subheader("📈 Coefficient Evolution: Training → Warmup → Holdout")
                        st.caption("Complete beta adaptation timeline showing warmup stabilization and holdout testing")

                        # Selector
                        available_combos = sorted(list(beta_history_store.keys()))
                        selected_combo = st.selectbox(
                            "Select Product/Brand + Model:",
                            options=available_combos,
                            key='beta_time_combo'
                        )

                        beta_data = beta_history_store[selected_combo]
                        feature_names = beta_data['feature_names']
                        warmup_snapshots = beta_data.get('warmup_beta_snapshots', [])
                        holdout_snapshots = beta_data.get('holdout_beta_snapshots', [])
                        n_warmup = beta_data.get('n_warmup', 0)
                        n_holdout = beta_data.get('n_holdout', 0)

                        # Build complete timeline
                        all_snapshots = warmup_snapshots + holdout_snapshots[1:]  # Skip duplicate at boundary

                        # Time labels
                        time_labels = (
                            ["Initial (Week 44)"] +
                            [f"Warmup Week {i+1}" for i in range(n_warmup)] +
                            [f"Holdout Week {i+1}" for i in range(n_holdout)]
                        )

                        # Chart
                        fig_beta_time = go.Figure()
                        colors_beta = px.colors.qualitative.Set2

                        for i, feature in enumerate(feature_names):
                            values = [snapshot[i] for snapshot in all_snapshots]

                            fig_beta_time.add_trace(go.Scatter(
                                x=time_labels,
                                y=values,
                                mode='lines+markers',
                                name=feature,
                                line=dict(color=colors_beta[i % len(colors_beta)], width=2),
                                marker=dict(size=8),
                                hovertemplate=f'<b>{feature}</b><br>%{{x}}<br>Beta: %{{y:.4f}}<extra></extra>'
                            ))

                        # Add vertical lines to separate phases
                        fig_beta_time.add_vline(
                            x=n_warmup,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="Warmup Ends"
                        )

                        fig_beta_time.add_vline(
                            x=n_warmup + 0.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Holdout Testing Starts"
                        )

                        fig_beta_time.update_layout(
                            title=f"Beta Evolution: {selected_combo}",
                            xaxis_title="Time Period",
                            yaxis_title="Coefficient Value",
                            height=600,
                            hovermode='x unified',
                            showlegend=True
                        )

                        st.plotly_chart(fig_beta_time, use_container_width=True)

                        # Summary table with 3 phases
                        st.markdown("---")
                        st.subheader("📋 Beta Changes Across Phases")

                        initial_betas = all_snapshots[0]
                        after_warmup_betas = all_snapshots[n_warmup]
                        final_betas = all_snapshots[-1]

                        summary_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Initial (Week 44)': initial_betas,
                            'After Warmup (Week 48)': after_warmup_betas,
                            'Final (Week 52)': final_betas,
                            'Warmup Change': after_warmup_betas - initial_betas,
                            'Holdout Change': final_betas - after_warmup_betas,
                            'Total Change': final_betas - initial_betas
                        })

                        summary_df = summary_df.round(4)
                        st.dataframe(summary_df, use_container_width=True)
                    else:
                        st.info("No beta history available.")
                else:
                    st.info("Beta tracking not available. Enable RLS and run models.")


if __name__ == "__main__":
    main()
