"""
Custom Regression Models
Contains all custom model classes for elasticity modeling
"""

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression


class CustomConstrainedRidge(BaseEstimator, RegressorMixin):
    """Ridge regression with coefficient sign constraints"""

    def __init__(self, l2_penalty=0.1, learning_rate=0.001, iterations=10000,
                adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8,
                non_positive_features=None, non_negative_features=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Store as tuples for sklearn compatibility (immutable)
        self.non_positive_features = tuple(non_positive_features) if non_positive_features else ()
        self.non_negative_features = tuple(non_negative_features) if non_negative_features else ()

    def fit(self, X, Y, feature_names):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

        configured_non_positive = set(self.non_positive_features) if self.non_positive_features else set()
        configured_non_negative = set(self.non_negative_features) if self.non_negative_features else set()

        self._non_positive_feature_names = {name for name in feature_names if name in configured_non_positive}
        self._non_negative_feature_names = {name for name in feature_names if name in configured_non_negative}
        self._non_positive_indices = [i for i, name in enumerate(feature_names) if name in self._non_positive_feature_names]
        self._non_negative_indices = [i for i, name in enumerate(feature_names) if name in self._non_negative_feature_names]
        # Don't modify constructor parameters - use internal attributes only

        if self.adam:
            self.m_W = np.zeros(self.n)
            self.v_W = np.zeros(self.n)
            self.m_b = 0
            self.v_b = 0
            self.t = 0

        for _ in range(self.iterations):
            self.update_weights()

        self.intercept_ = self.b
        self.coef_ = self.W
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        grad_w = (-(2 * (self.X.T).dot(self.Y - Y_pred)) + 2 * self.l2_penalty * self.W) / self.m
        grad_b = -(2 / self.m) * np.sum(self.Y - Y_pred)

        if self.adam:
            self.t += 1
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_w
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_w ** 2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

            m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            self.W -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

        if getattr(self, '_non_positive_indices', []):
            self.W[self._non_positive_indices] = np.minimum(self.W[self._non_positive_indices], 0)
        if getattr(self, '_non_negative_indices', []):
            self.W[self._non_negative_indices] = np.maximum(self.W[self._non_negative_indices], 0)

    def predict(self, X):
        return X.dot(self.W) + self.b


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression with coefficient sign constraints"""

    def __init__(self, learning_rate=0.001, iterations=10000,
                adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8,
                non_positive_features=None, non_negative_features=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Store as tuples for sklearn compatibility (immutable)
        self.non_positive_features = tuple(non_positive_features) if non_positive_features else ()
        self.non_negative_features = tuple(non_negative_features) if non_negative_features else ()

    def fit(self, X, Y, feature_names):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

        configured_non_positive = set(self.non_positive_features) if self.non_positive_features else set()
        configured_non_negative = set(self.non_negative_features) if self.non_negative_features else set()

        self._non_positive_feature_names = {name for name in feature_names if name in configured_non_positive}
        self._non_negative_feature_names = {name for name in feature_names if name in configured_non_negative}
        self._non_positive_indices = [i for i, name in enumerate(feature_names) if name in self._non_positive_feature_names]
        self._non_negative_indices = [i for i, name in enumerate(feature_names) if name in self._non_negative_feature_names]
        # Don't modify constructor parameters - use internal attributes only

        if self.adam:
            self.m_W = np.zeros(self.n)
            self.v_W = np.zeros(self.n)
            self.m_b = 0
            self.v_b = 0
            self.t = 0

        for _ in range(self.iterations):
            self.update_weights()

        self.intercept_ = self.b
        self.coef_ = self.W
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = -(2 * self.X.T.dot(self.Y - Y_pred)) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        if self.adam:
            self.t += 1
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

            m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        if getattr(self, '_non_positive_indices', []):
            self.W[self._non_positive_indices] = np.minimum(self.W[self._non_positive_indices], 0)
        if getattr(self, '_non_negative_indices', []):
            self.W[self._non_negative_indices] = np.maximum(self.W[self._non_negative_indices], 0)

    def predict(self, X):
        return X.dot(self.W) + self.b


class StackedInteractionModel(BaseEstimator, RegressorMixin):
    """Stacked model with interaction terms for group-specific coefficients"""

    def __init__(self, base_model, group_keys, enforce_combined_constraints=False):
        self.base_model = base_model
        self.group_keys = group_keys
        self.enforce_combined_constraints = enforce_combined_constraints
        self.group_mapping = None
        self.feature_names = None
        self.fitted_model = None
        self.base_features_count = None

    def fit(self, X, y, feature_names=None, groups_df=None):
        self.feature_names = feature_names if feature_names is not None else (
            list(X.columns) if hasattr(X, 'columns') else [f"X{i}" for i in range(X.shape[1])]
        )
        self.base_features_count = len(self.feature_names)

        if groups_df is None:
            raise ValueError("groups_df is required for stacked models")

        if not self.group_keys:
            self.fitted_model = clone(self.base_model)
            if isinstance(self.fitted_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                self.fitted_model.fit(X, y, self.feature_names)
            else:
                self.fitted_model.fit(X, y)
            self.group_mapping = {}
            return self

        missing_keys = [k for k in self.group_keys if k not in groups_df.columns]
        if missing_keys:
            raise ValueError(f"Group keys {missing_keys} not found in groups_df")

        if len(self.group_keys) == 1:
            group_combinations = groups_df[self.group_keys[0]].astype(str)
        else:
            group_data = groups_df[self.group_keys].astype(str)
            group_combinations = group_data.apply(lambda row: "_".join(row), axis=1)

        unique_groups = sorted(group_combinations.unique())

        if len(unique_groups) == 1:
            self.fitted_model = clone(self.base_model)
            if isinstance(self.fitted_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                self.fitted_model.fit(X, y, self.feature_names)
            else:
                self.fitted_model.fit(X, y)
            self.group_mapping = {unique_groups[0]: 0}
            self.reference_group = unique_groups[0]
            return self

        self.group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
        self.reference_group = unique_groups[0]

        dummy_matrix = np.zeros((len(X), len(unique_groups) - 1))
        for i, group in enumerate(group_combinations):
            group_idx = self.group_mapping[group]
            if group_idx > 0:
                dummy_matrix[i, group_idx - 1] = 1

        X_array = X.values if hasattr(X, 'values') else X
        interaction_features = []
        interaction_names = []

        for j, feat_name in enumerate(self.feature_names):
            for k in range(len(unique_groups) - 1):
                interaction = X_array[:, j] * dummy_matrix[:, k]
                interaction_features.append(interaction)
                group_name = unique_groups[k + 1]
                interaction_names.append(f"{feat_name}*{group_name}")

        X_stacked = np.hstack([
            X_array,
            dummy_matrix,
            np.column_stack(interaction_features) if interaction_features else np.empty((len(X), 0))
        ])

        dummy_names = [f"dummy_{unique_groups[i+1]}" for i in range(len(unique_groups) - 1)]
        self.all_feature_names = self.feature_names + dummy_names + interaction_names

        if isinstance(self.base_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
            self.fitted_model = self._fit_with_combined_constraints(X_stacked, y, unique_groups)
        else:
            self.fitted_model = clone(self.base_model)
            self.fitted_model.fit(X_stacked, y)

        return self

    def _fit_with_combined_constraints(self, X_stacked, y, unique_groups):
        model = clone(self.base_model)

        if self.enforce_combined_constraints:
            model._parent_stacked = self
            model._unique_groups = unique_groups

        if isinstance(model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
            model.fit(X_stacked, y, self.all_feature_names)
        else:
            model.fit(X_stacked, y)

        try:
            if self.enforce_combined_constraints and isinstance(model, CustomConstrainedRidge):
                parent = self
                n_base = parent.base_features_count
                n_groups = len(unique_groups)

                negative_feature_names = set(getattr(model, '_non_positive_feature_names', set()))
                positive_feature_names = set(getattr(model, '_non_negative_feature_names', set()))

                for g_idx in range(n_groups):
                    for f_idx, feat_name in enumerate(parent.feature_names[:n_base]):
                        combined_coef = model.W[f_idx]
                        interaction_idx = None

                        if g_idx > 0:
                            interaction_idx = n_base + (n_groups - 1) + (g_idx - 1) * n_base + f_idx
                            if interaction_idx < len(model.W):
                                combined_coef += model.W[interaction_idx]

                        if feat_name in negative_feature_names:
                            if combined_coef > 0:
                                if g_idx == 0:
                                    model.W[f_idx] = 0
                                else:
                                    correction = -combined_coef / 2
                                    model.W[f_idx] += correction
                                    if interaction_idx is not None and interaction_idx < len(model.W):
                                        model.W[interaction_idx] += correction

                        elif feat_name in positive_feature_names:
                            if combined_coef < 0:
                                if g_idx == 0:
                                    model.W[f_idx] = 0
                                else:
                                    correction = -combined_coef / 2
                                    model.W[f_idx] += correction
                                    if interaction_idx is not None and interaction_idx < len(model.W):
                                        model.W[interaction_idx] += correction
        except Exception:
            pass

        return model

    def predict(self, X, groups_df=None):
        if groups_df is None:
            raise ValueError("groups_df is required for prediction")

        if not self.group_keys or not self.group_mapping:
            return self.fitted_model.predict(X)

        if len(self.group_mapping) == 1:
            return self.fitted_model.predict(X)

        if len(self.group_keys) == 1:
            group_combinations = groups_df[self.group_keys[0]].astype(str)
        else:
            group_data = groups_df[self.group_keys].astype(str)
            group_combinations = group_data.apply(lambda row: "_".join(row), axis=1)

        dummy_matrix = np.zeros((len(X), len(self.group_mapping) - 1))
        for i, group in enumerate(group_combinations):
            if group in self.group_mapping:
                group_idx = self.group_mapping[group]
                if group_idx > 0:
                    dummy_matrix[i, group_idx - 1] = 1

        X_array = X.values if hasattr(X, 'values') else X
        interaction_features = []

        for j in range(X_array.shape[1]):
            for k in range(len(self.group_mapping) - 1):
                interaction = X_array[:, j] * dummy_matrix[:, k]
                interaction_features.append(interaction)

        X_stacked = np.hstack([
            X_array,
            dummy_matrix,
            np.column_stack(interaction_features) if interaction_features else np.empty((len(X), 0))
        ])

        return self.fitted_model.predict(X_stacked)

    def get_group_coefficients(self):
        if not hasattr(self.fitted_model, 'coef_'):
            return None

        if not self.group_keys or not self.group_mapping:
            return {
                'base': {
                    'intercept': self.fitted_model.intercept_,
                    'coefficients': dict(zip(self.feature_names, self.fitted_model.coef_[:self.base_features_count]))
                }
            }

        if len(self.group_mapping) == 1:
            group_name = list(self.group_mapping.keys())[0]
            return {
                group_name: {
                    'intercept': self.fitted_model.intercept_,
                    'coefficients': dict(zip(self.feature_names, self.fitted_model.coef_[:self.base_features_count]))
                }
            }

        coef_dict = {}
        n_features = len(self.feature_names)
        n_groups = len(self.group_mapping)

        sorted_groups = sorted(self.group_mapping.keys(), key=lambda x: self.group_mapping[x])

        for group_idx, group_name in enumerate(sorted_groups):
            combined_coefs = {}
            combined_intercept = self.fitted_model.intercept_

            if group_idx > 0:
                dummy_idx = n_features + group_idx - 1
                if dummy_idx < len(self.fitted_model.coef_):
                    combined_intercept += self.fitted_model.coef_[dummy_idx]

            for j, feat_name in enumerate(self.feature_names):
                base_coef = self.fitted_model.coef_[j]

                if group_idx > 0:
                    interaction_idx = n_features + (n_groups - 1) + (group_idx - 1) * n_features + j
                    if interaction_idx < len(self.fitted_model.coef_):
                        interaction_coef = self.fitted_model.coef_[interaction_idx]
                        combined_coefs[feat_name] = base_coef + interaction_coef
                    else:
                        combined_coefs[feat_name] = base_coef
                else:
                    combined_coefs[feat_name] = base_coef

            coef_dict[group_name] = {
                'intercept': combined_intercept,
                'coefficients': combined_coefs
            }

        return coef_dict


class StatsMixedEffectsModel(BaseEstimator, RegressorMixin):
    """Wrapper for statsmodels MixedLM"""

    def __init__(self, group_col='Brand', min_group_size=3):
        self.group_col = group_col
        self.min_group_size = min_group_size
        self._model_result = None
        self.fixed_coef_ = None
        self.intercept_ = 0.0
        self.random_effects_dict_ = {}
        self.feature_names_ = None
        self._fallback_model = None
        self._use_fallback = False

    def fit(self, X, y, groups):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

        self.feature_names_ = list(X.columns)

        groups_series = pd.Series(groups.values if hasattr(groups, 'values') else groups)
        group_counts = groups_series.value_counts()

        valid_groups = group_counts[group_counts >= self.min_group_size].index
        valid_mask = groups_series.isin(valid_groups)

        n_filtered = len(group_counts) - len(valid_groups)
        if n_filtered > 0:
            st.caption(f"Filtered {n_filtered} groups with < {self.min_group_size} observations")

        if valid_mask.sum() < len(X) * 0.5:
            self._use_fallback = True

        try:
            if not self._use_fallback and len(valid_groups) > 1:
                X_valid = X[valid_mask]
                y_valid = y[valid_mask] if hasattr(y, '__getitem__') else y[valid_mask]
                groups_valid = groups_series[valid_mask]

                X_with_const = sm.add_constant(X_valid, has_constant='add')

                mixed_model = sm.MixedLM(
                    endog=y_valid,
                    exog=X_with_const,
                    groups=groups_valid.values,
                    exog_re=None
                )

                try:
                    self._model_result = mixed_model.fit(method='lbfgs', reml=False)
                except (np.linalg.LinAlgError, ValueError):
                    try:
                        self._model_result = mixed_model.fit(method='bfgs', reml=False)
                    except (np.linalg.LinAlgError, ValueError):
                        self._model_result = mixed_model.fit(method='powell', reml=False)

                params = self._model_result.params
                self.intercept_ = params['const'] if 'const' in params else 0.0
                self.fixed_coef_ = params.drop('const').values

                self.random_effects_dict_ = {
                    group: effects.values[0]
                    for group, effects in self._model_result.random_effects.items()
                }

                for group in group_counts.index:
                    if group not in self.random_effects_dict_:
                        self.random_effects_dict_[group] = 0.0

            else:
                self._use_fallback = True

        except Exception as e:
            st.warning(f"Mixed effects failed: {str(e)}. Using fallback.")
            self._use_fallback = True

        if self._use_fallback:
            self._fallback_model = LinearRegression()
            self._fallback_model.fit(X, y)
            self.intercept_ = self._fallback_model.intercept_
            self.fixed_coef_ = self._fallback_model.coef_
            self.random_effects_dict_ = {group: 0.0 for group in group_counts.index}

        self.coef_ = self.fixed_coef_

        return self

    def predict(self, X, groups=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        if self._use_fallback and self._fallback_model is not None:
            return self._fallback_model.predict(X)
        else:
            X_with_const = sm.add_constant(X, has_constant='add')
            y_pred_fixed = X_with_const.values @ np.concatenate([[self.intercept_], self.fixed_coef_])

            if groups is None:
                return y_pred_fixed
            else:
                y_pred = y_pred_fixed.copy()
                groups_array = groups.values if hasattr(groups, 'values') else groups

                for i, group in enumerate(groups_array):
                    if group in self.random_effects_dict_:
                        y_pred[i] += self.random_effects_dict_[group]

                return y_pred


class RecursiveLeastSquares(BaseEstimator, RegressorMixin):
    """
    Recursive Least Squares with forgetting factor and constraints

    Parameters:
    -----------
    forgetting_factor : float, default=0.99
        Forgetting factor (0 < λ ≤ 1). Lower values give more weight to recent data.
        Common values: 0.95-0.99
    initial_P : float, default=1.0
        Initial covariance matrix scaling (P = initial_P * I)
        For standardized features, 1-100 is appropriate
    non_negative_features : list, default=None
        Feature names that must have non-negative (≥0) coefficients
    non_positive_features : list, default=None
        Feature names that must have non-positive (≤0) coefficients
    """
    def __init__(self, forgetting_factor=0.99, initial_P=1.0,
                 non_negative_features=None, non_positive_features=None):
        self.forgetting_factor = forgetting_factor
        self.initial_P = initial_P
        self.non_negative_features = non_negative_features or []
        self.non_positive_features = non_positive_features or []

        self.beta = None
        self.coef_ = None
        self.intercept_ = 0.0
        self.P = None
        self.n_features_ = None
        self.beta_history_ = []
        self.prediction_history_ = []
        self.feature_names_ = None
        self._non_negative_indices = []
        self._non_positive_indices = []

    def fit(self, X, y, feature_names=None, initial_beta=None, initial_intercept=None):
        """
        Initialize RLS with coefficients from another model (preferred) or OLS fallback
        """
        # Store feature names for constraint mapping
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names_ = feature_names

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        self.n_features_ = X.shape[1]

        # Map constraint feature names to indices (1-indexed because of intercept)
        if self.feature_names_:
            self._non_negative_indices = [
                i + 1 for i, name in enumerate(self.feature_names_)
                if name in self.non_negative_features
            ]
            self._non_positive_indices = [
                i + 1 for i, name in enumerate(self.feature_names_)
                if name in self.non_positive_features
            ]

        # Use provided betas WITHOUT running RLS updates
        if initial_beta is not None and initial_intercept is not None:
            # Use the trained model's coefficients directly
            self.beta = np.concatenate([[initial_intercept], initial_beta])

        else:
            # Fallback: Initialize with OLS on first few samples
            init_size = min(10, len(X))
            X_init = X_with_intercept[:init_size]
            y_init = y[:init_size]

            try:
                XtX = X_init.T @ X_init
                Xty = X_init.T @ y_init
                beta_init = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                beta_init = np.linalg.pinv(X_init) @ y_init

            self.beta = beta_init.copy()

            # Only run RLS updates if we used OLS initialization
            for i in range(init_size, len(X)):
                x_t = X_with_intercept[i]
                y_t = y[i]
                self._update(x_t, y_t)

        # Apply constraints to initial beta
        self._apply_constraints()

        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

        # Initialize covariance matrix P
        try:
            XtX = X_with_intercept.T @ X_with_intercept
            n_params = XtX.shape[0]

            # For pre-trained models, use smaller P for stability
            if initial_beta is not None and initial_intercept is not None:
                # Pre-trained model: Use inverse of data covariance for more stable updates
                regularization = 0.01 * np.eye(n_params)
                self.P = np.linalg.inv(XtX / len(X) + regularization)
            else:
                # Fresh start: Use larger P for faster learning
                self.P = self.initial_P * np.eye(n_params)

        except np.linalg.LinAlgError:
            self.P = self.initial_P * np.eye(len(self.beta))

        # Store initial state
        self.beta_history_ = [self.beta.copy()]

        return self


    def _apply_constraints(self):
        """Apply non-negative and non-positive constraints to beta coefficients"""
        for idx in self._non_negative_indices:
            if self.beta[idx] < 0:
                self.beta[idx] = 0.0

        for idx in self._non_positive_indices:
            if self.beta[idx] > 0:
                self.beta[idx] = 0.0

    def _update(self, x_t, y_t):
        """
        Internal method to update beta with a single observation

        Parameters:
        -----------
        x_t : np.ndarray
            Single observation with intercept [1, x1, x2, ...]
        y_t : float
            Target value for this observation
        """
        x_t_col = x_t.reshape(-1, 1)

        # Prediction error
        y_pred = x_t @ self.beta
        error = y_t - y_pred

        # RLS update: K = P @ x / (λ + x.T @ P @ x)
        P_x = self.P @ x_t_col
        denominator = self.forgetting_factor + (x_t_col.T @ self.P @ x_t_col)[0, 0]
        denominator = max(denominator, 1e-8)  # Numerical stability
        K = P_x / denominator

        # Update beta: beta = beta + K * error
        beta_update = (K * error).flatten()

        # Clip extremely large updates
        max_update = 10.0
        if np.abs(beta_update).max() > max_update:
            beta_update = np.clip(beta_update, -max_update, max_update)

        self.beta += beta_update

        # Clip beta values to prevent explosion
        max_beta = 1000.0
        if np.abs(self.beta).max() > max_beta:
            self.beta = np.clip(self.beta, -max_beta, max_beta)

        # Apply constraints
        self._apply_constraints()

        # Update P matrix: P_new = (1/λ) * (P - K @ x.T @ P)
        P_new = (1.0 / self.forgetting_factor) * self.P
        P_new = P_new - K @ (x_t_col.T @ P_new)

        # Numerical stability
        max_P_value = 100.0
        P_new = np.clip(P_new, -max_P_value, max_P_value)
        P_new = P_new + 0.01 * np.eye(len(P_new))
        self.P = P_new

        # Track history
        self.beta_history_.append(self.beta.copy())
        self.prediction_history_.append(y_pred)

        # Update intercept and coefficients
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

    def update(self, X, y):
        """
        Update RLS coefficients with new observation(s) without returning prediction
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        for i in range(len(X)):
            x_t = np.concatenate([[1], X[i]])
            y_t = y[i]
            self._update(x_t, y_t)

    def predict(self, X):
        """Predict using current beta estimates"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        return X_with_intercept @ self.beta

    def update_and_predict(self, X, y):
        """
        Update model with new data and return predictions
        Returns predictions BEFORE updating (true out-of-sample)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        predictions = []

        for i in range(len(X)):
            x_t = np.concatenate([[1], X[i]])
            y_pred = x_t @ self.beta
            predictions.append(y_pred)

            # Update with actual observation
            self._update(x_t, y[i])

        return np.array(predictions)

    def get_beta_history(self):
        """Return history of beta estimates"""
        return np.array(self.beta_history_)
