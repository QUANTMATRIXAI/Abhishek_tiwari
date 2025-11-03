"""
Standalone Elasticity Modeling App
Single-page Streamlit application for running regression models
"""

"""
Standalone Elasticity Modeling App
Single-page Streamlit application for running regression models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import statsmodels.api as sm
import warnings
import time

warnings.filterwarnings('ignore')


warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CLASSES
# ═══════════════════════════════════════════════════════════════════════════

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
                except:
                    try:
                        self._model_result = mixed_model.fit(method='bfgs', reml=False)
                    except:
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
            from sklearn.linear_model import LinearRegression
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

# ═══════════════════════════════════════════════════════════════════════════
# RECURSIVE LEAST SQUARES MODEL
# ═══════════════════════════════════════════════════════════════════════════

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
        
        # ═══════════════════════════════════════════════════════════════
        # FIXED: Use provided betas WITHOUT running RLS updates
        # ═══════════════════════════════════════════════════════════════
        if initial_beta is not None and initial_intercept is not None:
            # Use the trained model's coefficients directly
            self.beta = np.concatenate([[initial_intercept], initial_beta])
            
            # DO NOT run RLS updates on training data!
            # The betas are already optimized from the trained model
            
        else:
            # Fallback: Initialize with OLS on first few samples
            init_size = min(10, len(X))
            X_init = X_with_intercept[:init_size]
            y_init = y[:init_size]
            
            try:
                XtX = X_init.T @ X_init
                Xty = X_init.T @ y_init
                beta_init = np.linalg.solve(XtX, Xty)
            except:
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
        
        # ═══════════════════════════════════════════════════════════════
        # Initialize covariance matrix P
        # ═══════════════════════════════════════════════════════════════
        try:
            XtX = X_with_intercept.T @ X_with_intercept
            n_params = XtX.shape[0]
            
            # CRITICAL FIX: Better P initialization for stability
            if initial_beta is not None and initial_intercept is not None:
                # Pre-trained model: Use VERY conservative P for minimal drift
                # P represents uncertainty - low uncertainty = small updates
                regularization = 1.0 * np.eye(n_params)  # Increased regularization
                self.P = np.linalg.inv(XtX / len(X) + regularization)
                
                # Scale down P to make updates more conservative
                self.P = self.P * 0.1  # Make updates 10x smaller
            else:
                # Fresh start: Use moderate P
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
        
        # CRITICAL FIX: Correct RLS update formula
        # K = (P @ x) / (λ + x.T @ P @ x)
        P_x = self.P @ x_t_col
        denominator = self.forgetting_factor + (x_t_col.T @ P_x)[0, 0]
        denominator = max(denominator, 1e-10)  # Numerical stability
        K = P_x / denominator
        
        # Update beta: beta = beta + K * error
        beta_update = (K * error).flatten()
        
        # FIXED: More reasonable clipping based on current beta magnitude
        max_update = max(10.0, np.abs(self.beta).max() * 0.5)  # Allow 50% change max
        if np.abs(beta_update).max() > max_update:
            beta_update = np.clip(beta_update, -max_update, max_update)
        
        self.beta += beta_update
        
        # FIXED: More reasonable beta clipping
        max_beta = 10000.0  # Increased from 1000
        if np.abs(self.beta).max() > max_beta:
            self.beta = np.clip(self.beta, -max_beta, max_beta)
        
        # Apply constraints BEFORE updating P (important!)
        self._apply_constraints()
        
        # CRITICAL FIX: Correct P matrix update formula
        # P_new = (1/λ) * (P - (P @ x @ x.T @ P) / (λ + x.T @ P @ x))
        # Simplified: P_new = (1/λ) * (P - K @ x.T @ P)
        self.P = (1.0 / self.forgetting_factor) * (self.P - K @ x_t_col.T @ self.P)
        
        # FIXED: Better numerical stability for P matrix
        # Ensure P remains symmetric and positive definite
        self.P = (self.P + self.P.T) / 2  # Force symmetry
        
        # Add small regularization to maintain positive definiteness
        min_eigenval = np.min(np.linalg.eigvalsh(self.P))
        if min_eigenval < 1e-6:
            self.P += (1e-6 - min_eigenval + 1e-8) * np.eye(len(self.P))
        
        # Prevent P from growing too large
        max_P_trace = 1000.0
        current_trace = np.trace(self.P)
        if current_trace > max_P_trace:
            self.P = self.P * (max_P_trace / current_trace)
        
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
    
    def get_diagnostics(self):
        """Return diagnostic information about RLS state"""
        return {
            'current_beta': self.beta.copy(),
            'P_trace': np.trace(self.P),
            'P_condition_number': np.linalg.cond(self.P),
            'P_min_eigenvalue': np.min(np.linalg.eigvalsh(self.P)),
            'P_max_eigenvalue': np.max(np.linalg.eigvalsh(self.P)),
            'beta_magnitude': np.linalg.norm(self.beta),
            'n_updates': len(self.beta_history_) - 1
        }


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
# Values closer to 1.0 = more stable, less adaptation
# Values closer to 0.95 = more adaptation, less stable
DEFAULT_RLS_LAMBDA_GRID = [0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0]


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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════

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


