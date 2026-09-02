"""
Model Selection Framework.

Implementation of model architectures and selection framework for both
classification and regression tasks.

The model pool contains linear models, XGBoost, RandomForest, and MLP
configurations for classification and regression.

Pool size: K = 65 per task type (linear: 15, XGBoost: 20,
RandomForest: 15, MLP: 15),
The pool contains 65 configurations per task type.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score, log_loss, mean_squared_error,
    mean_absolute_error, r2_score
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is required by the model pool (XGBoost/RF/MLP). "
        "Install it with `pip install xgboost`."
    ) from exc

from .utils import RANDOM_SEED


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    name: str
    model_class: Any
    params: Dict[str, Any]
    family: str = "unknown"  # 'xgboost' | 'random_forest' | 'mlp' - used by LOFO splits


class ModelSelectionFramework:
    """
    Complete framework for model selection using synthetic data.

    The model pool H = F_linear ⊔ F_xgb ⊔ F_rf ⊔ F_mlp
    (four families, K=65 total)
    is the controlled substrate for:
      - WP1 Random vs. LOFO (leave-one-family-out) transfer experiments,
      - WP2 calibration-set (Hcal) selection strategies over loss profiles.
    """

    # Fixed pool sizes per family, per the research plan (Section 3.2).
    N_LINEAR = 15
    N_XGBOOST = 20
    N_RANDOM_FOREST = 15
    N_MLP = 15

    FAMILIES = ("linear", "xgboost", "random_forest", "mlp")

    def __init__(self, task_type: str = 'classification', loss_type: str = None):
        """
        Initialize the model selection framework.

        Args:
            task_type: 'classification' or 'regression'
            loss_type: Loss type for optimization. If None, auto-selects:
                       - 'accuracy' for classification
                       - 'mse' for regression
        """
        self.task_type = task_type

        if loss_type is None:
            loss_type = 'accuracy' if task_type == 'classification' else 'mse'
        self.loss_type = loss_type

        self.trained_models = []
        self.results = defaultdict(list)

    # ------------------------------------------------------------------
    # Architecture pool
    # ------------------------------------------------------------------

    def get_model_architectures(self) -> List[ModelConfig]:
        """Get the K=65 model pool (linear 15 + XGBoost 20 + RF 15 + MLP 15)."""
        if self.task_type == 'regression':
            return (
                self._get_linear_architectures_regression()
                + self._get_xgboost_architectures_regression()
                + self._get_random_forest_architectures_regression()
                + self._get_mlp_architectures_regression()
            )
        return (
            self._get_linear_architectures_classification()
            + self._get_xgboost_architectures_classification()
            + self._get_random_forest_architectures_classification()
            + self._get_mlp_architectures_classification()
        )

    def get_architectures_by_family(self, family: str) -> List[ModelConfig]:
        """Return only the architectures belonging to one family (for LOFO splits)."""
        if family not in self.FAMILIES:
            raise ValueError(f"Unknown family '{family}'. Expected one of {self.FAMILIES}.")
        return [a for a in self.get_model_architectures() if a.family == family]

    # ------------------------------------------------------------------
    # Linear models
    # ------------------------------------------------------------------

    def _get_linear_architectures_classification(self) -> List[ModelConfig]:
        """Return 15 deterministic linear classification configurations."""
        configs = [
            ("LogReg_L2_C0p01", LogisticRegression, {"C": 0.01, "penalty": "l2", "solver": "lbfgs"}),
            ("LogReg_L2_C0p1", LogisticRegression, {"C": 0.1, "penalty": "l2", "solver": "lbfgs"}),
            ("LogReg_L2_C1", LogisticRegression, {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}),
            ("LogReg_L2_C10", LogisticRegression, {"C": 10.0, "penalty": "l2", "solver": "lbfgs"}),
            ("LogReg_L1_C0p1", LogisticRegression, {"C": 0.1, "penalty": "l1", "solver": "liblinear"}),
            ("LogReg_L1_C1", LogisticRegression, {"C": 1.0, "penalty": "l1", "solver": "liblinear"}),
            ("LogReg_L1_C10", LogisticRegression, {"C": 10.0, "penalty": "l1", "solver": "liblinear"}),
            ("LogReg_Elastic_C0p1", LogisticRegression, {"C": 0.1, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.25}),
            ("LogReg_Elastic_C1", LogisticRegression, {"C": 1.0, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5}),
            ("LogReg_Elastic_C10", LogisticRegression, {"C": 10.0, "penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.75}),
            ("SGD_LogLoss_Alpha1e-5", SGDClassifier, {"loss": "log_loss", "alpha": 1e-5, "penalty": "l2"}),
            ("SGD_LogLoss_Alpha1e-4", SGDClassifier, {"loss": "log_loss", "alpha": 1e-4, "penalty": "l2"}),
            ("SGD_LogLoss_Alpha1e-3", SGDClassifier, {"loss": "log_loss", "alpha": 1e-3, "penalty": "l2"}),
            ("SGD_LogLoss_L1", SGDClassifier, {"loss": "log_loss", "alpha": 1e-4, "penalty": "l1"}),
            ("SGD_LogLoss_Elastic", SGDClassifier, {"loss": "log_loss", "alpha": 1e-4, "penalty": "elasticnet", "l1_ratio": 0.5}),
        ]
        return [
            ModelConfig(
                name=name,
                model_class=model_class,
                params={**params, "max_iter": 1000, "random_state": RANDOM_SEED},
                family="linear",
            )
            for name, model_class, params in configs
        ]

    def _get_linear_architectures_regression(self) -> List[ModelConfig]:
        """Return 15 deterministic linear regression configurations."""
        configs = [
            ("LinearRegression", LinearRegression, {}),
            ("Ridge_Alpha0p01", Ridge, {"alpha": 0.01}),
            ("Ridge_Alpha0p1", Ridge, {"alpha": 0.1}),
            ("Ridge_Alpha1", Ridge, {"alpha": 1.0}),
            ("Ridge_Alpha10", Ridge, {"alpha": 10.0}),
            ("Ridge_Alpha100", Ridge, {"alpha": 100.0}),
            ("Ridge_Alpha1000", Ridge, {"alpha": 1000.0}),
            ("Lasso_Alpha0p0001", Lasso, {"alpha": 0.0001}),
            ("Lasso_Alpha0p001", Lasso, {"alpha": 0.001}),
            ("Lasso_Alpha0p01", Lasso, {"alpha": 0.01}),
            ("Lasso_Alpha0p1", Lasso, {"alpha": 0.1}),
            ("ElasticNet_Alpha0p001", ElasticNet, {"alpha": 0.001, "l1_ratio": 0.25}),
            ("ElasticNet_Alpha0p01", ElasticNet, {"alpha": 0.01, "l1_ratio": 0.5}),
            ("ElasticNet_Alpha0p1", ElasticNet, {"alpha": 0.1, "l1_ratio": 0.75}),
            ("ElasticNet_Alpha1", ElasticNet, {"alpha": 1.0, "l1_ratio": 0.9}),
        ]
        return [
            ModelConfig(
                name=name,
                model_class=model_class,
                params={**params, "max_iter": 2000, "random_state": RANDOM_SEED}
                if model_class is not LinearRegression
                else params,
                family="linear",
            )
            for name, model_class, params in configs
        ]

    # ---------------------- XGBoost ----------------------

    # Grid builders - each yields exactly N configs via a deterministic,
    # evenly-spaced subsample of the full Cartesian grid (so diversity in
    # loss-space is preserved without combinatorial blow-up).
    # ------------------------------------------------------------------

    @staticmethod
    def _evenly_spaced_grid(grid: Dict[str, List[Any]], n: int, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
        """Deterministically subsample n combinations from a full grid product."""
        keys = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(combos))
        if n > len(combos):
            raise ValueError(f"Requested {n} configs but grid only has {len(combos)} combinations.")
        chosen = sorted(idx[:n].tolist())
        return [dict(zip(keys, combos[i])) for i in chosen]

    # ---------------------- XGBoost ----------------------

    def _get_xgboost_architectures_classification(self) -> List[ModelConfig]:
        grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 1.0],
        }
        combos = self._evenly_spaced_grid(grid, self.N_XGBOOST, seed=RANDOM_SEED)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'n_jobs': -1,
                'random_state': RANDOM_SEED,
                'use_label_encoder': False,
            })
            archs.append(ModelConfig(
                name=f'XGB_ne{params["n_estimators"]}_d{params["max_depth"]}'
                     f'_lr{params["learning_rate"]}_ss{params["subsample"]}_{i}',
                model_class=XGBClassifier,
                params=full_params,
                family='xgboost',
            ))
        return archs

    def _get_xgboost_architectures_regression(self) -> List[ModelConfig]:
        grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 1.0],
        }
        combos = self._evenly_spaced_grid(grid, self.N_XGBOOST, seed=RANDOM_SEED + 1)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({
                'colsample_bytree': 0.8,
                'n_jobs': -1,
                'random_state': RANDOM_SEED,
            })
            archs.append(ModelConfig(
                name=f'XGB_ne{params["n_estimators"]}_d{params["max_depth"]}'
                     f'_lr{params["learning_rate"]}_ss{params["subsample"]}_{i}',
                model_class=XGBRegressor,
                params=full_params,
                family='xgboost',
            ))
        return archs

    # ---------------------- Random Forest ----------------------

    def _get_random_forest_architectures_classification(self) -> List[ModelConfig]:
        grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 5],
        }
        combos = self._evenly_spaced_grid(grid, self.N_RANDOM_FOREST, seed=RANDOM_SEED)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({'n_jobs': -1, 'random_state': RANDOM_SEED})
            depth_tag = params['max_depth'] if params['max_depth'] is not None else 'None'
            archs.append(ModelConfig(
                name=f'RF_ne{params["n_estimators"]}_d{depth_tag}'
                     f'_{params["criterion"]}_msl{params["min_samples_leaf"]}_{i}',
                model_class=RandomForestClassifier,
                params=full_params,
                family='random_forest',
            ))
        return archs

    def _get_random_forest_architectures_regression(self) -> List[ModelConfig]:
        grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'criterion': ['squared_error', 'absolute_error'],
            'min_samples_leaf': [1, 5],
        }
        combos = self._evenly_spaced_grid(grid, self.N_RANDOM_FOREST, seed=RANDOM_SEED + 1)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({'n_jobs': -1, 'random_state': RANDOM_SEED})
            depth_tag = params['max_depth'] if params['max_depth'] is not None else 'None'
            archs.append(ModelConfig(
                name=f'RF_ne{params["n_estimators"]}_d{depth_tag}'
                     f'_{params["criterion"]}_msl{params["min_samples_leaf"]}_{i}',
                model_class=RandomForestRegressor,
                params=full_params,
                family='random_forest',
            ))
        return archs

    # ---------------------- MLP ----------------------

    def _get_mlp_architectures_classification(self) -> List[ModelConfig]:
        grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [1e-4, 1e-3],
        }
        combos = self._evenly_spaced_grid(grid, self.N_MLP, seed=RANDOM_SEED)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({'max_iter': 500, 'random_state': RANDOM_SEED})
            if params['solver'] == 'sgd':
                full_params['learning_rate'] = 'adaptive'
            hl_tag = 'x'.join(str(h) for h in params['hidden_layer_sizes'])
            archs.append(ModelConfig(
                name=f'MLP_{hl_tag}_{params["activation"]}_{params["solver"]}_{i}',
                model_class=MLPClassifier,
                params=full_params,
                family='mlp',
            ))
        return archs

    def _get_mlp_architectures_regression(self) -> List[ModelConfig]:
        grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [1e-4, 1e-3],
        }
        combos = self._evenly_spaced_grid(grid, self.N_MLP, seed=RANDOM_SEED + 1)
        archs = []
        for i, params in enumerate(combos):
            full_params = dict(params)
            full_params.update({'max_iter': 500, 'random_state': RANDOM_SEED})
            if params['solver'] == 'sgd':
                full_params['learning_rate'] = 'adaptive'
            hl_tag = 'x'.join(str(h) for h in params['hidden_layer_sizes'])
            archs.append(ModelConfig(
                name=f'MLP_{hl_tag}_{params["activation"]}_{params["solver"]}_{i}',
                model_class=MLPRegressor,
                params=full_params,
                family='mlp',
            ))
        return archs

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def train_model(self, config: ModelConfig, X_train: pd.DataFrame,
                     y_train: pd.Series, random_seed: Optional[int] = None) -> Any:
        """Train a single model with given configuration."""
        if self.task_type == 'classification' and len(y_train.unique()) < 2:
            model = DummyClassifier(strategy='most_frequent')
            model.fit(X_train, y_train)
            return model

        params = config.params.copy()
        if random_seed is not None and 'random_state' in params:
            params['random_state'] = random_seed

        try:
            model = config.model_class(**params)
            model.fit(X_train, y_train)
            return model
        except ValueError as e:
            print(f"Training failed for {config.name}: {str(e)}. Using Dummy model.")
            if self.task_type == 'classification':
                model = DummyClassifier(strategy='most_frequent')
            else:
                model = DummyRegressor(strategy='mean')
            model.fit(X_train, y_train)
            return model

    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.

        For classification: accuracy, log_loss
        For regression: MSE, MAE, R^2
        """
        predictions = model.predict(X)

        if self.task_type == 'regression':
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            rmse = np.sqrt(mse)

            loss = mae if self.loss_type == 'mae' else mse

            return {
                'loss': loss, 'mse': mse, 'rmse': rmse,
                'mae': mae, 'r2': r2, 'predictions': predictions
            }
        else:
            accuracy = accuracy_score(y, predictions)
            loss = 1.0 - accuracy

            if self.loss_type == 'log_loss':
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        if not np.any(np.isnan(proba)):
                            eps = 1e-15
                            proba = np.clip(proba, eps, 1 - eps)
                            loss = log_loss(y, proba)
                    except Exception:
                        pass

            return {
                'loss': loss, 'accuracy': accuracy, 'predictions': predictions
            }

    def random_seed_train(self, config: ModelConfig,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_synth: pd.DataFrame, y_synth: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           seed: int = 10) -> Dict:
        """Train model with a specific random seed and evaluate."""
        results = {
            'seed': [], 'synth_losses': [], 'test_losses': [],
        }

        model = self.train_model(config, X_train, y_train, random_seed=seed)

        synth_eval = self.evaluate_model(model, X_synth, y_synth)
        test_eval = self.evaluate_model(model, X_test, y_test)

        results['seed'].append(seed)
        results['synth_losses'].append(synth_eval['loss'])
        results['test_losses'].append(test_eval['loss'])

        return results
