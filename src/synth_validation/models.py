"""
Model Selection Framework.

Implementation of model architectures and selection framework for both
classification and regression tasks.
"""

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

# Ensemble Models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)

# Linear Models
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, LogisticRegressionCV,
    Ridge, Lasso, ElasticNet, LinearRegression, SGDRegressor
)

# Support Vector Machines
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, LinearSVR, NuSVR

# Neural Networks
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Tree-Based Models
from sklearn.tree import (
    DecisionTreeClassifier, ExtraTreeClassifier,
    DecisionTreeRegressor, ExtraTreeRegressor
)

# Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Semi-Supervised
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

# Calibration
from sklearn.calibration import CalibratedClassifierCV

from .utils import RANDOM_SEED


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    name: str
    model_class: Any
    params: Dict[str, Any]


class ModelSelectionFramework:
    """
    Complete framework for model selection using synthetic data.
    Supports both classification and regression tasks.
    """

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
        
        # Auto-select loss_type based on task if not specified
        if loss_type is None:
            loss_type = 'accuracy' if task_type == 'classification' else 'mse'
        self.loss_type = loss_type
        
        self.trained_models = []
        self.results = defaultdict(list)

    def get_model_architectures(self) -> List[ModelConfig]:
        """Get model architectures based on task type."""
        if self.task_type == 'regression':
            return self._get_regression_architectures()
        else:
            return self._get_classification_architectures()

    def _get_classification_architectures(self) -> List[ModelConfig]:
        """Define diverse classification model architectures."""
        architectures = [
            # ==================== ENSEMBLE METHODS ====================
            
            # Random Forest variants
            ModelConfig('RandomForest_Gini', RandomForestClassifier,
                {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_Entropy', RandomForestClassifier,
                {'n_estimators': 100, 'criterion': 'entropy', 'max_depth': 15, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_ShallowTrees', RandomForestClassifier,
                {'n_estimators': 200, 'max_depth': 5, 'min_samples_leaf': 5, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_DeepTrees', RandomForestClassifier,
                {'n_estimators': 150, 'max_depth': 25, 'min_samples_split': 2, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_Bootstrap', RandomForestClassifier,
                {'n_estimators': 100, 'criterion': 'gini', 'bootstrap': True, 'oob_score': True, 'n_jobs': -1, 'random_state': RANDOM_SEED}),

            # Gradient Boosting variants
            ModelConfig('GradientBoosting', GradientBoostingClassifier,
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': RANDOM_SEED}),
            ModelConfig('GradientBoosting_Deep', GradientBoostingClassifier,
                {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': RANDOM_SEED}),
            ModelConfig('HistGradientBoosting', HistGradientBoostingClassifier,
                {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': None, 'random_state': RANDOM_SEED}),

            # Extra Trees
            ModelConfig('ExtraTrees', ExtraTreesClassifier,
                {'n_estimators': 100, 'criterion': 'gini', 'bootstrap': False, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('ExtraTrees_Entropy', ExtraTreesClassifier,
                {'n_estimators': 100, 'criterion': 'entropy', 'bootstrap': False, 'n_jobs': -1, 'random_state': RANDOM_SEED}),

            # Bagging
            ModelConfig('Bagging_Tree', BaggingClassifier,
                {'estimator': DecisionTreeClassifier(), 'n_estimators': 50, 'n_jobs': 1, 'random_state': RANDOM_SEED}),
            ModelConfig('Bagging_LogReg', BaggingClassifier,
                {'estimator': LogisticRegression(max_iter=500), 'n_estimators': 20, 'n_jobs': 1, 'random_state': RANDOM_SEED}),

            # AdaBoost
            ModelConfig('AdaBoost_Tree', AdaBoostClassifier,
                {'n_estimators': 50, 'learning_rate': 1.0, 'random_state': RANDOM_SEED}),
            ModelConfig('AdaBoost_LogReg', AdaBoostClassifier,
                {'estimator': LogisticRegression(max_iter=500), 'n_estimators': 20, 'learning_rate': 0.5, 'random_state': RANDOM_SEED}),
            ModelConfig('AdaBoost_LowLR', AdaBoostClassifier,
                {'n_estimators': 100, 'learning_rate': 0.5, 'random_state': RANDOM_SEED}),

            # ==================== LINEAR MODELS ====================

            # Logistic Regression
            ModelConfig('LogReg_L2_LBFGS', LogisticRegression,
                {'penalty': 'l2', 'solver': 'lbfgs', 'C': 1.0, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('LogReg_L1_SAGA', LogisticRegression,
                {'penalty': 'l1', 'solver': 'saga', 'C': 1.0, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('LogReg_ElasticNet', LogisticRegression,
                {'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5, 'C': 1.0, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('LogReg_L2_LBFGS_CV', LogisticRegressionCV,
                {'penalty': 'l2', 'solver': 'lbfgs', 'Cs': 5, 'cv': 5, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('LogReg_None', LogisticRegression,
                {'penalty': None, 'solver': 'lbfgs', 'max_iter': 2000, 'random_state': RANDOM_SEED}),

            # SGD
            ModelConfig('SGD_Log_Loss', SGDClassifier,
                {'loss': 'log_loss', 'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': RANDOM_SEED}),

            # Calibrated linear models
            ModelConfig('Calibrated_LinearSVC', CalibratedClassifierCV,
                {'estimator': LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED), 'cv': 3, 'method': 'sigmoid'}),
            ModelConfig('Calibrated_SGD_Hinge', CalibratedClassifierCV,
                {'estimator': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, random_state=RANDOM_SEED),
                 'cv': 3, 'method': 'sigmoid'}),
            ModelConfig('Calibrated_LinearSVC_Isotonic', CalibratedClassifierCV,
                {'estimator': LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_SEED), 'cv': 3, 'method': 'isotonic'}),

            # ==================== SUPPORT VECTOR MACHINES ====================

            ModelConfig('SVC_RBF', SVC,
                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'probability': True, 'random_state': RANDOM_SEED}),
            ModelConfig('SVC_Poly', SVC,
                {'kernel': 'poly', 'degree': 2, 'C': 1.0, 'probability': True, 'random_state': RANDOM_SEED}),
            ModelConfig('SVC_Linear', SVC,
                {'kernel': 'linear', 'C': 1.0, 'probability': True, 'random_state': RANDOM_SEED}),
            ModelConfig('NuSVC', NuSVC,
                {'nu': 0.5, 'kernel': 'rbf', 'probability': True, 'random_state': RANDOM_SEED}),

            # ==================== NEAREST NEIGHBORS ====================

            ModelConfig('KNN_Uniform_5', KNeighborsClassifier,
                {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}),
            ModelConfig('KNN_Distance_10', KNeighborsClassifier,
                {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'auto'}),
            ModelConfig('KNN_Manhattan', KNeighborsClassifier,
                {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'manhattan'}),

            # ==================== NAIVE BAYES ====================

            ModelConfig('GaussianNB', GaussianNB, {}),
            ModelConfig('BernoulliNB', BernoulliNB, {'alpha': 1.0, 'binarize': 0.0}),
            ModelConfig('MultinomialNB_Alpha1', MultinomialNB, {'alpha': 1.0, 'fit_prior': True}),
            ModelConfig('ComplementNB_Alpha1', ComplementNB, {'alpha': 1.0, 'fit_prior': True}),

            # ==================== DISCRIMINANT ANALYSIS ====================

            ModelConfig('LDA_SVD', LinearDiscriminantAnalysis, {'solver': 'svd'}),
            ModelConfig('QDA', QuadraticDiscriminantAnalysis, {'reg_param': 0.0}),

            # ==================== DECISION TREES ====================

            ModelConfig('DecisionTree_Gini', DecisionTreeClassifier,
                {'criterion': 'gini', 'max_depth': None, 'random_state': RANDOM_SEED}),
            ModelConfig('DecisionTree_Entropy', DecisionTreeClassifier,
                {'criterion': 'entropy', 'max_depth': 15, 'random_state': RANDOM_SEED}),
            ModelConfig('ExtraTreeClassifier_Deep', ExtraTreeClassifier,
                {'max_depth': 20, 'splitter': 'best', 'random_state': RANDOM_SEED}),

            # ==================== NEURAL NETWORKS ====================

            ModelConfig('MLP_ReLU_Adam', MLPClassifier,
                {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Tanh_SGD', MLPClassifier,
                {'hidden_layer_sizes': (50, 25), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Deep', MLPClassifier,
                {'hidden_layer_sizes': (100, 50, 25), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Wide', MLPClassifier,
                {'hidden_layer_sizes': (200, 100), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),

            # ==================== ENSEMBLE META-MODELS ====================

            ModelConfig('VotingClassifier_Soft', VotingClassifier,
                {'estimators': [
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)),
                    ('svc', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
                ], 'voting': 'soft'}),

            ModelConfig('StackingClassifier_LogReg', StackingClassifier,
                {'estimators': [
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_SEED)),
                    ('svc', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)),
                ], 'final_estimator': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED), 'cv': 5}),
            ModelConfig('StackingClassifier_MLP', StackingClassifier,
                {'estimators': [
                    ('rf', RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)),
                    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_SEED)),
                    ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
                ], 'final_estimator': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=RANDOM_SEED), 'cv': 5}),

            # ==================== SEMI-SUPERVISED ====================

            ModelConfig('LabelSpreading_KNN', LabelSpreading,
                {'kernel': 'knn', 'n_neighbors': 10, 'max_iter': 1000}),
            ModelConfig('SelfTraining_LogReg', SelfTrainingClassifier,
                {'base_estimator': LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
                 'threshold': 0.8, 'verbose': False}),
        ]

        return architectures

    def _get_regression_architectures(self) -> List[ModelConfig]:
        """Define diverse regression model architectures."""
        
        architectures = [
            # ==================== ENSEMBLE METHODS ====================
            
            # Random Forest variants
            ModelConfig('RandomForest_MSE', RandomForestRegressor,
                {'n_estimators': 100, 'criterion': 'squared_error', 'max_depth': None, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_MAE', RandomForestRegressor,
                {'n_estimators': 100, 'criterion': 'absolute_error', 'max_depth': 15, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('RandomForest_Shallow', RandomForestRegressor,
                {'n_estimators': 50, 'max_depth': 8, 'min_samples_split': 10, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            
            # Gradient Boosting variants
            ModelConfig('GradientBoosting', GradientBoostingRegressor,
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': RANDOM_SEED}),
            ModelConfig('GradientBoosting_Deep', GradientBoostingRegressor,
                {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': RANDOM_SEED}),
            ModelConfig('HistGradientBoosting', HistGradientBoostingRegressor,
                {'max_iter': 100, 'learning_rate': 0.1, 'max_depth': None, 'random_state': RANDOM_SEED}),
            ModelConfig('HistGradientBoosting_L1', HistGradientBoostingRegressor,
                {'max_iter': 100, 'learning_rate': 0.1, 'loss': 'absolute_error', 'random_state': RANDOM_SEED}),
            
            # Extra Trees
            ModelConfig('ExtraTrees', ExtraTreesRegressor,
                {'n_estimators': 100, 'criterion': 'squared_error', 'bootstrap': False, 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            ModelConfig('ExtraTrees_MAE', ExtraTreesRegressor,
                {'n_estimators': 100, 'criterion': 'absolute_error', 'n_jobs': -1, 'random_state': RANDOM_SEED}),
            
            # Bagging
            ModelConfig('Bagging_Tree', BaggingRegressor,
                {'estimator': DecisionTreeRegressor(), 'n_estimators': 50, 'n_jobs': 1, 'random_state': RANDOM_SEED}),
            ModelConfig('Bagging_SVR', BaggingRegressor,
                {'estimator': SVR(kernel='rbf'), 'n_estimators': 10, 'max_samples': 0.5, 'n_jobs': 1, 'random_state': RANDOM_SEED}),
            
            # AdaBoost
            ModelConfig('AdaBoost_Tree', AdaBoostRegressor,
                {'n_estimators': 50, 'learning_rate': 1.0, 'random_state': RANDOM_SEED}),
            ModelConfig('AdaBoost_Linear', AdaBoostRegressor,
                {'n_estimators': 50, 'learning_rate': 0.5, 'loss': 'linear', 'random_state': RANDOM_SEED}),
            
            # ==================== LINEAR MODELS ====================
            
            ModelConfig('LinearRegression', LinearRegression, {}),
            
            ModelConfig('Ridge_Alpha1', Ridge, {'alpha': 1.0, 'random_state': RANDOM_SEED}),
            ModelConfig('Ridge_Alpha10', Ridge, {'alpha': 10.0, 'random_state': RANDOM_SEED}),
            ModelConfig('Ridge_Alpha01', Ridge, {'alpha': 0.1, 'random_state': RANDOM_SEED}),
            
            ModelConfig('Lasso_Alpha1', Lasso, {'alpha': 1.0, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('Lasso_Alpha01', Lasso, {'alpha': 0.1, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            
            ModelConfig('ElasticNet_L1_05', ElasticNet,
                {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('ElasticNet_L1_02', ElasticNet,
                {'alpha': 1.0, 'l1_ratio': 0.2, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            
            ModelConfig('SGD_L2', SGDRegressor,
                {'loss': 'squared_error', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': RANDOM_SEED}),
            ModelConfig('SGD_L1', SGDRegressor,
                {'loss': 'squared_error', 'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': RANDOM_SEED}),
            ModelConfig('SGD_Huber', SGDRegressor,
                {'loss': 'huber', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000, 'random_state': RANDOM_SEED}),
            
            # ==================== SUPPORT VECTOR MACHINES ====================
            
            ModelConfig('SVR_RBF', SVR, {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}),
            ModelConfig('SVR_Linear', LinearSVR, {'C': 1.0, 'max_iter': 2000, 'random_state': RANDOM_SEED}),
            ModelConfig('SVR_Poly', SVR, {'kernel': 'poly', 'degree': 2, 'C': 1.0}),
            ModelConfig('NuSVR', NuSVR, {'nu': 0.5, 'kernel': 'rbf', 'C': 1.0}),
            
            # ==================== NEAREST NEIGHBORS ====================
            
            ModelConfig('KNN_Uniform_5', KNeighborsRegressor,
                {'n_neighbors': 5, 'weights': 'uniform', 'algorithm': 'auto'}),
            ModelConfig('KNN_Distance_10', KNeighborsRegressor,
                {'n_neighbors': 10, 'weights': 'distance', 'algorithm': 'auto'}),
            ModelConfig('KNN_Manhattan', KNeighborsRegressor,
                {'n_neighbors': 5, 'weights': 'distance', 'metric': 'manhattan'}),
            
            # ==================== TREE-BASED MODELS ====================
            
            ModelConfig('DecisionTree_MSE', DecisionTreeRegressor,
                {'criterion': 'squared_error', 'max_depth': None, 'random_state': RANDOM_SEED}),
            ModelConfig('DecisionTree_Shallow', DecisionTreeRegressor,
                {'criterion': 'squared_error', 'max_depth': 10, 'min_samples_split': 10, 'random_state': RANDOM_SEED}),
            ModelConfig('DecisionTree_MAE', DecisionTreeRegressor,
                {'criterion': 'absolute_error', 'max_depth': 15, 'random_state': RANDOM_SEED}),
            ModelConfig('ExtraTreeRegressor', ExtraTreeRegressor,
                {'max_depth': None, 'splitter': 'random', 'random_state': RANDOM_SEED}),
            
            # ==================== NEURAL NETWORKS ====================
            
            ModelConfig('MLP_ReLU_Adam', MLPRegressor,
                {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Tanh_SGD', MLPRegressor,
                {'hidden_layer_sizes': (50, 25), 'activation': 'tanh', 'solver': 'sgd', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Deep', MLPRegressor,
                {'hidden_layer_sizes': (100, 50, 25), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            ModelConfig('MLP_Wide', MLPRegressor,
                {'hidden_layer_sizes': (200, 100), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500, 'random_state': RANDOM_SEED}),
            
            # ==================== ENSEMBLE META-MODELS ====================
            
            ModelConfig('VotingRegressor', VotingRegressor,
                {'estimators': [
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)),
                    ('svr', SVR(kernel='rbf')),
                    ('ridge', Ridge(alpha=1.0))
                ]}),
            
            ModelConfig('StackingRegressor_Ridge', StackingRegressor,
                {'estimators': [
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)),
                    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=RANDOM_SEED)),
                    ('svr', SVR(kernel='rbf')),
                ], 'final_estimator': Ridge(alpha=1.0), 'cv': 5}),
            
            ModelConfig('StackingRegressor_Linear', StackingRegressor,
                {'estimators': [
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)),
                    ('et', ExtraTreesRegressor(n_estimators=50, random_state=RANDOM_SEED)),
                    ('ridge', Ridge(alpha=1.0)),
                ], 'final_estimator': LinearRegression(), 'cv': 5}),
        ]

        return architectures
    
    def train_model(self, config: ModelConfig, X_train: pd.DataFrame,
               y_train: pd.Series, random_seed: Optional[int] = None) -> Any:
        """Train a single model with given configuration."""
        
        # For classification: check if only one class
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
        For regression: MSE, MAE, R²
        """
        predictions = model.predict(X)
        
        if self.task_type == 'regression':
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            rmse = np.sqrt(mse)
            
            if self.loss_type == 'mae':
                loss = mae
            else:
                loss = mse
            
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
