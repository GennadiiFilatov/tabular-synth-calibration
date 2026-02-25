"""
Data Loading and Preprocessing.

Functions to load and prepare tabular datasets for both classification and regression tasks.

Classification Datasets:
    adult, diabetes, breast_cancer, wine, iris, digits_binary, mushroom, german_credit,
    heart_disease, titanic, bank_marketing, spambase, ionosphere, sonar, imbalanced_synth, obesity

Regression Datasets:
    california_housing, diabetes_regression, ames_housing, diamond_prices, auto_mpg,
    bike_sharing, concrete_strength, energy_efficiency, wine_quality, abalone, regression_synth

Usage:
    # Load a classification dataset
    df = DataLoader.load_uci_dataset('breast_cancer')
    X_train, X_test, y_train, y_test, transformers = DataLoader.prepare_data(df, task_type='classification')

    # Load a regression dataset
    df = DataLoader.load_uci_dataset('california_housing')
    X_train, X_test, y_train, y_test, transformers = DataLoader.prepare_data(df, task_type='regression')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .utils import RANDOM_SEED


class DataLoader:
    """Handles loading and preprocessing of tabular datasets for classification and regression tasks."""
    
    # Classification datasets
    CLASSIFICATION_DATASETS = {
        'adult': {
            'description': 'UCI Adult Income dataset - predict income >50K',
            'loader': '_load_adult',
            'samples': 32561,
            'features': 14
        },
        'diabetes': {
            'description': 'Pima Indians Diabetes - binary classification',
            'loader': '_load_diabetes',
            'samples': 768,
            'features': 8
        },
        'breast_cancer': {
            'description': 'Breast Cancer Wisconsin - malignant vs benign',
            'loader': '_load_breast_cancer',
            'samples': 569,
            'features': 30
        },
        'wine': {
            'description': 'Wine recognition - 3 wine classes',
            'loader': '_load_wine',
            'samples': 178,
            'features': 13
        },
        'iris': {
            'description': 'Iris flower species classification',
            'loader': '_load_iris',
            'samples': 150,
            'features': 4
        },
        'digits_binary': {
            'description': 'Handwritten digits (0-4 vs 5-9) binary classification',
            'loader': '_load_digits_binary',
            'samples': 1797,
            'features': 64
        },
        'mushroom': {
            'description': 'Mushroom edibility classification',
            'loader': '_load_mushroom',
            'samples': 8124,
            'features': 22
        },
        'german_credit': {
            'description': 'German Credit Risk assessment',
            'loader': '_load_german_credit',
            'samples': 1000,
            'features': 20
        },
        'heart_disease': {
            'description': 'Heart disease presence prediction',
            'loader': '_load_heart_disease',
            'samples': 303,
            'features': 13
        },
        'titanic': {
            'description': 'Titanic passenger survival prediction',
            'loader': '_load_titanic',
            'samples': 891,
            'features': 11
        },
        'bank_marketing': {
            'description': 'Bank marketing campaign success',
            'loader': '_load_bank_marketing',
            'samples': 41188,
            'features': 20
        },
        'spambase': {
            'description': 'Spam email detection',
            'loader': '_load_spambase',
            'samples': 4601,
            'features': 57
        },
        'ionosphere': {
            'description': 'Ionosphere radar returns classification',
            'loader': '_load_ionosphere',
            'samples': 351,
            'features': 34
        },
        'sonar': {
            'description': 'Sonar signals - mine vs rock',
            'loader': '_load_sonar',
            'samples': 208,
            'features': 60
        },
        'imbalanced_synth': {
            'description': 'Synthetic imbalanced binary classification',
            'loader': '_load_imbalanced_synth',
            'samples': 500,
            'features': 20
        },
        'obesity': {
            'description': 'Obesity level classification',
            'loader': '_load_obesity',
            'samples': 2111,
            'features': 16
        },
    }
    
    # Regression datasets
    REGRESSION_DATASETS = {
        'california_housing': {
            'description': 'California house prices regression',
            'loader': '_load_california_housing',
            'samples': 20640,
            'features': 8
        },
        'diabetes_regression': {
            'description': 'Diabetes progression regression',
            'loader': '_load_diabetes_regression',
            'samples': 442,
            'features': 10
        },
        'ames_housing': {
            'description': 'Ames Iowa housing prices',
            'loader': '_load_ames_housing',
            'samples': 1460,
            'features': 80
        },
        'diamond_prices': {
            'description': 'Diamond price prediction',
            'loader': '_load_diamond_prices',
            'samples': 53940,
            'features': 9
        },
        'auto_mpg': {
            'description': 'Auto MPG fuel efficiency prediction',
            'loader': '_load_auto_mpg',
            'samples': 398,
            'features': 7
        },
        'bike_sharing': {
            'description': 'Bike sharing demand prediction',
            'loader': '_load_bike_sharing',
            'samples': 17379,
            'features': 16
        },
        'concrete_strength': {
            'description': 'Concrete compressive strength',
            'loader': '_load_concrete_strength',
            'samples': 1030,
            'features': 8
        },
        'energy_efficiency': {
            'description': 'Building energy efficiency prediction',
            'loader': '_load_energy_efficiency',
            'samples': 768,
            'features': 8
        },
        'wine_quality': {
            'description': 'Wine quality rating regression',
            'loader': '_load_wine_quality',
            'samples': 4898,
            'features': 11
        },
        'abalone': {
            'description': 'Abalone age prediction',
            'loader': '_load_abalone',
            'samples': 4177,
            'features': 8
        },
        'regression_synth': {
            'description': 'Synthetic regression dataset',
            'loader': '_load_regression_synth',
            'samples': 1000,
            'features': 20
        },
    }
    
    # Combined datasets dictionary for backward compatibility
    DATASETS = {**CLASSIFICATION_DATASETS, **REGRESSION_DATASETS}
    
    @staticmethod
    def get_task_type(dataset_name: str) -> str:
        """Return the task type for a given dataset."""
        dataset_name = dataset_name.lower()
        if dataset_name in DataLoader.CLASSIFICATION_DATASETS:
            return 'classification'
        elif dataset_name in DataLoader.REGRESSION_DATASETS:
            return 'regression'
        else:
            return 'unknown'
    
    @staticmethod
    def list_datasets(task_type: str = None) -> List[str]:
        """List available datasets."""
        if task_type == 'classification':
            return list(DataLoader.CLASSIFICATION_DATASETS.keys())
        elif task_type == 'regression':
            return list(DataLoader.REGRESSION_DATASETS.keys())
        else:
            return list(DataLoader.DATASETS.keys())
    
    @staticmethod
    def load_uci_dataset(dataset_name: str = 'adult') -> pd.DataFrame:
        """Load dataset from UCI or generate synthetic data for testing."""
        dataset_name = dataset_name.lower()
        
        if dataset_name not in DataLoader.DATASETS:
            print(f"Dataset '{dataset_name}' not found.")
            print(f"Available classification datasets: {', '.join(DataLoader.CLASSIFICATION_DATASETS.keys())}")
            print(f"Available regression datasets: {', '.join(DataLoader.REGRESSION_DATASETS.keys())}")
            print("Generating synthetic data...")
            return None
        
        dataset_info = DataLoader.DATASETS[dataset_name]
        loader_method = dataset_info['loader']
        loader = getattr(DataLoader, loader_method)
        return loader()
    
    # ==================== CLASSIFICATION DATASET LOADERS ====================
    
    @staticmethod
    def _load_breast_cancer() -> pd.DataFrame:
        """Breast Cancer dataset from sklearn"""
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df['target'] = data.target
        print(f"[Classification] Breast Cancer loaded: {df.shape}")
        return df
    
    @staticmethod
    def _load_wine() -> pd.DataFrame:
        """Wine dataset from sklearn"""
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame.copy()
        df['target'] = data.target
        print(f"[Classification] Wine loaded: {df.shape}")
        return df
    
    @staticmethod
    def _load_iris() -> pd.DataFrame:
        """Iris dataset from sklearn"""
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df['target'] = data.target
        print(f"[Classification] Iris loaded: {df.shape}")
        return df
    
    @staticmethod
    def _load_digits_binary() -> pd.DataFrame:
        """Digits Binary dataset from sklearn"""
        from sklearn.datasets import load_digits
        data = load_digits(as_frame=True)
        df = data.frame.copy()
        df['target'] = (data.target >= 5).astype(int)
        print(f"[Classification] Digits (binary) loaded: {df.shape}")
        return df
    
    @staticmethod
    def _load_imbalanced_synth() -> pd.DataFrame:
        """Synthesized imbalanced dataset"""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=15, n_redundant=3,
            n_classes=2, weights=[0.7, 0.3], random_state=RANDOM_SEED
        )
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        df['target'] = y
        print(f"[Classification] Imbalanced Synthetic loaded: {df.shape}")
        return df
    
    @staticmethod
    def _load_adult() -> pd.DataFrame:
        """Adult Income dataset"""
        try:
            adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                      'marital-status', 'occupation', 'relationship', 'race', 'sex',
                      'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
            df = pd.read_csv(adult_url, names=columns, skipinitialspace=True, na_values='?')
            print(f"[Classification] Adult dataset loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"Direct URL failed: {e}")
        
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
            print(f"[Classification] Adult dataset loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
            return None
    
    @staticmethod
    def _load_diabetes() -> pd.DataFrame:
        """Pima Indians Diabetes dataset (Classification)"""
        try:
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                    'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
            df = pd.read_csv(url, names=columns, header=None)
            df['target'] = df['outcome']
            df = df.drop(columns=['outcome'])
            print(f"[Classification] Diabetes loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Diabetes download failed: {e}")
            return None
    
    @staticmethod
    def _load_mushroom() -> pd.DataFrame:
        """Mushroom Classification dataset"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
            columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises',
                    'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                    'stalk-shape', 'stalk-root', 'stalk-surface-above', 'stalk-surface-below',
                    'stalk-color-above', 'stalk-color-below', 'veil-type', 'veil-color',
                    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
            df = pd.read_csv(url, names=columns, na_values='?')
            print(f"[Classification] Mushroom loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Mushroom download failed: {e}")
            return None
    
    @staticmethod
    def _load_german_credit() -> pd.DataFrame:
        """German Credit dataset"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=31, as_frame=True, parser='auto')
            print(f"[Classification] German Credit loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            columns = ['status_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                      'savings', 'employment', 'install_rate', 'personal_status', 'debtors',
                      'residence', 'property', 'age', 'plans', 'housing', 'credits', 'job',
                      'dependents', 'telephone', 'foreign', 'class']
            df = pd.read_csv(url, names=columns, sep=' ', header=None)
            print(f"[Classification] German Credit loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"UCI download failed: {e}")
            return None
    
    @staticmethod
    def _load_heart_disease() -> pd.DataFrame:
        """Heart Disease dataset"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=40682, as_frame=True, parser='auto')
            print(f"[Classification] Heart Disease loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(url, names=columns, na_values='?')
            print(f"[Classification] Heart Disease loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"UCI download failed: {e}")
            return None
    
    @staticmethod
    def _load_titanic() -> pd.DataFrame:
        """Titanic dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=40945, as_frame=True, parser='auto')
            df = data.frame.copy()
            df = df.dropna()
            print(f"[Classification] Titanic loaded from OpenML: {df.shape}")
            return df
        except Exception as e:
            print(f"OpenML failed: {e}")
            return None
    
    @staticmethod
    def _load_bank_marketing() -> pd.DataFrame:
        """Bank Marketing dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=1461, as_frame=True, parser='auto')
            print(f"[Classification] Bank Marketing loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
            return None
    
    @staticmethod
    def _load_spambase() -> pd.DataFrame:
        """Spambase dataset"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=44, as_frame=True, parser='auto')
            print(f"[Classification] Spambase loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            n_features = 57
            columns = [f'feature_{i}' for i in range(n_features)] + ['target']
            df = pd.read_csv(url, names=columns, header=None)
            print(f"[Classification] Spambase loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"UCI download failed: {e}")
            return None
    
    @staticmethod
    def _load_ionosphere() -> pd.DataFrame:
        """Ionosphere dataset"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=59, as_frame=True, parser='auto')
            print(f"[Classification] Ionosphere loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
            n_features = 34
            columns = [f'feature_{i}' for i in range(n_features)] + ['target']
            df = pd.read_csv(url, names=columns, header=None)
            print(f"[Classification] Ionosphere loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"UCI download failed: {e}")
            return None
    
    @staticmethod
    def _load_sonar() -> pd.DataFrame:
        """Sonar dataset"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=40, as_frame=True, parser='auto')
            print(f"[Classification] Sonar loaded from OpenML: {data.frame.shape}")
            return data.frame
        except Exception as e:
            print(f"OpenML failed: {e}")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
            n_features = 60
            columns = [f'feature_{i}' for i in range(n_features)] + ['target']
            df = pd.read_csv(url, names=columns, header=None)
            print(f"[Classification] Sonar loaded from UCI: {df.shape}")
            return df
        except Exception as e:
            print(f"UCI download failed: {e}")
            return None
    
    @staticmethod
    def _load_obesity() -> pd.DataFrame:
        """Obesity Levels dataset"""
        try:
            from ucimlrepo import fetch_ucirepo
            
            obesity = fetch_ucirepo(id=544)
            X = obesity.data.features
            y = obesity.data.targets
            
            df = X.copy()
            target_col = y.columns[0]
            df['target'] = y[target_col].values
            
            print(f"[Classification] Obesity dataset loaded from UCI ML Repo: {df.shape}")
            print(f"   Target classes: {df['target'].nunique()}")
            return df
            
        except ImportError:
            print("ucimlrepo not installed. Install: pip install ucimlrepo")
            return None
        except Exception as e:
            print(f"Obesity dataset load failed: {e}")
            return None
    
    # ==================== REGRESSION DATASET LOADERS ====================
    
    @staticmethod
    def _load_california_housing() -> pd.DataFrame:
        """California Housing dataset from sklearn"""
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        df['target'] = data.target
        print(f"[Regression] California Housing loaded: {df.shape}")
        print(f"   Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
        return df
    
    @staticmethod
    def _load_diabetes_regression() -> pd.DataFrame:
        """Diabetes dataset from sklearn (Regression version)"""
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['target'] = data.target
        print(f"[Regression] Diabetes (sklearn) loaded: {df.shape}")
        print(f"   Target range: [{df['target'].min():.0f}, {df['target'].max():.0f}]")
        return df
    
    @staticmethod
    def _load_ames_housing() -> pd.DataFrame:
        """Ames Housing dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=42165, as_frame=True, parser='auto')
            df = data.frame.copy()
            if 'SalePrice' in df.columns:
                df['target'] = df['SalePrice']
                df = df.drop(columns=['SalePrice'])
            print(f"[Regression] Ames Housing loaded from OpenML: {df.shape}")
            print(f"   Target range: [{df['target'].min():.0f}, {df['target'].max():.0f}]")
            return df
        except Exception as e:
            print(f"Ames Housing load failed: {e}")
            return None
    
    @staticmethod
    def _load_diamond_prices() -> pd.DataFrame:
        """Diamond Prices dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=42225, as_frame=True, parser='auto')
            df = data.frame.copy()
            if 'price' in df.columns:
                df['target'] = df['price']
                df = df.drop(columns=['price'])
            print(f"[Regression] Diamond Prices loaded from OpenML: {df.shape}")
            print(f"   Target range: [{df['target'].min():.0f}, {df['target'].max():.0f}]")
            return df
        except Exception as e:
            print(f"Diamond Prices load failed: {e}")
            return None
    
    @staticmethod
    def _load_auto_mpg() -> pd.DataFrame:
        """Auto MPG dataset from UCI"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
            columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin', 'car_name']
            df = pd.read_csv(url, names=columns, sep=r'\s+', na_values='?')
            df = df.drop(columns=['car_name'])
            df['target'] = df['mpg']
            df = df.drop(columns=['mpg'])
            df = df.dropna()
            print(f"[Regression] Auto MPG loaded from UCI: {df.shape}")
            print(f"   Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}] mpg")
            return df
        except Exception as e:
            print(f"Auto MPG download failed: {e}")
            return None
    
    @staticmethod
    def _load_bike_sharing() -> pd.DataFrame:
        """Bike Sharing Demand dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=42712, as_frame=True, parser='auto')
            df = data.frame.copy()
            if 'cnt' in df.columns:
                df['target'] = df['cnt']
                df = df.drop(columns=['cnt'])
            elif 'count' in df.columns:
                df['target'] = df['count']
                df = df.drop(columns=['count'])
            print(f"[Regression] Bike Sharing loaded from OpenML: {df.shape}")
            print(f"   Target range: [{df['target'].min():.0f}, {df['target'].max():.0f}]")
            return df
        except Exception as e:
            print(f"Bike Sharing load failed: {e}")
            try:
                from sklearn.datasets import fetch_openml
                data = fetch_openml(data_id=44063, as_frame=True, parser='auto')
                df = data.frame.copy()
                print(f"[Regression] Bike Sharing (alt) loaded from OpenML: {df.shape}")
                return df
            except Exception as e2:
                print(f"Alternative also failed: {e2}")
                return None
    
    @staticmethod
    def _load_concrete_strength() -> pd.DataFrame:
        """Concrete Compressive Strength dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=4353, as_frame=True, parser='auto')
            df = data.frame.copy()
            target_candidates = ['strength', 'Strength', 'concrete_compressive_strength', 
                               'csMPa', df.columns[-1]]
            for col in target_candidates:
                if col in df.columns:
                    df['target'] = df[col]
                    if col != 'target':
                        df = df.drop(columns=[col])
                    break
            print(f"[Regression] Concrete Strength loaded from OpenML: {df.shape}")
            print(f"   Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}] MPa")
            return df
        except Exception as e:
            print(f"Concrete Strength load failed: {e}")
            return None
    
    @staticmethod
    def _load_energy_efficiency() -> pd.DataFrame:
        """Energy Efficiency dataset from OpenML"""
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(data_id=1472, as_frame=True, parser='auto')
            df = data.frame.copy()
            if 'Y1' in df.columns:
                df['target'] = df['Y1']
                df = df.drop(columns=['Y1'])
            if 'Y2' in df.columns:
                df = df.drop(columns=['Y2'])
            print(f"[Regression] Energy Efficiency loaded from OpenML: {df.shape}")
            print(f"   Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}]")
            return df
        except Exception as e:
            print(f"Energy Efficiency load failed: {e}")
            return None
    
    @staticmethod
    def _load_wine_quality() -> pd.DataFrame:
        """Wine Quality dataset (red wine) from UCI"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=';')
            df['target'] = df['quality']
            df = df.drop(columns=['quality'])
            print(f"[Regression] Wine Quality (red) loaded from UCI: {df.shape}")
            print(f"   Target range: [{df['target'].min()}, {df['target'].max()}] (quality score)")
            return df
        except Exception as e:
            WINE_PATH = "data/winequality-red.csv"
            try:
                df = pd.read_csv(WINE_PATH, sep=';')
                df['target'] = df['quality']
                df = df.drop(columns=['quality'])
                print(f"   Target range: [{df['target'].min()}, {df['target'].max()}] (quality score)")
                return df
            except:
                print(f"Wine Quality download failed: {e}")
                return None
    
    @staticmethod
    def _load_abalone() -> pd.DataFrame:
        """Abalone dataset from UCI"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
            columns = ['sex', 'length', 'diameter', 'height', 'whole_weight',
                      'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
            df = pd.read_csv(url, names=columns, header=None)
            df['target'] = df['rings']
            df = df.drop(columns=['rings'])
            print(f"[Regression] Abalone loaded from UCI: {df.shape}")
            print(f"   Target range: [{df['target'].min()}, {df['target'].max()}] rings")
            return df
        except Exception as e:
            ABALONE_PATH = "data/abalone.data.csv"
            try:
                columns = ['sex', 'length', 'diameter', 'height', 'whole_weight',
                          'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
                df = pd.read_csv(ABALONE_PATH, names=columns, header=None)
                df['target'] = df['rings']
                df = df.drop(columns=['rings'])
                return df
            except:
                print(f"Abalone download failed: {e}")
                return None
    
    @staticmethod
    def _load_regression_synth() -> pd.DataFrame:
        """Synthesized regression dataset for testing"""
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=1000, n_features=20, n_informative=15, 
            noise=10.0, random_state=RANDOM_SEED
        )
        df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        df['target'] = y
        print(f"[Regression] Synthetic Regression loaded: {df.shape}")
        print(f"   Target range: [{df['target'].min():.1f}, {df['target'].max():.1f}]")
        return df
    
    # ==================== DATA PREPARATION ====================
    
    @staticmethod
    def prepare_data(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    task_type: str = 'auto') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Prepare data: handle missing values, encode categoricals, scale features.
        
        Correct order:
        1. Fit scaler/encoders on TRAIN only
        2. Transform TRAIN and TEST
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            task_type: 'classification', 'regression', or 'auto'
            
        Returns:
            X_train, X_test, y_train, y_test, transformers dict
        """
        transformers: Dict[str, Any] = {}
        
        # Handle numeric missing values
        numeric_cols_all = X_train.select_dtypes(include=['number']).columns
        median_values = X_train[numeric_cols_all].median()
        X_train = X_train.fillna(median_values)
        X_test = X_test.fillna(median_values)
        transformers['median_imputer'] = median_values

        # Encode categorical features
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            # Handle unseen categories in test
            test_vals = X_test[col].astype(str)
            known = set(le.classes_)
            mask_unknown = ~test_vals.isin(known)

            if mask_unknown.any():
                new_classes = np.append(le.classes_, "__UNKNOWN__")
                le.classes_ = new_classes
                test_vals = test_vals.where(~mask_unknown, "__UNKNOWN__")

            X_test[col] = le.transform(test_vals)
            encoders[col] = le

        transformers['label_encoders'] = encoders
        transformers['categorical_cols'] = categorical_cols

        # Encode target only for classification
        if task_type == 'classification':
            if y_train.dtype == 'object' or y_train.dtype.name == 'category':
                le = LabelEncoder()
                y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index)
                y_test = pd.Series(le.transform(y_test.astype(str)), index=y_test.index)
                transformers['target_encoder'] = le
            else:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)
        else:
            # For regression, ensure target is numeric and scale it
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)

            y_scaler = MinMaxScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

            y_train = pd.Series(y_train_scaled.ravel(), index=y_train.index, name=y_train.name)
            y_test = pd.Series(y_test_scaled.ravel(), index=y_test.index, name=y_test.name)
            transformers['y_scaler'] = y_scaler

        # Determine columns to scale
        cols_to_scale = []
        cols_categorical_int = []
        cols_boolean = []

        for col in numeric_cols_all:
            if pd.api.types.is_bool_dtype(X_train[col]):
                cols_boolean.append(col)
            elif X_train[col].nunique() < 20 and pd.api.types.is_integer_dtype(X_train[col]):
                cols_categorical_int.append(col)
            else:
                cols_to_scale.append(col)

        print(f"   Scaling {len(cols_to_scale)} features: {cols_to_scale}")
        print(f"   Skipping scaling for {len(cols_categorical_int)} int-categorical: {cols_categorical_int}")
        print(f"   Skipping scaling for {len(cols_boolean)} boolean: {cols_boolean}")

        if cols_to_scale:
            scaler = MinMaxScaler()
            X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

            transformers['x_scaler'] = scaler
            transformers['scaled_cols'] = cols_to_scale
            transformers['int_categorical_cols'] = cols_categorical_int
            transformers['boolean_cols'] = cols_boolean
            
        return X_train, X_test, y_train, y_test, transformers

    @staticmethod
    def create_train_splits(X: pd.DataFrame, y: pd.Series, 
                           split_ratios: List[float] = [0.2, 0.6, 1.0]) -> Dict:
        """
        Create multiple training splits of different sizes.
        
        Args:
            X, y: Full training data
            split_ratios: Ratios of data to use (e.g., [0.2, 0.6, 1.0])

        Returns:
            Dictionary with different split sizes
        """
        splits = {}
        n_samples = len(X)

        for ratio in split_ratios:
            n_split = int(n_samples * ratio)
            split_name = f"Train{int(ratio*100)}"

            X_split = X.iloc[:n_split]
            y_split = y.iloc[:n_split]

            splits[split_name] = {
                'X': X_split,
                'y': y_split,
                'size': n_split
            }

        return splits
