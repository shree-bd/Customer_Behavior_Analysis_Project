"""
Model Training Module for Customer Behavior Analysis

This module contains advanced machine learning models and training pipelines
for predicting customer purchase behavior.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import resample
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Advanced model training class with multiple algorithms and evaluation"""
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scalers = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize various machine learning models"""
        self.models = {
            'xgboost': XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'mlp': MLPClassifier(
                random_state=self.random_state,
                max_iter=500
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def prepare_data(self, df, target_column='is_purchased', 
                    feature_columns=None, balance_method='downsample'):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
            feature_columns (list): List of feature columns
            balance_method (str): Method to balance classes ('downsample', 'upsample', 'smote')
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Default feature columns if not provided
        if feature_columns is None:
            feature_columns = [
                'brand', 'price', 'hour', 'day_of_week', 
                'category_level1', 'category_level2', 'total_events',
                'unique_products', 'avg_price'
            ]
        
        # Select features that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No valid feature columns found in dataframe")
        
        logger.info(f"Using features: {available_features}")
        
        # Prepare features and target
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('unknown'))
            self.scalers[f'{col}_encoder'] = le
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Balance classes
        X_train, y_train = self._balance_classes(
            X_train, y_train, method=balance_method
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_columns = X_train.select_dtypes(include=[np.number]).columns
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
        self.scalers['standard_scaler'] = scaler
        
        logger.info(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def _balance_classes(self, X, y, method='downsample'):
        """
        Balance classes using various methods
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            method (str): Balancing method
            
        Returns:
            tuple: Balanced X, y
        """
        logger.info(f"Balancing classes using {method} method...")
        
        if method == 'downsample':
            # Combine X and y
            df_combined = X.copy()
            df_combined['target'] = y
            
            # Separate classes
            df_majority = df_combined[df_combined.target == 0]
            df_minority = df_combined[df_combined.target == 1]
            
            # Downsample majority class
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=self.random_state
            )
            
            # Combine
            df_balanced = pd.concat([df_majority_downsampled, df_minority])
            
            X_balanced = df_balanced.drop('target', axis=1)
            y_balanced = df_balanced['target']
            
        elif method == 'upsample':
            # Similar to downsample but upsample minority class
            df_combined = X.copy()
            df_combined['target'] = y
            
            df_majority = df_combined[df_combined.target == 0]
            df_minority = df_combined[df_combined.target == 1]
            
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=self.random_state
            )
            
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            
            X_balanced = df_balanced.drop('target', axis=1)
            y_balanced = df_balanced['target']
            
        elif method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
            
        else:
            X_balanced, y_balanced = X, y
        
        logger.info(f"Class distribution after balancing: {y_balanced.value_counts().to_dict()}")
        
        return X_balanced, y_balanced
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance
        
        Args:
            X_train, X_test, y_train, y_test: Training and test data
        """
        logger.info("Training all models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'metrics': metrics
                }
                
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                          f"AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model
        self._find_best_model()
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _find_best_model(self):
        """Find the best performing model"""
        if not self.results:
            logger.warning("No models trained yet")
            return
        
        best_score = 0
        best_name = None
        
        for name, result in self.results.items():
            # Use F1 score as the primary metric
            score = result['metrics']['f1']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model = {
            'name': best_name,
            'model': self.results[best_name]['model'],
            'score': best_score
        }
        
        logger.info(f"Best model: {best_name} (F1: {best_score:.4f})")
    
    def hyperparameter_tuning(self, model_name, X_train, y_train, param_grid=None):
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name (str): Name of the model to tune
            X_train, y_train: Training data
            param_grid (dict): Parameter grid for tuning
            
        Returns:
            Best estimator
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_name)
        
        logger.info(f"Tuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _get_default_param_grid(self, model_name):
        """Get default parameter grids for hyperparameter tuning"""
        param_grids = {
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        }
        
        return param_grids.get(model_name, {})
    
    def generate_report(self):
        """Generate comprehensive model comparison report"""
        if not self.results:
            logger.warning("No models trained yet")
            return None
        
        report_data = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            report_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC': f"{metrics.get('auc', 0):.4f}"
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('F1-Score', ascending=False)
        
        return report_df
    
    def save_model(self, model_name=None, filepath='best_model.pkl'):
        """
        Save trained model to file
        
        Args:
            model_name (str): Name of model to save (if None, saves best model)
            filepath (str): Path to save the model
        """
        if model_name is None:
            if self.best_model is None:
                logger.error("No best model found")
                return
            model_to_save = self.best_model['model']
            model_name = self.best_model['name']
        else:
            if model_name not in self.results:
                logger.error(f"Model {model_name} not found")
                return
            model_to_save = self.results[model_name]['model']
        
        # Save model and scalers
        joblib.dump({
            'model': model_to_save,
            'scalers': self.scalers,
            'model_name': model_name
        }, filepath)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            saved_data = joblib.load(filepath)
            self.best_model = {
                'name': saved_data['model_name'],
                'model': saved_data['model']
            }
            self.scalers = saved_data['scalers']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def predict(self, X):
        """
        Make predictions using the best model
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.array: Predictions
        """
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        # Apply same preprocessing as training
        X_processed = X.copy()
        
        # Handle categorical variables
        for col in X_processed.select_dtypes(include=['object']).columns:
            if f'{col}_encoder' in self.scalers:
                le = self.scalers[f'{col}_encoder']
                X_processed[col] = le.transform(X_processed[col].fillna('unknown'))
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Scale numerical features
        if 'standard_scaler' in self.scalers:
            scaler = self.scalers['standard_scaler']
            numerical_columns = X_processed.select_dtypes(include=[np.number]).columns
            X_processed[numerical_columns] = scaler.transform(X_processed[numerical_columns])
        
        return self.best_model['model'].predict(X_processed)


# Utility functions
def feature_importance_analysis(model, feature_names):
    """
    Analyze feature importance for tree-based models
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return None


if __name__ == "__main__":
    # Example usage
    logger.info("Model Training Example")
    
    # Note: This is example code - replace with actual data loading
    # df = pd.read_csv("processed_data.csv")
    # 
    # trainer = ModelTrainer()
    # X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    # trainer.train_all_models(X_train, X_test, y_train, y_test)
    # 
    # # Generate report
    # report = trainer.generate_report()
    # print(report)
    # 
    # # Save best model
    # trainer.save_model()
    
    pass 