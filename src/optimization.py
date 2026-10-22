"""
Model Optimization Module - Hyperparameter Tuning and Feature Engineering
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Hyperparameter optimization framework for CPS anomaly detection models
    """
    
    def __init__(self, config):
        """
        Initialize optimizer
        
        Args:
            config: OptimizationConfig object
        """
        self.config = config
        self.best_params = {}
        self.optimization_history = []
    
    def optimize_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                              cv_folds: int = 3) -> Dict:
        """
        Optimize Random Forest hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with best parameters and score
        """
        logger.info("\n🔧 Optimizing Random Forest hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        from sklearn.ensemble import RandomForestClassifier
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Optimization complete in {elapsed/60:.2f} minutes")
        logger.info(f"🏆 Best F1-Score: {grid_search.best_score_*100:.2f}%")
        logger.info(f"⚙️ Best Parameters: {grid_search.best_params_}")
        
        self.best_params['RandomForest'] = grid_search.best_params_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'optimization_time': elapsed
        }
    
    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv_folds: int = 3) -> Dict:
        """
        Optimize XGBoost hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with best parameters and score
        """
        logger.info("\n🔧 Optimizing XGBoost hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'scale_pos_weight': [1, 5, 10, 20],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        import xgboost as xgb
        
        xgb_model = xgb.XGBClassifier(
            random_state=42, 
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv_folds, scoring='f1',
            n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Optimization complete in {elapsed/60:.2f} minutes")
        logger.info(f"🏆 Best F1-Score: {grid_search.best_score_*100:.2f}%")
        logger.info(f"⚙️ Best Parameters: {grid_search.best_params_}")
        
        self.best_params['XGBoost'] = grid_search.best_params_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'optimization_time': elapsed
        }
    
    def optimize_model(self, model_name: str, model, X_train: np.ndarray,
                      y_train: np.ndarray, cv_folds: int = None) -> Dict:
        """
        Generic model optimization dispatcher
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            Optimization results dictionary
        """
        if cv_folds is None:
            cv_folds = self.config.cv_folds
        
        if model_name == 'RandomForest':
            return self.optimize_random_forest(X_train, y_train, cv_folds)
        elif model_name == 'XGBoost':
            return self.optimize_xgboost(X_train, y_train, cv_folds)
        else:
            logger.warning(f"⚠️ No optimization routine for {model_name}")
            return {}
    
    def save_optimization_results(self, results: Dict, filepath: str):
        """Save optimization results to file"""
        import json
        import pickle
        
        # Save best parameters as JSON
        with open(filepath.replace('.pkl', '.json'), 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save full results as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"💾 Optimization results saved to: {filepath}")