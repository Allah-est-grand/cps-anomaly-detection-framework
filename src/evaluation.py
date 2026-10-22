"""
Evaluation Module - Metrics, Cross-Validation, and Statistical Testing
"""
from typing import Dict, List, Tuple, Optional  # ✅ Gardez uniquement celle-ci
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, 
                             precision_recall_curve, confusion_matrix,
                             classification_report)
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
# ❌ SUPPRIMEZ la ligne duplicate: from typing import Dict, List, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CPSEvaluator:
    """
    Comprehensive evaluation framework for CPS anomaly detection models
    """
    
    def __init__(self, config):
        """
        Initialize evaluator with configuration
        
        Args:
            config: ValidationConfig object
        """
        self.config = config
        self.results = {}
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Calculate FPR and FNR
        cm = metrics['confusion_matrix']
        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            metrics['fpr'] = 0
            metrics['fnr'] = 0
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray,
                      model_name: str) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            model: Model to validate
            X: Features
            y: Labels
            model_name: Name of the model
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"\n🔄 Cross-validating {model_name} (k={self.config.k_folds})...")
        
        if self.config.use_stratified:
            cv = StratifiedKFold(n_splits=self.config.k_folds, 
                                shuffle=True, 
                                random_state=42)
        elif self.config.use_time_series:
            cv = TimeSeriesSplit(n_splits=self.config.k_folds)
        else:
            cv = self.config.k_folds
        
        # Compute scores for multiple metrics
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}
        
        for scoring in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                cv_results[f'{scoring}_mean'] = scores.mean()
                cv_results[f'{scoring}_std'] = scores.std()
                cv_results[f'{scoring}_scores'] = scores.tolist()
                logger.info(f"  {scoring}: {scores.mean()*100:.2f}% (+/- {scores.std()*2:.2f}%)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not compute {scoring}: {e}")
                cv_results[f'{scoring}_mean'] = 0.0
                cv_results[f'{scoring}_std'] = 0.0
        
        self.results[f'{model_name}_cv'] = cv_results
        return cv_results
    
    def generate_classification_report(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      target_names: List[str] = None) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for each class
            
        Returns:
            Formatted classification report string
        """
        if target_names is None:
            target_names = ['Normal', 'Attack']
        
        report = classification_report(y_true, y_pred, 
                                      target_names=target_names,
                                      digits=4)
        return report
    
    def compute_roc_curve(self, y_true: np.ndarray, 
                         y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve data
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return fpr, tpr, thresholds
    
    def compute_precision_recall_curve(self, y_true: np.ndarray,
                                      y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Precision-Recall curve data
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return precision, recall, thresholds
    
    def statistical_significance_test(self, scores1: np.ndarray, 
                                     scores2: np.ndarray,
                                     test_type: str = 'paired_t') -> Dict:
        """
        Perform statistical significance test between two models
        
        Args:
            scores1: Cross-validation scores for model 1
            scores2: Cross-validation scores for model 2
            test_type: Type of statistical test
            
        Returns:
            Dictionary with test results
        """
        from scipy import stats
        
        if test_type == 'paired_t':
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
        elif test_type == 'wilcoxon':
            t_stat, p_value = stats.wilcoxon(scores1, scores2)
        else:
            raise ValueError(f"Unknown test_type: {test_type}")
        
        result = {
            'test_type': test_type,
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        logger.info(f"📊 Statistical Test ({test_type}):")
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Significant (α=0.05): {'Yes' if result['significant'] else 'No'}")
        
        return result
    
    def save_results(self, results: Dict, filepath: str):
        """
        Save evaluation results to CSV
        
        Args:
            results: Dictionary with all results
            filepath: Output path
        """
        # Flatten results for CSV
        flat_results = {}
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat_results[f"{model_name}_{metric_name}"] = value
        
        df = pd.DataFrame([flat_results])
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Results saved to: {filepath}")