"""
Error Analyzer - Detailed Error Analysis Module
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements detailed error analysis to verify how errors change between
Baseline Detection Pipeline and Weighted Retraining Pipeline.

FOCUS AREAS:
------------
- False negatives (FN)
- False positives (FP)
- Corrected false negatives
- Newly introduced false positives
- Persistent errors
- Model-dependent error behavior

OUTPUT FILES:
-------------
- results/adaptive/final_validation/error_analysis/metrics/error_analysis_results.csv
- results/adaptive/final_validation/error_analysis/metrics/error_analysis_summary.json
- results/adaptive/final_validation/error_analysis/figures/error_transition_summary.png
- results/adaptive/final_validation/error_analysis/figures/false_negative_correction.png
- results/adaptive/final_validation/error_analysis/figures/false_positive_tradeoff.png
- results/adaptive/final_validation/error_analysis/figures/persistent_errors_by_model.png
- results/adaptive/final_validation/error_analysis/reports/error_analysis_report.txt
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
import sys
import time
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to sys.path for config import
# This file is at: src/adaptive_feedback/final_validation/error_analysis.py
# We need to go up 3 levels to reach project root where config.py exists
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now config.py can be imported from project root
from config import FrameworkConfig
from src.utils import setup_logging, set_random_seeds

# Data processing and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 2024, 2025, 3407]

# Only primary models for error analysis
MODELS_OF_INTEREST = ['XGBoost', 'RandomForest']

NORMAL_WEIGHT = 1.0
ANOMALY_WEIGHT = 3.0

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']

METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

# =============================================================================
# ERROR ANALYZER CLASS
# =============================================================================
class ErrorAnalyzer:
    """
    Implements detailed error analysis for adaptive feedback validation
    
    This class analyzes how errors change between baseline and weighted retraining
    pipelines, focusing on false negatives, false positives, and error transitions.
    """
    
    def __init__(self, base_dir: Path, analysis_dir: Optional[Path] = None):
        """
        Initialize Error Analyzer
        
        Args:
            base_dir: Base results directory
            analysis_dir: Path to save error analysis outputs
        """
        self.base_dir = Path(base_dir)
        self.analysis_dir = Path(analysis_dir) if analysis_dir else (
            self.base_dir / 'adaptive' / 'final_validation' / 'error_analysis'
        )
        
        # Create output directories
        (self.analysis_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / 'reports').mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.original_dataset = None
        self.results = []
        self.error_transitions = []
        self.statistics = {}
        self.analysis_status = 'pending'
        self.data_source_description = ''
        
        # Dataset path (same as previous modules)
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized ErrorAnalyzer")
        logger.info(f"📁 Analysis directory: {self.analysis_dir}")
        logger.info(f"🔢 Testing {len(SEEDS)} seeds: {SEEDS}")
    
    def load_resources(self) -> bool:
        """
        Load dataset and resources
        
        Returns:
            True if loaded successfully
        """
        logger.info("📊 Loading resources...")
        
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.dataset_path}")
            return False
        
        try:
            self.original_dataset = pd.read_csv(self.dataset_path)
            self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
            
            logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows, "
                       f"{len(self.original_dataset.columns)} columns")
            self.data_source_description = str(self.dataset_path)
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def _identify_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify label column"""
        for col_name in LABEL_COLUMN_NAMES:
            if col_name in df.columns:
                return col_name
        for col in df.columns:
            if col.lower() in [name.lower() for name in LABEL_COLUMN_NAMES]:
                return col
        return None
    
    def _convert_labels_to_binary(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        """Convert labels to binary (0=normal, 1=anomaly)"""
        labels_raw = df[label_col].values
        labels = np.zeros(len(labels_raw), dtype=int)
        normal_indicators = ['normal', '0', 'benign', 'ok', 'healthy', 'false', 'negative']
        
        for i, label in enumerate(labels_raw):
            if str(label).lower().strip() not in normal_indicators:
                labels[i] = 1
        
        return labels
    
    def _prepare_features(self, df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, List[str]]:
        """Prepare features excluding metadata and label columns"""
        exclude_cols = set(METADATA_COLUMNS)
        
        for label_name in LABEL_COLUMN_NAMES:
            exclude_cols.add(label_name.lower())
        
        exclude_cols.add(label_col.lower())
        
        feature_cols = []
        for col in df.columns:
            if col.lower() in exclude_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        if len(feature_cols) == 0:
            return None, []
        
        features = df[feature_cols].values
        return features, feature_cols
    
    def _create_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        """Create anomaly-sensitive sample weights"""
        weights = np.ones(len(labels))
        weights[labels == 1] = ANOMALY_WEIGHT
        return weights
    
    def run_single_seed(self, seed: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Run complete error analysis experiment for a single seed
        
        Args:
            seed: Random seed for this experiment
            
        Returns:
            Tuple of (results list, error transitions list)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running error analysis seed = {seed}")
        logger.info(f"{'='*80}")
        
        results = []
        error_transitions = []
        
        try:
            set_random_seeds(seed)
            
            # Identify label column
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return results, error_transitions
            
            # Convert labels
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            # Prepare features
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return results, error_transitions
            
            # Split data (no leakage)
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_features, all_labels,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=seed
            )
            
            X_train, X_retrain, y_train, y_retrain = train_test_split(
                X_temp, y_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=seed
            )
            
            # Preprocessing (fit on train ONLY)
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            # Train baseline models
            logger.info(f"  Training baseline models...")
            baseline_models = self.train_baseline_models(X_train_scaled, y_train)
            
            # Train weighted retraining models
            logger.info(f"  Training weighted retraining models...")
            X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
            y_combined = np.concatenate([y_train, y_retrain])
            sample_weights = self._create_sample_weights(y_combined)
            
            weighted_models = self.train_weighted_models(X_combined, y_combined, sample_weights)
            
            # Evaluate and compute error transitions for each model
            for model_name in MODELS_OF_INTEREST:
                if model_name not in baseline_models or model_name not in weighted_models:
                    continue
                
                baseline_model = baseline_models[model_name]
                weighted_model = weighted_models[model_name]
                
                # Get predictions
                baseline_pred = baseline_model.predict(X_test_scaled)
                weighted_pred = weighted_model.predict(X_test_scaled)
                
                # Compute metrics
                baseline_metrics = self.evaluate_metrics(y_test, baseline_pred, 'Baseline', seed, model_name)
                weighted_metrics = self.evaluate_metrics(y_test, weighted_pred, 'Weighted_Retraining', seed, model_name)
                
                results.append(baseline_metrics)
                results.append(weighted_metrics)
                
                # Compute error transitions
                error_transition = self.compute_error_transitions(
                    y_test, baseline_pred, weighted_pred, model_name, seed
                )
                error_transitions.append(error_transition)
                
                logger.info(f"    {model_name}: Baseline FN={baseline_metrics['false_negative']}, "
                           f"Weighted FN={weighted_metrics['false_negative']}")
            
            logger.info(f"  ✅ Seed {seed} completed: {len(results)} results, {len(error_transitions)} transitions")
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
        
        return results, error_transitions
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train baseline models"""
        models = {}
        
        # XGBoost
        try:
            xgb_model = xgb.XGBClassifier(**XGB_CONFIG)
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            logger.info(f"    ✅ Trained Baseline XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Baseline XGBoost failed: {e}")
        
        # Random Forest
        try:
            rf_model = RandomForestClassifier(**RF_CONFIG)
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
            logger.info(f"    ✅ Trained Baseline RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Baseline RandomForest failed: {e}")
        
        return models
    
    def train_weighted_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             sample_weights: np.ndarray) -> Dict[str, Any]:
        """Train weighted retrained models"""
        models = {}
        
        # XGBoost with sample weights
        try:
            xgb_model = xgb.XGBClassifier(**XGB_CONFIG)
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
            models['XGBoost'] = xgb_model
            logger.info(f"    ✅ Trained Weighted XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted XGBoost failed: {e}")
        
        # Random Forest with sample weights
        try:
            rf_model = RandomForestClassifier(**RF_CONFIG)
            rf_model.fit(X_train, y_train, sample_weight=sample_weights)
            models['RandomForest'] = rf_model
            logger.info(f"    ✅ Trained Weighted RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted RandomForest failed: {e}")
        
        return models
    
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        configuration: str, seed: int, model_name: str) -> Dict:
        """Compute evaluation metrics"""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'seed': seed,
            'model_name': model_name,
            'configuration': configuration,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'status': 'success'
        }
    
    def compute_error_transitions(self, y_true: np.ndarray, baseline_pred: np.ndarray,
                                  weighted_pred: np.ndarray, model_name: str, seed: int) -> Dict:
        """
        Compute detailed error transition analysis
        
        Args:
            y_true: True labels
            baseline_pred: Baseline predictions
            weighted_pred: Weighted retraining predictions
            model_name: Model name
            seed: Random seed
            
        Returns:
            Dictionary with error transition analysis
        """
        # Baseline errors
        baseline_fn_mask = (baseline_pred == 0) & (y_true == 1)
        baseline_fp_mask = (baseline_pred == 1) & (y_true == 0)
        
        # Weighted errors
        weighted_fn_mask = (weighted_pred == 0) & (y_true == 1)
        weighted_fp_mask = (weighted_pred == 1) & (y_true == 0)
        
        # Error transitions
        # Corrected FN: baseline predicted 0 (wrong), weighted predicted 1 (correct), true=1
        corrected_fn_mask = (baseline_pred == 0) & (weighted_pred == 1) & (y_true == 1)
        
        # Persistent FN: both predicted 0 (wrong), true=1
        persistent_fn_mask = (baseline_pred == 0) & (weighted_pred == 0) & (y_true == 1)
        
        # Newly introduced FN: baseline predicted 1 (correct), weighted predicted 0 (wrong), true=1
        newly_introduced_fn_mask = (baseline_pred == 1) & (weighted_pred == 0) & (y_true == 1)
        
        # Corrected FP: baseline predicted 1 (wrong), weighted predicted 0 (correct), true=0
        corrected_fp_mask = (baseline_pred == 1) & (weighted_pred == 0) & (y_true == 0)
        
        # Persistent FP: both predicted 1 (wrong), true=0
        persistent_fp_mask = (baseline_pred == 1) & (weighted_pred == 1) & (y_true == 0)
        
        # Newly introduced FP: baseline predicted 0 (correct), weighted predicted 1 (wrong), true=0
        newly_introduced_fp_mask = (baseline_pred == 0) & (weighted_pred == 1) & (y_true == 0)
        
        return {
            'seed': seed,
            'model_name': model_name,
            'baseline_false_negatives': int(np.sum(baseline_fn_mask)),
            'weighted_false_negatives': int(np.sum(weighted_fn_mask)),
            'false_negative_reduction': int(np.sum(baseline_fn_mask) - np.sum(weighted_fn_mask)),
            'corrected_false_negatives': int(np.sum(corrected_fn_mask)),
            'persistent_false_negatives': int(np.sum(persistent_fn_mask)),
            'newly_introduced_false_negatives': int(np.sum(newly_introduced_fn_mask)),
            'baseline_false_positives': int(np.sum(baseline_fp_mask)),
            'weighted_false_positives': int(np.sum(weighted_fp_mask)),
            'false_positive_change': int(np.sum(weighted_fp_mask) - np.sum(baseline_fp_mask)),
            'corrected_false_positives': int(np.sum(corrected_fp_mask)),
            'persistent_false_positives': int(np.sum(persistent_fp_mask)),
            'newly_introduced_false_positives': int(np.sum(newly_introduced_fp_mask)),
            'total_test_samples': len(y_true),
            'total_actual_anomalies': int(np.sum(y_true == 1)),
            'total_actual_normal': int(np.sum(y_true == 0))
        }
    
    def compute_statistics(self) -> Dict:
        """Compute statistical analysis across all seeds"""
        logger.info("\n📊 Computing statistics...")
        
        if not self.results or not self.error_transitions:
            logger.error("❌ No results available")
            return {}
        
        results_df = pd.DataFrame(self.results)
        transitions_df = pd.DataFrame(self.error_transitions)
        
        # Filter successful results
        successful = results_df[results_df['status'] == 'success']
        
        statistics = {}
        
        for model_name in MODELS_OF_INTEREST:
            model_data = successful[successful['model_name'] == model_name]
            model_transitions = transitions_df[transitions_df['model_name'] == model_name]
            
            if len(model_data) == 0 or len(model_transitions) == 0:
                continue
            
            statistics[model_name] = {}
            
            for config in ['Baseline', 'Weighted_Retraining']:
                config_data = model_data[model_data['configuration'] == config]
                
                if len(config_data) > 0:
                    statistics[model_name][config] = {
                        'n_seeds': len(config_data),
                        'mean_f1': round(config_data['f1_score'].mean(), 4),
                        'std_f1': round(config_data['f1_score'].std(), 4),
                        'mean_recall': round(config_data['recall'].mean(), 4),
                        'std_recall': round(config_data['recall'].std(), 4),
                        'mean_precision': round(config_data['precision'].mean(), 4),
                        'std_precision': round(config_data['precision'].std(), 4),
                        'mean_fn': round(config_data['false_negative'].mean(), 2),
                        'std_fn': round(config_data['false_negative'].std(), 2),
                        'mean_fp': round(config_data['false_positive'].mean(), 2),
                        'std_fp': round(config_data['false_positive'].std(), 2)
                    }
            
            # Error transition statistics
            if len(model_transitions) > 0:
                statistics[model_name]['error_transitions'] = {
                    'mean_fn_reduction': round(model_transitions['false_negative_reduction'].mean(), 2),
                    'std_fn_reduction': round(model_transitions['false_negative_reduction'].std(), 2),
                    'mean_corrected_fn': round(model_transitions['corrected_false_negatives'].mean(), 2),
                    'std_corrected_fn': round(model_transitions['corrected_false_negatives'].std(), 2),
                    'mean_persistent_fn': round(model_transitions['persistent_false_negatives'].mean(), 2),
                    'std_persistent_fn': round(model_transitions['persistent_false_negatives'].std(), 2),
                    'mean_newly_introduced_fn': round(model_transitions['newly_introduced_false_negatives'].mean(), 2),
                    'mean_fp_change': round(model_transitions['false_positive_change'].mean(), 2),
                    'mean_corrected_fp': round(model_transitions['corrected_false_positives'].mean(), 2),
                    'mean_persistent_fp': round(model_transitions['persistent_false_positives'].mean(), 2),
                    'mean_newly_introduced_fp': round(model_transitions['newly_introduced_false_positives'].mean(), 2)
                }
                
                # Compute improvements
                if 'Baseline' in statistics[model_name] and 'Weighted_Retraining' in statistics[model_name]:
                    baseline = statistics[model_name]['Baseline']
                    weighted = statistics[model_name]['Weighted_Retraining']
                    
                    statistics[model_name]['improvements'] = {
                        'f1_improvement': round(weighted['mean_f1'] - baseline['mean_f1'], 4),
                        'recall_improvement': round(weighted['mean_recall'] - baseline['mean_recall'], 4),
                        'precision_change': round(weighted['mean_precision'] - baseline['mean_precision'], 4),
                        'fn_reduction': round(baseline['mean_fn'] - weighted['mean_fn'], 2)
                    }
        
        self.statistics = statistics
        logger.info(f"✅ Statistics computed for {len(statistics)} models")
        
        return statistics
    
    def save_results(self) -> List[str]:
        """Save results to CSV and JSON"""
        logger.info("💾 Saving error analysis results...")
        
        saved_paths = []
        
        try:
            # Save results CSV
            if self.results:
                results_df = pd.DataFrame(self.results)
                csv_path = self.analysis_dir / 'metrics' / 'error_analysis_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"  ✅ Saved results CSV: {csv_path}")
            
            # Save error transitions CSV
            if self.error_transitions:
                transitions_df = pd.DataFrame(self.error_transitions)
                transitions_path = self.analysis_dir / 'metrics' / 'error_transitions.csv'
                transitions_df.to_csv(transitions_path, index=False)
                saved_paths.append(str(transitions_path))
                logger.info(f"  ✅ Saved error transitions CSV: {transitions_path}")
            
            # Save JSON summary
            if self.statistics:
                summary = {
                    'analysis_status': self.analysis_status,
                    'tested_seeds': SEEDS,
                    'evaluated_models': list(self.statistics.keys()),
                    'anomaly_weight': ANOMALY_WEIGHT,
                    'normal_weight': NORMAL_WEIGHT,
                    'statistics': self.statistics,
                    'total_experiments': len(self.results),
                    'total_error_transitions': len(self.error_transitions)
                }
                
                json_path = self.analysis_dir / 'metrics' / 'error_analysis_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"  ✅ Saved JSON: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def generate_figures(self) -> List[str]:
        """Generate publication-quality error analysis figures"""
        logger.info("🎨 Generating error analysis figures...")
        
        saved_paths = []
        
        if not self.statistics or not self.error_transitions:
            logger.warning("⚠️  No statistics for figure generation")
            return saved_paths
        
        transitions_df = pd.DataFrame(self.error_transitions)
        
        # FIGURE 1: Error Transition Summary (no error bars, simple bar chart)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            for model_idx, model_name in enumerate(MODELS_OF_INTEREST):
                if model_name not in self.statistics:
                    continue
                
                ax = axes[model_idx]
                model_transitions = transitions_df[transitions_df['model_name'] == model_name]
                
                if len(model_transitions) == 0:
                    continue
                
                # Aggregate across seeds
                mean_corrected_fn = model_transitions['corrected_false_negatives'].mean()
                mean_persistent_fn = model_transitions['persistent_false_negatives'].mean()
                mean_newly_fn = model_transitions['newly_introduced_false_negatives'].mean()
                mean_corrected_fp = model_transitions['corrected_false_positives'].mean()
                mean_newly_fp = model_transitions['newly_introduced_false_positives'].mean()
                
                categories = ['Corrected\nFN', 'Persistent\nFN', 'Newly\nFN', 'Corrected\nFP', 'Newly\nFP']
                values = [mean_corrected_fn, mean_persistent_fn, mean_newly_fn, mean_corrected_fp, mean_newly_fp]
                colors = ['#27ae60', '#e74c3c', '#e67e22', '#27ae60', '#e74c3c']
                
                x = range(len(categories))
                bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.8)
                
                # Dynamic y-axis limits and label placement
                y_min = min(values)
                y_max = max(values)
                y_range = y_max - y_min if y_max > y_min else max(abs(y_max), 1)
                
                # Add 25% margin for label clearance
                margin = y_range * 0.25
                ax.set_ylim(y_min - margin, y_max + margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # Add value labels with proper placement
                for bar, val in zip(bars, values):
                    if val >= 0:
                        label_y = val + offset
                        va = 'bottom'
                    else:
                        label_y = val - offset
                        va = 'top'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:.1f}', ha='center', va=va, fontsize=9, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=9)
                ax.set_ylabel('Mean Count', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: Error Transition Summary', fontsize=13, fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'error_transition_summary.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved error transition summary: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate error transition figure: {e}")
        
        # FIGURE 2: False Negative Correction (FIXED: label above bar + std + offset)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(MODELS_OF_INTEREST):
                if model_name not in self.statistics:
                    continue
                
                ax = axes[model_idx]
                stats = self.statistics[model_name]
                
                if 'Baseline' not in stats or 'Weighted_Retraining' not in stats:
                    continue
                
                baseline_fn = stats['Baseline']['mean_fn']
                weighted_fn = stats['Weighted_Retraining']['mean_fn']
                baseline_std = stats['Baseline']['std_fn']
                weighted_std = stats['Weighted_Retraining']['std_fn']
                
                categories = ['Baseline', 'Weighted\nRetraining']
                values = [baseline_fn, weighted_fn]
                stds = [baseline_std, weighted_std]
                colors = ['#3498db', '#27ae60']
                
                x = range(len(categories))
                bars = ax.bar(x, values, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
                
                # FIXED: Dynamic y-axis limits including error bars and label margin
                y_max_with_err = max(v + s for v, s in zip(values, stds))
                y_min_with_err = min(v - s for v, s in zip(values, stds))
                y_range = y_max_with_err - y_min_with_err if y_max_with_err > y_min_with_err else max(abs(y_max_with_err), 1)
                
                # Add margin for label clearance (label should be above bar + std + offset)
                label_margin = y_range * 0.25
                ax.set_ylim(max(0, y_min_with_err - label_margin), y_max_with_err + label_margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # FIXED: Place label above bar + std + offset (not touching error bar)
                for bar, val, std in zip(bars, values, stds):
                    label_y = val + std + offset  # Above bar + error bar + offset
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=11)
                ax.set_ylabel('False Negatives (mean ± std)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: FN Correction', fontsize=13, fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'false_negative_correction.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved FN correction figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FN correction figure: {e}")
        
        # FIGURE 3: False Positive Trade-off (FIXED: label above bar + std + offset)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(MODELS_OF_INTEREST):
                if model_name not in self.statistics:
                    continue
                
                ax = axes[model_idx]
                stats = self.statistics[model_name]
                
                if 'Baseline' not in stats or 'Weighted_Retraining' not in stats:
                    continue
                
                baseline_fp = stats['Baseline']['mean_fp']
                weighted_fp = stats['Weighted_Retraining']['mean_fp']
                baseline_std = stats['Baseline']['std_fp']
                weighted_std = stats['Weighted_Retraining']['std_fp']
                fp_change = weighted_fp - baseline_fp
                
                categories = ['Baseline', 'Weighted\nRetraining']
                values = [baseline_fp, weighted_fp]
                stds = [baseline_std, weighted_std]
                colors = ['#3498db', '#e74c3c']
                
                x = range(len(categories))
                bars = ax.bar(x, values, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
                
                # FIXED: Dynamic y-axis limits including error bars and label margin
                y_max_with_err = max(v + s for v, s in zip(values, stds))
                y_min_with_err = min(v - s for v, s in zip(values, stds))
                y_range = y_max_with_err - y_min_with_err if y_max_with_err > y_min_with_err else max(abs(y_max_with_err), 1)
                
                # Add margin for label clearance
                label_margin = y_range * 0.25
                ax.set_ylim(max(0, y_min_with_err - label_margin), y_max_with_err + label_margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # FIXED: Place label above bar + std + offset (not touching error bar)
                for bar, val, std in zip(bars, values, stds):
                    label_y = val + std + offset  # Above bar + error bar + offset
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                # Add FP change annotation (positioned in middle of plot, away from labels)
                annotation_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5
                ax.text(1.5, annotation_y, f'ΔFP: {fp_change:+.1f}',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=11)
                ax.set_ylabel('False Positives (mean ± std)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: FP Trade-off', fontsize=13, fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'false_positive_tradeoff.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved FP trade-off figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FP trade-off figure: {e}")
        
        # FIGURE 4: Persistent Errors by Model (FIXED: label above bar + std + offset)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(MODELS_OF_INTEREST):
                if model_name not in self.statistics:
                    continue
                
                ax = axes[model_idx]
                model_transitions = transitions_df[transitions_df['model_name'] == model_name]
                
                if len(model_transitions) == 0:
                    continue
                
                mean_persistent_fn = model_transitions['persistent_false_negatives'].mean()
                mean_persistent_fp = model_transitions['persistent_false_positives'].mean()
                std_persistent_fn = model_transitions['persistent_false_negatives'].std()
                std_persistent_fp = model_transitions['persistent_false_positives'].std()
                
                categories = ['Persistent\nFN', 'Persistent\nFP']
                values = [mean_persistent_fn, mean_persistent_fp]
                stds = [std_persistent_fn, std_persistent_fp]
                colors = ['#e74c3c', '#e67e22']
                
                x = range(len(categories))
                bars = ax.bar(x, values, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
                
                # FIXED: Dynamic y-axis limits including error bars and label margin
                y_max_with_err = max(v + s for v, s in zip(values, stds))
                y_min_with_err = min(v - s for v, s in zip(values, stds))
                y_range = y_max_with_err - y_min_with_err if y_max_with_err > y_min_with_err else max(abs(y_max_with_err), 1)
                
                # Add margin for label clearance
                label_margin = y_range * 0.25
                ax.set_ylim(max(0, y_min_with_err - label_margin), y_max_with_err + label_margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # FIXED: Place label above bar + std + offset (not touching error bar)
                for bar, val, std in zip(bars, values, stds):
                    label_y = val + std + offset  # Above bar + error bar + offset
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(categories, fontsize=11)
                ax.set_ylabel('Persistent Errors (mean ± std)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: Persistent Errors', fontsize=13, fontweight='bold', pad=15)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'persistent_errors_by_model.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved persistent errors figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate persistent errors figure: {e}")
        
        return saved_paths
    
    def generate_report(self) -> str:
        """Generate error analysis report"""
        logger.info("📝 Generating error analysis report...")
        
        lines = []
        lines.append("="*80)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("Detailed Error Behavior Analysis for Adaptive Feedback Framework")
        lines.append("="*80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Framework Version: 1.0.0")
        lines.append(f"Analysis Status: {self.analysis_status.upper()}")
        lines.append("")
        
        # Analysis overview
        lines.append("ANALYSIS OVERVIEW:")
        lines.append(f"  - Data source: {self.data_source_description}")
        lines.append(f"  - Random seeds: {SEEDS}")
        lines.append(f"  - Models evaluated: {', '.join(MODELS_OF_INTEREST)}")
        lines.append(f"  - Configurations: Baseline vs Weighted Retraining")
        lines.append(f"  - Anomaly weight: {ANOMALY_WEIGHT}")
        lines.append(f"  - Normal weight: {NORMAL_WEIGHT}")
        lines.append(f"  - Total experiments: {len(self.results)}")
        lines.append(f"  - Total error transitions analyzed: {len(self.error_transitions)}")
        lines.append("")
        
        # Purpose
        lines.append("PURPOSE OF ERROR ANALYSIS:")
        lines.append("  This error analysis aims to understand how errors change between")
        lines.append("  Baseline and Weighted Retraining pipelines, focusing on:")
        lines.append("  - False negative reduction (critical for CPS safety)")
        lines.append("  - False positive trade-offs")
        lines.append("  - Corrected errors (baseline wrong, weighted correct)")
        lines.append("  - Persistent errors (both configurations wrong)")
        lines.append("  - Newly introduced errors (baseline correct, weighted wrong)")
        lines.append("")
        
        # Results summary
        lines.append("RESULTS SUMMARY:")
        
        if self.statistics:
            for model_name in MODELS_OF_INTEREST:
                if model_name not in self.statistics:
                    continue
                
                stats = self.statistics[model_name]
                lines.append(f"  - {model_name}:")
                
                # Metrics comparison
                if 'Baseline' in stats and 'Weighted_Retraining' in stats:
                    baseline = stats['Baseline']
                    weighted = stats['Weighted_Retraining']
                    
                    lines.append(f"      Baseline: F1={baseline['mean_f1']:.4f}, Recall={baseline['mean_recall']:.4f}, FN={baseline['mean_fn']:.1f}")
                    lines.append(f"      Weighted: F1={weighted['mean_f1']:.4f}, Recall={weighted['mean_recall']:.4f}, FN={weighted['mean_fn']:.1f}")
                
                # Error transitions
                if 'error_transitions' in stats:
                    trans = stats['error_transitions']
                    lines.append(f"      Error Transitions:")
                    lines.append(f"          FN Reduction: {trans['mean_fn_reduction']:.1f}")
                    lines.append(f"          Corrected FN: {trans['mean_corrected_fn']:.1f}")
                    lines.append(f"          Persistent FN: {trans['mean_persistent_fn']:.1f}")
                    lines.append(f"          Newly Introduced FN: {trans['mean_newly_introduced_fn']:.1f}")
                    lines.append(f"          FP Change: {trans['mean_fp_change']:+.1f}")
                    lines.append(f"          Corrected FP: {trans['mean_corrected_fp']:.1f}")
                    lines.append(f"          Persistent FP: {trans['mean_persistent_fp']:.1f}")
                    lines.append(f"          Newly Introduced FP: {trans['mean_newly_introduced_fp']:.1f}")
                
                # Improvements
                if 'improvements' in stats:
                    imp = stats['improvements']
                    lines.append(f"      Improvements:")
                    lines.append(f"          F1: {imp['f1_improvement']:+.4f}")
                    lines.append(f"          Recall: {imp['recall_improvement']:+.4f}")
                    lines.append(f"          Precision: {imp['precision_change']:+.4f}")
        
        lines.append("")
        
        # Scientific interpretation
        lines.append("SCIENTIFIC INTERPRETATION:")
        lines.append("")
        
        if self.statistics:
            for model_name in MODELS_OF_INTEREST:
                if model_name not in self.statistics:
                    continue
                
                stats = self.statistics[model_name]
                
                lines.append(f"  {model_name} Error Behavior:")
                
                if 'improvements' in stats:
                    imp = stats['improvements']
                    trans = stats.get('error_transitions', {})
                    
                    # FN reduction
                    if imp.get('fn_reduction', 0) > 100:
                        lines.append(f"    - The results indicate substantial false negative reduction ({imp['fn_reduction']:.1f})")
                    elif imp.get('fn_reduction', 0) > 0:
                        lines.append(f"    - The results indicate moderate false negative reduction ({imp['fn_reduction']:.1f})")
                    else:
                        lines.append(f"    - False negative reduction is limited ({imp.get('fn_reduction', 0):.1f})")
                    
                    # Corrected vs persistent
                    if trans.get('mean_corrected_fn', 0) > trans.get('mean_persistent_fn', 0):
                        lines.append(f"    - More false negatives are corrected than remain persistent")
                    else:
                        lines.append(f"    - Many false negatives remain persistent, indicating need for further refinement")
                    
                    # FP trade-off
                    if trans.get('mean_fp_change', 0) > 50:
                        lines.append(f"    - False positives increase notably ({trans['mean_fp_change']:+.1f}), indicating a trade-off")
                    elif trans.get('mean_fp_change', 0) > 0:
                        lines.append(f"    - False positives show modest increase ({trans['mean_fp_change']:+.1f})")
                    else:
                        lines.append(f"    - False positives remain stable or decrease ({trans.get('mean_fp_change', 0):+.1f})")
                
                lines.append("")
        
        # General conclusions
        lines.append("  General Conclusions:")
        lines.append("    - The error reduction is model-dependent")
        lines.append("    - Weighted retraining reduces false negatives under the tested setting")
        lines.append("    - Some errors remain persistent, indicating that further refinement may be needed")
        lines.append("    - The false positive trade-off should be considered for deployment")
        lines.append("")
        
        lines.append("  ⚠️  CAUTION:")
        lines.append("  ⚠️  - These results are from error analysis on a single dataset")
        lines.append("  ⚠️  - Do not claim perfect detection or universal generalization")
        lines.append("  ⚠️  - The results should not be interpreted as universal improvement")
        lines.append("  ⚠️  - External validation is recommended before deployment claims")
        lines.append("")
        
        lines.append("="*80)
        lines.append("END OF ERROR ANALYSIS REPORT")
        lines.append("="*80)
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.analysis_dir / 'reports' / 'error_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return report_text
    
    def run_full_analysis(self) -> Dict:
        """Run complete error analysis"""
        logger.info("\n" + "="*80)
        logger.info("🔍 RUNNING ERROR ANALYSIS")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'analysis_complete': False
        }
        
        # Step 1: Load resources
        if not self.load_resources():
            logger.error("❌ Failed to load resources")
            results['status'] = 'failed'
            return results
        
        # Step 2: Run experiments for each seed
        for seed in SEEDS:
            seed_results, seed_transitions = self.run_single_seed(seed)
            self.results.extend(seed_results)
            self.error_transitions.extend(seed_transitions)
        
        if not self.results:
            logger.error("❌ No results generated")
            results['status'] = 'failed'
            return results
        
        # Step 3: Compute statistics
        self.compute_statistics()
        
        # Step 4: Set analysis status to complete BEFORE saving
        self.analysis_status = 'complete'
        results['status'] = 'complete'
        results['analysis_complete'] = True
        
        # Step 5: Save results
        saved_paths = self.save_results()
        results['outputs_saved'].extend(saved_paths)
        
        # Step 6: Generate figures
        fig_paths = self.generate_figures()
        results['outputs_saved'].extend(fig_paths)
        
        # Step 7: Generate report
        self.generate_report()
        results['outputs_saved'].append(str(self.analysis_dir / 'reports' / 'error_analysis_report.txt'))
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute error analysis"""
    global logger
    
    start_time = time.time()
    
    # Initialize configuration
    config = FrameworkConfig()
    
    # Setup logging
    logger = setup_logging(config.output.logs_dir, 'error_analysis')
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ERROR ANALYSIS - Detailed Error Behavior Study")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize paths
    base_dir = Path(config.output.base_dir)
    analysis_dir = base_dir / 'adaptive' / 'final_validation' / 'error_analysis'
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = ErrorAnalyzer(base_dir=base_dir, analysis_dir=analysis_dir)
    results = analyzer.run_full_analysis()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ERROR ANALYSIS COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()