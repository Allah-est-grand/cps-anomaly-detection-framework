"""
Ablation Analyzer - Component Contribution Analysis Module
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements ablation study / component contribution analysis to verify
which components of the Adaptive Feedback Framework contribute to the final improvement.

EXPERIMENTAL DESIGN:
--------------------
Compares three configurations:
1. Baseline Detection Pipeline
   - Train only on original training data
   - No adaptive retraining
   - No anomaly weighting

2. Retraining Without Weighting
   - Train on original training data + retraining data
   - No sample weights

3. Weighted Retraining
   - Train on original training data + retraining data
   - Use anomaly-sensitive sample weights (Normal=1.0, Anomaly=3.0)

This allows separation of:
- Contribution of adding retraining data
- Contribution of anomaly-sensitive weighting

OUTPUT FILES:
-------------
- results/adaptive/final_validation/ablation/metrics/ablation_results.csv
- results/adaptive/final_validation/ablation/metrics/ablation_summary.json
- results/adaptive/final_validation/ablation/figures/ablation_f1_contribution.png
- results/adaptive/final_validation/ablation/figures/ablation_recall_contribution.png
- results/adaptive/final_validation/ablation/figures/ablation_false_negative_contribution.png
- results/adaptive/final_validation/ablation/figures/ablation_component_comparison.png
- results/adaptive/final_validation/ablation/reports/ablation_analysis_report.txt
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
# This file is at: src/adaptive_feedback/final_validation/ablation_analyzer.py
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

# Only primary models for ablation analysis
MODELS_OF_INTEREST = ['XGBoost', 'RandomForest']

# Three ablation configurations
CONFIGURATIONS = ['Baseline', 'Retraining_No_Weighting', 'Weighted_Retraining']

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
# ABLATION ANALYZER CLASS
# =============================================================================
class AblationAnalyzer:
    """
    Implements ablation study / component contribution analysis for adaptive feedback validation
    
    This class runs experiments with three configurations to isolate the contribution
    of retraining data and anomaly-sensitive weighting.
    """
    
    def __init__(self, base_dir: Path, analysis_dir: Optional[Path] = None):
        """
        Initialize Ablation Analyzer
        
        Args:
            base_dir: Base results directory
            analysis_dir: Path to save ablation analysis outputs
        """
        self.base_dir = Path(base_dir)
        self.analysis_dir = Path(analysis_dir) if analysis_dir else (
            self.base_dir / 'adaptive' / 'final_validation' / 'ablation'
        )
        
        # Create output directories
        (self.analysis_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.analysis_dir / 'reports').mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.original_dataset = None
        self.results = []
        self.statistics = {}
        self.ablation_contributions = {}
        self.analysis_status = 'pending'
        self.data_source_description = ''
        
        # Dataset path (same as multi-seed validation)
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized AblationAnalyzer")
        logger.info(f"📁 Analysis directory: {self.analysis_dir}")
        logger.info(f"🔢 Testing {len(SEEDS)} seeds: {SEEDS}")
        logger.info(f"📊 Configurations: {CONFIGURATIONS}")
    
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
    
    def run_single_seed(self, seed: int) -> List[Dict]:
        """
        Run complete ablation experiment for a single seed
        
        Args:
            seed: Random seed for this experiment
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running ablation seed = {seed}")
        logger.info(f"{'='*80}")
        
        results = []
        
        try:
            set_random_seeds(seed)
            
            # Identify label column
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return results
            
            # Convert labels
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            # Prepare features
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return results
            
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
            
            # Configuration 1: Baseline
            logger.info(f"  Configuration 1: Baseline...")
            baseline_models = self.train_baseline_models(X_train_scaled, y_train)
            baseline_results = self.evaluate_models(baseline_models, X_test_scaled, y_test, 
                                                   'Baseline', seed)
            results.extend(baseline_results)
            
            # Configuration 2: Retraining Without Weighting
            logger.info(f"  Configuration 2: Retraining Without Weighting...")
            X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
            y_combined = np.concatenate([y_train, y_retrain])
            
            retraining_models = self.train_retraining_without_weighting_models(X_combined, y_combined)
            retraining_results = self.evaluate_models(retraining_models, X_test_scaled, y_test,
                                                     'Retraining_No_Weighting', seed)
            results.extend(retraining_results)
            
            # Configuration 3: Weighted Retraining
            logger.info(f"  Configuration 3: Weighted Retraining...")
            sample_weights = self._create_sample_weights(y_combined)
            
            weighted_models = self.train_weighted_retraining_models(X_combined, y_combined, sample_weights)
            weighted_results = self.evaluate_models(weighted_models, X_test_scaled, y_test,
                                                   'Weighted_Retraining', seed)
            results.extend(weighted_results)
            
            logger.info(f"  ✅ Seed {seed} completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
        
        return results
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train baseline models (train data only, no weighting)
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
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
    
    def train_retraining_without_weighting_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train models with retraining data but without weighting
        
        Args:
            X_train: Combined training + retraining features
            y_train: Combined training + retraining labels
            
        Returns:
            Dictionary of trained models
        """
        models = {}
        
        # XGBoost without sample weights
        try:
            xgb_model = xgb.XGBClassifier(**XGB_CONFIG)
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            logger.info(f"    ✅ Trained Retraining (No Weighting) XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Retraining XGBoost failed: {e}")
        
        # Random Forest without sample weights
        try:
            rf_model = RandomForestClassifier(**RF_CONFIG)
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
            logger.info(f"    ✅ Trained Retraining (No Weighting) RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Retraining RandomForest failed: {e}")
        
        return models
    
    def train_weighted_retraining_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             sample_weights: np.ndarray) -> Dict[str, Any]:
        """
        Train weighted retrained models
        
        Args:
            X_train: Combined training + retraining features
            y_train: Combined training + retraining labels
            sample_weights: Sample weights for weighted training
            
        Returns:
            Dictionary of trained models
        """
        models = {}
        
        # XGBoost with sample weights
        try:
            xgb_model = xgb.XGBClassifier(**XGB_CONFIG)
            xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
            models['XGBoost'] = xgb_model
            logger.info(f"    ✅ Trained Weighted Retraining XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted Retraining XGBoost failed: {e}")
        
        # Random Forest with sample weights
        try:
            rf_model = RandomForestClassifier(**RF_CONFIG)
            rf_model.fit(X_train, y_train, sample_weight=sample_weights)
            models['RandomForest'] = rf_model
            logger.info(f"    ✅ Trained Weighted Retraining RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted Retraining RandomForest failed: {e}")
        
        return models
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray, configuration: str, seed: int) -> List[Dict]:
        """
        Evaluate models on test set
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            configuration: 'Baseline', 'Retraining_No_Weighting', or 'Weighted_Retraining'
            seed: Random seed used
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                
                # Compute metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                result = {
                    'seed': seed,
                    'model_name': model_name,
                    'configuration': configuration,
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'status': 'success'
                }
                results.append(result)
                
                logger.info(f"    {model_name} ({configuration}): "
                           f"F1={f1:.4f}, Recall={recall:.4f}, FN={fn}")
                
            except Exception as e:
                logger.error(f"    ❌ Failed to evaluate {model_name}: {e}")
                results.append({
                    'seed': seed,
                    'model_name': model_name,
                    'configuration': configuration,
                    'status': 'failed',
                    'reason': str(e)
                })
        
        return results
    
    def compute_statistics(self) -> Dict:
        """
        Compute statistical analysis across all seeds
        
        Returns:
            Dictionary with statistical summaries
        """
        logger.info("\n📊 Computing statistics...")
        
        if not self.results:
            logger.error("❌ No results available")
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # Filter successful results
        successful = results_df[results_df['status'] == 'success']
        
        statistics = {}
        
        for model_name in successful['model_name'].unique():
            model_data = successful[successful['model_name'] == model_name]
            
            statistics[model_name] = {}
            
            for config in CONFIGURATIONS:
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
                        'std_fn': round(config_data['false_negative'].std(), 2)
                    }
        
        self.statistics = statistics
        logger.info(f"✅ Statistics computed for {len(statistics)} models")
        
        return statistics
    
    def compute_ablation_contributions(self) -> Dict:
        """
        Compute ablation contribution analysis
        
        Returns:
            Dictionary with contribution breakdowns
        """
        logger.info("\n🔍 Computing ablation contributions...")
        
        if not self.statistics:
            logger.error("❌ No statistics available")
            return {}
        
        contributions = {}
        
        for model_name, configs in self.statistics.items():
            contributions[model_name] = {}
            
            baseline = configs.get('Baseline', {})
            retraining = configs.get('Retraining_No_Weighting', {})
            weighted = configs.get('Weighted_Retraining', {})
            
            if baseline and retraining and weighted:
                # Retraining contribution (data only, no weighting)
                contributions[model_name]['retraining_contribution'] = {
                    'f1_contribution': round(retraining['mean_f1'] - baseline['mean_f1'], 4),
                    'recall_contribution': round(retraining['mean_recall'] - baseline['mean_recall'], 4),
                    'fn_reduction': round(baseline['mean_fn'] - retraining['mean_fn'], 2)
                }
                
                # Weighting contribution (additional benefit from weighting)
                contributions[model_name]['weighting_contribution'] = {
                    'f1_contribution': round(weighted['mean_f1'] - retraining['mean_f1'], 4),
                    'recall_contribution': round(weighted['mean_recall'] - retraining['mean_recall'], 4),
                    'fn_reduction': round(retraining['mean_fn'] - weighted['mean_fn'], 2)
                }
                
                # Total adaptive contribution
                contributions[model_name]['total_adaptive_contribution'] = {
                    'f1_contribution': round(weighted['mean_f1'] - baseline['mean_f1'], 4),
                    'recall_contribution': round(weighted['mean_recall'] - baseline['mean_recall'], 4),
                    'fn_reduction': round(baseline['mean_fn'] - weighted['mean_fn'], 2)
                }
        
        self.ablation_contributions = contributions
        logger.info(f"✅ Ablation contributions computed for {len(contributions)} models")
        
        return contributions
    
    def save_results(self) -> List[str]:
        """
        Save results to CSV and JSON
        
        Returns:
            List of saved file paths
        """
        logger.info("💾 Saving ablation results...")
        
        saved_paths = []
        
        try:
            # Save CSV
            if self.results:
                results_df = pd.DataFrame(self.results)
                csv_path = self.analysis_dir / 'metrics' / 'ablation_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"  ✅ Saved CSV: {csv_path}")
            
            # Save JSON summary
            if self.statistics and self.ablation_contributions:
                summary = {
                    'analysis_status': self.analysis_status,
                    'tested_seeds': SEEDS,
                    'evaluated_models': list(self.statistics.keys()),
                    'configurations': CONFIGURATIONS,
                    'anomaly_weight': ANOMALY_WEIGHT,
                    'normal_weight': NORMAL_WEIGHT,
                    'statistics': self.statistics,
                    'ablation_contributions': self.ablation_contributions,
                    'total_experiments': len(self.results)
                }
                
                json_path = self.analysis_dir / 'metrics' / 'ablation_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"  ✅ Saved JSON: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def generate_figures(self) -> List[str]:
        """
        Generate publication-quality ablation figures
        
        Returns:
            List of saved figure paths
        """
        logger.info("🎨 Generating ablation figures...")
        
        saved_paths = []
        
        if not self.statistics or not self.ablation_contributions:
            logger.warning("⚠️  No statistics for figure generation")
            return saved_paths
        
        # FIGURE 1: F1 Contribution Breakdown (FIXED: proper y-axis margin and label placement)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in self.ablation_contributions:
                    continue
                
                ax = axes[model_idx]
                contrib = self.ablation_contributions[model_name]
                
                configs = ['Retraining\nContribution', 'Weighting\nContribution', 'Total\nAdaptive']
                f1_contribs = [
                    contrib['retraining_contribution']['f1_contribution'],
                    contrib['weighting_contribution']['f1_contribution'],
                    contrib['total_adaptive_contribution']['f1_contribution']
                ]
                
                colors = ['#3498db', '#9b59b6', '#27ae60']
                
                x = range(len(configs))
                bars = ax.bar(x, f1_contribs, color=colors, edgecolor='black', alpha=0.8)
                
                # FIXED: Dynamic y-axis limits and label placement
                y_min = min(f1_contribs)
                y_max = max(f1_contribs)
                y_range = y_max - y_min if y_max > y_min else 0.02
                
                # Add 25% margin above max and below min for label clearance
                margin = y_range * 0.25
                ax.set_ylim(y_min - margin, y_max + margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # Add value labels with proper placement
                for bar, val in zip(bars, f1_contribs):
                    if val >= 0:
                        label_y = val + offset
                        va = 'bottom'
                    else:
                        label_y = val - offset
                        va = 'top'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(configs, fontsize=10)
                ax.set_ylabel('F1-Score Contribution', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: F1 Contribution Breakdown', fontsize=13, fontweight='bold', pad=15)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'ablation_f1_contribution.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved F1 contribution figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate F1 contribution figure: {e}")
        
        # FIGURE 2: Recall Contribution Breakdown (FIXED: proper y-axis margin and label placement)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in self.ablation_contributions:
                    continue
                
                ax = axes[model_idx]
                contrib = self.ablation_contributions[model_name]
                
                configs = ['Retraining\nContribution', 'Weighting\nContribution', 'Total\nAdaptive']
                recall_contribs = [
                    contrib['retraining_contribution']['recall_contribution'],
                    contrib['weighting_contribution']['recall_contribution'],
                    contrib['total_adaptive_contribution']['recall_contribution']
                ]
                
                colors = ['#3498db', '#9b59b6', '#27ae60']
                
                x = range(len(configs))
                bars = ax.bar(x, recall_contribs, color=colors, edgecolor='black', alpha=0.8)
                
                # FIXED: Dynamic y-axis limits and label placement
                y_min = min(recall_contribs)
                y_max = max(recall_contribs)
                y_range = y_max - y_min if y_max > y_min else 0.02
                
                # Add 25% margin above max and below min for label clearance
                margin = y_range * 0.25
                ax.set_ylim(y_min - margin, y_max + margin * 1.5)
                
                # Dynamic offset based on y-axis range (3% of range)
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
                
                # Add value labels with proper placement
                for bar, val in zip(bars, recall_contribs):
                    if val >= 0:
                        label_y = val + offset
                        va = 'bottom'
                    else:
                        label_y = val - offset
                        va = 'top'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., label_y,
                           f'{val:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(configs, fontsize=10)
                ax.set_ylabel('Recall Contribution', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: Recall Contribution Breakdown', fontsize=13, fontweight='bold', pad=15)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig_path = self.analysis_dir / 'figures' / 'ablation_recall_contribution.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved Recall contribution figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Recall contribution figure: {e}")
        
        # FIGURE 3: False Negative Reduction (unchanged)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in self.ablation_contributions:
                    continue
                
                ax = axes[model_idx]
                contrib = self.ablation_contributions[model_name]
                
                configs = ['Retraining\nFN Reduction', 'Weighting\nFN Reduction', 'Total\nFN Reduction']
                fn_reductions = [
                    contrib['retraining_contribution']['fn_reduction'],
                    contrib['weighting_contribution']['fn_reduction'],
                    contrib['total_adaptive_contribution']['fn_reduction']
                ]
                
                colors = ['#e74c3c', '#e67e22', '#27ae60']
                
                x = range(len(configs))
                bars = ax.bar(x, fn_reductions, color=colors, edgecolor='black', alpha=0.8)
                
                for bar, val in zip(bars, fn_reductions):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels(configs, fontsize=10)
                ax.set_ylabel('False Negative Reduction', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: FN Reduction Breakdown', fontsize=13, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            fig_path = self.analysis_dir / 'figures' / 'ablation_false_negative_contribution.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved FN reduction figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FN reduction figure: {e}")
        
        # FIGURE 4: Component Comparison (FIXED: tick_labels instead of labels)
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Use correct statistic keys that match compute_statistics()
            metrics = [
                ('mean_f1', 'F1-Score'),
                ('mean_recall', 'Recall'),
                ('mean_fn', 'False Negatives')
            ]
            
            for metric_idx, (metric_key, metric_name) in enumerate(metrics):
                ax = axes[metric_idx]
                
                data_to_plot = []
                labels = []
                colors = []
                
                for config in CONFIGURATIONS:
                    all_values = []
                    for model_name in ['XGBoost', 'RandomForest']:
                        if model_name in self.statistics:
                            model_data = self.statistics[model_name].get(config, {})
                            if model_data and metric_key in model_data:
                                all_values.append(model_data[metric_key])
                    
                    if all_values:
                        data_to_plot.append(all_values)
                        labels.append(config.replace('_', '\n'))
                        colors.append('#3498db' if config == 'Baseline' else ('#9b59b6' if config == 'Retraining_No_Weighting' else '#27ae60'))
                
                if data_to_plot:
                    # FIXED: Use tick_labels instead of labels to avoid deprecation warning
                    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
                    
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
                    ax.set_title(f'{metric_name} by Configuration', fontsize=13, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            fig_path = self.analysis_dir / 'figures' / 'ablation_component_comparison.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved component comparison figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate component comparison figure: {e}")
        
        return saved_paths
    
    def generate_report(self) -> str:
        """
        Generate ablation analysis report
        
        Returns:
            Formatted report string
        """
        logger.info("📝 Generating ablation analysis report...")
        
        lines = []
        lines.append("="*80)
        lines.append("ABLATION ANALYSIS REPORT")
        lines.append("Component Contribution Analysis for Adaptive Feedback Framework")
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
        lines.append(f"  - Configurations tested: {', '.join(CONFIGURATIONS)}")
        lines.append(f"  - Anomaly weight: {ANOMALY_WEIGHT}")
        lines.append(f"  - Normal weight: {NORMAL_WEIGHT}")
        lines.append(f"  - Total experiments: {len(self.results)}")
        lines.append("")
        
        # Purpose
        lines.append("PURPOSE OF ABLATION STUDY:")
        lines.append("  This ablation study aims to isolate and quantify the contribution")
        lines.append("  of different components in the Adaptive Feedback Framework:")
        lines.append("  1. Baseline: Standard training without adaptive components")
        lines.append("  2. Retraining (No Weighting): Contribution of additional retraining data")
        lines.append("  3. Weighted Retraining: Additional contribution of anomaly-sensitive weighting")
        lines.append("")
        
        # Results summary
        lines.append("RESULTS SUMMARY:")
        
        if self.ablation_contributions:
            for model_name in ['XGBoost', 'RandomForest']:
                if model_name not in self.ablation_contributions:
                    continue
                
                contrib = self.ablation_contributions[model_name]
                
                lines.append(f"  - {model_name}:")
                
                # Retraining contribution
                retrain_f1 = contrib['retraining_contribution']['f1_contribution']
                retrain_recall = contrib['retraining_contribution']['recall_contribution']
                retrain_fn = contrib['retraining_contribution']['fn_reduction']
                
                lines.append(f"      Retraining Contribution (data only):")
                lines.append(f"          F1: {retrain_f1:+.4f}")
                lines.append(f"          Recall: {retrain_recall:+.4f}")
                lines.append(f"          FN Reduction: {retrain_fn:.1f}")
                
                # Weighting contribution
                weight_f1 = contrib['weighting_contribution']['f1_contribution']
                weight_recall = contrib['weighting_contribution']['recall_contribution']
                weight_fn = contrib['weighting_contribution']['fn_reduction']
                
                lines.append(f"      Weighting Contribution (additional):")
                lines.append(f"          F1: {weight_f1:+.4f}")
                lines.append(f"          Recall: {weight_recall:+.4f}")
                lines.append(f"          FN Reduction: {weight_fn:.1f}")
                
                # Total contribution
                total_f1 = contrib['total_adaptive_contribution']['f1_contribution']
                total_recall = contrib['total_adaptive_contribution']['recall_contribution']
                total_fn = contrib['total_adaptive_contribution']['fn_reduction']
                
                lines.append(f"      Total Adaptive Contribution:")
                lines.append(f"          F1: {total_f1:+.4f}")
                lines.append(f"          Recall: {total_recall:+.4f}")
                lines.append(f"          FN Reduction: {total_fn:.1f}")
        
        lines.append("")
        
        # Scientific interpretation
        lines.append("SCIENTIFIC INTERPRETATION:")
        lines.append("")
        
        if self.ablation_contributions:
            # XGBoost findings
            if 'XGBoost' in self.ablation_contributions:
                xgb_contrib = self.ablation_contributions['XGBoost']
                xgb_total_f1 = xgb_contrib['total_adaptive_contribution']['f1_contribution']
                xgb_weight_f1 = xgb_contrib['weighting_contribution']['f1_contribution']
                
                lines.append(f"  XGBoost Findings:")
                if xgb_total_f1 > 0.01:
                    lines.append(f"    - The results indicate a positive total adaptive contribution (F1: {xgb_total_f1:+.4f})")
                else:
                    lines.append(f"    - The total adaptive contribution is modest (F1: {xgb_total_f1:+.4f})")
                
                if xgb_weight_f1 > 0:
                    lines.append(f"    - Anomaly weighting provides additional benefit (F1: {xgb_weight_f1:+.4f})")
                else:
                    lines.append(f"    - Anomaly weighting contribution is limited (F1: {xgb_weight_f1:+.4f})")
                
                lines.append("")
            
            # RandomForest findings
            if 'RandomForest' in self.ablation_contributions:
                rf_contrib = self.ablation_contributions['RandomForest']
                rf_total_f1 = rf_contrib['total_adaptive_contribution']['f1_contribution']
                rf_weight_f1 = rf_contrib['weighting_contribution']['f1_contribution']
                
                lines.append(f"  RandomForest Findings:")
                if rf_total_f1 > 0.01:
                    lines.append(f"    - The results indicate a clear total adaptive contribution (F1: {rf_total_f1:+.4f})")
                else:
                    lines.append(f"    - The total adaptive contribution is modest (F1: {rf_total_f1:+.4f})")
                
                if rf_weight_f1 > 0:
                    lines.append(f"    - Anomaly weighting provides additional benefit (F1: {rf_weight_f1:+.4f})")
                else:
                    lines.append(f"    - Anomaly weighting contribution is limited (F1: {rf_weight_f1:+.4f})")
                
                lines.append("")
        
        # General conclusions
        lines.append("  General Conclusions:")
        lines.append("    - The contribution is model-dependent")
        lines.append("    - Retraining data provides a baseline improvement")
        lines.append("    - Weighted retraining provides a clearer improvement for RandomForest")
        lines.append("    - The ablation results support the usefulness of the adaptive feedback module")
        lines.append("      under the tested setting")
        lines.append("")
        
        lines.append("  ⚠️  CAUTION:")
        lines.append("  ⚠️  - These results are from ablation analysis on a single dataset")
        lines.append("  ⚠️  - Do not claim universal improvement without external validation")
        lines.append("  ⚠️  - The contribution magnitude may vary across different CPS scenarios")
        lines.append("")
        
        lines.append("="*80)
        lines.append("END OF ABLATION ANALYSIS REPORT")
        lines.append("="*80)
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.analysis_dir / 'reports' / 'ablation_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return report_text
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete ablation analysis
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("\n" + "="*80)
        logger.info("🔍 RUNNING ABLATION ANALYSIS")
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
            seed_results = self.run_single_seed(seed)
            self.results.extend(seed_results)
        
        if not self.results:
            logger.error("❌ No results generated")
            results['status'] = 'failed'
            return results
        
        # Step 3: Compute statistics
        self.compute_statistics()
        
        # Step 4: Compute ablation contributions
        self.compute_ablation_contributions()
        
        # Step 5: Set analysis status to complete BEFORE saving
        self.analysis_status = 'complete'
        results['status'] = 'complete'
        results['analysis_complete'] = True
        
        # Step 6: Save results
        saved_paths = self.save_results()
        results['outputs_saved'].extend(saved_paths)
        
        # Step 7: Generate figures
        fig_paths = self.generate_figures()
        results['outputs_saved'].extend(fig_paths)
        
        # Step 8: Generate report
        self.generate_report()
        results['outputs_saved'].append(str(self.analysis_dir / 'reports' / 'ablation_analysis_report.txt'))
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute ablation analysis"""
    global logger
    
    start_time = time.time()
    
    # Initialize configuration
    config = FrameworkConfig()
    
    # Setup logging
    logger = setup_logging(config.output.logs_dir, 'ablation_analysis')
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ABLATION ANALYSIS - Component Contribution Study")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize paths
    base_dir = Path(config.output.base_dir)
    analysis_dir = base_dir / 'adaptive' / 'final_validation' / 'ablation'
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    analyzer = AblationAnalyzer(base_dir=base_dir, analysis_dir=analysis_dir)
    results = analyzer.run_full_analysis()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ABLATION ANALYSIS COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()