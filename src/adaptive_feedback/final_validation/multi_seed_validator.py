"""
Multi-Seed Validator - Stability Analysis Module
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements multi-seed stability analysis to verify whether
the improvement introduced by the Adaptive Feedback Module is stable
and reproducible across different random seeds.

EXPERIMENTAL DESIGN:
--------------------
Compares:
1. Baseline Detection Pipeline
   vs
2. Adaptive Weighted Retraining Pipeline

Across 5 random seeds: [42, 123, 2024, 2025, 3407]

OUTPUT FILES:
-------------
- results/adaptive/final_validation/metrics/multi_seed_results.csv
- results/adaptive/final_validation/metrics/multi_seed_summary.json
- results/adaptive/final_validation/figures/f1_stability_across_seeds.png
- results/adaptive/final_validation/figures/recall_stability_across_seeds.png
- results/adaptive/final_validation/figures/false_negative_reduction_across_seeds.png
- results/adaptive/final_validation/figures/baseline_vs_adaptive_distribution.png
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
# This file is at: src/adaptive_feedback/final_validation/multi_seed_validator.py
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

# Only primary models for multi-seed validation
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
# MULTI-SEED VALIDATOR CLASS
# =============================================================================
class MultiSeedValidator:
    """
    Implements multi-seed stability analysis for adaptive feedback validation
    
    This class runs complete independent experiments across multiple random seeds
    to verify that adaptive weighted retraining provides stable improvements.
    """
    
    def __init__(self, base_dir: Path, validation_dir: Optional[Path] = None):
        """
        Initialize Multi-Seed Validator
        
        Args:
            base_dir: Base results directory
            validation_dir: Path to save validation outputs
        """
        self.base_dir = Path(base_dir)
        self.validation_dir = Path(validation_dir) if validation_dir else (
            self.base_dir / 'adaptive' / 'final_validation'
        )
        
        # Create output directories
        (self.validation_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.validation_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.original_dataset = None
        self.results = []
        self.statistics = {}
        self.validation_status = 'pending'
        self.data_source_description = ''
        
        # Dataset path
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized MultiSeedValidator")
        logger.info(f"📁 Validation directory: {self.validation_dir}")
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
    
    def run_single_seed(self, seed: int) -> List[Dict]:
        """
        Run complete experiment for a single seed
        
        Args:
            seed: Random seed for this experiment
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running seed = {seed}")
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
            
            # Train and evaluate baseline models
            logger.info(f"  Training baseline models...")
            baseline_models = self.train_baseline_models(X_train_scaled, y_train)
            baseline_results = self.evaluate_models(baseline_models, X_test_scaled, y_test, 
                                                   'baseline', seed)
            results.extend(baseline_results)
            
            # Train and evaluate weighted retrained models
            logger.info(f"  Training weighted retrained models...")
            X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
            y_combined = np.concatenate([y_train, y_retrain])
            sample_weights = self._create_sample_weights(y_combined)
            
            weighted_models = self.train_weighted_models(X_combined, y_combined, sample_weights)
            weighted_results = self.evaluate_models(weighted_models, X_test_scaled, y_test,
                                                   'weighted_retraining', seed)
            results.extend(weighted_results)
            
            logger.info(f"  ✅ Seed {seed} completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
        
        return results
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train baseline models (XGBoost and RandomForest only)
        
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
            logger.info(f"    ✅ Trained XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  XGBoost unavailable: {e}")
        
        # Random Forest
        try:
            rf_model = RandomForestClassifier(**RF_CONFIG)
            rf_model.fit(X_train, y_train)
            models['RandomForest'] = rf_model
            logger.info(f"    ✅ Trained RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  RandomForest unavailable: {e}")
        
        return models
    
    def train_weighted_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             sample_weights: np.ndarray) -> Dict[str, Any]:
        """
        Train weighted retrained models (XGBoost and RandomForest only)
        
        Args:
            X_train: Training features
            y_train: Training labels
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
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                       y_test: np.ndarray, model_type: str, seed: int) -> List[Dict]:
        """
        Evaluate models on test set
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            model_type: 'baseline' or 'weighted_retraining'
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
                    'model_type': model_type,
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'status': 'success'
                }
                results.append(result)
                
                logger.info(f"    {model_name} ({model_type}): "
                           f"F1={f1:.4f}, Recall={recall:.4f}, FN={fn}")
                
            except Exception as e:
                logger.error(f"    ❌ Failed to evaluate {model_name}: {e}")
                results.append({
                    'seed': seed,
                    'model_name': model_name,
                    'model_type': model_type,
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
            
            for model_type in ['baseline', 'weighted_retraining']:
                type_data = model_data[model_data['model_type'] == model_type]
                
                if len(type_data) > 0:
                    statistics[model_name][model_type] = {
                        'n_seeds': len(type_data),
                        'mean_f1': round(type_data['f1_score'].mean(), 4),
                        'std_f1': round(type_data['f1_score'].std(), 4),
                        'mean_recall': round(type_data['recall'].mean(), 4),
                        'std_recall': round(type_data['recall'].std(), 4),
                        'mean_precision': round(type_data['precision'].mean(), 4),
                        'std_precision': round(type_data['precision'].std(), 4),
                        'mean_fn': round(type_data['false_negative'].mean(), 2),
                        'std_fn': round(type_data['false_negative'].std(), 2)
                    }
            
            # Compute improvements
            if 'baseline' in statistics[model_name] and 'weighted_retraining' in statistics[model_name]:
                baseline = statistics[model_name]['baseline']
                weighted = statistics[model_name]['weighted_retraining']
                
                statistics[model_name]['improvement'] = {
                    'f1_improvement': round(weighted['mean_f1'] - baseline['mean_f1'], 4),
                    'recall_improvement': round(weighted['mean_recall'] - baseline['mean_recall'], 4),
                    'fn_reduction': round(baseline['mean_fn'] - weighted['mean_fn'], 2)
                }
        
        self.statistics = statistics
        logger.info(f"✅ Statistics computed for {len(statistics)} models")
        
        return statistics
    
    def save_results(self) -> List[str]:
        """
        Save results to CSV and JSON
        
        Returns:
            List of saved file paths
        """
        logger.info("💾 Saving results...")
        
        saved_paths = []
        
        try:
            # Save CSV
            if self.results:
                results_df = pd.DataFrame(self.results)
                csv_path = self.validation_dir / 'metrics' / 'multi_seed_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"  ✅ Saved CSV: {csv_path}")
            
            # Save JSON summary
            if self.statistics:
                summary = {
                    'validation_status': self.validation_status,
                    'tested_seeds': SEEDS,
                    'evaluated_models': list(self.statistics.keys()),
                    'anomaly_weight': ANOMALY_WEIGHT,
                    'normal_weight': NORMAL_WEIGHT,
                    'statistics': self.statistics,
                    'total_experiments': len(self.results)
                }
                
                json_path = self.validation_dir / 'metrics' / 'multi_seed_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"  ✅ Saved JSON: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def generate_visualization_data(self) -> List[str]:
        """
        Generate publication-quality figures
        
        Returns:
            List of saved figure paths
        """
        logger.info("🎨 Generating visualization figures...")
        
        saved_paths = []
        
        if not self.results:
            logger.warning("⚠️  No results for visualization")
            return saved_paths
        
        results_df = pd.DataFrame(self.results)
        successful = results_df[results_df['status'] == 'success']
        
        # Create seed labels for x-axis
        seed_labels = [str(s) for s in SEEDS]
        x_positions = range(len(SEEDS))
        
        # FIGURE 1: F1-score Stability Across Seeds
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in successful['model_name'].values:
                    continue
                
                ax = axes[model_idx]
                model_data = successful[successful['model_name'] == model_name]
                
                for model_type in ['baseline', 'weighted_retraining']:
                    type_data = model_data[model_data['model_type'] == model_type]
                    
                    if len(type_data) > 0:
                        # Group by seed and compute mean/std
                        grouped = type_data.groupby('seed')['f1_score']
                        means = grouped.mean()
                        stds = grouped.std()
                        
                        # Map seeds to x positions
                        x_vals = [SEEDS.index(s) for s in means.index]
                        
                        ax.errorbar(x_vals, means.values, yerr=stds.values,
                                   marker='o', linewidth=2, capsize=5,
                                   label=model_type.replace('_', ' ').title())
                
                ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
                ax.set_ylabel('F1-score', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: F1-score Stability', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(seed_labels, rotation=45, ha='right')
                ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            fig_path = self.validation_dir / 'figures' / 'f1_stability_across_seeds.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved F1 stability figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate F1 figure: {e}")
        
        # FIGURE 2: Recall Stability Across Seeds
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in successful['model_name'].values:
                    continue
                
                ax = axes[model_idx]
                model_data = successful[successful['model_name'] == model_name]
                
                for model_type in ['baseline', 'weighted_retraining']:
                    type_data = model_data[model_data['model_type'] == model_type]
                    
                    if len(type_data) > 0:
                        grouped = type_data.groupby('seed')['recall']
                        means = grouped.mean()
                        stds = grouped.std()
                        
                        x_vals = [SEEDS.index(s) for s in means.index]
                        
                        ax.errorbar(x_vals, means.values, yerr=stds.values,
                                   marker='s', linewidth=2, capsize=5,
                                   label=model_type.replace('_', ' ').title())
                
                ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
                ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: Recall Stability', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(seed_labels, rotation=45, ha='right')
                ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            fig_path = self.validation_dir / 'figures' / 'recall_stability_across_seeds.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved Recall stability figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Recall figure: {e}")
        
        # FIGURE 3: False Negative Reduction
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for model_idx, model_name in enumerate(['XGBoost', 'RandomForest']):
                if model_name not in successful['model_name'].values:
                    continue
                
                ax = axes[model_idx]
                model_data = successful[successful['model_name'] == model_name]
                
                components = ['baseline', 'weighted_retraining']
                mean_fn = []
                std_fn = []
                
                for component in components:
                    type_data = model_data[model_data['model_type'] == component]
                    if len(type_data) > 0:
                        mean_fn.append(type_data['false_negative'].mean())
                        std_fn.append(type_data['false_negative'].std())
                
                x = range(len(components))
                ax.bar(x, mean_fn, yerr=std_fn, capsize=5, alpha=0.7, 
                      edgecolor='black', color='#e74c3c')
                ax.set_xticks(x)
                ax.set_xticklabels([c.replace('_', ' ').title() for c in components], 
                                  rotation=45, ha='right')
                ax.set_ylabel('False Negatives (mean ± std)', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name}: FN Reduction', fontsize=13, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            fig_path = self.validation_dir / 'figures' / 'false_negative_reduction_across_seeds.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved FN reduction figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FN figure: {e}")
        
        # FIGURE 4: Baseline vs Adaptive Distribution (Boxplot)
        # Only XGBoost and RandomForest, baseline vs weighted retraining
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = [('f1_score', 'F1-score'), ('recall', 'Recall'), ('false_negative', 'False Negatives')]
            
            for metric_idx, (metric_col, metric_name) in enumerate(metrics):
                ax = axes[metric_idx]
                
                data_to_plot = []
                labels = []
                colors = []
                
                # Only baseline and weighted_retraining for XGBoost and RandomForest
                for model_type in ['baseline', 'weighted_retraining']:
                    type_data = successful[successful['model_type'] == model_type]
                    if len(type_data) > 0:
                        data_to_plot.append(type_data[metric_col].values)
                        labels.append(model_type.replace('_', ' ').title())
                        colors.append('#3498db' if model_type == 'baseline' else '#27ae60')
                
                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
                    ax.set_title(f'{metric_name} Distribution', fontsize=13, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            fig_path = self.validation_dir / 'figures' / 'baseline_vs_adaptive_distribution.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            saved_paths.append(str(fig_path))
            logger.info(f"  ✅ Saved distribution figure: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate distribution figure: {e}")
        
        return saved_paths
    
    def run_full_validation(self) -> Dict:
        """
        Run complete multi-seed validation
        
        Returns:
            Dictionary with validation results
        """
        logger.info("\n" + "="*80)
        logger.info("🔢 RUNNING MULTI-SEED STABILITY ANALYSIS")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'validation_complete': False
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
        
        # Step 4: Set validation status to complete BEFORE saving
        self.validation_status = 'complete'
        results['status'] = 'complete'
        results['validation_complete'] = True
        
        # Step 5: Save results (now with correct validation_status)
        saved_paths = self.save_results()
        results['outputs_saved'].extend(saved_paths)
        
        # Step 6: Generate figures
        fig_paths = self.generate_visualization_data()
        results['outputs_saved'].extend(fig_paths)
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute multi-seed validation"""
    global logger
    
    start_time = time.time()
    
    # Initialize configuration
    config = FrameworkConfig()
    
    # Setup logging
    logger = setup_logging(config.output.logs_dir, 'multi_seed_validation')
    
    logger.info("\n" + "="*80)
    logger.info("🔢 MULTI-SEED STABILITY ANALYSIS")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize paths
    base_dir = Path(config.output.base_dir)
    validation_dir = base_dir / 'adaptive' / 'final_validation'
    
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    validator = MultiSeedValidator(base_dir=base_dir, validation_dir=validation_dir)
    results = validator.run_full_validation()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 MULTI-SEED VALIDATION COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()