"""
One-Class SVM Extension - Final Validation
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module extends the existing four-model final-validation structure to include
One-Class SVM using the same controlled protocol.

IMPORTANT SCIENTIFIC NOTES:
- One-Class SVM is UNSUPERVISED (no y during model.fit)
- Variant A (baseline): fit on X_train only
- Variant B (adaptive_refit): fit on X_train + X_retrain concatenated (fresh model)
- Labels used ONLY for evaluation, NOT for training
- Historical April 2026 baseline is CONTEXT ONLY
- NOT used as validation target for new 60/20/20 experiment

OUTPUT FILES:
-------------
- results/adaptive/final_validation_extended/one_class_svm/metrics/one_class_svm_results.csv
- results/adaptive/final_validation_extended/one_class_svm/metrics/one_class_svm_summary.csv
- results/adaptive/final_validation_extended/one_class_svm/metrics/one_class_svm_summary.json
- results/adaptive/final_validation_extended/one_class_svm/metrics/one_class_svm_transitions.csv
- results/adaptive/final_validation_extended/one_class_svm/predictions/one_class_svm_seed{seed}_predictions.npz
- results/adaptive/final_validation_extended/one_class_svm/reports/one_class_svm_report.txt
- results/adaptive/final_validation_extended/one_class_svm/checkpoints/one_class_svm_checkpoint.json
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
from __future__ import annotations

import sys
import os
import time
import logging
import json
import gc
import hashlib
import traceback
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FrameworkConfig
from src.utils import setup_logging, set_random_seeds

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import OneClassSVM

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 2024, 2025, 3407]

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

# One-Class SVM configuration (from src/models.py historical configuration)
OCSVM_KERNEL = 'rbf'
OCSVM_GAMMA = 'scale'
OCSVM_NU = 0.1

# Require exactly 51 features
EXPECTED_FEATURE_COUNT = 51

CHECKPOINT_VERSION = '1.0'

# =============================================================================
# ONE-CLASS SVM EXTENSION CLASS
# =============================================================================
class OneClassSVMExtension:
    """
    Implements One-Class SVM extension to final-validation structure.
    """
    
    def __init__(self, base_dir: Path, extension_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.extension_dir = Path(extension_dir) if extension_dir else (
            self.base_dir / 'adaptive' / 'final_validation_extended' / 'one_class_svm'
        )
        
        # Create output directories
        (self.extension_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.extension_dir / 'predictions').mkdir(parents=True, exist_ok=True)
        (self.extension_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.extension_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.seed_results = []
        self.transition_results = []
        self.multiseed_summary = {}
        self.data_source_description = ''
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        # Checkpoint management
        self.checkpoint_path = self.extension_dir / 'checkpoints' / 'one_class_svm_checkpoint.json'
        self.completed_seeds = []
        self.seed_artifacts = {}
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized OneClassSVMExtension")
        logger.info(f"📁 Extension directory: {self.extension_dir}")
        logger.info(f"🔢 Seeds: {SEEDS}")
        logger.info(f"🔧 One-Class SVM: kernel={OCSVM_KERNEL}, gamma={OCSVM_GAMMA}, nu={OCSVM_NU}")
        logger.warning("⚠️  WARNING: Full RBF One-Class SVM may require many hours per fit.")
    
    def _compute_dataset_fingerprint(self) -> str:
        """Compute streaming SHA-256 hash of ENTIRE dataset file."""
        if not self.dataset_path.exists():
            return 'dataset_not_found'
        
        hasher = hashlib.sha256()
        try:
            with open(self.dataset_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
            
            fingerprint = hasher.hexdigest()[:32]
            logger.info(f"📊 Dataset fingerprint (SHA-256): {fingerprint}")
            return fingerprint
        except Exception as e:
            logger.warning(f"⚠️  Failed to compute dataset fingerprint: {e}")
            return 'fingerprint_computation_failed'
    
    def _get_config_hash(self) -> str:
        """Generate deterministic SHA-256 fingerprint of experimental configuration."""
        config_dict = {
            'seeds': SEEDS,
            'dataset_path': str(self.dataset_path),
            'dataset_fingerprint': self.dataset_fingerprint,
            'train_ratio': TRAIN_RATIO,
            'retrain_ratio': RETRAIN_RATIO,
            'test_ratio': TEST_RATIO,
            'expected_feature_count': EXPECTED_FEATURE_COUNT,
            'ocsvm_kernel': OCSVM_KERNEL,
            'ocsvm_gamma': OCSVM_GAMMA,
            'ocsvm_nu': OCSVM_NU,
            'label_column_names': sorted(LABEL_COLUMN_NAMES),
            'metadata_columns': sorted(METADATA_COLUMNS),
            'preprocessing': 'SimpleImputer_mean_StandardScaler',
            'split_method': 'stratified_train_test_split'
        }
        
        config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        fingerprint = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:32]
        
        return fingerprint
    
    def _save_atomically(self, filepath: Path, data: Any):
        """Save file atomically using temporary file + os.replace()."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=filepath.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def _save_checkpoint(self):
        """Save checkpoint with per-seed artifacts."""
        checkpoint = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'config_fingerprint': self.config_fingerprint,
            'dataset_fingerprint': self.dataset_fingerprint,
            'completed_seeds': self.completed_seeds,
            'seed_artifacts': self.seed_artifacts,
            'saved_at': datetime.now().isoformat()
        }
        
        self._save_atomically(self.checkpoint_path, checkpoint)
    
    def _load_checkpoint(self) -> bool:
        """Load and validate checkpoint with per-seed artifacts."""
        if not self.checkpoint_path.exists():
            logger.info("ℹ️   No checkpoint found. Starting fresh.")
            return False
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
                logger.warning(f"⚠️  Checkpoint version mismatch")
                return False
            
            if checkpoint.get('config_fingerprint') != self.config_fingerprint:
                logger.error("❌ Checkpoint config fingerprint mismatch")
                return False
            
            if checkpoint.get('dataset_fingerprint') != self.dataset_fingerprint:
                logger.error("❌ Checkpoint dataset fingerprint mismatch")
                return False
            
            self.completed_seeds = checkpoint.get('completed_seeds', [])
            self.seed_artifacts = checkpoint.get('seed_artifacts', {})
            
            for seed_str, artifacts in self.seed_artifacts.items():
                seed = int(seed_str)
                if 'baseline_metrics' in artifacts and 'adaptive_metrics' in artifacts:
                    self.seed_results.append({
                        'baseline': artifacts['baseline_metrics'],
                        'adaptive_refit': artifacts['adaptive_metrics']
                    })
                if 'transitions' in artifacts:
                    self.transition_results.append(artifacts['transitions'])
            
            logger.info(f"📥 Loaded checkpoint: {len(self.completed_seeds)} completed seeds")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            return False
    
    def _validate_seed_artifact(self, seed: int, artifacts: Dict) -> bool:
        """Validate all artifacts exist and are consistent."""
        try:
            if 'baseline_metrics' not in artifacts or 'adaptive_metrics' not in artifacts:
                logger.warning(f"⚠️  Seed {seed}: Missing metrics")
                return False
            
            if 'transitions' not in artifacts:
                logger.warning(f"⚠️  Seed {seed}: Missing transitions")
                return False
            
            npz_path = artifacts.get('npz_path')
            if not npz_path or not Path(npz_path).exists():
                logger.warning(f"⚠️  Seed {seed}: NPZ file missing")
                return False
            
            npz = np.load(npz_path)
            required_arrays = ['y_true', 'baseline_y_pred', 'adaptive_y_pred', 'test_indices']
            for arr_name in required_arrays:
                if arr_name not in npz:
                    logger.warning(f"⚠️  Seed {seed}: NPZ missing {arr_name}")
                    npz.close()
                    return False
            
            n_samples = len(npz['y_true'])
            if len(npz['baseline_y_pred']) != n_samples:
                logger.warning(f"⚠️  Seed {seed}: baseline_y_pred length mismatch")
                npz.close()
                return False
            
            if len(npz['adaptive_y_pred']) != n_samples:
                logger.warning(f"⚠️  Seed {seed}: adaptive_y_pred length mismatch")
                npz.close()
                return False
            
            if len(npz['test_indices']) != n_samples:
                logger.warning(f"⚠️  Seed {seed}: test_indices length mismatch")
                npz.close()
                return False
            
            if len(np.unique(npz['test_indices'])) != n_samples:
                logger.warning(f"⚠️  Seed {seed}: test_indices not unique")
                npz.close()
                return False
            
            # Validate binary predictions
            if not np.all(np.isin(npz['baseline_y_pred'], [0, 1])):
                logger.warning(f"⚠️  Seed {seed}: baseline_y_pred not binary")
                npz.close()
                return False
            
            if not np.all(np.isin(npz['adaptive_y_pred'], [0, 1])):
                logger.warning(f"⚠️  Seed {seed}: adaptive_y_pred not binary")
                npz.close()
                return False
            
            y_true = npz['y_true']
            baseline_pred = npz['baseline_y_pred']
            adaptive_pred = npz['adaptive_y_pred']
            
            # Recompute transitions and validate identities
            anomaly_mask = (y_true == 1)
            normal_mask = (y_true == 0)
            
            corrected_fn = int(np.sum(anomaly_mask & (baseline_pred == 0) & (adaptive_pred == 1)))
            persistent_fn = int(np.sum(anomaly_mask & (baseline_pred == 0) & (adaptive_pred == 0)))
            newly_introduced_fn = int(np.sum(anomaly_mask & (baseline_pred == 1) & (adaptive_pred == 0)))
            
            baseline_fn = int(np.sum(anomaly_mask & (baseline_pred == 0)))
            adaptive_fn = int(np.sum(anomaly_mask & (adaptive_pred == 0)))
            
            corrected_fp = int(np.sum(normal_mask & (baseline_pred == 1) & (adaptive_pred == 0)))
            persistent_fp = int(np.sum(normal_mask & (baseline_pred == 1) & (adaptive_pred == 1)))
            newly_introduced_fp = int(np.sum(normal_mask & (baseline_pred == 0) & (adaptive_pred == 1)))
            
            baseline_fp = int(np.sum(normal_mask & (baseline_pred == 1)))
            adaptive_fp = int(np.sum(normal_mask & (adaptive_pred == 1)))
            
            if baseline_fn != corrected_fn + persistent_fn:
                logger.warning(f"⚠️  Seed {seed}: FN identity failed")
                npz.close()
                return False
            
            if adaptive_fn != persistent_fn + newly_introduced_fn:
                logger.warning(f"⚠️  Seed {seed}: Adaptive FN identity failed")
                npz.close()
                return False
            
            if baseline_fp != corrected_fp + persistent_fp:
                logger.warning(f"⚠️  Seed {seed}: FP identity failed")
                npz.close()
                return False
            
            if adaptive_fp != persistent_fp + newly_introduced_fp:
                logger.warning(f"⚠️  Seed {seed}: Adaptive FP identity failed")
                npz.close()
                return False
            
            npz.close()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Seed {seed} artifact validation failed: {e}")
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
        
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_cols)}")
        
        features = df[feature_cols].values
        return features, feature_cols
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate predictions and return metrics."""
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
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'evaluation_samples': int(len(y_true))
        }
    
    def _compute_transitions(self, y_true: np.ndarray, 
                            baseline_pred: np.ndarray, 
                            adaptive_pred: np.ndarray) -> Dict:
        """Compute sample-level transition counts."""
        anomaly_mask = (y_true == 1)
        normal_mask = (y_true == 0)
        
        corrected_fn = int(np.sum(anomaly_mask & (baseline_pred == 0) & (adaptive_pred == 1)))
        persistent_fn = int(np.sum(anomaly_mask & (baseline_pred == 0) & (adaptive_pred == 0)))
        newly_introduced_fn = int(np.sum(anomaly_mask & (baseline_pred == 1) & (adaptive_pred == 0)))
        
        baseline_fn = int(np.sum(anomaly_mask & (baseline_pred == 0)))
        adaptive_fn = int(np.sum(anomaly_mask & (adaptive_pred == 0)))
        
        corrected_fp = int(np.sum(normal_mask & (baseline_pred == 1) & (adaptive_pred == 0)))
        persistent_fp = int(np.sum(normal_mask & (baseline_pred == 1) & (adaptive_pred == 1)))
        newly_introduced_fp = int(np.sum(normal_mask & (baseline_pred == 0) & (adaptive_pred == 1)))
        
        baseline_fp = int(np.sum(normal_mask & (baseline_pred == 1)))
        adaptive_fp = int(np.sum(normal_mask & (adaptive_pred == 1)))
        
        if baseline_fn != corrected_fn + persistent_fn:
            raise ValueError(f"FN identity failed: {baseline_fn} != {corrected_fn} + {persistent_fn}")
        
        if adaptive_fn != persistent_fn + newly_introduced_fn:
            raise ValueError(f"Adaptive FN identity failed: {adaptive_fn} != {persistent_fn} + {newly_introduced_fn}")
        
        if baseline_fp != corrected_fp + persistent_fp:
            raise ValueError(f"FP identity failed: {baseline_fp} != {corrected_fp} + {persistent_fp}")
        
        if adaptive_fp != persistent_fp + newly_introduced_fp:
            raise ValueError(f"Adaptive FP identity failed: {adaptive_fp} != {persistent_fp} + {newly_introduced_fp}")
        
        return {
            'corrected_fn': corrected_fn,
            'persistent_fn': persistent_fn,
            'newly_introduced_fn': newly_introduced_fn,
            'baseline_fn': baseline_fn,
            'adaptive_fn': adaptive_fn,
            'corrected_fp': corrected_fp,
            'persistent_fp': persistent_fp,
            'newly_introduced_fp': newly_introduced_fp,
            'baseline_fp': baseline_fp,
            'adaptive_fp': adaptive_fp,
            'fn_reduction': baseline_fn - adaptive_fn,
            'fp_reduction': baseline_fp - adaptive_fp
        }
    
    def _save_seed_outputs(self, seed: int, baseline_metrics: Dict, adaptive_metrics: Dict, 
                          transitions: Dict, y_true: np.ndarray, baseline_pred: np.ndarray, 
                          adaptive_pred: np.ndarray, test_indices: np.ndarray,
                          baseline_decision_score: np.ndarray, adaptive_decision_score: np.ndarray):
        """Save outputs after each seed."""
        npz_path = self.extension_dir / 'predictions' / f'one_class_svm_seed{seed}_predictions.npz'
        np.savez(
            npz_path,
            y_true=y_true,
            baseline_y_pred=baseline_pred,
            adaptive_y_pred=adaptive_pred,
            test_indices=test_indices,
            baseline_decision_score=baseline_decision_score,
            adaptive_decision_score=adaptive_decision_score
        )
        
        self.seed_artifacts[str(seed)] = {
            'baseline_metrics': baseline_metrics,
            'adaptive_metrics': adaptive_metrics,
            'transitions': transitions,
            'npz_path': str(npz_path)
        }
        
        self._update_results_csv()
        self._update_transitions_csv()
        self._save_checkpoint()
        
        logger.info(f"   ✅ Seed {seed} outputs saved")
    
    def _update_results_csv(self):
        """Update results CSV incrementally."""
        rows = []
        for seed_artifacts in self.seed_artifacts.values():
            if 'baseline_metrics' in seed_artifacts:
                rows.append(seed_artifacts['baseline_metrics'])
            if 'adaptive_metrics' in seed_artifacts:
                rows.append(seed_artifacts['adaptive_metrics'])
        
        if rows:
            metrics_df = pd.DataFrame(rows)
            metrics_path = self.extension_dir / 'metrics' / 'one_class_svm_results.csv'
            metrics_df.to_csv(metrics_path, index=False)
    
    def _update_transitions_csv(self):
        """Update transitions CSV incrementally."""
        if self.seed_artifacts:
            transitions_list = [artifacts['transitions'] for artifacts in self.seed_artifacts.values() 
                               if 'transitions' in artifacts]
            if transitions_list:
                transitions_df = pd.DataFrame(transitions_list)
                transitions_path = self.extension_dir / 'metrics' / 'one_class_svm_transitions.csv'
                transitions_df.to_csv(transitions_path, index=False)
    
    def _compute_multiseed_summary(self):
        """Compute multi-seed summary statistics."""
        logger.info("📊 Computing multi-seed summary statistics...")
        
        if not self.seed_results:
            self.multiseed_summary = {
                'status': 'no_results',
                'note': 'No results available'
            }
            return
        
        summary_stats = {}
        
        for variant in ['baseline', 'adaptive_refit']:
            variant_data = [r[variant] for r in self.seed_results if variant in r]
            
            if not variant_data:
                continue
            
            def safe_mean_std(values):
                valid = [v for v in values if v is not None and np.isfinite(v)]
                if len(valid) == 0:
                    return None, None
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                return mean, std
            
            summary_stats[variant] = {
                'accuracy_mean': safe_mean_std([m['accuracy'] for m in variant_data])[0],
                'accuracy_std': safe_mean_std([m['accuracy'] for m in variant_data])[1],
                'precision_mean': safe_mean_std([m['precision'] for m in variant_data])[0],
                'precision_std': safe_mean_std([m['precision'] for m in variant_data])[1],
                'recall_mean': safe_mean_std([m['recall'] for m in variant_data])[0],
                'recall_std': safe_mean_std([m['recall'] for m in variant_data])[1],
                'f1_mean': safe_mean_std([m['f1_score'] for m in variant_data])[0],
                'f1_std': safe_mean_std([m['f1_score'] for m in variant_data])[1],
                'fp_mean': safe_mean_std([m['false_positive'] for m in variant_data])[0],
                'fp_std': safe_mean_std([m['false_positive'] for m in variant_data])[1],
                'fn_mean': safe_mean_std([m['false_negative'] for m in variant_data])[0],
                'fn_std': safe_mean_std([m['false_negative'] for m in variant_data])[1],
                'train_time_seconds_mean': safe_mean_std([m['train_time_seconds'] for m in variant_data])[0],
                'train_time_seconds_std': safe_mean_std([m['train_time_seconds'] for m in variant_data])[1],
            }
        
        if 'baseline' in summary_stats and 'adaptive_refit' in summary_stats:
            summary_stats['mean_changes'] = {
                'accuracy_change': summary_stats['adaptive_refit']['accuracy_mean'] - summary_stats['baseline']['accuracy_mean'],
                'precision_change': summary_stats['adaptive_refit']['precision_mean'] - summary_stats['baseline']['precision_mean'],
                'recall_change': summary_stats['adaptive_refit']['recall_mean'] - summary_stats['baseline']['recall_mean'],
                'f1_change': summary_stats['adaptive_refit']['f1_mean'] - summary_stats['baseline']['f1_mean'],
                'fp_reduction': summary_stats['baseline']['fp_mean'] - summary_stats['adaptive_refit']['fp_mean'],
                'fn_reduction': summary_stats['baseline']['fn_mean'] - summary_stats['adaptive_refit']['fn_mean'],
            }
        
        self.multiseed_summary = {
            'total_seeds': len(SEEDS),
            'completed_seeds': len(self.completed_seeds),
            'seeds': SEEDS,
            'model': 'OneClassSVM',
            'hyperparameters': {
                'kernel': OCSVM_KERNEL,
                'gamma': OCSVM_GAMMA,
                'nu': OCSVM_NU,
            },
            'variant_statistics': summary_stats,
            'scientific_note': 'One-Class SVM is unsupervised. Adaptive variant uses unsupervised refit on train+retrain. Historical April 2026 baseline is context only, not validation target.',
            'reproducibility_metadata': {
                'python_version': sys.version.split()[0],
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__,
                'sklearn_version': None,
                'dataset_path': str(self.dataset_path),
                'dataset_row_count': len(self.original_dataset) if self.original_dataset is not None else None,
                'feature_count': EXPECTED_FEATURE_COUNT,
                'split_ratios': {
                    'train': TRAIN_RATIO,
                    'retrain': RETRAIN_RATIO,
                    'test': TEST_RATIO
                },
                'preprocessing': 'SimpleImputer_mean_StandardScaler_fitted_on_train_only'
            }
        }
        
        try:
            import sklearn
            self.multiseed_summary['reproducibility_metadata']['sklearn_version'] = sklearn.__version__
        except:
            pass
    
    def _generate_report(self) -> str:
        """Generate comprehensive report."""
        logger.info("📝 Generating report...")
        
        report = []
        report.append("="*80)
        report.append("ONE-CLASS SVM EXTENSION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 1.0.0")
        report.append(f"Thesis Project: CPS Anomaly Detection Framework")
        report.append("")
        
        report.append("EXPERIMENTAL CONFIGURATION:")
        report.append(f"  - Model: One-Class SVM")
        report.append(f"  - Kernel: {OCSVM_KERNEL}")
        report.append(f"  - Gamma: {OCSVM_GAMMA}")
        report.append(f"  - Nu: {OCSVM_NU}")
        report.append(f"  - Seeds: {SEEDS}")
        report.append(f"  - Completed seeds: {self.completed_seeds}")
        report.append(f"  - Train/Retrain/Test ratios: {TRAIN_RATIO}/{RETRAIN_RATIO}/{TEST_RATIO}")
        report.append("")
        
        report.append("VARIANT DESCRIPTION:")
        report.append(f"  - baseline: fit on X_train only (unsupervised)")
        report.append(f"  - adaptive_refit: fit on X_train + X_retrain (unsupervised, fresh model)")
        report.append(f"  - adaptation_method: unsupervised_refit_train_plus_retrain")
        report.append("")
        
        report.append("MULTI-SEED SUMMARY STATISTICS:")
        if self.multiseed_summary and 'variant_statistics' in self.multiseed_summary:
            for variant, stats in self.multiseed_summary['variant_statistics'].items():
                if variant == 'mean_changes':
                    continue
                report.append(f"  {variant}:")
                report.append(f"    - Accuracy: {stats.get('accuracy_mean', 'N/A')} ± {stats.get('accuracy_std', 0):.4f}")
                report.append(f"    - Precision: {stats.get('precision_mean', 'N/A')} ± {stats.get('precision_std', 0):.4f}")
                report.append(f"    - Recall: {stats.get('recall_mean', 'N/A')} ± {stats.get('recall_std', 0):.4f}")
                report.append(f"    - F1: {stats.get('f1_mean', 'N/A')} ± {stats.get('f1_std', 0):.4f}")
                report.append(f"    - FP: {stats.get('fp_mean', 'N/A')} ± {stats.get('fp_std', 0):.2f}")
                report.append(f"    - FN: {stats.get('fn_mean', 'N/A')} ± {stats.get('fn_std', 0):.2f}")
            
            if 'mean_changes' in self.multiseed_summary['variant_statistics']:
                changes = self.multiseed_summary['variant_statistics']['mean_changes']
                report.append(f"  Mean Changes (adaptive - baseline):")
                report.append(f"    - Accuracy change: {changes.get('accuracy_change', 0):.4f}")
                report.append(f"    - Precision change: {changes.get('precision_change', 0):.4f}")
                report.append(f"    - Recall change: {changes.get('recall_change', 0):.4f}")
                report.append(f"    - F1 change: {changes.get('f1_change', 0):.4f}")
                report.append(f"    - FP reduction: {changes.get('fp_reduction', 0):.2f} (positive=improvement)")
                report.append(f"    - FN reduction: {changes.get('fn_reduction', 0):.2f} (positive=improvement)")
        
        report.append("")
        report.append("SCIENTIFIC NOTES:")
        report.append(f"  - One-Class SVM is UNSUPERVISED (no y during model.fit)")
        report.append(f"  - Historical April 2026 baseline is CONTEXT ONLY")
        report.append(f"  - NOT used as validation target for new 60/20/20 experiment")
        report.append(f"  - Positive FP/FN reduction means improvement")
        report.append("")
        
        report.append("="*80)
        report.append("END OF ONE-CLASS SVM EXTENSION REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        report_path = self.extension_dir / 'reports' / 'one_class_svm_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return str(report_path)
    
    def run_single_seed(self, seed: int) -> Tuple[bool, Dict, Dict, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run complete analysis for one seed.
        
        Returns:
            (success, baseline_metrics, adaptive_metrics, transitions, y_true, 
             baseline_pred, adaptive_pred, test_indices, baseline_decision_score, 
             adaptive_decision_score)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING SEED {seed}")
        logger.info(f"{'='*80}")
        
        try:
            set_random_seeds(seed)
            
            if self.original_dataset is None:
                if not self.dataset_path.exists():
                    logger.error(f"❌ Dataset not found: {self.dataset_path}")
                    return False, {}, {}, {}, None, None, None, None, None, None
                
                self.original_dataset = pd.read_csv(self.dataset_path)
                self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
                logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows")
            
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            n_features = len(feature_cols)
            logger.info(f"📊 Features: {n_features} columns")
            
            all_indices = np.arange(len(self.original_dataset))
            
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                all_features, all_labels, all_indices,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=seed
            )
            
            X_train, X_retrain, y_train, y_retrain, idx_train, idx_retrain = train_test_split(
                X_temp, y_temp, idx_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=seed
            )
            
            logger.info(f"📊 Split sizes: Train={len(X_train)}, Retrain={len(X_retrain)}, Test={len(X_test)}")
            logger.info(f"📊 Test indices: {len(idx_test)} unique indices")
            
            if len(np.unique(idx_test)) != len(idx_test):
                logger.error("❌ Test indices are not unique")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            train_set = set(idx_train)
            retrain_set = set(idx_retrain)
            test_set = set(idx_test)
            
            if train_set.intersection(retrain_set):
                logger.error("❌ Train and retrain sets overlap")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            if train_set.intersection(test_set):
                logger.error("❌ Train and test sets overlap")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            if retrain_set.intersection(test_set):
                logger.error("❌ Retrain and test sets overlap")
                return False, {}, {}, {}, None, None, None, None, None, None
            
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            logger.info(f"✅ Preprocessing complete (fit on train only)")
            
            # =========================================================
            # Variant A: BASELINE (fit on X_train only)
            # =========================================================
            logger.info(f"📊 Training baseline One-Class SVM...")
            logger.info(f"   Seed: {seed}, Variant: baseline")
            logger.info(f"   Training samples: {len(X_train_scaled)}, Features: {n_features}")
            baseline_start = time.time()
            
            model_baseline = OneClassSVM(
                kernel=OCSVM_KERNEL,
                gamma=OCSVM_GAMMA,
                nu=OCSVM_NU
            )
            
            model_baseline.fit(X_train_scaled)
            
            baseline_time = time.time() - baseline_start
            logger.info(f"   Elapsed training seconds: {baseline_time:.2f}")
            
            # Predict on X_test
            baseline_raw = model_baseline.predict(X_test_scaled)
            baseline_pred = np.where(baseline_raw == -1, 1, 0)
            baseline_decision_score = model_baseline.decision_function(X_test_scaled)
            
            # =========================================================
            # Variant B: ADAPTIVE_REFIT (fresh model on X_train + X_retrain)
            # =========================================================
            logger.info(f"📊 Training adaptive_refit One-Class SVM...")
            logger.info(f"   Seed: {seed}, Variant: adaptive_refit")
            X_train_plus_retrain = np.vstack([X_train_scaled, X_retrain_scaled])
            logger.info(f"   Training samples: {len(X_train_plus_retrain)}, Features: {n_features}")
            adaptive_start = time.time()
            
            model_adaptive = OneClassSVM(
                kernel=OCSVM_KERNEL,
                gamma=OCSVM_GAMMA,
                nu=OCSVM_NU
            )
            
            model_adaptive.fit(X_train_plus_retrain)
            
            adaptive_time = time.time() - adaptive_start
            logger.info(f"   Elapsed training seconds: {adaptive_time:.2f}")
            
            # Predict on SAME X_test
            adaptive_raw = model_adaptive.predict(X_test_scaled)
            adaptive_pred = np.where(adaptive_raw == -1, 1, 0)
            adaptive_decision_score = model_adaptive.decision_function(X_test_scaled)
            
            # =========================================================
            # Evaluate both variants
            # =========================================================
            baseline_metrics = self._evaluate_predictions(y_test, baseline_pred)
            baseline_metrics['variant'] = 'baseline'
            baseline_metrics['adaptation_method'] = 'unsupervised_fit_train_only'
            baseline_metrics['seed'] = seed
            baseline_metrics['train_time_seconds'] = baseline_time
            baseline_metrics['test_samples'] = len(y_test)
            
            adaptive_metrics = self._evaluate_predictions(y_test, adaptive_pred)
            adaptive_metrics['variant'] = 'adaptive_refit'
            adaptive_metrics['adaptation_method'] = 'unsupervised_refit_train_plus_retrain'
            adaptive_metrics['seed'] = seed
            adaptive_metrics['train_time_seconds'] = adaptive_time
            adaptive_metrics['test_samples'] = len(y_test)
            
            # =========================================================
            # Compute transitions
            # =========================================================
            transitions = self._compute_transitions(y_test, baseline_pred, adaptive_pred)
            transitions['seed'] = seed
            
            logger.info(f"✅ Seed {seed} completed")
            logger.info(f"   Baseline F1: {baseline_metrics['f1_score']:.4f}")
            logger.info(f"   Adaptive F1: {adaptive_metrics['f1_score']:.4f}")
            logger.info(f"   FN reduction: {transitions['fn_reduction']}, FP reduction: {transitions['fp_reduction']}")
            
            return True, baseline_metrics, adaptive_metrics, transitions, y_test, baseline_pred, adaptive_pred, idx_test, baseline_decision_score, adaptive_decision_score
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
            return False, {}, {}, {}, None, None, None, None, None, None
    
    def run_full_extension(self) -> Dict:
        """Run complete One-Class SVM extension."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING ONE-CLASS SVM EXTENSION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'extension_complete': False
        }
        
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.dataset_path}")
            results['status'] = 'failed'
            return results
        
        try:
            self.original_dataset = pd.read_csv(self.dataset_path)
            self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
            logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows")
            
            self.dataset_fingerprint = self._compute_dataset_fingerprint()
            self.config_fingerprint = self._get_config_hash()
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            results['status'] = 'failed'
            return results
        
        self._load_checkpoint()
        
        for seed in SEEDS:
            if seed in self.completed_seeds:
                if str(seed) in self.seed_artifacts:
                    if self._validate_seed_artifact(seed, self.seed_artifacts[str(seed)]):
                        logger.info(f"⏭️  Seed {seed} already completed and validated - skipping")
                        continue
                    else:
                        logger.info(f"⚠️  Seed {seed} artifacts invalid - recomputing")
                        self.completed_seeds.remove(seed)
                        del self.seed_artifacts[str(seed)]
                        self.seed_results = [r for r in self.seed_results 
                                           if r.get('baseline', {}).get('seed') != seed]
                        self.transition_results = [t for t in self.transition_results if t.get('seed') != seed]
                else:
                    logger.info(f"⚠️  Seed {seed} in completed_seeds but no artifacts - recomputing")
                    self.completed_seeds.remove(seed)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 Processing seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
            logger.info(f"{'='*80}")
            
            success, baseline_metrics, adaptive_metrics, transitions, y_true, baseline_pred, adaptive_pred, test_indices, baseline_decision_score, adaptive_decision_score = self.run_single_seed(seed)
            
            if success:
                self.seed_results.append({
                    'baseline': baseline_metrics,
                    'adaptive_refit': adaptive_metrics
                })
                
                self.transition_results.append(transitions)
                
                self.completed_seeds.append(seed)
                
                self._save_seed_outputs(seed, baseline_metrics, adaptive_metrics, transitions, 
                                       y_true, baseline_pred, adaptive_pred, test_indices,
                                       baseline_decision_score, adaptive_decision_score)
                
                logger.info(f"✅ Seed {seed} completed successfully")
            else:
                logger.error(f"❌ Seed {seed} failed")
                results['status'] = 'failed'
                return results
        
        self._compute_multiseed_summary()
        
        if self.multiseed_summary:
            summary_json_path = self.extension_dir / 'metrics' / 'one_class_svm_summary.json'
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.multiseed_summary, f, indent=2)
            results['outputs_saved'].append(str(summary_json_path))
        
        report_path = self._generate_report()
        results['outputs_saved'].append(str(report_path))
        
        if self.multiseed_summary and 'variant_statistics' in self.multiseed_summary:
            summary_rows = []
            for variant, stats in self.multiseed_summary['variant_statistics'].items():
                if variant != 'mean_changes':
                    row = {'variant': variant}
                    row.update(stats)
                    summary_rows.append(row)
            
            if 'mean_changes' in self.multiseed_summary['variant_statistics']:
                row = {'variant': 'mean_changes'}
                row.update(self.multiseed_summary['variant_statistics']['mean_changes'])
                summary_rows.append(row)
            
            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = self.extension_dir / 'metrics' / 'one_class_svm_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            results['outputs_saved'].append(str(summary_csv_path))
        
        if len(self.completed_seeds) == len(SEEDS):
            results['status'] = 'complete'
            results['extension_complete'] = True
        else:
            results['status'] = 'partial'
            results['extension_complete'] = False
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute One-Class SVM extension."""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'one_class_svm_extension')
    
    logger.info("\n" + "="*80)
    logger.info("🔧 ONE-CLASS SVM EXTENSION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    extension_dir = base_dir / 'adaptive' / 'final_validation_extended' / 'one_class_svm'
    extension_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Extension directory: {extension_dir}")
    logger.info(f"⚠️  Does NOT modify existing four-model, Isolation Forest, or Autoencoder result files")
    logger.info(f"⚠️  Historical April 2026 baseline is CONTEXT ONLY")
    logger.info(f"⚠️  Checkpoint/resume enabled for long-running analysis")
    logger.warning("⚠️  WARNING: Full RBF One-Class SVM may require many hours per fit.")
    
    extension = OneClassSVMExtension(base_dir=base_dir, extension_dir=extension_dir)
    results = extension.run_full_extension()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ONE-CLASS SVM EXTENSION COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs saved to: {extension_dir}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()