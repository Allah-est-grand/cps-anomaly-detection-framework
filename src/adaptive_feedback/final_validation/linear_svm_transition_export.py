"""
Linear SVM Transition Export - Paired Prediction Reproducibility
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This script regenerates paired Linear SVM predictions under the SAME validated
protocol so that sample-level error transitions can be exported.

IMPORTANT:
- This is NOT a new experiment
- This is NOT model tuning
- This is NOT a new validation protocol
- This is a reproducibility/export run for Linear SVM ONLY
- Uses EXACT same configuration as validated four-model ablation

OUTPUT FILES:
-------------
- results/adaptive/final_validation_four_model/error_analysis/linear_svm_prediction_exports/linear_svm_seed{seed}_predictions.npz
- results/adaptive/final_validation_four_model/error_analysis/metrics/linear_svm_transitions.csv
- results/adaptive/final_validation_four_model/error_analysis/metrics/linear_svm_transitions_summary.csv
- results/adaptive/final_validation_four_model/error_analysis/checkpoints/linear_svm_transition_export_checkpoint.json
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
from sklearn.svm import LinearSVC

# =============================================================================
# CONFIGURATION (from validated four_model_ablation_extension.py)
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

# FIX #1: Linear SVM configuration (from validated four_model_ablation_extension.py)
# random_state=42 FIXED for ALL seeds - seed controls splits ONLY
SVM_C = 1.0
SVM_CLASS_WEIGHT = None
SVM_MAX_ITER = 10000
SVM_RANDOM_STATE = 42  # FIXED - do NOT use random_state=seed

# Anomaly weight for weighted_retraining (from validated ablation)
ANOMALY_WEIGHT = 3.0

# Require exactly 51 features
EXPECTED_FEATURE_COUNT = 51

CHECKPOINT_VERSION = '1.0'

# Existing validated ablation results path (MUST exist for metric consistency check)
EXISTING_ABLATION_RESULTS_PATH = None  # Will be set in __init__

# =============================================================================
# LINEAR SVM TRANSITION EXPORT CLASS
# =============================================================================
class LinearSVMTransitionExport:
    """
    Regenerates Linear SVM paired predictions for transition export.
    """
    
    def __init__(self, base_dir: Path, export_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.export_dir = Path(export_dir) if export_dir else (
            self.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'linear_svm_prediction_exports'
        )
        self.metrics_dir = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'metrics'
        self.checkpoint_dir = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'checkpoints'
        
        # Create output directories
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.seed_results = []
        self.transition_results = []
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        # Checkpoint management
        self.checkpoint_path = self.checkpoint_dir / 'linear_svm_transition_export_checkpoint.json'
        self.completed_seeds = []
        self.seed_artifacts = {}
        
        # FIX #8: Reproduction status per seed:variant (dictionary, not single string)
        self.reproduction_status = {}
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        # FIX #2: Existing ablation results for metric consistency check (MUST exist)
        self.ablation_results_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
        EXISTING_ABLATION_RESULTS_PATH = self.ablation_results_path
        self.existing_ablation_df = None
        
        logger.info(f"✅ Initialized LinearSVMTransitionExport")
        logger.info(f"📁 Export directory: {self.export_dir}")
        logger.info(f"🔢 Seeds: {SEEDS}")
        logger.info(f"🔧 Linear SVM: C={SVM_C}, max_iter={SVM_MAX_ITER}, random_state={SVM_RANDOM_STATE} (FIXED), anomaly_weight={ANOMALY_WEIGHT}")
        logger.info(f"📊 Existing ablation results: {self.ablation_results_path}")
    
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
            'svm_c': SVM_C,
            'svm_class_weight': SVM_CLASS_WEIGHT,
            'svm_max_iter': SVM_MAX_ITER,
            'svm_random_state': SVM_RANDOM_STATE,
            'anomaly_weight': ANOMALY_WEIGHT,
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
            'reproduction_status': self.reproduction_status,  # FIX #8: Dictionary per seed:variant
            'saved_at': datetime.now().isoformat()
        }
        
        self._save_atomically(self.checkpoint_path, checkpoint)
    
    def _load_checkpoint(self) -> bool:
        """FIX #3 & #9: Load and validate checkpoint with per-seed artifacts, restoring BOTH seed_results and transition_results."""
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
            
            # FIX #9: Restore reproduction_status from checkpoint
            self.reproduction_status = checkpoint.get('reproduction_status', {})
            if not isinstance(self.reproduction_status, dict):
                logger.warning(f"⚠️  Invalid reproduction_status in checkpoint, resetting to {{}}")
                self.reproduction_status = {}
            
            # FIX #3: Restore BOTH seed_results AND transition_results from artifacts
            for seed_str, artifacts in self.seed_artifacts.items():
                seed = int(seed_str)
                if 'baseline_metrics' in artifacts and 'retraining_metrics' in artifacts and 'weighted_metrics' in artifacts:
                    self.seed_results.append({
                        'baseline': artifacts['baseline_metrics'],
                        'retraining': artifacts['retraining_metrics'],
                        'weighted': artifacts['weighted_metrics']
                    })
                if 'transitions' in artifacts:
                    # FIX #4: Restore transitions with seed and model_name fields
                    for trans in artifacts['transitions']:
                        trans_copy = trans.copy()
                        trans_copy['seed'] = seed
                        trans_copy['model_name'] = 'LinearSVM'
                        self.transition_results.append(trans_copy)
            
            logger.info(f"📥 Loaded checkpoint: {len(self.completed_seeds)} completed seeds")
            logger.info(f"📥 Restored {len(self.seed_results)} seed results and {len(self.transition_results)} transition records")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            return False
    
    def _validate_seed_artifact(self, seed: int, artifacts: Dict) -> bool:
        """FIX #3 & #6: Validate all artifacts exist and are consistent before skipping seed."""
        try:
            # Check all three variant metrics exist
            if 'baseline_metrics' not in artifacts or 'retraining_metrics' not in artifacts or 'weighted_metrics' not in artifacts:
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
            required_arrays = ['y_true', 'baseline_y_pred', 'retraining_without_weighting_y_pred', 'weighted_retraining_y_pred', 'test_indices']
            for arr_name in required_arrays:
                if arr_name not in npz:
                    logger.warning(f"⚠️  Seed {seed}: NPZ missing {arr_name}")
                    npz.close()
                    return False
            
            n_samples = len(npz['y_true'])
            for arr_name in ['baseline_y_pred', 'retraining_without_weighting_y_pred', 'weighted_retraining_y_pred', 'test_indices']:
                if len(npz[arr_name]) != n_samples:
                    logger.warning(f"⚠️  Seed {seed}: {arr_name} length mismatch")
                    npz.close()
                    return False
            
            if len(np.unique(npz['test_indices'])) != n_samples:
                logger.warning(f"⚠️  Seed {seed}: test_indices not unique")
                npz.close()
                return False
            
            # FIX #3: Validate predictions are binary
            for arr_name in ['baseline_y_pred', 'retraining_without_weighting_y_pred', 'weighted_retraining_y_pred']:
                if not np.all(np.isin(npz[arr_name], [0, 1])):
                    logger.warning(f"⚠️  Seed {seed}: {arr_name} not binary")
                    npz.close()
                    return False
            
            # FIX #5 & #6: Recompute and validate ALL saved transition identities (FN and FP)
            y_true = npz['y_true']
            baseline_pred = npz['baseline_y_pred']
            retraining_pred = npz['retraining_without_weighting_y_pred']
            weighted_pred = npz['weighted_retraining_y_pred']
            
            anomaly_mask = (y_true == 1)
            normal_mask = (y_true == 0)
            
            # ===== VALIDATE FN IDENTITIES =====
            # baseline -> retraining
            corrected_fn_1 = int(np.sum(anomaly_mask & (baseline_pred == 0) & (retraining_pred == 1)))
            persistent_fn_1 = int(np.sum(anomaly_mask & (baseline_pred == 0) & (retraining_pred == 0)))
            newly_introduced_fn_1 = int(np.sum(anomaly_mask & (baseline_pred == 1) & (retraining_pred == 0)))
            baseline_fn = int(np.sum(anomaly_mask & (baseline_pred == 0)))
            retraining_fn = int(np.sum(anomaly_mask & (retraining_pred == 0)))
            
            if baseline_fn != corrected_fn_1 + persistent_fn_1:
                logger.warning(f"⚠️  Seed {seed}: baseline->retraining FN identity failed")
                npz.close()
                return False
            
            if retraining_fn != persistent_fn_1 + newly_introduced_fn_1:
                logger.warning(f"⚠️  Seed {seed}: baseline->retraining target FN identity failed")
                npz.close()
                return False
            
            # retraining -> weighted
            corrected_fn_2 = int(np.sum(anomaly_mask & (retraining_pred == 0) & (weighted_pred == 1)))
            persistent_fn_2 = int(np.sum(anomaly_mask & (retraining_pred == 0) & (weighted_pred == 0)))
            newly_introduced_fn_2 = int(np.sum(anomaly_mask & (retraining_pred == 1) & (weighted_pred == 0)))
            weighted_fn = int(np.sum(anomaly_mask & (weighted_pred == 0)))
            
            if retraining_fn != corrected_fn_2 + persistent_fn_2:
                logger.warning(f"⚠️  Seed {seed}: retraining->weighted FN identity failed")
                npz.close()
                return False
            
            if weighted_fn != persistent_fn_2 + newly_introduced_fn_2:
                logger.warning(f"⚠️  Seed {seed}: retraining->weighted target FN identity failed")
                npz.close()
                return False
            
            # baseline -> weighted
            corrected_fn_3 = int(np.sum(anomaly_mask & (baseline_pred == 0) & (weighted_pred == 1)))
            persistent_fn_3 = int(np.sum(anomaly_mask & (baseline_pred == 0) & (weighted_pred == 0)))
            newly_introduced_fn_3 = int(np.sum(anomaly_mask & (baseline_pred == 1) & (weighted_pred == 0)))
            
            if baseline_fn != corrected_fn_3 + persistent_fn_3:
                logger.warning(f"⚠️  Seed {seed}: baseline->weighted FN identity failed")
                npz.close()
                return False
            
            if weighted_fn != persistent_fn_3 + newly_introduced_fn_3:
                logger.warning(f"⚠️  Seed {seed}: baseline->weighted target FN identity failed")
                npz.close()
                return False
            
            # ===== FIX #6: VALIDATE FP IDENTITIES (COMPLETE, NOT ABBREVIATED) =====
            # baseline -> retraining
            corrected_fp_1 = int(np.sum(normal_mask & (baseline_pred == 1) & (retraining_pred == 0)))
            persistent_fp_1 = int(np.sum(normal_mask & (baseline_pred == 1) & (retraining_pred == 1)))
            newly_introduced_fp_1 = int(np.sum(normal_mask & (baseline_pred == 0) & (retraining_pred == 1)))
            baseline_fp = int(np.sum(normal_mask & (baseline_pred == 1)))
            retraining_fp = int(np.sum(normal_mask & (retraining_pred == 1)))
            
            if baseline_fp != corrected_fp_1 + persistent_fp_1:
                logger.warning(f"⚠️  Seed {seed}: baseline->retraining FP identity failed")
                npz.close()
                return False
            
            if retraining_fp != persistent_fp_1 + newly_introduced_fp_1:
                logger.warning(f"⚠️  Seed {seed}: baseline->retraining target FP identity failed")
                npz.close()
                return False
            
            # retraining -> weighted
            corrected_fp_2 = int(np.sum(normal_mask & (retraining_pred == 1) & (weighted_pred == 0)))
            persistent_fp_2 = int(np.sum(normal_mask & (retraining_pred == 1) & (weighted_pred == 1)))
            newly_introduced_fp_2 = int(np.sum(normal_mask & (retraining_pred == 0) & (weighted_pred == 1)))
            weighted_fp = int(np.sum(normal_mask & (weighted_pred == 1)))
            
            if retraining_fp != corrected_fp_2 + persistent_fp_2:
                logger.warning(f"⚠️  Seed {seed}: retraining->weighted FP identity failed")
                npz.close()
                return False
            
            if weighted_fp != persistent_fp_2 + newly_introduced_fp_2:
                logger.warning(f"⚠️  Seed {seed}: retraining->weighted target FP identity failed")
                npz.close()
                return False
            
            # baseline -> weighted
            corrected_fp_3 = int(np.sum(normal_mask & (baseline_pred == 1) & (weighted_pred == 0)))
            persistent_fp_3 = int(np.sum(normal_mask & (baseline_pred == 1) & (weighted_pred == 1)))
            newly_introduced_fp_3 = int(np.sum(normal_mask & (baseline_pred == 0) & (weighted_pred == 1)))
            
            if baseline_fp != corrected_fp_3 + persistent_fp_3:
                logger.warning(f"⚠️  Seed {seed}: baseline->weighted FP identity failed")
                npz.close()
                return False
            
            if weighted_fp != persistent_fp_3 + newly_introduced_fp_3:
                logger.warning(f"⚠️  Seed {seed}: baseline->weighted target FP identity failed")
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
                            source_pred: np.ndarray, 
                            target_pred: np.ndarray,
                            source_metrics: Dict,
                            target_metrics: Dict,
                            source_variant: str,
                            target_variant: str) -> Dict:
        """Compute paired transition counts."""
        anomaly_mask = (y_true == 1)
        normal_mask = (y_true == 0)
        
        # For anomalies
        corrected_fn = int(np.sum(anomaly_mask & (source_pred == 0) & (target_pred == 1)))
        persistent_fn = int(np.sum(anomaly_mask & (source_pred == 0) & (target_pred == 0)))
        newly_introduced_fn = int(np.sum(anomaly_mask & (source_pred == 1) & (target_pred == 0)))
        
        source_fn = int(np.sum(anomaly_mask & (source_pred == 0)))
        target_fn = int(np.sum(anomaly_mask & (target_pred == 0)))
        
        # For normals
        corrected_fp = int(np.sum(normal_mask & (source_pred == 1) & (target_pred == 0)))
        persistent_fp = int(np.sum(normal_mask & (source_pred == 1) & (target_pred == 1)))
        newly_introduced_fp = int(np.sum(normal_mask & (source_pred == 0) & (target_pred == 1)))
        
        source_fp = int(np.sum(normal_mask & (source_pred == 1)))
        target_fp = int(np.sum(normal_mask & (target_pred == 1)))
        
        # Validate identities
        if source_fn != corrected_fn + persistent_fn:
            raise ValueError(f"FN identity failed: {source_fn} != {corrected_fn} + {persistent_fn}")
        
        if target_fn != persistent_fn + newly_introduced_fn:
            raise ValueError(f"Target FN identity failed: {target_fn} != {persistent_fn} + {newly_introduced_fn}")
        
        if source_fp != corrected_fp + persistent_fp:
            raise ValueError(f"FP identity failed: {source_fp} != {corrected_fp} + {persistent_fp}")
        
        if target_fp != persistent_fp + newly_introduced_fp:
            raise ValueError(f"Target FP identity failed: {target_fp} != {persistent_fp} + {newly_introduced_fp}")
        
        return {
            'corrected_fn': corrected_fn,
            'persistent_fn': persistent_fn,
            'newly_introduced_fn': newly_introduced_fn,
            'source_fn': source_fn,
            'target_fn': target_fn,
            'fn_reduction': source_fn - target_fn,
            'corrected_fp': corrected_fp,
            'persistent_fp': persistent_fp,
            'newly_introduced_fp': newly_introduced_fp,
            'source_fp': source_fp,
            'target_fp': target_fp,
            'fp_reduction': source_fp - target_fp,
            'source_f1': source_metrics['f1_score'],
            'target_f1': target_metrics['f1_score'],
            'f1_change': target_metrics['f1_score'] - source_metrics['f1_score'],
            'source_variant': source_variant,
            'target_variant': target_variant
        }
    
    def _load_existing_ablation_results(self) -> bool:
        """FIX #2: Load existing validated ablation results for metric consistency check. FAIL CLOSED if file doesn't exist."""
        # FIX #2: File MUST exist
        if not Path(self.ablation_results_path).exists():
            logger.error(f"❌ Existing ablation results NOT FOUND: {self.ablation_results_path}")
            logger.error(f"❌ Cannot proceed without validated ablation results for metric consistency check")
            return False
        
        try:
            self.existing_ablation_df = pd.read_csv(self.ablation_results_path)
            logger.info(f"✅ Loaded existing ablation results: {len(self.existing_ablation_df)} rows")
            
            # Verify file has expected LinearSVM data
            linear_svm_rows = self.existing_ablation_df[self.existing_ablation_df['model_name'] == 'LinearSVM']
            if len(linear_svm_rows) == 0:
                logger.error(f"❌ No LinearSVM results found in existing ablation results")
                return False
            
            logger.info(f"✅ Found {len(linear_svm_rows)} LinearSVM rows in existing ablation results")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing ablation results: {e}")
            return False
    
    def _check_metric_consistency(self, seed: int, variant: str, regenerated_metrics: Dict) -> bool:
        """
        Check regenerated metrics against existing validated ablation results.
        
        CORRECTED POLICY:
        - baseline and retraining_without_weighting: strict matching
        - weighted_retraining: allow at most ONE test sample classification difference
        - precision/recall/F1 tolerances NOT applied (denominators differ from N_test)
        """
        # existing_ablation_df MUST be loaded
        if self.existing_ablation_df is None:
            logger.error(f"❌ Cannot check metric consistency: existing ablation results not loaded")
            return False  # FAIL CLOSED
        
        try:
            existing_row = self.existing_ablation_df[
                (self.existing_ablation_df['seed'] == seed) &
                (self.existing_ablation_df['model_name'] == 'LinearSVM') &
                (self.existing_ablation_df['variant'] == variant)
            ]
            
            # Must find EXACTLY one matching row
            if len(existing_row) == 0:
                logger.error(f"❌ No existing ablation result found for seed={seed}, model=LinearSVM, variant={variant}")
                return False  # FAIL CLOSED
            
            if len(existing_row) > 1:
                logger.error(f"❌ Multiple ({len(existing_row)}) existing ablation results found for seed={seed}, model=LinearSVM, variant={variant}")
                return False  # FAIL CLOSED
            
            existing_row = existing_row.iloc[0]
            
            # Get evaluation samples for tolerance calculation
            evaluation_samples = regenerated_metrics.get('evaluation_samples', 288344)
            one_sample_accuracy_tolerance = 1.0 / evaluation_samples + 1e-12
            
            # FIX #1: Verify historical FP and FN values exist and are finite (do NOT use default 0)
            existing_fp = existing_row.get('false_positive')
            existing_fn = existing_row.get('false_negative')
            
            if existing_fp is None or not np.isfinite(existing_fp):
                logger.error(f"❌ Historical FP is None or not finite for seed={seed}, variant={variant}")
                return False  # FAIL CLOSED
            
            if existing_fn is None or not np.isfinite(existing_fn):
                logger.error(f"❌ Historical FN is None or not finite for seed={seed}, variant={variant}")
                return False  # FAIL CLOSED
            
            # Calculate confusion matrix discrepancies
            fp_diff = abs(regenerated_metrics.get('false_positive', 0) - existing_fp)
            fn_diff = abs(regenerated_metrics.get('false_negative', 0) - existing_fn)
            classification_count_difference = fp_diff + fn_diff
            
            # Calculate accuracy difference
            accuracy_diff = abs(regenerated_metrics.get('accuracy', 0) - existing_row.get('accuracy', 0))
            
            # FIX #1: Define exact reproduction only if ALL metrics match within 1e-10
            is_exact = (
                fp_diff == 0 and
                fn_diff == 0 and
                abs(regenerated_metrics.get('accuracy', 0) - existing_row.get('accuracy', 0)) <= 1e-10 and
                abs(regenerated_metrics.get('precision', 0) - existing_row.get('precision', 0)) <= 1e-10 and
                abs(regenerated_metrics.get('recall', 0) - existing_row.get('recall', 0)) <= 1e-10 and
                abs(regenerated_metrics.get('f1_score', 0) - existing_row.get('f1_score', 0)) <= 1e-10
            )
            
            # Check all metrics for finiteness
            has_critical_discrepancy = False
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                regenerated_val = regenerated_metrics.get(metric)
                existing_val = existing_row.get(metric)
                
                if regenerated_val is None:
                    logger.error(f"❌ Regenerated metric {metric} is None for seed={seed}, variant={variant}")
                    return False  # FAIL CLOSED
                
                if existing_val is None:
                    logger.error(f"❌ Existing metric {metric} is None for seed={seed}, variant={variant}")
                    return False  # FAIL CLOSED
                
                if not np.isfinite(regenerated_val):
                    logger.error(f"❌ Regenerated metric {metric} is not finite for seed={seed}, variant={variant}")
                    return False  # FAIL CLOSED
                
                if not np.isfinite(existing_val):
                    logger.error(f"❌ Existing metric {metric} is not finite for seed={seed}, variant={variant}")
                    return False  # FAIL CLOSED
            
            # Apply variant-specific matching rules
            if variant in ['baseline', 'retraining_without_weighting']:
                # STRICT matching for baseline and retraining_without_weighting
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    diff = abs(regenerated_metrics.get(metric, 0) - existing_row.get(metric, 0))
                    if diff > 1e-10:
                        logger.error(f"❌ Strict metric mismatch for seed={seed}, variant={variant}, metric={metric}")
                        logger.error(f"   Regenerated: {regenerated_metrics.get(metric)}, Existing: {existing_row.get(metric)}, Diff: {diff}")
                        has_critical_discrepancy = True
                
                # FP/FN must match exactly
                if fp_diff > 0 or fn_diff > 0:
                    logger.error(f"❌ Confusion matrix mismatch for seed={seed}, variant={variant}")
                    logger.error(f"   FP diff: {fp_diff}, FN diff: {fn_diff}")
                    has_critical_discrepancy = True
            
            elif variant == 'weighted_retraining':
                # ONE-SAMPLE TOLERANCE for weighted_retraining only
                if classification_count_difference > 1:
                    logger.error(f"❌ Weighted_retraining classification count difference exceeds one sample")
                    logger.error(f"   FP diff: {fp_diff}, FN diff: {fn_diff}, Total: {classification_count_difference}")
                    has_critical_discrepancy = True
                
                if accuracy_diff > one_sample_accuracy_tolerance:
                    logger.error(f"❌ Weighted_retraining accuracy difference exceeds one-sample tolerance")
                    logger.error(f"   Accuracy diff: {accuracy_diff}, Tolerance: {one_sample_accuracy_tolerance}")
                    has_critical_discrepancy = True
                
                # Report precision/recall/F1 differences but do NOT apply tolerance
                # (their denominators differ from N_test)
                for metric in ['precision', 'recall', 'f1_score']:
                    diff = abs(regenerated_metrics.get(metric, 0) - existing_row.get(metric, 0))
                    logger.info(f"   {metric} diff: {diff} (reported, tolerance not applied)")
            
            if has_critical_discrepancy:
                return False  # FAIL CLOSED
            
            # Determine and store reproduction status
            if is_exact:
                status = 'exact'
                logger.info(f"✅ Metric consistency EXACT for seed={seed}, variant={variant}")
            else:
                status = 'accepted_one_sample_numerical_tolerance'
                logger.warning(f"⚠️  Weighted LinearSVM accepted within one-sample numerical reproducibility tolerance; historical aggregate results remain authoritative.")
                logger.warning(f"   FP diff: {fp_diff}, FN diff: {fn_diff}, Accuracy diff: {accuracy_diff}")
            
            # Store status per seed:variant (not global)
            status_key = f"{seed}:{variant}"
            self.reproduction_status[status_key] = status
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to check metric consistency: {e}")
            return False  # FAIL CLOSED
    
    def _save_seed_outputs(self, seed: int, baseline_metrics: Dict, retraining_metrics: Dict, 
                          weighted_metrics: Dict, transitions: List[Dict], y_true: np.ndarray,
                          baseline_pred: np.ndarray, retraining_pred: np.ndarray,
                          weighted_pred: np.ndarray, test_indices: np.ndarray,
                          baseline_decision: np.ndarray, retraining_decision: np.ndarray,
                          weighted_decision: np.ndarray):
        """Save outputs after each seed."""
        npz_path = self.export_dir / f'linear_svm_seed{seed}_predictions.npz'
        np.savez(
            npz_path,
            y_true=y_true,
            baseline_y_pred=baseline_pred,
            retraining_without_weighting_y_pred=retraining_pred,
            weighted_retraining_y_pred=weighted_pred,
            test_indices=test_indices,
            baseline_decision_score=baseline_decision,
            retraining_without_weighting_decision_score=retraining_decision,
            weighted_retraining_decision_score=weighted_decision
        )
        
        self.seed_artifacts[str(seed)] = {
            'baseline_metrics': baseline_metrics,
            'retraining_metrics': retraining_metrics,
            'weighted_metrics': weighted_metrics,
            'transitions': transitions,
            'npz_path': str(npz_path)
        }
        
        self._update_transitions_csv()
        self._save_checkpoint()
        
        logger.info(f"   ✅ Seed {seed} outputs saved")
    
    def _update_transitions_csv(self):
        """Update transitions CSV incrementally."""
        if self.seed_artifacts:
            transitions_list = []
            for seed_str, artifacts in self.seed_artifacts.items():
                if 'transitions' in artifacts:
                    for trans in artifacts['transitions']:
                        trans_copy = trans.copy()
                        trans_copy['seed'] = int(seed_str)
                        trans_copy['model_name'] = 'LinearSVM'
                        transitions_list.append(trans_copy)
            
            if transitions_list:
                transitions_df = pd.DataFrame(transitions_list)
                transitions_path = self.metrics_dir / 'linear_svm_transitions.csv'
                transitions_df.to_csv(transitions_path, index=False)
    
    def _compute_summary(self) -> pd.DataFrame:
        """Compute summary statistics across seeds."""
        logger.info("📊 Computing summary statistics...")
        
        if not self.transition_results:
            return pd.DataFrame()
        
        summary_rows = []
        comparisons = set(t['comparison'] for t in self.transition_results)
        
        for comparison in comparisons:
            comp_data = [t for t in self.transition_results if t['comparison'] == comparison]
            
            if not comp_data:
                continue
            
            def safe_mean_std(values):
                valid = [v for v in values if v is not None and np.isfinite(v)]
                if len(valid) == 0:
                    return None, None
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                return mean, std
            
            summary_rows.append({
                'comparison': comparison,
                'corrected_fn_mean': safe_mean_std([t['corrected_fn'] for t in comp_data])[0],
                'corrected_fn_std': safe_mean_std([t['corrected_fn'] for t in comp_data])[1],
                'persistent_fn_mean': safe_mean_std([t['persistent_fn'] for t in comp_data])[0],
                'persistent_fn_std': safe_mean_std([t['persistent_fn'] for t in comp_data])[1],
                'newly_introduced_fn_mean': safe_mean_std([t['newly_introduced_fn'] for t in comp_data])[0],
                'newly_introduced_fn_std': safe_mean_std([t['newly_introduced_fn'] for t in comp_data])[1],
                'fn_reduction_mean': safe_mean_std([t['fn_reduction'] for t in comp_data])[0],
                'fn_reduction_std': safe_mean_std([t['fn_reduction'] for t in comp_data])[1],
                'corrected_fp_mean': safe_mean_std([t['corrected_fp'] for t in comp_data])[0],
                'corrected_fp_std': safe_mean_std([t['corrected_fp'] for t in comp_data])[1],
                'persistent_fp_mean': safe_mean_std([t['persistent_fp'] for t in comp_data])[0],
                'persistent_fp_std': safe_mean_std([t['persistent_fp'] for t in comp_data])[1],
                'newly_introduced_fp_mean': safe_mean_std([t['newly_introduced_fp'] for t in comp_data])[0],
                'newly_introduced_fp_std': safe_mean_std([t['newly_introduced_fp'] for t in comp_data])[1],
                'fp_reduction_mean': safe_mean_std([t['fp_reduction'] for t in comp_data])[0],
                'fp_reduction_std': safe_mean_std([t['fp_reduction'] for t in comp_data])[1],
                'f1_change_mean': safe_mean_std([t['f1_change'] for t in comp_data])[0],
                'f1_change_std': safe_mean_std([t['f1_change'] for t in comp_data])[1],
            })
        
        return pd.DataFrame(summary_rows)
    
    def run_single_seed(self, seed: int) -> Tuple[bool, Dict, Dict, Dict, List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run complete analysis for one seed.
        
        Returns:
            (success, baseline_metrics, retraining_metrics, weighted_metrics, transitions,
             y_true, baseline_pred, retraining_pred, weighted_pred, test_indices,
             baseline_decision, retraining_decision, weighted_decision)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING SEED {seed}")
        logger.info(f"{'='*80}")
        
        try:
            set_random_seeds(seed)
            
            if self.original_dataset is None:
                if not self.dataset_path.exists():
                    logger.error(f"❌ Dataset not found: {self.dataset_path}")
                    return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
                
                self.original_dataset = pd.read_csv(self.dataset_path)
                self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
                logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows")
            
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            n_features = len(feature_cols)
            logger.info(f"📊 Features: {n_features} columns")
            
            all_indices = np.arange(len(self.original_dataset))
            
            # Two-stage stratified split (same as validated protocol)
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
            
            # Validate disjointness
            train_set = set(idx_train)
            retrain_set = set(idx_retrain)
            test_set = set(idx_test)
            
            if train_set.intersection(retrain_set) or train_set.intersection(test_set) or retrain_set.intersection(test_set):
                logger.error("❌ Train/retrain/test sets are not mutually disjoint")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            if len(np.unique(idx_test)) != len(idx_test):
                logger.error("❌ Test indices are not unique")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            # Preprocessing (fit on TRAIN ONLY)
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
            logger.info(f"📊 Training baseline Linear SVM...")
            logger.info(f"   Seed: {seed}, Variant: baseline")
            logger.info(f"   Training samples: {len(X_train_scaled)}, Features: {n_features}")
            baseline_start = time.time()
            
            # FIX #1: Use SVM_RANDOM_STATE=42 (FIXED) for ALL seeds
            model_baseline = LinearSVC(
                C=SVM_C,
                class_weight=SVM_CLASS_WEIGHT,
                max_iter=SVM_MAX_ITER,
                random_state=SVM_RANDOM_STATE  # FIXED = 42, NOT seed
            )
            
            model_baseline.fit(X_train_scaled, y_train)
            
            baseline_time = time.time() - baseline_start
            logger.info(f"   Elapsed training seconds: {baseline_time:.2f}")
            
            baseline_pred = model_baseline.predict(X_test_scaled)
            baseline_decision = model_baseline.decision_function(X_test_scaled)
            baseline_metrics = self._evaluate_predictions(y_test, baseline_pred)
            baseline_metrics['variant'] = 'baseline'
            baseline_metrics['seed'] = seed
            
            # FIX #2: Check metric consistency - FAIL CLOSED
            if not self._check_metric_consistency(seed, 'baseline', baseline_metrics):
                logger.error(f"❌ Baseline metric consistency check failed for seed {seed}")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            # =========================================================
            # Variant B: RETRAINING_WITHOUT_WEIGHTING
            # =========================================================
            logger.info(f"📊 Training retraining_without_weighting Linear SVM...")
            logger.info(f"   Seed: {seed}, Variant: retraining_without_weighting")
            X_train_plus_retrain = np.vstack([X_train_scaled, X_retrain_scaled])
            y_train_plus_retrain = np.concatenate([y_train, y_retrain])
            logger.info(f"   Training samples: {len(X_train_plus_retrain)}, Features: {n_features}")
            retraining_start = time.time()
            
            # FIX #1: Use SVM_RANDOM_STATE=42 (FIXED) for ALL seeds
            model_retraining = LinearSVC(
                C=SVM_C,
                class_weight=SVM_CLASS_WEIGHT,
                max_iter=SVM_MAX_ITER,
                random_state=SVM_RANDOM_STATE  # FIXED = 42, NOT seed
            )
            
            model_retraining.fit(X_train_plus_retrain, y_train_plus_retrain)
            
            retraining_time = time.time() - retraining_start
            logger.info(f"   Elapsed training seconds: {retraining_time:.2f}")
            
            retraining_pred = model_retraining.predict(X_test_scaled)
            retraining_decision = model_retraining.decision_function(X_test_scaled)
            retraining_metrics = self._evaluate_predictions(y_test, retraining_pred)
            retraining_metrics['variant'] = 'retraining_without_weighting'
            retraining_metrics['seed'] = seed
            
            # FIX #2: Check metric consistency - FAIL CLOSED
            if not self._check_metric_consistency(seed, 'retraining_without_weighting', retraining_metrics):
                logger.error(f"❌ Retraining metric consistency check failed for seed {seed}")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            # =========================================================
            # Variant C: WEIGHTED_RETRAINING
            # =========================================================
            logger.info(f"📊 Training weighted_retraining Linear SVM...")
            logger.info(f"   Seed: {seed}, Variant: weighted_retraining")
            logger.info(f"   Training samples: {len(X_train_plus_retrain)}, Features: {n_features}")
            logger.info(f"   Anomaly weight: {ANOMALY_WEIGHT}")
            weighted_start = time.time()
            
            # Create sample weights (exact validated logic)
            sample_weights = np.ones(len(y_train_plus_retrain))
            sample_weights[y_train_plus_retrain == 1] = ANOMALY_WEIGHT
            
            # FIX #1: Use SVM_RANDOM_STATE=42 (FIXED) for ALL seeds
            model_weighted = LinearSVC(
                C=SVM_C,
                class_weight=SVM_CLASS_WEIGHT,
                max_iter=SVM_MAX_ITER,
                random_state=SVM_RANDOM_STATE  # FIXED = 42, NOT seed
            )
            
            model_weighted.fit(X_train_plus_retrain, y_train_plus_retrain, sample_weight=sample_weights)
            
            weighted_time = time.time() - weighted_start
            logger.info(f"   Elapsed training seconds: {weighted_time:.2f}")
            
            weighted_pred = model_weighted.predict(X_test_scaled)
            weighted_decision = model_weighted.decision_function(X_test_scaled)
            weighted_metrics = self._evaluate_predictions(y_test, weighted_pred)
            weighted_metrics['variant'] = 'weighted_retraining'
            weighted_metrics['seed'] = seed
            
            # FIX #2: Check metric consistency - FAIL CLOSED
            if not self._check_metric_consistency(seed, 'weighted_retraining', weighted_metrics):
                logger.error(f"❌ Weighted retraining metric consistency check failed for seed {seed}")
                return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
            
            # =========================================================
            # Compute transitions
            # =========================================================
            transitions = []
            
            # baseline -> retraining_without_weighting
            trans1 = self._compute_transitions(
                y_test, baseline_pred, retraining_pred,
                baseline_metrics, retraining_metrics,
                'baseline', 'retraining_without_weighting'
            )
            trans1['comparison'] = 'baseline_vs_retraining'
            transitions.append(trans1)
            
            # retraining_without_weighting -> weighted_retraining
            trans2 = self._compute_transitions(
                y_test, retraining_pred, weighted_pred,
                retraining_metrics, weighted_metrics,
                'retraining_without_weighting', 'weighted_retraining'
            )
            trans2['comparison'] = 'retraining_vs_weighted'
            transitions.append(trans2)
            
            # baseline -> weighted_retraining
            trans3 = self._compute_transitions(
                y_test, baseline_pred, weighted_pred,
                baseline_metrics, weighted_metrics,
                'baseline', 'weighted_retraining'
            )
            trans3['comparison'] = 'baseline_vs_weighted'
            transitions.append(trans3)
            
            logger.info(f"✅ Seed {seed} completed")
            logger.info(f"   Baseline F1: {baseline_metrics['f1_score']:.4f}")
            logger.info(f"   Retraining F1: {retraining_metrics['f1_score']:.4f}")
            logger.info(f"   Weighted F1: {weighted_metrics['f1_score']:.4f}")
            
            return True, baseline_metrics, retraining_metrics, weighted_metrics, transitions, y_test, baseline_pred, retraining_pred, weighted_pred, idx_test, baseline_decision, retraining_decision, weighted_decision
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
            return False, {}, {}, {}, [], None, None, None, None, None, None, None, None
    
    def run_full_export(self) -> Dict:
        """Run complete Linear SVM transition export."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING LINEAR SVM TRANSITION EXPORT")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'export_complete': False
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
        
        # FIX #2: Load existing ablation results for metric consistency check - MUST succeed
        if not self._load_existing_ablation_results():
            logger.error(f"❌ Cannot proceed without existing validated ablation results")
            results['status'] = 'failed'
            return results
        
        # Load checkpoint (FIX #3: restores BOTH seed_results and transition_results)
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
                        # Remove from restored results
                        self.seed_results = [r for r in self.seed_results if not (
                            r.get('baseline', {}).get('seed') == seed or
                            r.get('retraining', {}).get('seed') == seed or
                            r.get('weighted', {}).get('seed') == seed
                        )]
                        self.transition_results = [t for t in self.transition_results if t.get('seed') != seed]
                else:
                    logger.info(f"⚠️  Seed {seed} in completed_seeds but no artifacts - recomputing")
                    self.completed_seeds.remove(seed)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 Processing seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
            logger.info(f"{'='*80}")
            
            success, baseline_metrics, retraining_metrics, weighted_metrics, transitions, y_true, baseline_pred, retraining_pred, weighted_pred, test_indices, baseline_decision, retraining_decision, weighted_decision = self.run_single_seed(seed)
            
            if success:
                self.seed_results.append({
                    'baseline': baseline_metrics,
                    'retraining': retraining_metrics,
                    'weighted': weighted_metrics
                })
                
                self.transition_results.extend(transitions)
                
                self.completed_seeds.append(seed)
                
                self._save_seed_outputs(seed, baseline_metrics, retraining_metrics, weighted_metrics, transitions,
                                       y_true, baseline_pred, retraining_pred, weighted_pred, test_indices,
                                       baseline_decision, retraining_decision, weighted_decision)
                
                logger.info(f"✅ Seed {seed} completed successfully")
            else:
                logger.error(f"❌ Seed {seed} failed")
                results['status'] = 'failed'
                return results
        
        # Compute summary
        summary_df = self._compute_summary()
        if len(summary_df) > 0:
            summary_path = self.metrics_dir / 'linear_svm_transitions_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            results['outputs_saved'].append(str(summary_path))
            logger.info(f"   ✅ Saved summary: {summary_path}")
        
        if len(self.completed_seeds) == len(SEEDS):
            results['status'] = 'complete'
            results['export_complete'] = True
        else:
            results['status'] = 'partial'
            results['export_complete'] = False
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute Linear SVM transition export."""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'linear_svm_transition_export')
    
    logger.info("\n" + "="*80)
    logger.info("🔧 LINEAR SVM TRANSITION EXPORT")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    
    logger.info(f"📁 Export directory: {base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'linear_svm_prediction_exports'}")
    logger.info(f"⚠️  Does NOT modify existing four-model or ablation result files")
    logger.info(f"⚠️  This is a REPRODUCIBILITY/EXPORT run for Linear SVM ONLY")
    logger.info(f"⚠️  Checkpoint/resume enabled")
    logger.info(f"⚠️  Metric consistency check: FAIL CLOSED")
    logger.info(f"⚠️  LinearSVC random_state={SVM_RANDOM_STATE} (FIXED for all seeds)")
    
    export = LinearSVMTransitionExport(base_dir=base_dir)
    results = export.run_full_export()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 LINEAR SVM TRANSITION EXPORT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()