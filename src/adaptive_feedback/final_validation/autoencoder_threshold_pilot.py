"""
Autoencoder Threshold Calibration Pilot - Final Exploratory Study
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements the final exploratory Autoencoder experiment to evaluate
whether supervised threshold calibration improves the precision-recall trade-off
of the linear-output Autoencoder.

This is the FINAL planned exploratory Autoencoder experiment before consolidating
results and proceeding to Isolation Forest analysis.

IMPORTANT SCIENTIFIC NOTE:
--------------------------
This experiment evaluates SUPERVISED THRESHOLD CALIBRATION using a held-out
calibration subset. The calibrated threshold is selected based on F1-score
optimization on calibration data, then evaluated on a separate held-out
calibration-evaluation subset.

This is NOT an independent final test because the overall Autoencoder development
has already involved earlier evaluations. Results should be interpreted as
developmental evidence only.

EXPERIMENTAL DESIGN:
--------------------
Seed: 42 (consistent with activation pilot)

Compares:
1. Historical Threshold (95th percentile of normal TRAIN reconstruction errors)
   vs
2. Calibrated Threshold (best F1 from candidate percentiles on normal calibration subset)

Both evaluated on SAME held-out calibration-evaluation subset.

OUTPUT FILES:
-------------
- results/adaptive/autoencoder_threshold_pilot/metrics/threshold_pilot_results.csv
- results/adaptive/autoencoder_threshold_pilot/metrics/threshold_pilot_summary.json
- results/adaptive/autoencoder_threshold_pilot/metrics/threshold_candidate_metrics.csv
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
from __future__ import annotations

import sys
import time
import logging
import json
import traceback
import gc
import math  # FIX: Added for partition size calculation
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to sys.path for config import
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

# TensorFlow for Autoencoder
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42  # Consistent with activation pilot

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

# Retrain subdivision (from activation pilot)
ACTIVATION_PILOT_VAL_RATIO_WITHIN_RETRAIN = 0.3  # 30% excluded (activation pilot development validation)
RESERVED_RATIO_WITHIN_RETRAIN = 0.7  # 70% reserved for this threshold experiment

# Split reserved 70% into two equal subsets
CALIBRATION_RATIO = 0.5
CALIBRATION_EVAL_RATIO = 0.5

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']

METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

# Autoencoder training configuration (from activation pilot)
AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_VAL_SPLIT = 0.1
AE_LEARNING_RATE = 0.001

# Prediction batch size for memory safety
PREDICTION_BATCH_SIZE = 2048

# Historical threshold percentile
HISTORICAL_THRESHOLD_PERCENTILE = 95

# Candidate threshold percentiles for calibration
CANDIDATE_PERCENTILES = [90, 92, 94, 95, 96, 97, 98, 99, 99.5]

EXPECTED_RECORD_COUNT = 2  # (historical_threshold) + (calibrated_threshold)
EXPECTED_CANDIDATE_COUNT = 9  # Number of candidate percentiles

# =============================================================================
# AUTOENCODER THRESHOLD PILOT CLASS
# =============================================================================
class AutoencoderThresholdPilot:
    """
    Implements final exploratory threshold calibration study for Autoencoder.
    """
    
    def __init__(self, base_dir: Path, pilot_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.pilot_dir = Path(pilot_dir) if pilot_dir else (
            self.base_dir / 'adaptive' / 'autoencoder_threshold_pilot'
        )
        
        (self.pilot_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.results = []
        self.calibration_candidates = []
        self.pilot_status = 'pending'
        self.data_source_description = ''
        
        # Initialize actual value storage attributes
        self.feature_count = 0
        self.train_size = 0
        self.retrain_size = 0
        self.test_size = 0
        self.activation_pilot_val_size = 0
        self.calibration_size = 0
        self.calibration_normal_count = 0
        self.calibration_anomaly_count = 0
        self.calibration_eval_size = 0
        self.calibration_eval_normal_count = 0
        self.calibration_eval_anomaly_count = 0
        
        # FIX: Split integrity tracking with original indices
        self.split_integrity_checks = {}
        self.original_indices = {
            'train': None,
            'retrain_reserved': None,
            'activation_pilot_val': None,
            'calibration': None,
            'calibration_eval': None,
            'test': None
        }
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized AutoencoderThresholdPilot")
        logger.info(f"📁 Pilot directory: {self.pilot_dir}")
        logger.info(f"🔢 Seed: {SEED}")
        logger.info(f"📈 Expected records: {EXPECTED_RECORD_COUNT}")
        logger.info(f"📈 Expected candidates: {EXPECTED_CANDIDATE_COUNT}")
    
    def load_resources(self) -> bool:
        logger.info("📊 Loading resources...")
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.dataset_path}")
            return False
        
        try:
            self.original_dataset = pd.read_csv(self.dataset_path)
            self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
            logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows, {len(self.original_dataset.columns)} columns")
            self.data_source_description = str(self.dataset_path)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def _identify_label_column(self, df: pd.DataFrame) -> Optional[str]:
        for col_name in LABEL_COLUMN_NAMES:
            if col_name in df.columns:
                return col_name
        for col in df.columns:
            if col.lower() in [name.lower() for name in LABEL_COLUMN_NAMES]:
                return col
        return None
    
    def _convert_labels_to_binary(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        labels_raw = df[label_col].values
        labels = np.zeros(len(labels_raw), dtype=int)
        normal_indicators = ['normal', '0', 'benign', 'ok', 'healthy', 'false', 'negative']
        for i, label in enumerate(labels_raw):
            if str(label).lower().strip() not in normal_indicators:
                labels[i] = 1
        logger.info(f"📊 Labels: {np.sum(labels == 0)} normal, {np.sum(labels == 1)} anomaly")
        return labels
    
    def _prepare_features(self, df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, List[str]]:
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
        
        logger.info(f"📊 Feature selection: {len(feature_cols)} numeric features retained")
        return df[feature_cols].values, feature_cols
    
    def _create_autoencoder_model(self, n_features: int) -> Any:
        """
        Create Autoencoder model with LINEAR output activation (from activation pilot)
        """
        # Encoder
        encoder_input = layers.Input(shape=(n_features,))
        x = layers.Dense(64, activation='relu')(encoder_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(16, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        decoded = layers.Dense(n_features, activation='linear')(x)  # LINEAR output
        
        # Autoencoder
        autoencoder = models.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
            loss='mse'
        )
        
        logger.debug(f"📊 Autoencoder model created (linear output): {autoencoder.count_params()} parameters")
        return autoencoder
    
    def _compute_reconstruction_errors_batched(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction MSE in batches without storing full reconstruction matrix.
        Memory-safe batch-by-batch MSE computation.
        FIX: Removed per-batch gc.collect(); cleanup after stage completion instead.
        """
        all_mse = []
        
        for i in range(0, len(X), PREDICTION_BATCH_SIZE):
            batch = X[i:i+PREDICTION_BATCH_SIZE]
            recon_batch = model.predict(batch, verbose=0)
            
            # Compute MSE for this batch only
            batch_mse = np.mean(np.power(batch - recon_batch, 2), axis=1)
            all_mse.append(batch_mse)
            
            # Clean up batch reconstruction immediately (no gc.collect per batch)
            del recon_batch, batch
        
        # Concatenate only the 1-D MSE arrays
        reconstruction_errors = np.concatenate(all_mse)
        del all_mse
        
        return reconstruction_errors
    
    def _evaluate_threshold(self, reconstruction_errors: np.ndarray, y_true: np.ndarray, 
                           threshold: float, threshold_name: str) -> Dict:
        """
        Evaluate a specific threshold on given data.
        FIX: Full-precision values internally, enhanced metrics for candidate CSV.
        """
        # Binary predictions
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Full-precision metrics internally
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Verify confusion matrix sums to evaluation population
        cm_sum = tn + fp + fn + tp
        evaluation_n = len(y_true)
        if cm_sum != evaluation_n:
            logger.warning(f"⚠️  Confusion matrix sum ({cm_sum}) != evaluation_n ({evaluation_n})")
        
        return {
            'threshold_name': threshold_name,
            'threshold_value': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive_rate': float(fpr),
            'evaluation_samples': int(evaluation_n),
            'confusion_matrix_sum': int(cm_sum)
        }
    
    def run_pilot(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Run complete Autoencoder threshold calibration pilot.
        FIX #1: Preserve original indices throughout all splits.
        FIX #2: Reproduce activation pilot partition exactly (70% reserved, 30% excluded).
        
        Returns:
            Tuple of (results, calibration_candidates)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING AUTOENCODER THRESHOLD CALIBRATION PILOT (Seed={SEED})")
        logger.info(f"{'='*80}")
        
        results = []
        calibration_candidates = []
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("❌ TensorFlow not available. Cannot run Autoencoder pilot.")
            return results, calibration_candidates
        
        try:
            set_random_seeds(SEED)
            
            # Load dataset
            if not self.load_resources():
                return results, calibration_candidates
            
            # Identify label column
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return results, calibration_candidates
            
            # Convert labels
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            # Prepare features
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return results, calibration_candidates
            
            # Store actual feature count
            self.feature_count = len(feature_cols)
            n_features = all_features.shape[1]
            logger.info(f"📊 Features: {all_features.shape[0]} samples, {n_features} features")
            
            # FIX #1: Create original row-index array BEFORE any splitting
            original_indices = np.arange(len(all_features))
            logger.info(f"📊 Original indices created: {len(original_indices)} indices")
            
            # =====================================================================
            # STAGE 1: Two-stage stratified split (same as activation pilot)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 1: Two-stage stratified split")
            logger.info(f"{'='*80}")
            
            # Stage 1: 80% temporary (train+retrain), 20% historical test (NOT USED)
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                all_features, all_labels, original_indices,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=SEED
            )
            
            # Stage 2: Of the 80% temporary: 75% -> train (60% overall), 25% -> retrain (20% overall)
            X_train, X_retrain, y_train, y_retrain, idx_train, idx_retrain = train_test_split(
                X_temp, y_temp, idx_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=SEED
            )
            
            # Store actual split sizes
            self.train_size = len(X_train)
            self.retrain_size = len(X_retrain)
            self.test_size = len(X_test)
            
            # FIX #1: Store original indices for integrity verification
            self.original_indices['train'] = idx_train
            self.original_indices['test'] = idx_test
            
            logger.info(f"📊 Split sizes: Train={self.train_size}, Retrain={self.retrain_size}, Historical Test={self.test_size}")
            logger.info(f"⚠️  Historical test partition NOT USED in this threshold calibration experiment")
            
            # =====================================================================
            # STAGE 2: Subdivide RETRAIN (FIX #2: Reproduce activation pilot EXACTLY)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 2: Subdivide RETRAIN (reproduce activation pilot)")
            logger.info(f"{'='*80}")
            
            # FIX #2: EXACT reproduction of activation pilot split
            # First output: 70% reserved for threshold pilot
            # Second output: 30% activation_pilot_val (EXCLUDED)
            (
                X_retrain_reserved,
                X_retrain_act_val,
                y_retrain_reserved,
                y_retrain_act_val,
                idx_retrain_reserved,
                idx_retrain_act_val
            ) = train_test_split(
                X_retrain,
                y_retrain,
                idx_retrain,
                test_size=0.3,  # 30% excluded (activation pilot development validation)
                stratify=y_retrain,
                random_state=SEED
            )
            
            self.activation_pilot_val_size = len(X_retrain_act_val)
            self.retrain_reserved_size = len(X_retrain_reserved)
            
            # FIX #1: Store original indices
            self.original_indices['activation_pilot_val'] = idx_retrain_act_val
            self.original_indices['retrain_reserved'] = idx_retrain_reserved
            
            logger.info(f"📊 Activation pilot validation (excluded): {self.activation_pilot_val_size} samples")
            logger.info(f"📊 Reserved for threshold pilot: {self.retrain_reserved_size} samples")
            
            # =====================================================================
            # STAGE 3: Split reserved 70% into calibration + calibration-evaluation
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 3: Split reserved into calibration subsets")
            logger.info(f"{'='*80}")
            
            # Split reserved 70% into two equal stratified subsets
            X_calibration, X_calibration_eval, y_calibration, y_calibration_eval, idx_calibration, idx_calibration_eval = train_test_split(
                X_retrain_reserved, y_retrain_reserved, idx_retrain_reserved,
                test_size=CALIBRATION_EVAL_RATIO,  # 0.5
                stratify=y_retrain_reserved,
                random_state=SEED
            )
            
            self.calibration_size = len(X_calibration)
            self.calibration_normal_count = int(np.sum(y_calibration == 0))
            self.calibration_anomaly_count = int(np.sum(y_calibration == 1))
            self.calibration_eval_size = len(X_calibration_eval)
            self.calibration_eval_normal_count = int(np.sum(y_calibration_eval == 0))
            self.calibration_eval_anomaly_count = int(np.sum(y_calibration_eval == 1))
            
            # FIX #1: Store original indices
            self.original_indices['calibration'] = idx_calibration
            self.original_indices['calibration_eval'] = idx_calibration_eval
            
            logger.info(f"📊 Threshold calibration subset: {self.calibration_size} samples")
            logger.info(f"📊   Normal: {self.calibration_normal_count}, Anomaly: {self.calibration_anomaly_count}")
            logger.info(f"📊 Calibration-evaluation subset: {self.calibration_eval_size} samples")
            logger.info(f"📊   Normal: {self.calibration_eval_normal_count}, Anomaly: {self.calibration_eval_anomaly_count}")
            
            # =====================================================================
            # STAGE 4: Split integrity checks (FIX #1 & #4)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 4: Split integrity checks (original indices)")
            logger.info(f"{'='*80}")
            
            integrity_checks_passed = True
            
            # FIX: Verify partition sizes are correct (70% reserved, 30% excluded)
            # FIX: Use math.ceil to match sklearn's train_test_split rounding behavior
            expected_act_val = math.ceil(
                self.retrain_size * ACTIVATION_PILOT_VAL_RATIO_WITHIN_RETRAIN
            )
            expected_reserved = self.retrain_size - expected_act_val
            
            reserved_size_check = (self.retrain_reserved_size == expected_reserved)
            act_val_size_check = (self.activation_pilot_val_size == expected_act_val)
            
            self.split_integrity_checks['reserved_size_correct'] = reserved_size_check
            self.split_integrity_checks['activation_pilot_val_size_correct'] = act_val_size_check
            self.split_integrity_checks['reserved_size_expected'] = expected_reserved
            self.split_integrity_checks['reserved_size_actual'] = self.retrain_reserved_size
            self.split_integrity_checks['activation_pilot_val_size_expected'] = expected_act_val
            self.split_integrity_checks['activation_pilot_val_size_actual'] = self.activation_pilot_val_size
            
            logger.info(f"{'✅' if reserved_size_check else '❌'} Reserved size correct: {self.retrain_reserved_size} (expected {expected_reserved})")
            logger.info(f"{'✅' if act_val_size_check else '❌'} Activation pilot val size correct: {self.activation_pilot_val_size} (expected {expected_act_val})")
            
            if not reserved_size_check or not act_val_size_check:
                integrity_checks_passed = False
            
            # FIX: Verify all calibration/evaluation indices belong to reserved 70%
            reserved_set = set(idx_retrain_reserved)
            calib_set = set(idx_calibration)
            calib_eval_set = set(idx_calibration_eval)
            
            all_calib_in_reserved = calib_set.issubset(reserved_set)
            all_calib_eval_in_reserved = calib_eval_set.issubset(reserved_set)
            
            self.split_integrity_checks['all_calibration_in_reserved'] = all_calib_in_reserved
            self.split_integrity_checks['all_calibration_eval_in_reserved'] = all_calib_eval_in_reserved
            
            logger.info(f"{'✅' if all_calib_in_reserved else '❌'} All calibration indices in reserved 70%")
            logger.info(f"{'✅' if all_calib_eval_in_reserved else '❌'} All calibration-evaluation indices in reserved 70%")
            
            if not all_calib_in_reserved or not all_calib_eval_in_reserved:
                integrity_checks_passed = False
            
            # Check calibration and calibration-evaluation are disjoint
            overlap_calib_eval = len(calib_set.intersection(calib_eval_set))
            self.split_integrity_checks['calibration_eval_disjoint'] = (overlap_calib_eval == 0)
            self.split_integrity_checks['calibration_eval_overlap_count'] = int(overlap_calib_eval)
            logger.info(f"{'✅' if overlap_calib_eval == 0 else '❌'} Calibration and calibration-evaluation disjoint: overlap={overlap_calib_eval}")
            if overlap_calib_eval > 0:
                integrity_checks_passed = False
            
            # Check neither overlaps with TRAIN
            train_set = set(idx_train)
            overlap_train_calib = len(train_set.intersection(calib_set))
            overlap_train_calib_eval = len(train_set.intersection(calib_eval_set))
            self.split_integrity_checks['train_calibration_disjoint'] = (overlap_train_calib == 0)
            self.split_integrity_checks['train_calibration_eval_disjoint'] = (overlap_train_calib_eval == 0)
            self.split_integrity_checks['train_calibration_overlap_count'] = int(overlap_train_calib)
            self.split_integrity_checks['train_calibration_eval_overlap_count'] = int(overlap_train_calib_eval)
            logger.info(f"{'✅' if overlap_train_calib == 0 else '❌'} TRAIN and calibration disjoint: overlap={overlap_train_calib}")
            logger.info(f"{'✅' if overlap_train_calib_eval == 0 else '❌'} TRAIN and calibration-evaluation disjoint: overlap={overlap_train_calib_eval}")
            if overlap_train_calib > 0 or overlap_train_calib_eval > 0:
                integrity_checks_passed = False
            
            # Check neither overlaps with historical TEST
            test_set = set(idx_test)
            overlap_test_calib = len(test_set.intersection(calib_set))
            overlap_test_calib_eval = len(test_set.intersection(calib_eval_set))
            self.split_integrity_checks['test_calibration_disjoint'] = (overlap_test_calib == 0)
            self.split_integrity_checks['test_calibration_eval_disjoint'] = (overlap_test_calib_eval == 0)
            self.split_integrity_checks['test_calibration_overlap_count'] = int(overlap_test_calib)
            self.split_integrity_checks['test_calibration_eval_overlap_count'] = int(overlap_test_calib_eval)
            logger.info(f"{'✅' if overlap_test_calib == 0 else '❌'} Historical TEST and calibration disjoint: overlap={overlap_test_calib}")
            logger.info(f"{'✅' if overlap_test_calib_eval == 0 else '❌'} Historical TEST and calibration-evaluation disjoint: overlap={overlap_test_calib_eval}")
            if overlap_test_calib > 0 or overlap_test_calib_eval > 0:
                integrity_checks_passed = False
            
            # Check activation pilot val is disjoint from calibration subsets
            act_val_set = set(idx_retrain_act_val)
            overlap_actval_calib = len(act_val_set.intersection(calib_set))
            overlap_actval_calib_eval = len(act_val_set.intersection(calib_eval_set))
            self.split_integrity_checks['activation_pilot_val_calibration_disjoint'] = (overlap_actval_calib == 0)
            self.split_integrity_checks['activation_pilot_val_calibration_eval_disjoint'] = (overlap_actval_calib_eval == 0)
            self.split_integrity_checks['activation_pilot_val_calibration_overlap_count'] = int(overlap_actval_calib)
            self.split_integrity_checks['activation_pilot_val_calibration_eval_overlap_count'] = int(overlap_actval_calib_eval)
            logger.info(f"{'✅' if overlap_actval_calib == 0 else '❌'} Activation pilot val and calibration disjoint: overlap={overlap_actval_calib}")
            logger.info(f"{'✅' if overlap_actval_calib_eval == 0 else '❌'} Activation pilot val and calibration-evaluation disjoint: overlap={overlap_actval_calib_eval}")
            if overlap_actval_calib > 0 or overlap_actval_calib_eval > 0:
                integrity_checks_passed = False
            
            # FIX #4: Do not allow empty integrity-check dictionary to count as success
            self.split_integrity_checks['all_required_checks_performed'] = (len(self.split_integrity_checks) >= 10)
            self.split_integrity_checks['all_integrity_checks_passed'] = integrity_checks_passed
            
            logger.info(f"📊 Total integrity checks performed: {len(self.split_integrity_checks)}")
            logger.info(f"{'✅' if integrity_checks_passed else '❌'} All split integrity checks passed: {integrity_checks_passed}")
            
            if not integrity_checks_passed:
                logger.error("❌ Split integrity checks failed. Cannot proceed.")
                return results, calibration_candidates
            
            # =====================================================================
            # STAGE 5: Preprocessing (fit on TRAIN ONLY)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 5: Preprocessing (fit on TRAIN ONLY)")
            logger.info(f"{'='*80}")
            
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_calibration_imp = imputer.transform(X_calibration)
            X_calibration_eval_imp = imputer.transform(X_calibration_eval)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_calibration_scaled = scaler.transform(X_calibration_imp)
            X_calibration_eval_scaled = scaler.transform(X_calibration_eval_imp)
            
            # Convert to float32 for TensorFlow memory efficiency
            X_train_scaled = X_train_scaled.astype(np.float32)
            X_calibration_scaled = X_calibration_scaled.astype(np.float32)
            X_calibration_eval_scaled = X_calibration_eval_scaled.astype(np.float32)
            
            logger.info(f"✅ Preprocessing fitted on train set only (no leakage)")
            
            # =====================================================================
            # STAGE 6: Extract NORMAL samples for training and calibration
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 6: Extract NORMAL samples")
            logger.info(f"{'='*80}")
            
            X_train_normal = X_train_scaled[y_train == 0]
            n_train_normal = len(X_train_normal)
            logger.info(f"📊 Training on {n_train_normal} normal samples only")
            
            X_calibration_normal = X_calibration_scaled[y_calibration == 0]
            n_calibration_normal = len(X_calibration_normal)
            logger.info(f"📊 Calibration threshold calibration on {n_calibration_normal} normal samples")
            
            # =====================================================================
            # STAGE 7: Train Autoencoder
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 STAGE 7: Train Autoencoder (linear output)")
            logger.info(f"{'='*80}")
            
            try:
                # Reset TensorFlow state and seed
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Create FRESH Autoencoder with LINEAR output
                model = self._create_autoencoder_model(n_features)
                
                # Train on normal samples only
                history = model.fit(
                    X_train_normal, X_train_normal,
                    epochs=AE_EPOCHS,
                    batch_size=AE_BATCH_SIZE,
                    validation_split=AE_VAL_SPLIT,
                    verbose=0
                )
                
                epochs_trained = len(history.history['loss'])
                logger.info(f"✅ Autoencoder trained for {epochs_trained} epochs")
                
                # =====================================================================
                # STAGE 8: Compute HISTORICAL threshold (95th percentile of normal TRAIN)
                # =====================================================================
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 STAGE 8: Compute HISTORICAL threshold")
                logger.info(f"{'='*80}")
                
                train_normal_errors = self._compute_reconstruction_errors_batched(model, X_train_normal)
                historical_threshold = np.percentile(train_normal_errors, HISTORICAL_THRESHOLD_PERCENTILE)
                logger.info(f"📊 Historical threshold ({HISTORICAL_THRESHOLD_PERCENTILE}th percentile of normal TRAIN): {historical_threshold:.6f}")
                
                # Cleanup after reconstruction stage (not per batch)
                del train_normal_errors
                gc.collect()
                
                # =====================================================================
                # STAGE 9: Threshold CALIBRATION (on normal CALIBRATION subset)
                # =====================================================================
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 STAGE 9: Threshold CALIBRATION")
                logger.info(f"{'='*80}")
                
                # Compute reconstruction errors on CALIBRATION subset (both normal and anomaly)
                calibration_errors = self._compute_reconstruction_errors_batched(model, X_calibration_scaled)
                calibration_normal_errors = calibration_errors[y_calibration == 0]
                
                # Test candidate percentiles on NORMAL calibration errors
                logger.info(f"📊 Testing {len(CANDIDATE_PERCENTILES)} candidate percentiles on normal calibration errors")
                
                best_f1 = -1.0
                best_recall = -1.0
                best_threshold = None
                best_percentile = None
                
                for percentile in CANDIDATE_PERCENTILES:
                    threshold = np.percentile(calibration_normal_errors, percentile)
                    
                    # Evaluate on CALIBRATION subset (supervised threshold selection)
                    y_pred = (calibration_errors > threshold).astype(int)
                    
                    # Compute full metrics for candidate CSV
                    cm = confusion_matrix(y_calibration, y_pred, labels=[0, 1])
                    if cm.shape == (2, 2):
                        tn_c, fp_c, fn_c, tp_c = cm.ravel()
                    else:
                        tn_c = fp_c = fn_c = tp_c = 0
                    
                    f1 = f1_score(y_calibration, y_pred, zero_division=0)
                    recall = recall_score(y_calibration, y_pred, zero_division=0)
                    precision = precision_score(y_calibration, y_pred, zero_division=0)
                    fpr = fp_c / (fp_c + tn_c) if (fp_c + tn_c) > 0 else 0.0
                    
                    # FIX: Store full metrics for candidate CSV
                    candidate = {
                        'percentile': percentile,
                        'threshold_value': float(threshold),
                        'f1_score': float(f1),
                        'recall': float(recall),
                        'precision': float(precision),
                        'false_positive': int(fp_c),
                        'false_negative': int(fn_c),
                        'false_positive_rate': float(fpr),
                        'selection_rule': 'highest F1, then highest recall, then highest threshold'
                    }
                    calibration_candidates.append(candidate)
                    
                    logger.info(f"   Percentile {percentile}: Threshold={threshold:.6f}, F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
                    
                    # Deterministic tie-breaking (highest F1, then highest recall, then highest threshold)
                    if (f1 > best_f1) or (f1 == best_f1 and recall > best_recall) or (f1 == best_f1 and recall == best_recall and threshold > best_threshold):
                        best_f1 = f1
                        best_recall = recall
                        best_threshold = threshold
                        best_percentile = percentile
                
                logger.info(f"✅ Selected calibrated threshold: {best_percentile}th percentile = {best_threshold:.6f} (F1={best_f1:.4f})")
                
                # Cleanup after reconstruction stage
                del calibration_errors, calibration_normal_errors
                gc.collect()
                
                # =====================================================================
                # STAGE 10: Independent CALIBRATION-EVALUATION
                # =====================================================================
                logger.info(f"\n{'='*80}")
                logger.info(f"📊 STAGE 10: Independent CALIBRATION-EVALUATION")
                logger.info(f"{'='*80}")
                
                # Compute reconstruction errors on CALIBRATION-EVALUATION subset
                calibration_eval_errors = self._compute_reconstruction_errors_batched(model, X_calibration_eval_scaled)
                
                # Evaluate HISTORICAL threshold
                historical_result = self._evaluate_threshold(
                    calibration_eval_errors, y_calibration_eval,
                    historical_threshold, 'historical_95th_percentile'
                )
                historical_result['seed'] = SEED
                historical_result['variant'] = 'historical_threshold'
                historical_result['epochs_trained'] = epochs_trained
                historical_result['training_samples'] = n_train_normal
                historical_result['calibration_samples'] = self.calibration_size
                historical_result['calibration_normal_count'] = self.calibration_normal_count
                historical_result['calibration_eval_samples'] = self.calibration_eval_size
                historical_result['calibration_eval_normal_count'] = self.calibration_eval_normal_count
                historical_result['calibration_eval_anomaly_count'] = self.calibration_eval_anomaly_count
                historical_result['status'] = 'success'
                results.append(historical_result)
                
                logger.info(f"📊 Historical threshold on calibration-evaluation: F1={historical_result['f1_score']:.4f}, Precision={historical_result['precision']:.4f}, Recall={historical_result['recall']:.4f}")
                
                # Evaluate CALIBRATED threshold
                calibrated_result = self._evaluate_threshold(
                    calibration_eval_errors, y_calibration_eval,
                    best_threshold, 'calibrated_threshold'
                )
                calibrated_result['seed'] = SEED
                calibrated_result['variant'] = 'calibrated_threshold'
                calibrated_result['calibrated_percentile'] = best_percentile
                calibrated_result['epochs_trained'] = epochs_trained
                calibrated_result['training_samples'] = n_train_normal
                calibrated_result['calibration_samples'] = self.calibration_size
                calibrated_result['calibration_normal_count'] = self.calibration_normal_count
                calibrated_result['calibration_eval_samples'] = self.calibration_eval_size
                calibrated_result['calibration_eval_normal_count'] = self.calibration_eval_normal_count
                calibrated_result['calibration_eval_anomaly_count'] = self.calibration_eval_anomaly_count
                calibrated_result['status'] = 'success'
                results.append(calibrated_result)
                
                logger.info(f"📊 Calibrated threshold on calibration-evaluation: F1={calibrated_result['f1_score']:.4f}, Precision={calibrated_result['precision']:.4f}, Recall={calibrated_result['recall']:.4f}")
                
                # Cleanup after reconstruction stage
                del calibration_eval_errors, model
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Pilot failed: {e}")
                logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"❌ Pilot failed: {e}")
            logger.error(traceback.format_exc())
        
        return results, calibration_candidates
    
    def _verify_completeness(self, results: List[Dict], calibration_candidates: List[Dict]) -> Tuple[bool, List[str]]:
        """
        FIX #4: Strengthened completeness verification.
        Verify completeness with exactly 2 expected variants, 9 candidates, and all split-integrity checks passed.
        """
        missing_items = []
        
        if not results:
            missing_items.append("No results generated")
            return False, missing_items
        
        if not calibration_candidates:
            missing_items.append("No calibration candidates recorded")
            return False, missing_items
        
        # FIX #4: Verify exactly 9 candidates
        if len(calibration_candidates) != EXPECTED_CANDIDATE_COUNT:
            missing_items.append(f"Expected {EXPECTED_CANDIDATE_COUNT} candidates, got {len(calibration_candidates)}")
        
        # FIX #4: Verify each expected percentile occurs exactly once
        candidate_percentiles = [c.get('percentile') for c in calibration_candidates]
        for expected_percentile in CANDIDATE_PERCENTILES:
            count = candidate_percentiles.count(expected_percentile)
            if count != 1:
                missing_items.append(f"Percentile {expected_percentile} occurs {count} times (expected 1)")
        
        # FIX #4: Verify exactly 2 successful evaluation records
        successful = [r for r in results if r.get('status') == 'success']
        if len(successful) != EXPECTED_RECORD_COUNT:
            missing_items.append(f"Expected {EXPECTED_RECORD_COUNT} successful records, got {len(successful)}")
        
        # Extract unique variants from successful results
        actual_variants = set(r.get('variant') for r in successful)
        
        # Define expected variants
        expected_variants = {'historical_threshold', 'calibrated_threshold'}
        
        # Check for missing expected variants
        missing_variants = expected_variants - actual_variants
        if missing_variants:
            missing_items.append(f"Missing expected variants: {missing_variants}")
        
        # Check for unexpected variants
        unexpected_variants = actual_variants - expected_variants
        if unexpected_variants:
            missing_items.append(f"Unexpected variants: {unexpected_variants}")
        
        # FIX #4: Verify both thresholds evaluated on SAME evaluation samples
        if len(results) >= 2:
            eval_samples_1 = results[0].get('calibration_eval_samples')
            eval_samples_2 = results[1].get('calibration_eval_samples')
            if eval_samples_1 != eval_samples_2:
                missing_items.append(f"Thresholds evaluated on different sample counts: {eval_samples_1} vs {eval_samples_2}")
            
            # FIX #4: Verify confusion matrix sums match evaluation_n
            for r in results:
                cm_sum = r.get('confusion_matrix_sum')
                eval_n = r.get('evaluation_samples')
                if cm_sum is not None and eval_n is not None and cm_sum != eval_n:
                    missing_items.append(f"Confusion matrix sum ({cm_sum}) != evaluation_n ({eval_n}) for {r.get('variant')}")
        
        # FIX #4: Verify all required numerical outputs are present and finite
        for r in results:
            for key in ['threshold_value', 'accuracy', 'precision', 'recall', 'f1_score']:
                val = r.get(key)
                if val is None or not np.isfinite(val):
                    missing_items.append(f"Missing or non-finite {key} in {r.get('variant')}")
        
        # FIX #4: Verify split integrity checks were performed and passed
        if not self.split_integrity_checks:
            missing_items.append("Split integrity checks dictionary is empty")
        elif not self.split_integrity_checks.get('all_integrity_checks_passed', False):
            missing_items.append("Split integrity checks did not all pass")
        elif not self.split_integrity_checks.get('all_required_checks_performed', False):
            missing_items.append("Not all required integrity checks were performed")
        
        is_complete = (len(missing_items) == 0 and len(actual_variants) == EXPECTED_RECORD_COUNT)
        
        return is_complete, missing_items
    
    def save_results(self, results: List[Dict], calibration_candidates: List[Dict]) -> List[str]:
        logger.info("💾 Saving pilot results...")
        saved_paths = []
        
        try:
            if results:
                # Save main results CSV
                results_df = pd.DataFrame(results)
                csv_path = self.pilot_dir / 'metrics' / 'threshold_pilot_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"   ✅ Saved CSV: {csv_path}")
                
                # FIX: Save calibration candidates CSV with enhanced metrics
                if calibration_candidates:
                    candidates_df = pd.DataFrame(calibration_candidates)
                    candidates_csv_path = self.pilot_dir / 'metrics' / 'threshold_candidate_metrics.csv'
                    candidates_df.to_csv(candidates_csv_path, index=False)
                    saved_paths.append(str(candidates_csv_path))
                    logger.info(f"   ✅ Saved calibration candidates CSV: {candidates_csv_path}")
                
                # Verify completeness
                is_complete, missing_items = self._verify_completeness(results, calibration_candidates)
                n_success = sum(1 for r in results if r.get('status') == 'success')
                
                # Save JSON summary
                summary = {
                    'pilot_status': 'complete' if is_complete else 'incomplete',
                    'validation_complete': is_complete,
                    'seed': SEED,
                    'dataset_rows': len(self.original_dataset) if self.original_dataset is not None else 0,
                    'feature_count': self.feature_count,
                    'train_size': self.train_size,
                    'retrain_size': self.retrain_size,
                    'test_size': self.test_size,
                    'activation_pilot_val_size': self.activation_pilot_val_size,
                    'calibration_size': self.calibration_size,
                    'calibration_normal_count': self.calibration_normal_count,
                    'calibration_anomaly_count': self.calibration_anomaly_count,
                    'calibration_eval_size': self.calibration_eval_size,
                    'calibration_eval_normal_count': self.calibration_eval_normal_count,
                    'calibration_eval_anomaly_count': self.calibration_eval_anomaly_count,
                    'expected_record_count': EXPECTED_RECORD_COUNT,
                    'successful_record_count': n_success,
                    'expected_candidate_count': EXPECTED_CANDIDATE_COUNT,
                    'actual_candidate_count': len(calibration_candidates),
                    'expected_variants': ['historical_threshold', 'calibrated_threshold'],
                    'actual_variants': list(set(r.get('variant') for r in results if r.get('status') == 'success')),
                    'historical_threshold_percentile': HISTORICAL_THRESHOLD_PERCENTILE,
                    'candidate_percentiles': CANDIDATE_PERCENTILES,
                    'split_integrity_checks': self.split_integrity_checks,
                    'original_indices_stored': {k: (len(v) if v is not None else None) for k, v in self.original_indices.items()},
                    'threshold_policy': 'Historical: 95th percentile of normal TRAIN reconstruction errors. Calibrated: best F1 from candidate percentiles on normal CALIBRATION reconstruction errors.',
                    'training_config': {
                        'epochs': AE_EPOCHS,
                        'batch_size': AE_BATCH_SIZE,
                        'validation_split': AE_VAL_SPLIT,
                        'learning_rate': AE_LEARNING_RATE,
                        'loss': 'mse',
                        'optimizer': 'Adam',
                        'output_activation': 'linear',
                        'training_population': 'normal_train_samples_only'
                    },
                    'evaluation_population': 'held-out_calibration_evaluation_subset',
                    'historical_test_used': False,
                    'scientific_note': 'This is a DEVELOPMENTAL evaluation using held-out calibration data. The calibrated threshold is selected via supervised F1 optimization on calibration subset, then evaluated on separate calibration-evaluation subset. This is NOT an independent final test because overall Autoencoder development has involved earlier evaluations. Results should be interpreted as developmental evidence only.',
                    'experiment_scope': 'autoencoder_threshold_calibration_final_exploratory',
                    'results': results,
                    'calibration_candidates': calibration_candidates
                }
                
                json_path = self.pilot_dir / 'metrics' / 'threshold_pilot_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"   ✅ Saved JSON: {json_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def run_full_pilot(self) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING AUTOENCODER THRESHOLD CALIBRATION PILOT")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'pilot_complete': False
        }
        
        # Run pilot
        pilot_results, calibration_candidates = self.run_pilot()
        
        if not pilot_results:
            logger.error("❌ No pilot results generated")
            results['status'] = 'failed'
            return results
        
        # Verify completeness before saving
        is_complete, missing_items = self._verify_completeness(pilot_results, calibration_candidates)
        if not is_complete:
            logger.error("❌ Completeness check FAILED:")
            for item in missing_items:
                logger.error(f"   - {item}")
            results['status'] = 'incomplete'
        else:
            logger.info("✅ Completeness check PASSED: 2/2 variants succeeded with identical evaluation samples")
            logger.info(f"✅ All split-integrity checks passed: {self.split_integrity_checks.get('all_integrity_checks_passed', False)}")
        
        # Save results
        saved_paths = self.save_results(pilot_results, calibration_candidates)
        results['outputs_saved'] = saved_paths
        
        # Set final status
        if is_complete:
            self.pilot_status = 'complete'
            results['status'] = 'complete'
            results['pilot_complete'] = True
        else:
            self.pilot_status = 'incomplete'
            results['status'] = 'incomplete'
            results['pilot_complete'] = False
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'autoencoder_threshold_pilot')
    
    logger.info("\n" + "="*80)
    logger.info("🎯 AUTOENCODER THRESHOLD CALIBRATION PILOT")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    pilot_dir = base_dir / 'adaptive' / 'autoencoder_threshold_pilot'
    pilot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Pilot directory: {pilot_dir}")
    logger.info(f"⚠️  This is the FINAL exploratory Autoencoder experiment")
    logger.info(f"⚠️  Evaluates SUPERVISED threshold calibration (NOT unsupervised)")
    logger.info(f"⚠️  Historical test partition NOT USED")
    logger.info(f"⚠️  Does NOT modify validated four-model or other Autoencoder pilots")
    logger.info(f"⚠️  Results are DEVELOPMENTAL evidence only (not independent final test)")
    
    pilot = AutoencoderThresholdPilot(base_dir=base_dir, pilot_dir=pilot_dir)
    results = pilot.run_full_pilot()
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🎉 AUTOENCODER THRESHOLD CALIBRATION PILOT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()