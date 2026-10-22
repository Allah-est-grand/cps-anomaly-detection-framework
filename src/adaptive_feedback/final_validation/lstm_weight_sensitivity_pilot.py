"""
LSTM Weight Sensitivity Pilot - Anomaly Weight Selection Study
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module runs a controlled LSTM anomaly-weight sensitivity pilot to determine
the candidate anomaly weight for weighted retraining WITHOUT using the final test set.

This is a DEVELOPMENTAL pilot study, NOT a final evaluation.

KEY SCIENTIFIC REQUIREMENT:
---------------------------
- Do NOT select the best weight using the final test set
- Split the retraining subset internally into fitting and validation parts
- Use the internal validation part to compare weights
- Keep the final test set completely untouched during weight selection
- Report validation metrics for each weight
- Identify selected candidate weight based on validation F1
- Do NOT perform final test evaluation in this pilot

TESTED ANOMALY WEIGHTS:
-----------------------
[1.0, 1.5, 2.0, 2.5, 3.0]

Normal weight is fixed at 1.0 for all experiments.

OUTPUT FILES:
-------------
- results/adaptive/lstm_weight_sensitivity_pilot/metrics/weight_sensitivity_results.csv
- results/adaptive/lstm_weight_sensitivity_pilot/metrics/weight_sensitivity_summary.json
- results/adaptive/lstm_weight_sensitivity_pilot/reports/weight_sensitivity_report.txt

SCIENTIFIC NOTE:
----------------
Weight 1.0 represents train + retrain_fit with NO anomaly weighting.
This allows comparison of the effect of extra retraining data versus class weighting.
Selection criterion: highest internal-validation F1 → higher recall → lower anomaly weight (for ties).
This is a "selected candidate weight", NOT an "optimal weight".
Final test evaluation will be performed in a separate experiment after weight selection is complete.
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
import math
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

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

# TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("⚠️  TensorFlow not available. LSTM pilot cannot run.")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42  # Single seed for this pilot

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

# Internal validation split from retraining data (for weight selection)
INTERNAL_VAL_RATIO = 0.3  # 30% of retraining data for internal validation

# Anomaly weights to test
ANOMALY_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]
NORMAL_WEIGHT = 1.0  # Fixed

# LSTM architecture (preserved from multi_seed_validator_four_model.py)
TIME_STEPS = 10
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# Training configuration (preserved from multi_seed_validator_four_model.py)
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
PATIENCE = 10

# Threshold for binary predictions
PREDICTION_THRESHOLD = 0.5

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']

METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

# =============================================================================
# LSTM WEIGHT SENSITIVITY PILOT CLASS
# =============================================================================
class LSTMWeightSensitivityPilot:
    """
    Runs controlled LSTM anomaly-weight sensitivity pilot.
    
    Key scientific requirement:
    - Do NOT use final test set for weight selection
    - Split retraining data internally for validation
    - Use internal validation to compare weights
    - Keep final test set completely untouched
    """
    
    def __init__(self, base_dir: Path, pilot_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.pilot_dir = Path(pilot_dir) if pilot_dir else (
            self.base_dir / 'adaptive' / 'lstm_weight_sensitivity_pilot'
        )
        
        # Create output directories
        (self.pilot_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.pilot_dir / 'reports').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.weight_results = []
        self.selected_weight = None
        self.data_source_description = ''
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized LSTMWeightSensitivityPilot")
        logger.info(f"📁 Pilot directory: {self.pilot_dir}")
        logger.info(f"🔢 Seed: {SEED}")
        logger.info(f"⚖️  Anomaly weights to test: {ANOMALY_WEIGHTS}")
        logger.info(f"📊 Normal weight (fixed): {NORMAL_WEIGHT}")
    
    def _compute_dataset_fingerprint(self) -> str:
        """Compute streaming SHA-256 hash of dataset file."""
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
            'seed': SEED,
            'dataset_path': str(self.dataset_path),
            'dataset_fingerprint': self.dataset_fingerprint,
            'train_ratio': TRAIN_RATIO,
            'retrain_ratio': RETRAIN_RATIO,
            'test_ratio': TEST_RATIO,
            'internal_val_ratio': INTERNAL_VAL_RATIO,
            'anomaly_weights': ANOMALY_WEIGHTS,
            'normal_weight': NORMAL_WEIGHT,
            'time_steps': TIME_STEPS,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dense_units': DENSE_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'lstm_epochs': LSTM_EPOCHS,
            'lstm_batch_size': LSTM_BATCH_SIZE,
            'patience': PATIENCE,
            'prediction_threshold': PREDICTION_THRESHOLD,
            'preprocessing': 'SimpleImputer_mean_StandardScaler',
            'split_method': 'stratified_train_test_split'
        }
        
        config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        fingerprint = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:32]
        
        return fingerprint
    
    def _create_sequences_efficient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input using memory-efficient numpy operations.
        Based on multi_seed_validator_four_model.py pattern.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            X_seq: Sequences (n_sequences, time_steps, n_features)
            y_seq: Labels (n_sequences,) - aligned with sequences
        """
        n_samples = len(X)
        
        if n_samples < TIME_STEPS:
            logger.warning(f"⚠️  Not enough samples ({n_samples}) for time_steps={TIME_STEPS}")
            return np.array([]).reshape(0, TIME_STEPS, X.shape[1]), np.array([])
        
        n_sequences = n_samples - TIME_STEPS + 1
        
        # Use numpy sliding_window_view for memory efficiency
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X_seq = sliding_window_view(X, window_shape=TIME_STEPS, axis=0)
            # Correct output orientation - move axis to get (n_sequences, TIME_STEPS, n_features)
            X_seq = np.moveaxis(X_seq, -1, 1)
            X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)
        except Exception:
            # Fallback to manual creation if sliding_window_view not available
            X_seq = np.zeros((n_sequences, TIME_STEPS, X.shape[1]), dtype=np.float32)
            for i in range(n_sequences):
                X_seq[i] = X[i:i + TIME_STEPS]
        
        # Align labels with sequences (label at end of each sequence)
        y_seq = y[TIME_STEPS - 1:].astype(np.float32)
        
        logger.debug(f"📊 Created {n_sequences} sequences from {n_samples} samples")
        logger.debug(f"📊 Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def load_resources(self) -> bool:
        """Load dataset BEFORE computing fingerprints."""
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
            
            # Compute fingerprints AFTER loading
            self.dataset_fingerprint = self._compute_dataset_fingerprint()
            self.config_fingerprint = self._get_config_hash()
            
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
    
    def _create_lstm_model(self, n_features: int) -> Any:
        """Create LSTM model with preserved architecture"""
        model = models.Sequential([
            layers.Input(shape=(TIME_STEPS, n_features)),
            layers.LSTM(LSTM_UNITS_1, return_sequences=True),
            layers.Dropout(DROPOUT_RATE),
            layers.LSTM(LSTM_UNITS_2),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(DENSE_UNITS, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sample_weights(self, labels: np.ndarray, anomaly_weight: float) -> np.ndarray:
        """Create anomaly-sensitive sample weights"""
        weights = np.ones(len(labels)) * NORMAL_WEIGHT
        weights[labels == 1] = anomaly_weight
        return weights
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate predictions and return metrics"""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Explicit np.isfinite checks for all metrics
        metrics_valid = (
            np.isfinite(accuracy) and 
            np.isfinite(precision) and 
            np.isfinite(recall) and 
            np.isfinite(f1)
        )
        
        if not metrics_valid:
            logger.warning(f"⚠️  Invalid metrics detected: accuracy={accuracy}, precision={precision}, recall={recall}, f1={f1}")
        
        return {
            'accuracy': float(accuracy) if np.isfinite(accuracy) else None,
            'precision': float(precision) if np.isfinite(precision) else None,
            'recall': float(recall) if np.isfinite(recall) else None,
            'f1_score': float(f1) if np.isfinite(f1) else None,
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'metrics_valid': metrics_valid
        }
    
    def run_weight_sensitivity_analysis(self) -> Dict:
        """
        Run complete weight sensitivity analysis.
        
        CRITICAL: Do NOT use final test set for weight selection.
        Use internal validation split from retraining data only.
        """
        logger.info("\n" + "="*80)
        logger.info("⚖️  RUNNING LSTM WEIGHT SENSITIVITY ANALYSIS")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'weight_results': [],
            'selected_weight': None,
            'outputs_saved': [],
            'pilot_complete': False
        }
        
        # Assert exactly 5 unique anomaly weights
        assert ANOMALY_WEIGHTS == [1.0, 1.5, 2.0, 2.5, 3.0], f"ANOMALY_WEIGHTS must be [1.0, 1.5, 2.0, 2.5, 3.0], got {ANOMALY_WEIGHTS}"
        assert len(set(ANOMALY_WEIGHTS)) == 5, f"Must have 5 unique weights, got {len(set(ANOMALY_WEIGHTS))}"
        logger.info(f"✅ Verified {len(ANOMALY_WEIGHTS)} unique anomaly weights: {ANOMALY_WEIGHTS}")
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("❌ TensorFlow not available. Cannot run LSTM pilot.")
            results['status'] = 'failed'
            results['error'] = 'TensorFlow not available'
            return results
        
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        try:
            set_random_seeds(SEED)
            
            # Identify label column
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                results['status'] = 'failed'
                return results
            
            # Convert labels
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            # Prepare features
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                results['status'] = 'failed'
                return results
            
            n_features = len(feature_cols)
            logger.info(f"📊 Number of features: {n_features}")
            
            # Split data (preserved from multi_seed_validator_four_model.py)
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_features, all_labels,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=SEED
            )
            
            X_train, X_retrain, y_train, y_retrain = train_test_split(
                X_temp, y_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=SEED
            )
            
            # Log split sizes for sanity check
            logger.info(f"📊 Split sizes:")
            logger.info(f"   Train: {len(X_train)} samples")
            logger.info(f"   Retrain: {len(X_retrain)} samples")
            logger.info(f"   Test: {len(X_test)} samples (NOT USED for weight selection)")
            
            # CRITICAL: Confirm test set is NOT used for weight selection
            logger.info(f"⚠️  SANITY CHECK: Test set ({len(X_test)} samples) is RESERVED for final evaluation")
            logger.info(f"⚠️  Weight selection will use ONLY internal validation from retraining data")
            
            # Preprocessing (fit on TRAIN ONLY - preserved from multi_seed_validator_four_model.py)
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            
            # CRITICAL: Split retraining data internally for weight selection
            # This ensures final test set is NOT used for model selection
            X_retrain_fit, X_retrain_val, y_retrain_fit, y_retrain_val = train_test_split(
                X_retrain_scaled, y_retrain,
                test_size=INTERNAL_VAL_RATIO,
                stratify=y_retrain,
                random_state=SEED
            )
            
            logger.info(f"📊 Internal validation split from retraining data:")
            logger.info(f"   Fitting part: {len(X_retrain_fit)} samples")
            logger.info(f"   Validation part: {len(X_retrain_val)} samples ({INTERNAL_VAL_RATIO*100:.0f}%)")
            logger.info(f"   Final test set: {len(X_test)} samples (UNTOUCHED)")
            
            # Create sequences separately for X_train and X_retrain_fit, then concatenate
            # Do NOT create sequences across the boundary between X_train and X_retrain_fit
            logger.info(f"📊 Creating sequences for training pool (X_train + X_retrain_fit)...")
            
            X_train_seq, y_train_seq = self._create_sequences_efficient(X_train_scaled, y_train)
            X_retrain_fit_seq, y_retrain_fit_seq = self._create_sequences_efficient(X_retrain_fit, y_retrain_fit)
            
            # Concatenate sequences (not raw features)
            if len(X_train_seq) > 0 and len(X_retrain_fit_seq) > 0:
                X_combined_seq = np.concatenate([X_train_seq, X_retrain_fit_seq], axis=0)
                y_combined_seq = np.concatenate([y_train_seq, y_retrain_fit_seq], axis=0)
            elif len(X_train_seq) > 0:
                X_combined_seq = X_train_seq
                y_combined_seq = y_train_seq
            else:
                X_combined_seq = X_retrain_fit_seq
                y_combined_seq = y_retrain_fit_seq
            
            del X_train_seq, y_train_seq, X_retrain_fit_seq, y_retrain_fit_seq
            gc.collect()
            
            # Create validation sequences (RESERVED for weight comparison, NOT for EarlyStopping)
            X_val_seq, y_val_seq = self._create_sequences_efficient(X_retrain_val, y_retrain_val)
            
            logger.info(f"📊 LSTM sequence shapes:")
            logger.info(f"   Combined training sequences: {X_combined_seq.shape}")
            logger.info(f"   Validation sequences (reserved): {X_val_seq.shape}")
            
            # Log class counts
            n_train_normal = int(np.sum(y_train == 0))
            n_train_anomaly = int(np.sum(y_train == 1))
            n_retrain_fit_normal = int(np.sum(y_retrain_fit == 0))
            n_retrain_fit_anomaly = int(np.sum(y_retrain_fit == 1))
            n_val_normal = int(np.sum(y_retrain_val == 0))
            n_val_anomaly = int(np.sum(y_retrain_val == 1))
            
            logger.info(f"📊 Class counts:")
            logger.info(f"   Train: {n_train_normal} normal, {n_train_anomaly} anomaly")
            logger.info(f"   Retrain fit: {n_retrain_fit_normal} normal, {n_retrain_fit_anomaly} anomaly")
            logger.info(f"   Validation: {n_val_normal} normal, {n_val_anomaly} anomaly")
            
            # Test each anomaly weight
            logger.info(f"\n⚖️  Testing {len(ANOMALY_WEIGHTS)} anomaly weights...")
            
            for anomaly_weight in ANOMALY_WEIGHTS:
                logger.info(f"\n   Testing anomaly weight = {anomaly_weight}...")
                
                # Correct order for reproducibility
                tf.keras.backend.clear_session()
                gc.collect()
                set_random_seeds(SEED)
                tf.keras.utils.set_random_seed(SEED)
                
                # Create sample weights for fitting
                sample_weights = self._create_sample_weights(y_combined_seq, anomaly_weight)
                
                # Create and train LSTM model
                model = self._create_lstm_model(n_features)
                
                # Do NOT use X_val_seq/y_val_seq for EarlyStopping
                # Use validation_split=0.1 from training data instead
                # X_val_seq/y_val_seq remain completely reserved for weight comparison
                history = model.fit(
                    X_combined_seq, y_combined_seq,
                    sample_weight=sample_weights,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    validation_split=0.1,
                    callbacks=[
                        callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=PATIENCE,
                            restore_best_weights=True
                        )
                    ],
                    verbose=0
                )
                
                # Evaluate on RESERVED validation set (NOT final test set)
                y_val_pred_proba = model.predict(X_val_seq, verbose=0).flatten()
                y_val_pred = (y_val_pred_proba > PREDICTION_THRESHOLD).astype(int)
                
                val_metrics = self._evaluate_predictions(y_val_seq, y_val_pred)
                
                # Fail candidate if any metric is NaN/inf
                if not val_metrics['metrics_valid']:
                    logger.warning(f"⚠️  Weight {anomaly_weight}: Invalid metrics detected - marking as failed")
                    weight_result = {
                        'anomaly_weight': anomaly_weight,
                        'normal_weight': NORMAL_WEIGHT,
                        'validation_accuracy': val_metrics['accuracy'],
                        'validation_precision': val_metrics['precision'],
                        'validation_recall': val_metrics['recall'],
                        'validation_f1_score': val_metrics['f1_score'],
                        'validation_false_positive': val_metrics['false_positive'],
                        'validation_false_negative': val_metrics['false_negative'],
                        'validation_true_positive': val_metrics['true_positive'],
                        'validation_true_negative': val_metrics['true_negative'],
                        'epochs_trained': len(history.history['loss']),
                        'final_test_used': False,
                        'metrics_valid': False,
                        'failure_reason': 'Invalid metrics (NaN or inf)'
                    }
                else:
                    weight_result = {
                        'anomaly_weight': anomaly_weight,
                        'normal_weight': NORMAL_WEIGHT,
                        'validation_accuracy': val_metrics['accuracy'],
                        'validation_precision': val_metrics['precision'],
                        'validation_recall': val_metrics['recall'],
                        'validation_f1_score': val_metrics['f1_score'],
                        'validation_false_positive': val_metrics['false_positive'],
                        'validation_false_negative': val_metrics['false_negative'],
                        'validation_true_positive': val_metrics['true_positive'],
                        'validation_true_negative': val_metrics['true_negative'],
                        'epochs_trained': len(history.history['loss']),
                        'final_test_used': False,
                        'metrics_valid': True,
                        'failure_reason': None
                    }
                
                self.weight_results.append(weight_result)
                
                if val_metrics['metrics_valid']:
                    logger.info(f"      Validation F1: {val_metrics['f1_score']:.4f}")
                    logger.info(f"      Validation Recall: {val_metrics['recall']:.4f}")
                    logger.info(f"      Validation FN: {val_metrics['false_negative']}")
                    logger.info(f"      Epochs trained: {len(history.history['loss'])}")
                else:
                    logger.info(f"      FAILED: {weight_result['failure_reason']}")
                
                # Cleanup
                del model, sample_weights
                tf.keras.backend.clear_session()
                gc.collect()
            
            # Select best weight based on RESERVED VALIDATION F1
            # Only consider candidates with valid metrics
            valid_results = [r for r in self.weight_results if r.get('metrics_valid', False)]
            
            if valid_results:
                # Selection criteria: highest F1 → higher recall → lower anomaly weight (for ties)
                best_result = max(
                    valid_results,
                    key=lambda x: (x['validation_f1_score'], x['validation_recall'], -x['anomaly_weight'])
                )
                self.selected_weight = best_result['anomaly_weight']
                
                logger.info(f"\n✅ Selected candidate weight: {self.selected_weight}")
                logger.info(f"   Based on validation F1: {best_result['validation_f1_score']:.4f}")
                logger.info(f"   Validation recall: {best_result['validation_recall']:.4f}")
                logger.info(f"   Final test set was NOT used for selection")
            else:
                logger.error("❌ No valid weight results for selection")
                results['status'] = 'failed'
                results['error'] = 'No valid weight results'
                return results
            
            results['weight_results'] = self.weight_results
            results['selected_weight'] = self.selected_weight
            results['status'] = 'complete'
            results['pilot_complete'] = True
            
            # Save split metadata
            results['split_metadata'] = {
                'train_samples': len(X_train),
                'retrain_samples': len(X_retrain),
                'test_samples': len(X_test),
                'retrain_fit_samples': len(X_retrain_fit),
                'retrain_val_samples': len(X_retrain_val),
                'train_normal': n_train_normal,
                'train_anomaly': n_train_anomaly,
                'retrain_fit_normal': n_retrain_fit_normal,
                'retrain_fit_anomaly': n_retrain_fit_anomaly,
                'val_normal': n_val_normal,
                'val_anomaly': n_val_anomaly,
                'combined_sequences': len(X_combined_seq),
                'val_sequences': len(X_val_seq),
                'time_steps': TIME_STEPS,
                'n_features': n_features
            }
            
        except Exception as e:
            logger.error(f"❌ Weight sensitivity analysis failed: {e}")
            logger.error(traceback.format_exc())
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def save_results(self, pilot_results: Dict) -> List[str]:
        """Save pilot results to CSV and JSON"""
        logger.info("💾 Saving pilot results...")
        saved_paths = []
        
        try:
            # Save CSV
            if self.weight_results:
                results_df = pd.DataFrame(self.weight_results)
                csv_path = self.pilot_dir / 'metrics' / 'weight_sensitivity_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"   ✅ Saved CSV: {csv_path}")
            
            # Save JSON summary
            if pilot_results:
                summary = {
                    'pilot_status': pilot_results.get('status', 'unknown'),
                    'seed': SEED,
                    'tested_anomaly_weights': ANOMALY_WEIGHTS,
                    'normal_weight': NORMAL_WEIGHT,
                    'selected_weight': self.selected_weight,
                    'internal_val_ratio': INTERNAL_VAL_RATIO,
                    'final_test_preserved': True,
                    'final_test_samples': pilot_results.get('split_metadata', {}).get('test_samples', None),
                    'weight_results': self.weight_results,
                    'split_metadata': pilot_results.get('split_metadata', {}),
                    'dataset_fingerprint': self.dataset_fingerprint,
                    'config_fingerprint': self.config_fingerprint,
                    'scientific_note': 'Weight selection based on internal validation only. Final test set preserved for independent evaluation. Weight 1.0 represents train + retrain_fit with NO anomaly weighting.'
                }
                
                json_path = self.pilot_dir / 'metrics' / 'weight_sensitivity_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"   ✅ Saved JSON: {json_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def generate_report(self, pilot_results: Dict) -> str:
        """Generate comprehensive pilot report"""
        logger.info("📝 Generating pilot report...")
        
        report = []
        report.append("="*80)
        report.append("LSTM WEIGHT SENSITIVITY PILOT REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 1.0.0")
        report.append(f"Thesis Project: CPS Anomaly Detection Framework")
        report.append("")
        
        report.append("EXPERIMENTAL CONFIGURATION:")
        report.append(f"  - Seed: {SEED}")
        report.append(f"  - Tested anomaly weights: {ANOMALY_WEIGHTS}")
        report.append(f"  - Normal weight (fixed): {NORMAL_WEIGHT}")
        report.append(f"  - Internal validation ratio: {INTERNAL_VAL_RATIO}")
        report.append(f"  - LSTM time steps: {TIME_STEPS}")
        report.append(f"  - Prediction threshold: {PREDICTION_THRESHOLD}")
        report.append("")
        
        report.append("DATA SPLITS:")
        if 'split_metadata' in pilot_results:
            meta = pilot_results['split_metadata']
            report.append(f"  - Train: {meta['train_samples']} samples ({meta['train_normal']} normal, {meta['train_anomaly']} anomaly)")
            report.append(f"  - Retrain: {meta['retrain_samples']} samples")
            report.append(f"  - Retrain fit: {meta['retrain_fit_samples']} samples ({meta['retrain_fit_normal']} normal, {meta['retrain_fit_anomaly']} anomaly)")
            report.append(f"  - Retrain val: {meta['retrain_val_samples']} samples ({meta['val_normal']} normal, {meta['val_anomaly']} anomaly)")
            report.append(f"  - Test: {meta['test_samples']} samples (PRESERVED - NOT USED)")
            report.append(f"  - Combined sequences: {meta['combined_sequences']}")
            report.append(f"  - Validation sequences: {meta['val_sequences']}")
        report.append("")
        
        report.append("SCIENTIFIC SAFETY CHECKS:")
        report.append(f"  - Final test set used for weight selection: NO")
        report.append(f"  - Weight selection method: Internal validation from retraining data")
        report.append(f"  - Internal validation ratio: {INTERNAL_VAL_RATIO*100:.0f}% of retraining data")
        report.append(f"  - Final test preservation: CONFIRMED")
        report.append(f"  - Preprocessing fitted on train only: YES")
        report.append(f"  - Training and validation indices disjoint: YES")
        report.append("")
        
        report.append("WEIGHT SENSITIVITY RESULTS:")
        for result in self.weight_results:
            report.append(f"  - Anomaly weight {result['anomaly_weight']}:")
            if result.get('metrics_valid', False):
                report.append(f"      Validation F1: {result['validation_f1_score']:.4f}")
                report.append(f"      Validation Recall: {result['validation_recall']:.4f}")
                report.append(f"      Validation FN: {result['validation_false_negative']}")
                report.append(f"      Validation FP: {result['validation_false_positive']}")
                report.append(f"      Epochs trained: {result['epochs_trained']}")
            else:
                report.append(f"      FAILED: {result.get('failure_reason', 'Unknown')}")
            report.append(f"      Final test used: {result['final_test_used']}")
        report.append("")
        
        report.append("SELECTED CANDIDATE WEIGHT:")
        report.append(f"  - Selected anomaly weight: {self.selected_weight}")
        if self.weight_results:
            valid_results = [r for r in self.weight_results if r.get('metrics_valid', False)]
            if valid_results:
                best_result = max(
                    valid_results,
                    key=lambda x: (x['validation_f1_score'], x['validation_recall'], -x['anomaly_weight'])
                )
                report.append(f"  - Based on validation F1: {best_result['validation_f1_score']:.4f}")
                report.append(f"  - Validation recall: {best_result['validation_recall']:.4f}")
                report.append(f"  - Validation FN: {best_result['validation_false_negative']}")
                report.append(f"  - Selection criteria: Highest F1 → Higher recall → Lower weight (for ties)")
        report.append("")
        
        report.append("NEXT STEPS:")
        report.append(f"  - Use selected weight ({self.selected_weight}) for final LSTM evaluation")
        report.append(f"  - Perform final test evaluation in separate experiment")
        report.append(f"  - Compare with other models using same weight selection protocol")
        report.append("")
        
        report.append("SCIENTIFIC NOTES:")
        report.append(f"  - This pilot determines candidate anomaly weight using internal validation only")
        report.append(f"  - Final test set is completely preserved for independent evaluation")
        report.append(f"  - No data leakage from test set into weight selection")
        report.append(f"  - Weight selection based on validation F1 (balanced metric)")
        report.append(f"  - Weight 1.0 represents train + retrain_fit with NO anomaly weighting")
        report.append(f"  - Results are developmental and require final test confirmation")
        report.append(f"  - This is a 'selected candidate weight', NOT an 'optimal weight'")
        report.append("")
        
        report.append("="*80)
        report.append("END OF LSTM WEIGHT SENSITIVITY PILOT REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        report_path = self.pilot_dir / 'reports' / 'weight_sensitivity_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return report_text
    
    def run_full_pilot(self) -> Dict:
        """Run complete weight sensitivity pilot"""
        pilot_results = self.run_weight_sensitivity_analysis()
        
        if pilot_results.get('status') == 'complete':
            saved_paths = self.save_results(pilot_results)
            pilot_results['outputs_saved'] = saved_paths
            
            self.generate_report(pilot_results)
            pilot_results['outputs_saved'].append(
                str(self.pilot_dir / 'reports' / 'weight_sensitivity_report.txt')
            )
        
        return pilot_results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute LSTM weight sensitivity pilot"""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'lstm_weight_sensitivity_pilot')
    
    logger.info("\n" + "="*80)
    logger.info("⚖️  LSTM WEIGHT SENSITIVITY PILOT")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    pilot_dir = base_dir / 'adaptive' / 'lstm_weight_sensitivity_pilot'
    pilot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Pilot directory: {pilot_dir}")
    logger.info(f"⚠️  This is a DEVELOPMENTAL pilot study")
    logger.info(f"⚠️  Final test set is PRESERVED (not used for weight selection)")
    logger.info(f"⚠️  Weight selection based on internal validation only")
    
    pilot = LSTMWeightSensitivityPilot(base_dir=base_dir, pilot_dir=pilot_dir)
    results = pilot.run_full_pilot()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 LSTM WEIGHT SENSITIVITY PILOT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results.get('status', 'unknown')}")
    logger.info(f"⚖️  Selected candidate weight: {pilot.selected_weight}")
    logger.info(f"💾 Outputs: {len(results.get('outputs_saved', []))} files")
    
    for output in results.get('outputs_saved', []):
        logger.info(f"   - {output}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()