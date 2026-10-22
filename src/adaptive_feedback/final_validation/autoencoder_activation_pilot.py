"""
Autoencoder Activation Pilot - Output Activation Isolation Study
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements a controlled single-seed development experiment to investigate
whether the sigmoid output activation limits Autoencoder reconstruction performance
when inputs are standardized using StandardScaler.

Compares two otherwise identical architectures:
1. Historical Sigmoid-Output Autoencoder
2. Experimental Linear-Output Autoencoder

Change ONLY the final decoder activation.

IMPORTANT SCIENTIFIC NOTE:
--------------------------
StandardScaler produces feature values with mean=0, std=1, which can extend
significantly beyond the [0, 1] range that sigmoid activation can reconstruct.
This experiment isolates whether linear output activation improves reconstruction
capability for standardized inputs.

This is a controlled single-seed development experiment, NOT final independent
test evidence. Historical test partition is NOT used.

EXPERIMENTAL DESIGN:
--------------------
Seed: 42 (single-seed development only)

Compares:
1. Sigmoid-Output Autoencoder (historical architecture)
   vs
2. Linear-Output Autoencoder (experimental architecture)

OUTPUT FILES:
-------------
- results/adaptive/autoencoder_activation_pilot/metrics/activation_pilot_results.csv
- results/adaptive/autoencoder_activation_pilot/metrics/activation_pilot_summary.json
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
SEED = 42  # Single-seed development experiment only

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

# Additional split within RETRAIN for development validation
DEV_VAL_RATIO_WITHIN_RETRAIN = 0.3  # 30% of RETRAIN for development validation

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']

METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

# Autoencoder training configuration (from src/models.py)
AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_VAL_SPLIT = 0.1
AE_LEARNING_RATE = 0.001

# Prediction batch size for memory safety
PREDICTION_BATCH_SIZE = 2048

# Threshold percentile
THRESHOLD_PERCENTILE = 95

EXPECTED_RECORD_COUNT = 2  # (seed=42, sigmoid_normal_only) + (seed=42, linear_normal_only)
EXPECTED_VARIANTS = ['sigmoid_normal_only', 'linear_normal_only']

# =============================================================================
# AUTOENCODER ACTIVATION PILOT CLASS
# =============================================================================
class AutoencoderActivationPilot:
    """
    Implements controlled single-seed activation isolation study for Autoencoder.
    """
    
    def __init__(self, base_dir: Path, pilot_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.pilot_dir = Path(pilot_dir) if pilot_dir else (
            self.base_dir / 'adaptive' / 'autoencoder_activation_pilot'
        )
        
        (self.pilot_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.results = []
        self.pilot_status = 'pending'
        self.data_source_description = ''
        
        # Initialize actual value storage attributes
        self.feature_count = 0
        self.train_size = 0
        self.retrain_size = 0
        self.test_size = 0
        self.dev_val_size = 0
        self.dev_val_normal_count = 0
        self.dev_val_anomaly_count = 0
        
        # FIX #1: Initialize scaling diagnosis storage
        self.scaling_diagnosis = {}
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized AutoencoderActivationPilot")
        logger.info(f"📁 Pilot directory: {self.pilot_dir}")
        logger.info(f"🔢 Seed: {SEED}")
        logger.info(f"📈 Expected records: {EXPECTED_RECORD_COUNT}")
    
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
    
    def _diagnose_input_scaling(self, X_scaled: np.ndarray) -> Dict:
        """
        Diagnose input scaling after StandardScaler preprocessing.
        Documents the representational mismatch being investigated.
        """
        total_values = X_scaled.size
        below_zero = np.sum(X_scaled < 0)
        above_one = np.sum(X_scaled > 1)
        within_range = np.sum((X_scaled >= 0) & (X_scaled <= 1))
        
        pct_below_zero = (below_zero / total_values) * 100
        pct_above_one = (above_one / total_values) * 100
        pct_within_range = (within_range / total_values) * 100
        
        # Per-feature min/max (summarized)
        feature_mins = X_scaled.min(axis=0)
        feature_maxs = X_scaled.max(axis=0)
        
        logger.info(f"📊 Input Scaling Diagnosis:")
        logger.info(f"   Total values: {total_values}")
        logger.info(f"   Below 0: {below_zero} ({pct_below_zero:.2f}%)")
        logger.info(f"   Above 1: {above_one} ({pct_above_one:.2f}%)")
        logger.info(f"   Within [0,1]: {within_range} ({pct_within_range:.2f}%)")
        logger.info(f"   Per-feature min range: [{feature_mins.min():.3f}, {feature_mins.max():.3f}]")
        logger.info(f"   Per-feature max range: [{feature_maxs.min():.3f}, {feature_maxs.max():.3f}]")
        
        # FIX #5: Verify percentages sum to approximately 100%
        pct_sum = pct_below_zero + pct_above_one + pct_within_range
        if abs(pct_sum - 100.0) > 0.1:
            logger.warning(f"⚠️  Scaling percentages sum to {pct_sum:.2f}% (expected ~100%)")
        
        return {
            'total_values': int(total_values),
            'below_zero': int(below_zero),
            'above_one': int(above_one),
            'within_range': int(within_range),
            'pct_below_zero': round(pct_below_zero, 2),
            'pct_above_one': round(pct_above_one, 2),
            'pct_within_range': round(pct_within_range, 2),
            'pct_sum': round(pct_sum, 2),
            'feature_min_range': [round(float(feature_mins.min()), 3), round(float(feature_mins.max()), 3)],
            'feature_max_range': [round(float(feature_maxs.min()), 3), round(float(feature_maxs.max()), 3)]
        }
    
    def _create_autoencoder_model(self, n_features: int, output_activation: str = 'sigmoid') -> Any:
        """
        Create Autoencoder model with specified output activation.
        
        Args:
            n_features: Number of input/output features
            output_activation: 'sigmoid' or 'linear'
        
        Returns:
            Compiled Keras model
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
        decoded = layers.Dense(n_features, activation=output_activation)(x)
        
        # Autoencoder
        autoencoder = models.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
            loss='mse'
        )
        
        logger.debug(f"📊 Autoencoder model created ({output_activation} output): {autoencoder.count_params()} parameters")
        return autoencoder
    
    def _compute_reconstruction_errors_batched(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction MSE in batches without storing full reconstruction matrix.
        
        Args:
            model: Trained Autoencoder
            X: Input features
        
        Returns:
            1-D array of per-sample reconstruction MSE
        """
        all_mse = []
        
        for i in range(0, len(X), PREDICTION_BATCH_SIZE):
            batch = X[i:i+PREDICTION_BATCH_SIZE]
            recon_batch = model.predict(batch, verbose=0)
            
            # Compute MSE for this batch only
            batch_mse = np.mean(np.power(batch - recon_batch, 2), axis=1)
            all_mse.append(batch_mse)
            
            # Clean up batch reconstruction immediately
            del recon_batch, batch
        
        # Concatenate only the 1-D MSE arrays
        reconstruction_errors = np.concatenate(all_mse)
        del all_mse
        
        return reconstruction_errors
    
    def _compute_threshold_from_normal(self, reconstruction_errors: np.ndarray) -> float:
        """
        Compute anomaly threshold from specified percentile of normal reconstruction errors.
        
        Args:
            reconstruction_errors: 1-D array of reconstruction MSE from normal samples
        
        Returns:
            Threshold value (95th percentile by default)
        """
        if len(reconstruction_errors) == 0:
            logger.warning("⚠️  No normal samples available for threshold computation")
            return 0.0
        
        threshold = np.percentile(reconstruction_errors, THRESHOLD_PERCENTILE)
        logger.info(f"📊 Threshold computed from {len(reconstruction_errors)} normal samples ({THRESHOLD_PERCENTILE}th percentile): {threshold:.6f}")
        
        return float(threshold)
    
    def _evaluate_autoencoder(self, model: Any, X_val: np.ndarray, y_val: np.ndarray, 
                             threshold: float, variant: str) -> Dict:
        """
        Evaluate Autoencoder on development validation partition.
        
        Args:
            model: Trained Autoencoder
            X_val: Validation features
            y_val: Validation labels
            threshold: Reconstruction error threshold
            variant: Model variant name
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Compute reconstruction errors memory-safely
        reconstruction_errors = self._compute_reconstruction_errors_batched(model, X_val)
        
        # Binary predictions
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Compute metrics
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Summary reconstruction-error statistics by class
        normal_errors = reconstruction_errors[y_val == 0]
        anomaly_errors = reconstruction_errors[y_val == 1]
        
        # Cleanup
        del reconstruction_errors, y_pred
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive_rate': round(fpr, 4),
            'normal_error_mean': round(float(normal_errors.mean()), 6) if len(normal_errors) > 0 else None,
            'normal_error_std': round(float(normal_errors.std()), 6) if len(normal_errors) > 0 else None,
            'anomaly_error_mean': round(float(anomaly_errors.mean()), 6) if len(anomaly_errors) > 0 else None,
            'anomaly_error_std': round(float(anomaly_errors.std()), 6) if len(anomaly_errors) > 0 else None
        }
    
    def run_pilot(self) -> List[Dict]:
        """
        Run complete Autoencoder activation pilot with two variants.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING AUTOENCODER ACTIVATION PILOT (Seed={SEED})")
        logger.info(f"{'='*80}")
        
        results = []
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("❌ TensorFlow not available. Cannot run Autoencoder pilot.")
            return results
        
        try:
            set_random_seeds(SEED)
            
            # Load dataset
            if not self.load_resources():
                return results
            
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
            
            # Store actual feature count
            self.feature_count = len(feature_cols)
            n_features = all_features.shape[1]
            logger.info(f"📊 Features: {all_features.shape[0]} samples, {n_features} features")
            
            # Two-stage stratified split (same as validated four-model experiment)
            # Stage 1: 80% temporary (train+retrain), 20% historical test (NOT USED)
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_features, all_labels,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=SEED
            )
            
            # Stage 2: Of the 80% temporary: 75% -> train (60% overall), 25% -> retrain (20% overall)
            X_train, X_retrain, y_train, y_retrain = train_test_split(
                X_temp, y_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=SEED
            )
            
            # Store actual split sizes
            self.train_size = len(X_train)
            self.retrain_size = len(X_retrain)
            self.test_size = len(X_test)
            
            logger.info(f"📊 Split sizes: Train={self.train_size}, Retrain={self.retrain_size}, Historical Test={self.test_size}")
            logger.info(f"⚠️  Historical test partition NOT USED in this activation study")
            
            # Additional split within RETRAIN for development validation
            X_retrain_reserved, X_dev_val, y_retrain_reserved, y_dev_val = train_test_split(
                X_retrain, y_retrain,
                test_size=DEV_VAL_RATIO_WITHIN_RETRAIN,
                stratify=y_retrain,
                random_state=SEED
            )
            
            self.dev_val_size = len(X_dev_val)
            self.dev_val_normal_count = int(np.sum(y_dev_val == 0))
            self.dev_val_anomaly_count = int(np.sum(y_dev_val == 1))
            
            logger.info(f"📊 Development validation partition: {self.dev_val_size} samples")
            logger.info(f"📊   Normal: {self.dev_val_normal_count}, Anomaly: {self.dev_val_anomaly_count}")
            
            if self.dev_val_normal_count == 0 or self.dev_val_anomaly_count == 0:
                logger.error("❌ Development validation partition must contain both normal and anomaly samples")
                return results
            
            # Preprocessing (fit on TRAIN ONLY)
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_dev_val_imp = imputer.transform(X_dev_val)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_dev_val_scaled = scaler.transform(X_dev_val_imp)
            
            # Convert to float32 for TensorFlow memory efficiency
            X_train_scaled = X_train_scaled.astype(np.float32)
            X_dev_val_scaled = X_dev_val_scaled.astype(np.float32)
            
            logger.info(f"✅ Preprocessing fitted on train set only (no leakage)")
            
            # FIX #2: Diagnose input scaling and store in instance variable
            self.scaling_diagnosis = self._diagnose_input_scaling(X_train_scaled)
            
            # Extract NORMAL training samples only
            X_train_normal = X_train_scaled[y_train == 0]
            n_train_normal = len(X_train_normal)
            logger.info(f"📊 Training on {n_train_normal} normal samples only")
            
            # =====================================================================
            # VARIANT 1: SIGMOID-OUTPUT AUTOENCODER (HISTORICAL)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 VARIANT 1: SIGMOID-OUTPUT AUTOENCODER")
            logger.info(f"{'='*80}")
            
            sigmoid_result = {
                'seed': SEED,
                'variant': 'sigmoid_normal_only',
                'output_activation': 'sigmoid',
                'evaluation_n': int(self.dev_val_size),
                'threshold': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive': 0,
                'false_negative': 0,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive_rate': 0.0,
                'epochs_trained': 0,
                'training_samples': n_train_normal,
                'validation_samples': self.dev_val_size,
                'normal_validation_samples': self.dev_val_normal_count,
                'anomaly_validation_samples': self.dev_val_anomaly_count,
                'normal_error_mean': None,
                'normal_error_std': None,
                'anomaly_error_mean': None,
                'anomaly_error_std': None,
                'execution_time_sec': 0.0,
                'status': 'failed'
            }
            
            try:
                variant_start_time = time.time()
                
                # Reset TensorFlow state and seed
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Create FRESH Autoencoder with sigmoid output
                model = self._create_autoencoder_model(n_features, output_activation='sigmoid')
                
                # Train on normal samples only
                history = model.fit(
                    X_train_normal, X_train_normal,
                    epochs=AE_EPOCHS,
                    batch_size=AE_BATCH_SIZE,
                    validation_split=AE_VAL_SPLIT,
                    verbose=0
                )
                
                epochs_trained = len(history.history['loss'])
                sigmoid_result['epochs_trained'] = epochs_trained
                logger.info(f"✅ Sigmoid Autoencoder trained for {epochs_trained} epochs")
                
                # Compute threshold from NORMAL training samples only
                train_normal_errors = self._compute_reconstruction_errors_batched(model, X_train_normal)
                threshold = self._compute_threshold_from_normal(train_normal_errors)
                sigmoid_result['threshold'] = threshold
                del train_normal_errors
                
                # Evaluate on development validation partition
                metrics = self._evaluate_autoencoder(model, X_dev_val_scaled, y_dev_val, threshold, 'sigmoid_normal_only')
                sigmoid_result.update(metrics)
                sigmoid_result['status'] = 'success'
                
                variant_end_time = time.time()
                sigmoid_result['execution_time_sec'] = round(variant_end_time - variant_start_time, 2)
                
                logger.info(f"📊 Sigmoid: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                
                # Cleanup
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Sigmoid Autoencoder failed: {e}")
                logger.error(traceback.format_exc())
                sigmoid_result['status'] = 'failed'
            
            results.append(sigmoid_result)
            
            # =====================================================================
            # VARIANT 2: LINEAR-OUTPUT AUTOENCODER (EXPERIMENTAL)
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 VARIANT 2: LINEAR-OUTPUT AUTOENCODER")
            logger.info(f"{'='*80}")
            
            linear_result = {
                'seed': SEED,
                'variant': 'linear_normal_only',
                'output_activation': 'linear',
                'evaluation_n': int(self.dev_val_size),
                'threshold': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive': 0,
                'false_negative': 0,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive_rate': 0.0,
                'epochs_trained': 0,
                'training_samples': n_train_normal,
                'validation_samples': self.dev_val_size,
                'normal_validation_samples': self.dev_val_normal_count,
                'anomaly_validation_samples': self.dev_val_anomaly_count,
                'normal_error_mean': None,
                'normal_error_std': None,
                'anomaly_error_mean': None,
                'anomaly_error_std': None,
                'execution_time_sec': 0.0,
                'status': 'failed'
            }
            
            try:
                variant_start_time = time.time()
                
                # Reset TensorFlow state and seed
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Create FRESH Autoencoder with linear output
                model = self._create_autoencoder_model(n_features, output_activation='linear')
                
                # Train on normal samples only
                history = model.fit(
                    X_train_normal, X_train_normal,
                    epochs=AE_EPOCHS,
                    batch_size=AE_BATCH_SIZE,
                    validation_split=AE_VAL_SPLIT,
                    verbose=0
                )
                
                epochs_trained = len(history.history['loss'])
                linear_result['epochs_trained'] = epochs_trained
                logger.info(f"✅ Linear Autoencoder trained for {epochs_trained} epochs")
                
                # Compute threshold from NORMAL training samples only
                train_normal_errors = self._compute_reconstruction_errors_batched(model, X_train_normal)
                threshold = self._compute_threshold_from_normal(train_normal_errors)
                linear_result['threshold'] = threshold
                del train_normal_errors
                
                # Evaluate on SAME development validation partition
                metrics = self._evaluate_autoencoder(model, X_dev_val_scaled, y_dev_val, threshold, 'linear_normal_only')
                linear_result.update(metrics)
                linear_result['status'] = 'success'
                
                variant_end_time = time.time()
                linear_result['execution_time_sec'] = round(variant_end_time - variant_start_time, 2)
                
                logger.info(f"📊 Linear: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                
                # Cleanup
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Linear Autoencoder failed: {e}")
                logger.error(traceback.format_exc())
                linear_result['status'] = 'failed'
            
            results.append(linear_result)
            
        except Exception as e:
            logger.error(f"❌ Pilot failed: {e}")
            logger.error(traceback.format_exc())
        
        return results
    
    def _verify_completeness(self, results: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Verify completeness with exactly 2 expected variants.
        """
        missing_items = []
        
        if not results:
            missing_items.append("No results generated")
            return False, missing_items
        
        # Check we have exactly 2 records
        if len(results) != EXPECTED_RECORD_COUNT:
            missing_items.append(f"Expected {EXPECTED_RECORD_COUNT} records, got {len(results)}")
        
        # Extract unique (seed, variant) keys from successful results
        successful = [r for r in results if r.get('status') == 'success']
        actual_unique_variants = set()
        for r in successful:
            key = (r.get('seed'), r.get('variant'))
            actual_unique_variants.add(key)
        
        # Define expected keys
        expected_keys = {(SEED, 'sigmoid_normal_only'), (SEED, 'linear_normal_only')}
        
        # Check for missing expected keys
        missing_keys = expected_keys - actual_unique_variants
        if missing_keys:
            missing_items.append(f"Missing expected variants: {missing_keys}")
        
        # Check for unexpected keys
        unexpected_keys = actual_unique_variants - expected_keys
        if unexpected_keys:
            missing_items.append(f"Unexpected variants: {unexpected_keys}")
        
        # Check both expected variants are successful
        for expected_variant in EXPECTED_VARIANTS:
            variant_success = any(r.get('status') == 'success' and r.get('variant') == expected_variant for r in results)
            if not variant_success:
                missing_items.append(f"{expected_variant} variant did not succeed")
        
        # FIX #4 & #6: Verify scaling diagnosis is present and valid
        if not self.scaling_diagnosis:
            missing_items.append("Scaling diagnosis is missing or empty")
        else:
            # Check required fields
            required_fields = ['pct_below_zero', 'pct_above_one', 'pct_within_range', 
                             'feature_min_range', 'feature_max_range']
            for field in required_fields:
                if field not in self.scaling_diagnosis:
                    missing_items.append(f"Scaling diagnosis missing required field: {field}")
            
            # FIX #5: Verify percentages sum to approximately 100%
            if 'pct_sum' in self.scaling_diagnosis:
                pct_sum = self.scaling_diagnosis['pct_sum']
                if abs(pct_sum - 100.0) > 1.0:
                    missing_items.append(f"Scaling percentages sum to {pct_sum:.2f}% (expected ~100%)")
        
        is_complete = (len(missing_items) == 0 and len(actual_unique_variants) == EXPECTED_RECORD_COUNT)
        
        return is_complete, missing_items
    
    def save_results(self, results: List[Dict]) -> List[str]:
        logger.info("💾 Saving pilot results...")
        saved_paths = []
        
        try:
            if results:
                # Save CSV
                results_df = pd.DataFrame(results)
                csv_path = self.pilot_dir / 'metrics' / 'activation_pilot_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"   ✅ Saved CSV: {csv_path}")
                
                # Verify completeness
                is_complete, missing_items = self._verify_completeness(results)
                n_success = sum(1 for r in results if r.get('status') == 'success')
                
                # Extract actual unique variants
                actual_unique_variants = set((r.get('seed'), r.get('variant')) for r in results if r.get('status') == 'success')
                
                # FIX #3 & #6: Save JSON summary with ACTUAL scaling diagnosis
                summary = {
                    'pilot_status': 'complete' if is_complete else 'incomplete',
                    'validation_complete': is_complete,
                    'seed': SEED,
                    'dataset_rows': len(self.original_dataset) if self.original_dataset is not None else 0,
                    'feature_count': self.feature_count,
                    'train_size': self.train_size,
                    'retrain_size': self.retrain_size,
                    'test_size': self.test_size,
                    'dev_val_size': self.dev_val_size,
                    'dev_val_normal_count': self.dev_val_normal_count,
                    'dev_val_anomaly_count': self.dev_val_anomaly_count,
                    'expected_record_count': EXPECTED_RECORD_COUNT,
                    'successful_record_count': n_success,
                    'expected_variants': EXPECTED_VARIANTS,
                    'actual_unique_variants': list(actual_unique_variants),
                    # FIX #3: Use actual scaling diagnosis instead of empty dict
                    'scaling_diagnosis': self.scaling_diagnosis,
                    'architecture': {
                        'encoder': ['Dense(64, relu)', 'Dropout(0.2)', 'Dense(32, relu)', 'Dropout(0.2)', 'Dense(16, relu)'],
                        'decoder': ['Dense(32, relu)', 'Dropout(0.2)', 'Dense(64, relu)', 'Dropout(0.2)', 'Dense(n_features, activation)'],
                        'output_activations_tested': ['sigmoid', 'linear']
                    },
                    'training_config': {
                        'epochs': AE_EPOCHS,
                        'batch_size': AE_BATCH_SIZE,
                        'validation_split': AE_VAL_SPLIT,
                        'learning_rate': AE_LEARNING_RATE,
                        'loss': 'mse',
                        'optimizer': 'Adam',
                        'training_population': 'normal_train_samples_only'
                    },
                    'threshold_policy': f'{THRESHOLD_PERCENTILE}th percentile of reconstruction error on normal training samples only',
                    'evaluation_population': 'development_validation_partition_from_retrain',
                    'historical_test_used': False,
                    'results': results,
                    'scientific_note': 'This experiment isolates the effect of final output activation (sigmoid vs linear) on Autoencoder reconstruction performance with StandardScaler-preprocessed inputs. StandardScaler produces values beyond [0,1] range, which may limit sigmoid reconstruction capability.',
                    'experiment_scope': 'autoencoder_activation_isolation_single_seed_development'
                }
                
                json_path = self.pilot_dir / 'metrics' / 'activation_pilot_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"   ✅ Saved JSON: {json_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def run_full_pilot(self) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING AUTOENCODER ACTIVATION PILOT")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'pilot_complete': False
        }
        
        # Run pilot
        pilot_results = self.run_pilot()
        
        if not pilot_results:
            logger.error("❌ No pilot results generated")
            results['status'] = 'failed'
            return results
        
        # FIX #3: Verify completeness before saving (includes scaling diagnosis check)
        is_complete, missing_items = self._verify_completeness(pilot_results)
        if not is_complete:
            logger.error("❌ Completeness check FAILED:")
            for item in missing_items:
                logger.error(f"   - {item}")
            results['status'] = 'incomplete'
        else:
            logger.info("✅ Completeness check PASSED: 2/2 variants succeeded with correct keys")
            # Verify scaling diagnosis is present
            if self.scaling_diagnosis:
                logger.info(f"✅ Scaling diagnosis verified: {self.scaling_diagnosis.get('pct_sum', 0):.2f}% total")
        
        # Save results
        saved_paths = self.save_results(pilot_results)
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
    logger = setup_logging(config.output.logs_dir, 'autoencoder_activation_pilot')
    
    logger.info("\n" + "="*80)
    logger.info("🎯 AUTOENCODER ACTIVATION PILOT")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    pilot_dir = base_dir / 'adaptive' / 'autoencoder_activation_pilot'
    pilot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Pilot directory: {pilot_dir}")
    logger.info(f"⚠️  This is a SINGLE-SEED development experiment (Seed={SEED})")
    logger.info(f"⚠️  Isolates output activation effect ONLY (sigmoid vs linear)")
    logger.info(f"⚠️  Historical test partition NOT USED")
    logger.info(f"⚠️  Does NOT modify validated four-model or other Autoencoder pilots")
    
    pilot = AutoencoderActivationPilot(base_dir=base_dir, pilot_dir=pilot_dir)
    results = pilot.run_full_pilot()
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🎉 AUTOENCODER ACTIVATION PILOT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()