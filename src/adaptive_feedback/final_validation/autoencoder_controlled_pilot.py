"""
Autoencoder Normal-Only Pilot - Single-Seed Feasibility Study
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements a ONE-SEED controlled feasibility pilot testing
three Autoencoder training protocols for reconstruction-based anomaly detection.

This pilot is NOT yet final multi-seed evidence.

It tests three Autoencoder training protocols:
1. Historical Baseline (train on ALL X_train samples)
2. Normal-Only Baseline (train on NORMAL X_train samples only)
3. Normal-Only Adaptive (train on NORMAL X_train + NORMAL X_retrain samples)

IMPORTANT SCIENTIFIC DISTINCTION:
---------------------------------
The normal-only Autoencoder variants are model-specific reconstruction-based
comparators. They are NOT equivalent to supervised anomaly-weighted retraining.

Unlike supervised classifiers, the Autoencoder is trained only on normal
reference samples so that anomalies are expected to produce larger
reconstruction errors. The adaptive variant expands the normal reference
population using normal samples from the retraining partition.

EXPERIMENTAL DESIGN:
--------------------
Seed: 42 (single-seed pilot only)

Compares:
1. Historical Baseline Autoencoder
   vs
2. Normal-Only Baseline Autoencoder
   vs
3. Normal-Only Adaptive Autoencoder

OUTPUT FILES:
-------------
- results/adaptive/autoencoder_normal_only_pilot/metrics/autoencoder_normal_only_pilot_results.csv
- results/adaptive/autoencoder_normal_only_pilot/metrics/autoencoder_normal_only_pilot_summary.json
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
SEED = 42  # Single-seed pilot only

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

# Autoencoder training configuration (from src/models.py)
AE_EPOCHS = 100
AE_BATCH_SIZE = 256
AE_VAL_SPLIT = 0.1
AE_LEARNING_RATE = 0.001

# Prediction batch size for memory safety
PREDICTION_BATCH_SIZE = 2048

EXPECTED_RECORD_COUNT = 3  # (seed=42, historical_baseline) + (seed=42, normal_only_baseline) + (seed=42, normal_only_adaptive)
EXPECTED_VARIANTS = ['historical_baseline', 'normal_only_baseline', 'normal_only_adaptive']

# =============================================================================
# AUTOENCODER NORMAL-ONLY PILOT CLASS
# =============================================================================
class AutoencoderNormalOnlyPilot:
    """
    Implements single-seed controlled feasibility pilot for Autoencoder
    with three normal-reference training protocols.
    """
    
    def __init__(self, base_dir: Path, pilot_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.pilot_dir = Path(pilot_dir) if pilot_dir else (
            self.base_dir / 'adaptive' / 'autoencoder_normal_only_pilot'
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
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized AutoencoderNormalOnlyPilot")
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
    
    def _create_autoencoder_model(self, n_features: int) -> Any:
        """
        Create Autoencoder model (exact architecture from src/models.py)
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
        decoded = layers.Dense(n_features, activation='sigmoid')(x)
        
        # Autoencoder
        autoencoder = models.Model(encoder_input, decoded)
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
            loss='mse'
        )
        
        logger.debug(f"📊 Autoencoder model created")
        return autoencoder
    
    def _compute_reconstruction_errors(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction MSE without storing full reconstruction matrix.
        Memory-safe batch-by-batch MSE computation.
        
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
            gc.collect()
        
        # Concatenate only the 1-D MSE arrays
        reconstruction_errors = np.concatenate(all_mse)
        del all_mse
        
        return reconstruction_errors
    
    def _compute_threshold_from_normal(self, reconstruction_errors: np.ndarray) -> float:
        """
        Compute anomaly threshold from 95th percentile of normal reconstruction errors.
        
        Args:
            reconstruction_errors: 1-D array of reconstruction MSE from normal samples
        
        Returns:
            Threshold value (95th percentile)
        """
        if len(reconstruction_errors) == 0:
            logger.warning("⚠️  No normal samples available for threshold computation")
            return 0.0
        
        threshold = np.percentile(reconstruction_errors, 95)
        logger.info(f"📊 Threshold computed from {len(reconstruction_errors)} normal samples: {threshold:.6f}")
        
        return float(threshold)
    
    def _evaluate_autoencoder(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                             threshold: float, variant: str) -> Dict:
        """
        Evaluate Autoencoder on test set.
        """
        # Compute reconstruction errors memory-safely
        reconstruction_errors = self._compute_reconstruction_errors(model, X_test)
        
        # Binary predictions
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Compute metrics
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Cleanup
        del reconstruction_errors, y_pred
        gc.collect()
        
        return {
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn)
        }
    
    def run_pilot(self) -> List[Dict]:
        """
        Run complete Autoencoder normal-only pilot with three variants.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING AUTOENCODER NORMAL-ONLY PILOT (Seed={SEED})")
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
            # Stage 1: 80% temporary (train+retrain), 20% independent test
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
            
            logger.info(f"📊 Split sizes: Train={self.train_size}, Retrain={self.retrain_size}, Test={self.test_size}")
            
            # Preprocessing (fit on TRAIN ONLY)
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            # Convert to float32 for TensorFlow memory efficiency
            X_train_scaled = X_train_scaled.astype(np.float32)
            X_retrain_scaled = X_retrain_scaled.astype(np.float32)
            X_test_scaled = X_test_scaled.astype(np.float32)
            
            logger.info(f"✅ Preprocessing fitted on train set only (no leakage)")
            
            # =====================================================================
            # VARIANT 1: HISTORICAL BASELINE
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 VARIANT 1: HISTORICAL BASELINE")
            logger.info(f"{'='*80}")
            
            historical_result = {
                'seed': SEED,
                'variant': 'historical_baseline',
                'evaluation_n': int(len(X_test_scaled)),
                'threshold': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive': 0,
                'false_negative': 0,
                'true_positive': 0,
                'true_negative': 0,
                'epochs_trained': 0,
                'training_samples': int(len(X_train_scaled)),
                'normal_training_samples': int(np.sum(y_train == 0)),
                'anomaly_training_samples': int(np.sum(y_train == 1)),
                'training_population_description': 'All training samples (normal + anomaly)',
                'status': 'failed'
            }
            
            try:
                # Reset TensorFlow state
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Create FRESH Autoencoder
                model = self._create_autoencoder_model(n_features)
                
                # Train on ALL X_train_scaled (no weights)
                history = model.fit(
                    X_train_scaled, X_train_scaled,
                    epochs=AE_EPOCHS,
                    batch_size=AE_BATCH_SIZE,
                    validation_split=AE_VAL_SPLIT,
                    verbose=0
                )
                
                epochs_trained = len(history.history['loss'])
                historical_result['epochs_trained'] = epochs_trained
                logger.info(f"✅ Historical Baseline Autoencoder trained for {epochs_trained} epochs")
                
                # Compute threshold from NORMAL training samples only
                X_train_normal = X_train_scaled[y_train == 0]
                normal_errors = self._compute_reconstruction_errors(model, X_train_normal)
                threshold = self._compute_threshold_from_normal(normal_errors)
                historical_result['threshold'] = threshold
                del normal_errors, X_train_normal
                gc.collect()
                
                # Evaluate on test set
                metrics = self._evaluate_autoencoder(model, X_test_scaled, y_test, threshold, 'historical_baseline')
                historical_result.update(metrics)
                historical_result['status'] = 'success'
                
                logger.info(f"📊 Historical Baseline: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                
                # Cleanup
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Historical Baseline Autoencoder failed: {e}")
                logger.error(traceback.format_exc())
                historical_result['status'] = 'failed'
            
            results.append(historical_result)
            
            # =====================================================================
            # VARIANT 2: NORMAL-ONLY BASELINE
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 VARIANT 2: NORMAL-ONLY BASELINE")
            logger.info(f"{'='*80}")
            
            normal_only_baseline_result = {
                'seed': SEED,
                'variant': 'normal_only_baseline',
                'evaluation_n': int(len(X_test_scaled)),
                'threshold': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive': 0,
                'false_negative': 0,
                'true_positive': 0,
                'true_negative': 0,
                'epochs_trained': 0,
                'training_samples': int(np.sum(y_train == 0)),
                'normal_training_samples': int(np.sum(y_train == 0)),
                'anomaly_training_samples': 0,
                'training_population_description': 'Normal training samples only',
                'status': 'failed'
            }
            
            try:
                # Reset TensorFlow state
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Train on NORMAL samples only
                X_train_normal = X_train_scaled[y_train == 0]
                logger.info(f"📊 Normal-only training: {len(X_train_normal)} normal samples")
                
                # Create FRESH Autoencoder
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
                normal_only_baseline_result['epochs_trained'] = epochs_trained
                logger.info(f"✅ Normal-Only Baseline Autoencoder trained for {epochs_trained} epochs")
                
                # Compute threshold from the SAME normal-only training population
                normal_errors = self._compute_reconstruction_errors(model, X_train_normal)
                threshold = self._compute_threshold_from_normal(normal_errors)
                normal_only_baseline_result['threshold'] = threshold
                del normal_errors
                gc.collect()
                
                # Evaluate on SAME test set
                metrics = self._evaluate_autoencoder(model, X_test_scaled, y_test, threshold, 'normal_only_baseline')
                normal_only_baseline_result.update(metrics)
                normal_only_baseline_result['status'] = 'success'
                
                logger.info(f"📊 Normal-Only Baseline: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                
                # Cleanup
                del model, X_train_normal
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Normal-Only Baseline Autoencoder failed: {e}")
                logger.error(traceback.format_exc())
                normal_only_baseline_result['status'] = 'failed'
            
            results.append(normal_only_baseline_result)
            
            # =====================================================================
            # VARIANT 3: NORMAL-ONLY ADAPTIVE
            # =====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 VARIANT 3: NORMAL-ONLY ADAPTIVE")
            logger.info(f"{'='*80}")
            
            normal_only_adaptive_result = {
                'seed': SEED,
                'variant': 'normal_only_adaptive',
                'evaluation_n': int(len(X_test_scaled)),
                'threshold': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'false_positive': 0,
                'false_negative': 0,
                'true_positive': 0,
                'true_negative': 0,
                'epochs_trained': 0,
                'training_samples': 0,
                'normal_training_samples': 0,
                'anomaly_training_samples': 0,
                'training_population_description': 'Normal training + normal retrain samples',
                'status': 'failed'
            }
            
            try:
                # Reset TensorFlow state
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(SEED)
                gc.collect()
                
                # Build adaptive normal reference population
                X_train_normal = X_train_scaled[y_train == 0]
                X_retrain_normal = X_retrain_scaled[y_retrain == 0]
                X_adaptive_normal = np.vstack([X_train_normal, X_retrain_normal])
                
                n_adaptive_normal = len(X_adaptive_normal)
                n_train_normal = len(X_train_normal)
                n_retrain_normal = len(X_retrain_normal)
                
                normal_only_adaptive_result['training_samples'] = n_adaptive_normal
                normal_only_adaptive_result['normal_training_samples'] = n_adaptive_normal
                normal_only_adaptive_result['anomaly_training_samples'] = 0
                
                logger.info(f"📊 Normal-Only Adaptive training: {n_adaptive_normal} normal samples ({n_train_normal} train + {n_retrain_normal} retrain)")
                
                # Create FRESH Autoencoder
                model = self._create_autoencoder_model(n_features)
                
                # Train on adaptive normal samples only
                history = model.fit(
                    X_adaptive_normal, X_adaptive_normal,
                    epochs=AE_EPOCHS,
                    batch_size=AE_BATCH_SIZE,
                    validation_split=AE_VAL_SPLIT,
                    verbose=0
                )
                
                epochs_trained = len(history.history['loss'])
                normal_only_adaptive_result['epochs_trained'] = epochs_trained
                logger.info(f"✅ Normal-Only Adaptive Autoencoder trained for {epochs_trained} epochs")
                
                # Compute threshold from the SAME adaptive normal training population
                adaptive_normal_errors = self._compute_reconstruction_errors(model, X_adaptive_normal)
                threshold = self._compute_threshold_from_normal(adaptive_normal_errors)
                normal_only_adaptive_result['threshold'] = threshold
                del adaptive_normal_errors
                gc.collect()
                
                # Evaluate on SAME test set
                metrics = self._evaluate_autoencoder(model, X_test_scaled, y_test, threshold, 'normal_only_adaptive')
                normal_only_adaptive_result.update(metrics)
                normal_only_adaptive_result['status'] = 'success'
                
                logger.info(f"📊 Normal-Only Adaptive: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                
                # Cleanup
                del model, X_adaptive_normal, X_train_normal, X_retrain_normal
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Normal-Only Adaptive Autoencoder failed: {e}")
                logger.error(traceback.format_exc())
                normal_only_adaptive_result['status'] = 'failed'
            
            results.append(normal_only_adaptive_result)
            
        except Exception as e:
            logger.error(f"❌ Pilot failed: {e}")
            logger.error(traceback.format_exc())
        
        return results
    
    def _verify_completeness(self, results: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Verify completeness with exactly 3 expected variants.
        """
        missing_items = []
        
        if not results:
            missing_items.append("No results generated")
            return False, missing_items
        
        # Check we have exactly 3 records
        if len(results) != EXPECTED_RECORD_COUNT:
            missing_items.append(f"Expected {EXPECTED_RECORD_COUNT} records, got {len(results)}")
        
        # Extract unique (seed, variant) keys from successful results
        successful = [r for r in results if r.get('status') == 'success']
        actual_unique_variants = set()
        for r in successful:
            key = (r.get('seed'), r.get('variant'))
            actual_unique_variants.add(key)
        
        # Define expected keys
        expected_keys = {(SEED, 'historical_baseline'), (SEED, 'normal_only_baseline'), (SEED, 'normal_only_adaptive')}
        
        # Check for missing expected keys
        missing_keys = expected_keys - actual_unique_variants
        if missing_keys:
            missing_items.append(f"Missing expected variants: {missing_keys}")
        
        # Check for unexpected keys
        unexpected_keys = actual_unique_variants - expected_keys
        if unexpected_keys:
            missing_items.append(f"Unexpected variants: {unexpected_keys}")
        
        # Check all three expected variants are successful
        for expected_variant in EXPECTED_VARIANTS:
            variant_success = any(r.get('status') == 'success' and r.get('variant') == expected_variant for r in results)
            if not variant_success:
                missing_items.append(f"{expected_variant} variant did not succeed")
        
        is_complete = (len(missing_items) == 0 and len(actual_unique_variants) == EXPECTED_RECORD_COUNT)
        
        return is_complete, missing_items
    
    def save_results(self, results: List[Dict]) -> List[str]:
        logger.info("💾 Saving pilot results...")
        saved_paths = []
        
        try:
            if results:
                # Save CSV
                results_df = pd.DataFrame(results)
                csv_path = self.pilot_dir / 'metrics' / 'autoencoder_normal_only_pilot_results.csv'
                results_df.to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"   ✅ Saved CSV: {csv_path}")
                
                # Verify completeness
                is_complete, missing_items = self._verify_completeness(results)
                n_success = sum(1 for r in results if r.get('status') == 'success')
                
                # Extract actual unique variants
                actual_unique_variants = set((r.get('seed'), r.get('variant')) for r in results if r.get('status') == 'success')
                
                # Save JSON summary with ACTUAL values
                summary = {
                    'pilot_status': 'complete' if is_complete else 'incomplete',
                    'validation_complete': is_complete,
                    'seed': SEED,
                    'dataset_rows': len(self.original_dataset) if self.original_dataset is not None else 0,
                    'feature_count': self.feature_count,
                    'train_size': self.train_size,
                    'retrain_size': self.retrain_size,
                    'test_size': self.test_size,
                    'expected_record_count': EXPECTED_RECORD_COUNT,
                    'successful_record_count': n_success,
                    'expected_variants': EXPECTED_VARIANTS,
                    'actual_unique_variants': list(actual_unique_variants),
                    'threshold_policy': '95th percentile of reconstruction error on the normal training-reference population only',
                    'historical_baseline_protocol': 'Train on ALL X_train samples (normal + anomaly), threshold from 95th percentile of normal training reconstruction errors',
                    'normal_only_baseline_protocol': 'Train on NORMAL X_train samples only, threshold from 95th percentile of normal training reconstruction errors',
                    'normal_only_adaptive_protocol': 'Train on NORMAL X_train + NORMAL X_retrain samples, threshold from 95th percentile of adaptive normal reconstruction errors',
                    'results': results,
                    'scientific_note': 'The normal-only Autoencoder variants are model-specific reconstruction-based comparators. They are not equivalent to supervised anomaly-weighted retraining.',
                    'experiment_scope': 'autoencoder_normal_only_single_seed_pilot'
                }
                
                json_path = self.pilot_dir / 'metrics' / 'autoencoder_normal_only_pilot_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"   ✅ Saved JSON: {json_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        
        return saved_paths
    
    def run_full_pilot(self) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING AUTOENCODER NORMAL-ONLY PILOT")
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
        
        # Verify completeness before saving
        is_complete, missing_items = self._verify_completeness(pilot_results)
        if not is_complete:
            logger.error("❌ Completeness check FAILED:")
            for item in missing_items:
                logger.error(f"   - {item}")
            results['status'] = 'incomplete'
        else:
            logger.info("✅ Completeness check PASSED: 3/3 variants succeeded with correct keys")
        
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
    logger = setup_logging(config.output.logs_dir, 'autoencoder_normal_only_pilot')
    
    logger.info("\n" + "="*80)
    logger.info("🎯 AUTOENCODER NORMAL-ONLY PILOT")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    pilot_dir = base_dir / 'adaptive' / 'autoencoder_normal_only_pilot'
    pilot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Pilot directory: {pilot_dir}")
    logger.info(f"⚠️  This is a SINGLE-SEED feasibility pilot (Seed={SEED})")
    logger.info(f"⚠️  Tests THREE normal-reference Autoencoder protocols")
    logger.info(f"⚠️  Does NOT modify validated four-model experiment")
    
    pilot = AutoencoderNormalOnlyPilot(base_dir=base_dir, pilot_dir=pilot_dir)
    results = pilot.run_full_pilot()
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🎉 AUTOENCODER NORMAL-ONLY PILOT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()