"""
Export LSTM Baseline Predictions - Seed 42 Windows Native
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This script reproduces ONLY the historical LSTM baseline for seed 42
on native Windows and saves its sample-level predictions for later
transition analysis.

This script does NOT modify four_model_error_analysis.py.

OUTPUT FILES:
-------------
- results/adaptive/final_validation_four_model/error_analysis/lstm_prediction_exports/lstm_baseline_seed42_windows.npz
- results/adaptive/final_validation_four_model/error_analysis/lstm_prediction_exports/lstm_baseline_seed42_windows_metadata.json

SCIENTIFIC NOTE:
----------------
This reproduces the EXACT historical baseline protocol from
multi_seed_validator_four_model.py using tf.data lazy window generation.
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
import traceback  # FIX #1: Add traceback import
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.error("❌ TensorFlow not available. Cannot run LSTM export.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
TIME_STEPS = 10
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
PATIENCE = 10
PREDICTION_THRESHOLD = 0.5

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

# Reference validation paths
MULTISEED_RESULTS_PATH = None  # Will be set in main()
LSTM_EXPORT_DIR = None  # Will be set in main()

# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _identify_label_column(df: pd.DataFrame) -> Optional[str]:
    """Identify label column"""
    for col_name in LABEL_COLUMN_NAMES:
        if col_name in df.columns:
            return col_name
    for col in df.columns:
        if col.lower() in [name.lower() for name in LABEL_COLUMN_NAMES]:
            return col
    return None


def _convert_labels_to_binary(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Convert labels to binary (0=normal, 1=anomaly)"""
    labels_raw = df[label_col].values
    labels = np.zeros(len(labels_raw), dtype=int)
    normal_indicators = ['normal', '0', 'benign', 'ok', 'healthy', 'false', 'negative']
    
    for i, label in enumerate(labels_raw):
        if str(label).lower().strip() not in normal_indicators:
            labels[i] = 1
    
    return labels


def _prepare_features(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, List[str]]:
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


def _create_lstm_model(n_features: int) -> Any:
    """Create LSTM model with validated architecture."""
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


def _create_lstm_baseline_tf_dataset(X: np.ndarray, y: np.ndarray, 
                                     sequence_length: int = 10, 
                                     batch_size: int = 64,
                                     seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """
    Create LSTM baseline datasets using historical tf.data pipeline 
    (from multi_seed_validator_four_model.py).
    
    This reproduces the EXACT historical baseline protocol:
    - Lazily generates sliding windows
    - Splits last 10% as validation
    - Shuffles training with bounded buffer (10000)
    - Uses seed and reshuffle_each_iteration=True
    - Batches at 64, prefetches AUTOTUNE
    """
    n = len(X) - sequence_length + 1
    
    X_tf = tf.constant(X, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.float32)
    w_tf = tf.constant(np.ones(len(y), dtype=np.float32), dtype=tf.float32)
    
    def map_fn(i):
        return (
            X_tf[i:i + sequence_length],
            y_tf[i + sequence_length - 1],
            w_tf[i + sequence_length - 1]
        )
    
    ds = tf.data.Dataset.range(n).map(
        map_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    n_val = int(n * 0.1)
    n_train = n - n_val
    
    # Training dataset MUST use historical shuffle parameters
    train_dataset = ds.take(n_train).shuffle(
        buffer_size=min(10000, n_train),
        seed=seed,
        reshuffle_each_iteration=True
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset (no shuffle)
    val_dataset = ds.skip(n_train).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, n_train


def _create_lstm_test_dataset(X: np.ndarray, y: np.ndarray, 
                              sequence_length: int = 10,
                              batch_size: int = 64) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Create LSTM test dataset using historical tf.data pipeline.
    
    Returns:
        test_dataset: tf.data.Dataset
        y_true: numpy array of true labels (y[sequence_length-1:])
        test_endpoint_indices: numpy array of endpoint indices
    """
    n = len(X) - sequence_length + 1
    
    X_tf = tf.constant(X, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.float32)
    
    def test_map_fn(i):
        return (
            X_tf[i:i + sequence_length],
            y_tf[i + sequence_length - 1]
        )
    
    test_dataset = tf.data.Dataset.range(n).map(
        test_map_fn,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # y_true is y[sequence_length - 1:]
    y_true = y[sequence_length - 1:]
    
    # Test endpoint indices
    test_endpoint_indices = np.arange(sequence_length - 1, len(y))
    
    return test_dataset, y_true, test_endpoint_indices


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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


def _validate_against_reference(metrics: Dict, seed: int = 42) -> Tuple[bool, str]:
    """
    Validate metrics against multi_seed_four_model_results.csv reference.
    
    Requires:
    - All float metrics match at 4 decimals
    - FP and FN match exactly
    """
    global MULTISEED_RESULTS_PATH
    
    if not Path(MULTISEED_RESULTS_PATH).exists():
        return False, f"Reference file not found: {MULTISEED_RESULTS_PATH}"
    
    try:
        ref_df = pd.read_csv(MULTISEED_RESULTS_PATH)
        
        # Filter for seed=42, model_name=LSTM, model_type=baseline
        ref_record = ref_df[
            (ref_df['seed'] == seed) &
            (ref_df['model_name'] == 'LSTM') &
            (ref_df['model_type'] == 'baseline')
        ]
        
        if len(ref_record) != 1:
            return False, f"Reference record not found for seed={seed}, model=LSTM, type=baseline"
        
        ref_record = ref_record.iloc[0]
        
        # Validate float metrics at 4-decimal precision
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            generated_val = metrics.get(metric)
            reference_val = ref_record.get(metric)
            
            if generated_val is not None and reference_val is not None:
                if round(float(generated_val), 4) != round(float(reference_val), 4):
                    return False, f"{metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
        
        # Validate FP/FN exactly (integers)
        for cm_metric in ['false_positive', 'false_negative']:
            generated_val = metrics.get(cm_metric)
            reference_val = ref_record.get(cm_metric)
            
            if generated_val is not None and reference_val is not None:
                if int(generated_val) != int(reference_val):
                    return False, f"{cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
        
        return True, "Validation passed"
        
    except Exception as e:
        return False, f"Reference validation failed: {str(e)}"


def run_lstm_baseline_export() -> Dict:
    """
    Run LSTM baseline export for seed 42.
    
    Returns:
        Dictionary with status and output paths
    """
    global logger, MULTISEED_RESULTS_PATH, LSTM_EXPORT_DIR
    
    logger.info("\n" + "="*80)
    logger.info("🎯 EXPORTING LSTM BASELINE PREDICTIONS - SEED 42")
    logger.info("="*80)
    
    results = {
        'status': 'pending',
        'npz_path': None,
        'json_path': None,
        'validation_passed': False,
        'error': None
    }
    
    try:
        # FIX #2: Set random seeds immediately (reproduces historical run_single_seed)
        set_random_seeds(SEED)
        
        # Load dataset
        config = FrameworkConfig()
        dataset_path = config.output.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        MULTISEED_RESULTS_PATH = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'metrics' / 'multi_seed_four_model_results.csv'
        LSTM_EXPORT_DIR = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'lstm_prediction_exports'
        
        LSTM_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Loading dataset: {dataset_path}")
        original_dataset = pd.read_csv(dataset_path)
        original_dataset.columns = original_dataset.columns.astype(str).str.strip()
        logger.info(f"✅ Loaded dataset: {len(original_dataset)} rows")
        
        # Identify label column
        label_col = _identify_label_column(original_dataset)
        if label_col is None:
            raise ValueError("No label column found")
        
        # Convert labels
        all_labels = _convert_labels_to_binary(original_dataset, label_col)
        
        # Prepare features
        all_features, feature_cols = _prepare_features(original_dataset, label_col)
        if all_features is None:
            raise ValueError("No features found")
        
        n_features = len(feature_cols)
        logger.info(f"📊 Features: {n_features} columns")
        
        # Split data (stratified, seed=42)
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
        
        logger.info(f"📊 Split sizes: Train={len(X_train)}, Retrain={len(X_retrain)}, Test={len(X_test)}")
        
        # Preprocessing (fit on TRAIN ONLY)
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        logger.info(f"✅ Preprocessing complete (mean imputer + StandardScaler fitted on train)")
        
        # =========================================================
        # LSTM BASELINE: Historical tf.data reference protocol
        # =========================================================
        logger.info(f"📊 LSTM baseline protocol: historical tf.data reference protocol")
        
        # Create baseline datasets using historical tf.data pipeline
        baseline_train_dataset, baseline_val_dataset, n_train_baseline = _create_lstm_baseline_tf_dataset(
            X_train_scaled, y_train,
            sequence_length=TIME_STEPS,
            batch_size=LSTM_BATCH_SIZE,
            seed=SEED
        )
        
        logger.info(f"📊 Training sequences: {n_train_baseline}, Validation sequences: {len(X_train_scaled) - TIME_STEPS + 1 - n_train_baseline}")
        
        # Reset random state in exact historical order
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)
        
        # Create model
        model = _create_lstm_model(n_features)
        
        # Train
        logger.info(f"📊 Training for {LSTM_EPOCHS} epochs...")
        history = model.fit(
            baseline_train_dataset,
            validation_data=baseline_val_dataset,
            epochs=LSTM_EPOCHS,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=PATIENCE,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        # Create test dataset
        test_dataset, y_true, test_endpoint_indices = _create_lstm_test_dataset(
            X_test_scaled, y_test,
            sequence_length=TIME_STEPS,
            batch_size=LSTM_BATCH_SIZE
        )
        
        logger.info(f"📊 Test sequences: {len(test_endpoint_indices)} (expected 288335)")
        
        # Predict
        y_proba = model.predict(test_dataset, verbose=0).flatten()
        y_pred = (y_proba > PREDICTION_THRESHOLD).astype(int)
        
        # Evaluate
        metrics = _evaluate_predictions(y_true, y_pred)
        logger.info(f"📊 Metrics: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.info(f"📊 FP={metrics['false_positive']}, FN={metrics['false_negative']}")
        
        # Validate against reference
        validation_passed, validation_msg = _validate_against_reference(metrics, SEED)
        results['validation_passed'] = validation_passed
        
        if not validation_passed:
            logger.error(f"❌ Reference validation FAILED: {validation_msg}")
            results['status'] = 'failed'
            results['error'] = validation_msg
            return results
        
        logger.info(f"✅ Reference validation PASSED: {validation_msg}")
        
        # Save predictions
        npz_path = LSTM_EXPORT_DIR / 'lstm_baseline_seed42_windows.npz'
        np.savez(
            npz_path,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            test_endpoint_indices=test_endpoint_indices
        )
        results['npz_path'] = str(npz_path)
        logger.info(f"💾 Saved predictions: {npz_path}")
        
        # Save metadata
        metadata = {
            'seed': SEED,
            'environment': 'windows_native',
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__,
            'tensorflow_version': tf.__version__,
            'evaluation_n': metrics['evaluation_samples'],
            'metrics': metrics,
            'reference_validation_status': 'passed',
            'reference_validation_message': validation_msg,
            'timestamp': datetime.now().isoformat(),
            'protocol': 'historical_tf.data_baseline',
            'time_steps': TIME_STEPS,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dense_units': DENSE_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'epochs': LSTM_EPOCHS,
            'batch_size': LSTM_BATCH_SIZE,
            'patience': PATIENCE,
            'threshold': PREDICTION_THRESHOLD
        }
        
        json_path = LSTM_EXPORT_DIR / 'lstm_baseline_seed42_windows_metadata.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        results['json_path'] = str(json_path)
        logger.info(f"💾 Saved metadata: {json_path}")
        
        results['status'] = 'complete'
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        logger.error(traceback.format_exc())
        results['status'] = 'failed'
        results['error'] = str(e)
        return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute LSTM baseline export."""
    global logger
    
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'export_lstm_baseline_predictions')
    
    logger.info("\n" + "="*80)
    logger.info("🔬 EXPORT LSTM BASELINE PREDICTIONS - SEED 42")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    results = run_lstm_baseline_export()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 EXPORT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"✅ Validation passed: {results['validation_passed']}")
    
    if results['npz_path']:
        logger.info(f"💾 NPZ file: {results['npz_path']}")
    if results['json_path']:
        logger.info(f"💾 JSON file: {results['json_path']}")
    if results['error']:
        logger.error(f"❌ Error: {results['error']}")
    
    logger.info("="*80)
    
    # Exit with error code if validation failed
    if not results['validation_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()