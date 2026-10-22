"""
Export LSTM Adaptive Predictions - Seed 42 WSL2 GPU
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This script reproduces ONLY the LSTM adaptive configuration corresponding to:
- variant = retraining_without_weighting
- seed = 42
- anomaly weight = 1.0

This script is designed to be executed under WSL2 in the cps-gpu environment,
because the reference four-model ablation LSTM results were generated there.

The script saves sample-level predictions only if it exactly matches
the existing ablation reference and aligns with the Windows baseline.

OUTPUT FILES:
-------------
- results/adaptive/final_validation_four_model/error_analysis/lstm_prediction_exports/lstm_adaptive_seed42_wsl.npz
- results/adaptive/final_validation_four_model/error_analysis/lstm_prediction_exports/lstm_adaptive_seed42_wsl_metadata.json

SCIENTIFIC NOTE:
----------------
This reproduces the EXACT ablation retraining_without_weighting protocol
using NumPy sequence arrays with validation_split=0.1, anomaly weight=1.0.
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
import traceback
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
ABLATION_RESULTS_PATH = None  # Will be set in main()
LSTM_EXPORT_DIR = None  # Will be set in main()
BASELINE_NPZ_PATH = None  # Will be set in main()

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


def _create_sequences_efficient(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input without crossing boundaries.
    Exact implementation from four_model_ablation_extension.py.
    """
    n_samples = len(X)
    if n_samples < TIME_STEPS:
        return np.array([]).reshape(0, TIME_STEPS, X.shape[1]), np.array([])
    
    n_sequences = n_samples - TIME_STEPS + 1
    
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        X_seq = sliding_window_view(X, window_shape=TIME_STEPS, axis=0)
        X_seq = np.moveaxis(X_seq, -1, 1)
        X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)
    except Exception:
        X_seq = np.zeros((n_sequences, TIME_STEPS, X.shape[1]), dtype=np.float32)
        for i in range(n_sequences):
            X_seq[i] = X[i:i + TIME_STEPS]
    
    y_seq = y[TIME_STEPS - 1:].astype(np.float32)
    
    return X_seq, y_seq


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


def _validate_against_ablation_reference(metrics: Dict, seed: int = 42) -> Tuple[bool, str]:
    """
    Validate metrics against four_model_ablation_results.csv reference.
    
    Requires:
    - seed = 42
    - model_name = LSTM
    - variant = retraining_without_weighting
    - All float metrics match at 4 decimals
    - FP and FN match exactly
    """
    global ABLATION_RESULTS_PATH
    
    if not Path(ABLATION_RESULTS_PATH).exists():
        return False, f"Reference file not found: {ABLATION_RESULTS_PATH}"
    
    try:
        ref_df = pd.read_csv(ABLATION_RESULTS_PATH)
        
        # Filter for seed=42, model_name=LSTM, variant=retraining_without_weighting
        ref_record = ref_df[
            (ref_df['seed'] == seed) &
            (ref_df['model_name'] == 'LSTM') &
            (ref_df['variant'] == 'retraining_without_weighting')
        ]
        
        if len(ref_record) != 1:
            return False, f"Reference record not found for seed={seed}, model=LSTM, variant=retraining_without_weighting"
        
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
        
        return True, "Ablation reference validation passed"
        
    except Exception as e:
        return False, f"Reference validation failed: {str(e)}"


def _validate_baseline_alignment(y_true: np.ndarray, test_endpoint_indices: np.ndarray) -> Tuple[bool, str]:
    """
    Validate alignment with Windows baseline predictions.
    
    Requires:
    - y_true matches baseline y_true exactly
    - test_endpoint_indices matches baseline test_endpoint_indices exactly
    """
    global BASELINE_NPZ_PATH
    
    if not Path(BASELINE_NPZ_PATH).exists():
        return False, f"Baseline file not found: {BASELINE_NPZ_PATH}"
    
    try:
        baseline_npz = np.load(BASELINE_NPZ_PATH)
        
        # Validate y_true alignment
        if not np.array_equal(y_true, baseline_npz['y_true']):
            return False, "y_true does not match baseline y_true"
        
        # Validate test_endpoint_indices alignment
        if not np.array_equal(test_endpoint_indices, baseline_npz['test_endpoint_indices']):
            return False, "test_endpoint_indices does not match baseline test_endpoint_indices"
        
        return True, "Baseline alignment validation passed"
        
    except Exception as e:
        return False, f"Baseline alignment validation failed: {str(e)}"


def run_lstm_adaptive_export() -> Dict:
    """
    Run LSTM adaptive export for seed 42.
    
    Returns:
        Dictionary with status and output paths
    """
    global logger, ABLATION_RESULTS_PATH, LSTM_EXPORT_DIR, BASELINE_NPZ_PATH
    
    logger.info("\n" + "="*80)
    logger.info("🎯 EXPORTING LSTM ADAPTIVE PREDICTIONS - SEED 42")
    logger.info("="*80)
    
    results = {
        'status': 'pending',
        'npz_path': None,
        'json_path': None,
        'ablation_validation_passed': False,
        'baseline_alignment_passed': False,
        'error': None
    }
    
    try:
        # Set random seeds at start (reproduces historical run_single_seed)
        set_random_seeds(SEED)
        
        # Load dataset
        config = FrameworkConfig()
        dataset_path = config.output.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        ABLATION_RESULTS_PATH = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
        LSTM_EXPORT_DIR = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis' / 'lstm_prediction_exports'
        BASELINE_NPZ_PATH = LSTM_EXPORT_DIR / 'lstm_baseline_seed42_windows.npz'
        
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
        X_retrain_imp = imputer.transform(X_retrain)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_retrain_scaled = scaler.transform(X_retrain_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        logger.info(f"✅ Preprocessing complete (mean imputer + StandardScaler fitted on train)")
        
        # =========================================================
        # LSTM ADAPTIVE: Ablation retraining_without_weighting protocol
        # =========================================================
        logger.info(f"📊 LSTM adaptive protocol: ablation retraining_without_weighting protocol")
        logger.info(f"📊 Anomaly weight: 1.0 (no up-weighting)")
        
        # Create sequences separately (never cross boundaries)
        X_train_seq_part, y_train_seq_part = _create_sequences_efficient(X_train_scaled, y_train)
        X_retrain_seq_part, y_retrain_seq_part = _create_sequences_efficient(X_retrain_scaled, y_retrain)
        
        logger.info(f"📊 Train sequences: {len(X_train_seq_part)}, Retrain sequences: {len(X_retrain_seq_part)}")
        
        # Concatenate SEQUENCE ARRAYS (not raw features)
        X_train_seq = np.concatenate([X_train_seq_part, X_retrain_seq_part], axis=0)
        y_train_seq = np.concatenate([y_train_seq_part, y_retrain_seq_part], axis=0)
        
        logger.info(f"📊 Combined training sequences: {len(X_train_seq)}")
        
        # Create test sequences
        X_test_seq, y_test_seq = _create_sequences_efficient(X_test_scaled, y_test)
        
        logger.info(f"📊 Test sequences: {len(X_test_seq)} (expected 288335)")
        
        # Reset random state in exact historical order
        tf.keras.backend.clear_session()
        gc.collect()
        set_random_seeds(SEED)
        tf.keras.utils.set_random_seed(SEED)
        
        # Create model
        model = _create_lstm_model(n_features)
        
        # Train with validation_split=0.1 (ablation-compatible)
        logger.info(f"📊 Training for {LSTM_EPOCHS} epochs with validation_split=0.1...")
        history = model.fit(
            X_train_seq,
            y_train_seq,
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
            verbose=1
        )
        
        # Predict
        y_proba = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_proba > PREDICTION_THRESHOLD).astype(int)
        
        # y_true is y_test_seq
        y_true = y_test_seq
        
        # Test endpoint indices
        test_endpoint_indices = np.arange(TIME_STEPS - 1, len(y_test))
        
        # Evaluate
        metrics = _evaluate_predictions(y_true, y_pred)
        logger.info(f"📊 Metrics: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        logger.info(f"📊 FP={metrics['false_positive']}, FN={metrics['false_negative']}")
        
        # Validate against ablation reference
        ablation_passed, ablation_msg = _validate_against_ablation_reference(metrics, SEED)
        results['ablation_validation_passed'] = ablation_passed
        
        if not ablation_passed:
            logger.error(f"❌ Ablation reference validation FAILED: {ablation_msg}")
            results['status'] = 'failed'
            results['error'] = ablation_msg
            return results
        
        logger.info(f"✅ Ablation reference validation PASSED: {ablation_msg}")
        
        # Validate baseline alignment
        alignment_passed, alignment_msg = _validate_baseline_alignment(y_true, test_endpoint_indices)
        results['baseline_alignment_passed'] = alignment_passed
        
        if not alignment_passed:
            logger.error(f"❌ Baseline alignment validation FAILED: {alignment_msg}")
            results['status'] = 'failed'
            results['error'] = alignment_msg
            return results
        
        logger.info(f"✅ Baseline alignment validation PASSED: {alignment_msg}")
        
        # Save predictions (only after BOTH validations pass)
        npz_path = LSTM_EXPORT_DIR / 'lstm_adaptive_seed42_wsl.npz'
        np.savez(
            npz_path,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            test_endpoint_indices=test_endpoint_indices
        )
        results['npz_path'] = str(npz_path)
        logger.info(f"💾 Saved predictions: {npz_path}")
        
        # Detect GPU devices
        gpu_devices = []
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_devices = [gpu.name for gpu in gpus]
        except:
            pass
        
        # Save metadata
        metadata = {
            'seed': SEED,
            'environment': 'wsl_gpu',
            'platform': sys.platform,
            'python_version': sys.version.split()[0],
            'numpy_version': np.__version__,
            'tensorflow_version': tf.__version__,
            'sklearn_version': None,
            'gpu_devices': gpu_devices,
            'evaluation_n': metrics['evaluation_samples'],
            'metrics': metrics,
            'ablation_reference_validation_status': 'passed',
            'ablation_reference_validation_message': ablation_msg,
            'baseline_alignment_status': 'passed',
            'baseline_alignment_message': alignment_msg,
            'timestamp': datetime.now().isoformat(),
            'protocol': 'ablation_retraining_without_weighting',
            'anomaly_weight': 1.0,
            'time_steps': TIME_STEPS,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'dense_units': DENSE_UNITS,
            'dropout_rate': DROPOUT_RATE,
            'learning_rate': LEARNING_RATE,
            'epochs': LSTM_EPOCHS,
            'batch_size': LSTM_BATCH_SIZE,
            'patience': PATIENCE,
            'threshold': PREDICTION_THRESHOLD,
            'validation_split': 0.1
        }
        
        # Try to get sklearn version
        try:
            import sklearn
            metadata['sklearn_version'] = sklearn.__version__
        except:
            pass
        
        json_path = LSTM_EXPORT_DIR / 'lstm_adaptive_seed42_wsl_metadata.json'
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
    """Execute LSTM adaptive export."""
    global logger
    
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'export_lstm_adaptive_predictions')
    
    logger.info("\n" + "="*80)
    logger.info("🔬 EXPORT LSTM ADAPTIVE PREDICTIONS - SEED 42")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    results = run_lstm_adaptive_export()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 EXPORT COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"✅ Ablation validation passed: {results['ablation_validation_passed']}")
    logger.info(f"✅ Baseline alignment passed: {results['baseline_alignment_passed']}")
    
    if results['npz_path']:
        logger.info(f"💾 NPZ file: {results['npz_path']}")
    if results['json_path']:
        logger.info(f"💾 JSON file: {results['json_path']}")
    if results['error']:
        logger.error(f"❌ Error: {results['error']}")
    
    logger.info("="*80)
    
    # Exit with error code if validation failed
    if not results['ablation_validation_passed'] or not results['baseline_alignment_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()