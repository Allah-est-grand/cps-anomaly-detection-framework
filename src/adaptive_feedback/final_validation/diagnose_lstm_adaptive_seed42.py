"""
LSTM Determinism Diagnostic - Seed 42
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This script diagnoses whether the exact historical LSTM adaptive protocol for seed 42
is reproducible at sample level under the current WSL2 / cps-gpu environment.

IMPORTANT:
- Does NOT test alternative scientific protocols
- Does NOT concatenate raw train/retrain data before sequence generation
- Does NOT change validation_split
- Does NOT change shuffle behavior
- Does NOT change architecture, threshold, dataset, split, preprocessing, or sequence logic

This runs the EXACT SAME historical adaptive LSTM training protocol TWO TIMES
inside the same script to test reproducibility.

OUTPUT FILE:
------------
- results/adaptive/final_validation_four_model/error_analysis/lstm_determinism_diagnostic_seed42.json
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
    logging.error("❌ TensorFlow not available. Cannot run LSTM diagnostic.")
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

# Reference validation path
ABLATION_RESULTS_PATH = None  # Will be set in main()
LSTM_DIAGNOSTIC_DIR = None  # Will be set in main()

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


def _compute_hash(data: np.ndarray) -> str:
    """Compute SHA256 hash of numpy array."""
    return hashlib.sha256(data.tobytes()).hexdigest()


def _get_environment_audit() -> Dict:
    """Audit environment for reproducibility diagnostics."""
    env_audit = {
        'platform': sys.platform,
        'python_version': sys.version.split()[0],
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }
    
    try:
        import sklearn
        env_audit['sklearn_version'] = sklearn.__version__
    except:
        env_audit['sklearn_version'] = 'unavailable'
    
    try:
        env_audit['tensorflow_version'] = tf.__version__
    except:
        env_audit['tensorflow_version'] = 'unavailable'
    
    try:
        import keras
        env_audit['keras_version'] = keras.__version__
    except:
        env_audit['keras_version'] = 'unavailable'
    
    # CUDA/cuDNN information
    try:
        env_audit['cuda_build_info'] = tf.test.is_built_with_cuda()
    except:
        env_audit['cuda_build_info'] = 'unavailable'
    
    try:
        from tensorflow.python.platform import build_info as tf_build_info
        env_audit['cudnn_version'] = getattr(tf_build_info, 'build_info', {}).get('cudnn_version', 'unavailable')
        env_audit['cuda_version'] = getattr(tf_build_info, 'build_info', {}).get('cuda_version', 'unavailable')
    except:
        env_audit['cudnn_version'] = 'unavailable'
        env_audit['cuda_version'] = 'unavailable'
    
    # GPU devices
    try:
        gpus = tf.config.list_physical_devices('GPU')
        env_audit['visible_gpu_devices'] = len(gpus)
        env_audit['gpu_device_names'] = [gpu.name for gpu in gpus]
    except:
        env_audit['visible_gpu_devices'] = 0
        env_audit['gpu_device_names'] = []
    
    # Environment variables affecting determinism
    env_audit['TF_ENABLE_ONEDNN_OPTS'] = os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'not set')
    env_audit['TF_DETERMINISTIC_OPS'] = os.environ.get('TF_DETERMINISTIC_OPS', 'not set')
    env_audit['CUBLAS_WORKSPACE_CONFIG'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'not set')
    env_audit['PYTHONHASHSEED'] = os.environ.get('PYTHONHASHSEED', 'not set')
    
    return env_audit


def _run_lstm_adaptive_repeat(seed: int = 42) -> Dict:
    """
    Run ONE repeat of the exact historical LSTM adaptive protocol.
    
    This reproduces the EXACT protocol from four_model_ablation_extension.py:
    1. set_random_seeds(seed) at start
    2. train/test split
    3. train/retrain split
    4. imputer fit on train only
    5. scaler fit on train only
    6. create train sequences separately
    7. create retrain sequences separately
    8. concatenate sequence arrays
    9. create test sequences
    10. immediately before model creation:
        tf.keras.backend.clear_session()
        gc.collect()
        set_random_seeds(seed)
        tf.keras.utils.set_random_seed(seed)
    11. model.fit with validation_split=0.1
    12. prediction threshold > 0.5
    
    FIX #3: Returns arrays for in-memory diagnostic comparison (not saved to JSON).
    """
    global logger
    
    result = {
        'metrics': {},
        'epochs_trained': 0,
        'best_val_loss': None,
        'y_pred_hash': None,
        'y_proba_hash': None,
        'y_pred_array': None,  # FIX #3: For in-memory comparison only
        'y_proba_array': None,  # FIX #3: For in-memory comparison only
        'y_true_array': None,   # FIX #3: For in-memory comparison only
        'error': None
    }
    
    try:
        # Step 1: Set random seeds at start
        set_random_seeds(seed)
        
        # Load dataset
        config = FrameworkConfig()
        dataset_path = config.output.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"📊 Loading dataset...")
        original_dataset = pd.read_csv(dataset_path)
        original_dataset.columns = original_dataset.columns.astype(str).str.strip()
        
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
        
        # FIX #2: Steps 2-3: Split data (stratified, seed=42 - deterministic)
        # Both repeats use random_state=SEED, so splits are identical
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
        
        # Steps 4-5: Preprocessing (fit on TRAIN ONLY)
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)
        X_retrain_imp = imputer.transform(X_retrain)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_retrain_scaled = scaler.transform(X_retrain_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # FIX #2: Steps 6-9: Create sequences
        # Step 6: Create train sequences separately
        X_train_seq_part, y_train_seq_part = _create_sequences_efficient(X_train_scaled, y_train)
        
        # Step 7: Create retrain sequences separately
        X_retrain_seq_part, y_retrain_seq_part = _create_sequences_efficient(X_retrain_scaled, y_retrain)
        
        # Step 8: Concatenate sequence arrays (NOT raw features)
        X_train_seq = np.concatenate([X_train_seq_part, X_retrain_seq_part], axis=0)
        y_train_seq = np.concatenate([y_train_seq_part, y_retrain_seq_part], axis=0)
        
        # Step 9: Create test sequences
        X_test_seq, y_test_seq = _create_sequences_efficient(X_test_scaled, y_test)
        
        # Step 10: Reset random state in EXACT historical order
        tf.keras.backend.clear_session()
        gc.collect()
        set_random_seeds(seed)
        tf.keras.utils.set_random_seed(seed)
        
        # Create model
        model = _create_lstm_model(n_features)
        
        # Step 11: Train with validation_split=0.1
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
            verbose=0
        )
        
        # Record training information
        result['epochs_trained'] = len(history.history['loss'])
        result['best_val_loss'] = float(min(history.history['val_loss']))
        
        # Step 12: Predict
        y_proba = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_proba > PREDICTION_THRESHOLD).astype(int)
        
        # FIX #3: Store arrays for in-memory comparison (will be removed before JSON save)
        result['y_pred_array'] = y_pred.copy()
        result['y_proba_array'] = y_proba.astype(np.float32).copy()
        result['y_true_array'] = y_test_seq.copy()
        
        # Evaluate
        metrics = _evaluate_predictions(y_test_seq, y_pred)
        result['metrics'] = metrics
        
        # Compute hashes
        result['y_pred_hash'] = _compute_hash(y_pred)
        result['y_proba_hash'] = _compute_hash(y_proba.astype(np.float32))
        
        # Cleanup
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Repeat failed: {e}")
        logger.error(traceback.format_exc())
        result['error'] = str(e)
        return result


def run_diagnostics() -> Dict:
    """
    Run complete determinism diagnostics.
    
    Runs TWO repeats of the exact same protocol and compares them.
    Also compares both against historical reference.
    
    FIX #2: Both repeats use identical data splits (random_state=42).
    FIX #4: Direct comparison of y_pred and y_proba arrays.
    FIX #6: Use round(float, 4) comparison for float metrics.
    """
    global logger, ABLATION_RESULTS_PATH, LSTM_DIAGNOSTIC_DIR
    
    logger.info("\n" + "="*80)
    logger.info("🎯 LSTM DETERMINISM DIAGNOSTIC - SEED 42")
    logger.info("="*80)
    
    diagnostics = {
        'reference': None,
        'repeat_1': None,
        'repeat_2': None,
        'repeat_comparison': {},
        'environment': {},
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Load environment audit
        logger.info(f"📊 Auditing environment...")
        diagnostics['environment'] = _get_environment_audit()
        
        for key, value in diagnostics['environment'].items():
            logger.info(f"   {key}: {value}")
        
        # Load historical reference
        logger.info(f"📊 Loading historical reference...")
        ref_df = pd.read_csv(ABLATION_RESULTS_PATH)
        
        ref_record = ref_df[
            (ref_df['seed'] == SEED) &
            (ref_df['model_name'] == 'LSTM') &
            (ref_df['variant'] == 'retraining_without_weighting')
        ]
        
        if len(ref_record) != 1:
            logger.error(f"❌ Reference record not found")
            diagnostics['reference'] = {'error': 'Reference record not found'}
        else:
            ref_record = ref_record.iloc[0]
            diagnostics['reference'] = {
                'accuracy': float(ref_record.get('accuracy', None)),
                'precision': float(ref_record.get('precision', None)),
                'recall': float(ref_record.get('recall', None)),
                'f1_score': float(ref_record.get('f1_score', None)),
                'false_positive': int(ref_record.get('false_positive', None)) if pd.notna(ref_record.get('false_positive')) else None,
                'false_negative': int(ref_record.get('false_negative', None)) if pd.notna(ref_record.get('false_negative')) else None,
                'true_positive': int(ref_record.get('true_positive', None)) if pd.notna(ref_record.get('true_positive')) else None,
                'true_negative': int(ref_record.get('true_negative', None)) if pd.notna(ref_record.get('true_negative')) else None,
            }
            
            logger.info(f"✅ Historical reference loaded")
            for key, value in diagnostics['reference'].items():
                logger.info(f"   {key}: {value}")
        
        # Run repeat 1
        logger.info(f"\n📊 Running repeat 1...")
        diagnostics['repeat_1'] = _run_lstm_adaptive_repeat(SEED)
        
        if diagnostics['repeat_1'].get('error'):
            logger.error(f"❌ Repeat 1 failed: {diagnostics['repeat_1']['error']}")
        else:
            logger.info(f"✅ Repeat 1 completed")
            logger.info(f"   Epochs trained: {diagnostics['repeat_1']['epochs_trained']}")
            logger.info(f"   Best val_loss: {diagnostics['repeat_1']['best_val_loss']:.6f}")
            logger.info(f"   F1: {diagnostics['repeat_1']['metrics'].get('f1_score', 'N/A')}")
            logger.info(f"   y_pred hash: {diagnostics['repeat_1']['y_pred_hash'][:16]}...")
        
        # Run repeat 2 (ALWAYS run, regardless of repeat 1 result)
        logger.info(f"\n📊 Running repeat 2...")
        diagnostics['repeat_2'] = _run_lstm_adaptive_repeat(SEED)
        
        if diagnostics['repeat_2'].get('error'):
            logger.error(f"❌ Repeat 2 failed: {diagnostics['repeat_2']['error']}")
        else:
            logger.info(f"✅ Repeat 2 completed")
            logger.info(f"   Epochs trained: {diagnostics['repeat_2']['epochs_trained']}")
            logger.info(f"   Best val_loss: {diagnostics['repeat_2']['best_val_loss']:.6f}")
            logger.info(f"   F1: {diagnostics['repeat_2']['metrics'].get('f1_score', 'N/A')}")
            logger.info(f"   y_pred hash: {diagnostics['repeat_2']['y_pred_hash'][:16]}...")
        
        # FIX #4: Compare repeats directly (both use random_state=42, so splits are identical)
        logger.info(f"\n📊 Comparing repeats...")
        
        if diagnostics['repeat_1'].get('error') or diagnostics['repeat_2'].get('error'):
            diagnostics['repeat_comparison'] = {
                'error': 'One or both repeats failed',
                'repeat_1_error': diagnostics['repeat_1'].get('error'),
                'repeat_2_error': diagnostics['repeat_2'].get('error')
            }
        else:
            # FIX #4: Direct array comparison (both repeats use identical data splits)
            y_true_identical = np.array_equal(
                diagnostics['repeat_1']['y_true_array'],
                diagnostics['repeat_2']['y_true_array']
            )
            
            y_pred_identical = np.array_equal(
                diagnostics['repeat_1']['y_pred_array'],
                diagnostics['repeat_2']['y_pred_array']
            )
            
            different_prediction_count = int(np.count_nonzero(
                diagnostics['repeat_1']['y_pred_array'] !=
                diagnostics['repeat_2']['y_pred_array']
            ))
            
            max_probability_difference = float(np.max(
                np.abs(
                    diagnostics['repeat_1']['y_proba_array'] -
                    diagnostics['repeat_2']['y_proba_array']
                )
            ))
            
            mean_probability_difference = float(np.mean(
                np.abs(
                    diagnostics['repeat_1']['y_proba_array'] -
                    diagnostics['repeat_2']['y_proba_array']
                )
            ))
            
            diagnostics['repeat_comparison'] = {
                'y_true_identical': y_true_identical,
                'y_pred_identical': y_pred_identical,
                'different_prediction_count': different_prediction_count,
                'max_probability_difference': max_probability_difference,
                'mean_probability_difference': mean_probability_difference,
                'y_pred_hash_identical': diagnostics['repeat_1']['y_pred_hash'] == diagnostics['repeat_2']['y_pred_hash'],
                'y_proba_hash_identical': diagnostics['repeat_1']['y_proba_hash'] == diagnostics['repeat_2']['y_proba_hash'],
                'epochs_trained_identical': diagnostics['repeat_1']['epochs_trained'] == diagnostics['repeat_2']['epochs_trained'],
                'best_val_loss_difference': abs(diagnostics['repeat_1']['best_val_loss'] - diagnostics['repeat_2']['best_val_loss']),
                'note': 'Both repeats use random_state=42, so train/retrain/test splits are deterministic and identical.'
            }
            
            logger.info(f"   y_true identical: {y_true_identical}")
            logger.info(f"   y_pred identical: {y_pred_identical}")
            logger.info(f"   Different predictions: {different_prediction_count}")
            logger.info(f"   Max probability diff: {max_probability_difference:.6f}")
            logger.info(f"   Mean probability diff: {mean_probability_difference:.6f}")
            logger.info(f"   y_pred hash identical: {diagnostics['repeat_comparison']['y_pred_hash_identical']}")
            logger.info(f"   y_proba hash identical: {diagnostics['repeat_comparison']['y_proba_hash_identical']}")
        
        # FIX #6: Compare each repeat against reference using round(float, 4)
        if diagnostics['reference'] and not diagnostics['repeat_1'].get('error'):
            ref_metrics = diagnostics['reference']
            r1_metrics = diagnostics['repeat_1']['metrics']
            
            diagnostics['repeat_1_vs_reference'] = {
                'accuracy_match': round(float(r1_metrics.get('accuracy', 0)), 4) == round(float(ref_metrics.get('accuracy', 0)), 4) if ref_metrics.get('accuracy') is not None else False,
                'precision_match': round(float(r1_metrics.get('precision', 0)), 4) == round(float(ref_metrics.get('precision', 0)), 4) if ref_metrics.get('precision') is not None else False,
                'recall_match': round(float(r1_metrics.get('recall', 0)), 4) == round(float(ref_metrics.get('recall', 0)), 4) if ref_metrics.get('recall') is not None else False,
                'f1_match': round(float(r1_metrics.get('f1_score', 0)), 4) == round(float(ref_metrics.get('f1_score', 0)), 4) if ref_metrics.get('f1_score') is not None else False,
                'fp_match': r1_metrics.get('false_positive') == ref_metrics.get('false_positive') if ref_metrics.get('false_positive') is not None else False,
                'fn_match': r1_metrics.get('false_negative') == ref_metrics.get('false_negative') if ref_metrics.get('false_negative') is not None else False,
            }
        
        if diagnostics['reference'] and not diagnostics['repeat_2'].get('error'):
            ref_metrics = diagnostics['reference']
            r2_metrics = diagnostics['repeat_2']['metrics']
            
            diagnostics['repeat_2_vs_reference'] = {
                'accuracy_match': round(float(r2_metrics.get('accuracy', 0)), 4) == round(float(ref_metrics.get('accuracy', 0)), 4) if ref_metrics.get('accuracy') is not None else False,
                'precision_match': round(float(r2_metrics.get('precision', 0)), 4) == round(float(ref_metrics.get('precision', 0)), 4) if ref_metrics.get('precision') is not None else False,
                'recall_match': round(float(r2_metrics.get('recall', 0)), 4) == round(float(ref_metrics.get('recall', 0)), 4) if ref_metrics.get('recall') is not None else False,
                'f1_match': round(float(r2_metrics.get('f1_score', 0)), 4) == round(float(ref_metrics.get('f1_score', 0)), 4) if ref_metrics.get('f1_score') is not None else False,
                'fp_match': r2_metrics.get('false_positive') == ref_metrics.get('false_positive') if ref_metrics.get('false_positive') is not None else False,
                'fn_match': r2_metrics.get('false_negative') == ref_metrics.get('false_negative') if ref_metrics.get('false_negative') is not None else False,
            }
        
        # FIX #5: Remove arrays from repeat_1 and repeat_2 before saving JSON
        if not diagnostics['repeat_1'].get('error'):
            del diagnostics['repeat_1']['y_pred_array']
            del diagnostics['repeat_1']['y_proba_array']
            del diagnostics['repeat_1']['y_true_array']
        
        if not diagnostics['repeat_2'].get('error'):
            del diagnostics['repeat_2']['y_pred_array']
            del diagnostics['repeat_2']['y_proba_array']
            del diagnostics['repeat_2']['y_true_array']
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"❌ Diagnostics failed: {e}")
        logger.error(traceback.format_exc())
        diagnostics['error'] = str(e)
        return diagnostics


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute LSTM determinism diagnostics."""
    global logger, ABLATION_RESULTS_PATH, LSTM_DIAGNOSTIC_DIR
    
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'diagnose_lstm_determinism_seed42')
    
    logger.info("\n" + "="*80)
    logger.info("🔬 LSTM DETERMINISM DIAGNOSTIC - SEED 42")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Set paths
    ABLATION_RESULTS_PATH = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
    LSTM_DIAGNOSTIC_DIR = config.output.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis'
    
    LSTM_DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run diagnostics
    diagnostics = run_diagnostics()
    
    # Save results
    json_path = LSTM_DIAGNOSTIC_DIR / 'lstm_determinism_diagnostic_seed42.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2)
    
    logger.info(f"\n💾 Saved diagnostics: {json_path}")
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 DIAGNOSTICS COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"💾 Output: {json_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()