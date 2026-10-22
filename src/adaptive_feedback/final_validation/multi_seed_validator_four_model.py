"""
Multi-Seed Validator (Four-Model Extension) - Stability Analysis Module
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements multi-seed stability analysis for FOUR models:
1. XGBoost
2. Random Forest
3. Linear SVM
4. LSTM

This is a SEPARATE controlled experiment extending the validated three-model protocol.
LSTM is treated strictly as a "deep-learning comparator under the retained data representation".

IMPORTANT:
- This module does NOT modify the existing three-model validation
- This module uses a SEPARATE output directory
- Historical outputs remain untouched
- No figures are generated in this stage (numerical validation only)
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
from __future__ import annotations  # FIX #2: Makes TensorFlow-unavailable import safe

import sys
import time
import logging
import json
import traceback
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to sys.path for config import
# This file is at: src/adaptive_feedback/final_validation/multi_seed_validator_four_model.py
# We need to go up 3 levels to reach project root where config.py exists
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now config.py can be imported from project root
from config import FrameworkConfig
from src.utils import setup_logging, set_random_seeds

# Data processing
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb

# TensorFlow for LSTM
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 2024, 2025, 3407]

MODELS_OF_INTEREST = ['XGBoost', 'RandomForest', 'LinearSVM', 'LSTM']

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

SVM_CONFIG = {
    'C': 1.0,
    'class_weight': None,
    'max_iter': 10000,
    'random_state': 42
}

# FIX #1: Bound the shuffle buffer to prevent enormous memory allocation
LSTM_SHUFFLE_BUFFER = 10000  # Memory-safety adaptation for very large dataset

EXPECTED_RECORD_COUNT = 40  # 4 models × 5 seeds × 2 variants

# =============================================================================
# MULTI-SEED VALIDATOR FOUR-MODEL CLASS
# =============================================================================
class MultiSeedValidatorFourModel:
    """
    Implements multi-seed stability analysis for four models.
    """
    
    def __init__(self, base_dir: Path, validation_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.validation_dir = Path(validation_dir) if validation_dir else (
            self.base_dir / 'adaptive' / 'final_validation_four_model'
        )
        
        (self.validation_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.results = []
        self.statistics = {}
        self.validation_status = 'pending'
        self.data_source_description = ''
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized MultiSeedValidatorFourModel")
        logger.info(f"📁 Validation directory: {self.validation_dir}")
        logger.info(f"🔢 Testing {len(SEEDS)} seeds: {SEEDS}")
        logger.info(f"📊 Models: {MODELS_OF_INTEREST}")
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
            logger.error("❌ No numeric feature columns found")
            return None, []
            
        logger.info(f"📊 Feature selection: {len(feature_cols)} numeric features retained")
        return df[feature_cols].values, feature_cols
    
    def _create_sample_weights(self, labels: np.ndarray) -> np.ndarray:
        weights = np.ones(len(labels), dtype=np.float32)
        weights[labels == 1] = ANOMALY_WEIGHT
        return weights

    def _create_lstm_tf_dataset(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, 
                                sequence_length: int = 10, batch_size: int = 64, 
                                seed: Optional[int] = None, val_split: float = 0.1) -> Tuple[Any, Any, int]:
        """
        Creates a memory-efficient tf.data pipeline for LSTM training.
        Sliding windows are generated lazily to avoid materializing huge (N-9, 10, 51) NumPy arrays.
        
        FIX #1: Shuffle buffer is bounded to prevent enormous memory allocation.
        """
        n = len(X) - sequence_length + 1
        logger.info(f"    ℹ️   Number of valid sequences: {n}")
        
        X_tf = tf.constant(X, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)
        w_tf = tf.constant(sample_weights, dtype=tf.float32)
        
        def map_fn(i):
            return (
                X_tf[i : i + sequence_length],
                y_tf[i + sequence_length - 1],
                w_tf[i + sequence_length - 1]
            )
        
        ds = tf.data.Dataset.range(n).map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        n_val = int(n * val_split)
        n_train = n - n_val
        
        # FIX #1: Bound shuffle buffer to LSTM_SHUFFLE_BUFFER (10000) to prevent enormous memory allocation
        # This is a memory-safety adaptation for the very large dataset
        train_ds = ds.take(n_train).shuffle(
            buffer_size=min(LSTM_SHUFFLE_BUFFER, n_train),
            seed=seed,
            reshuffle_each_iteration=True
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Skip to the last n_val for validation (no shuffle for validation)
        val_ds = ds.skip(n_train).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, n_train

    def _create_combined_lstm_tf_dataset(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray,
                                         X_retrain: np.ndarray, y_retrain: np.ndarray, w_retrain: np.ndarray,
                                         sequence_length: int = 10, batch_size: int = 64, 
                                         seed: Optional[int] = None, val_split: float = 0.1) -> Tuple[Any, Any, int]:
        """
        Creates a memory-efficient tf.data pipeline for weighted LSTM training.
        Train and retrain sequences are constructed separately to prevent windows from crossing boundaries.
        They are then logically concatenated, and the final 10% of this combined logical population 
        is reserved for internal validation, equivalent to Keras validation_split=0.1, without 
        materializing a giant concatenated NumPy array.
        
        FIX #1: Shuffle buffer is bounded to prevent enormous memory allocation.
        """
        n_train_seq = len(X_train) - sequence_length + 1
        n_retrain_seq = len(X_retrain) - sequence_length + 1
        n_total = n_train_seq + n_retrain_seq
        logger.info(f"    ℹ️   Train sequences: {n_train_seq}, Retrain sequences: {n_retrain_seq}, Total logical sequences: {n_total}")
        
        X_train_tf = tf.constant(X_train, dtype=tf.float32)
        y_train_tf = tf.constant(y_train, dtype=tf.float32)
        w_train_tf = tf.constant(w_train, dtype=tf.float32)
        
        X_retrain_tf = tf.constant(X_retrain, dtype=tf.float32)
        y_retrain_tf = tf.constant(y_retrain, dtype=tf.float32)
        w_retrain_tf = tf.constant(w_retrain, dtype=tf.float32)
        
        def make_map_fn(X_tf, y_tf, w_tf):
            def map_fn(i):
                return (
                    X_tf[i : i + sequence_length],
                    y_tf[i + sequence_length - 1],
                    w_tf[i + sequence_length - 1]
                )
            return map_fn

        ds_train = tf.data.Dataset.range(n_train_seq).map(make_map_fn(X_train_tf, y_train_tf, w_train_tf), num_parallel_calls=tf.data.AUTOTUNE)
        ds_retrain = tf.data.Dataset.range(n_retrain_seq).map(make_map_fn(X_retrain_tf, y_retrain_tf, w_retrain_tf), num_parallel_calls=tf.data.AUTOTUNE)
        
        # Concatenate datasets logically without materializing a giant NumPy array
        combined_ds = ds_train.concatenate(ds_retrain)
        
        n_val = int(n_total * val_split)
        n_train_final = n_total - n_val
        
        # FIX #1: Bound shuffle buffer to LSTM_SHUFFLE_BUFFER (10000) to prevent enormous memory allocation
        # This is a memory-safety adaptation for the very large dataset
        final_train_ds = combined_ds.take(n_train_final).shuffle(
            buffer_size=min(LSTM_SHUFFLE_BUFFER, n_train_final),
            seed=seed,
            reshuffle_each_iteration=True
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Skip to the last n_val for validation (no shuffle for validation)
        final_val_ds = combined_ds.skip(n_train_final).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return final_train_ds, final_val_ds, n_train_final

    def _build_lstm_model(self, n_features: int, time_steps: int) -> Any:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(time_steps, n_features)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def run_single_seed(self, seed: int) -> List[Dict]:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running seed = {seed}")
        logger.info(f"{'='*80}")
        
        results = []
        try:
            set_random_seeds(seed)
            
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return results
            
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return results
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_features, all_labels, test_size=TEST_RATIO, stratify=all_labels, random_state=seed
            )
            X_train, X_retrain, y_train, y_retrain = train_test_split(
                X_temp, y_temp, test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO), stratify=y_temp, random_state=seed
            )
            
            logger.info(f"📊 Split sizes: Train={len(X_train)}, Retrain={len(X_retrain)}, Test={len(X_test)}")
            
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            logger.info(f"✅ Preprocessing fitted on train set only (no leakage)")
            
            # 1. Classical Models Baseline
            baseline_models = self.train_baseline_classical_models(X_train_scaled, y_train)
            results.extend(self.evaluate_classical_models(baseline_models, X_test_scaled, y_test, 'baseline', seed))
            
            # 2. Classical Models Weighted
            X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
            y_combined = np.concatenate([y_train, y_retrain])
            sample_weights = self._create_sample_weights(y_combined)
            
            weighted_models = self.train_weighted_classical_models(X_combined, y_combined, sample_weights)
            results.extend(self.evaluate_classical_models(weighted_models, X_test_scaled, y_test, 'weighted_retraining', seed))
            
            # 3. LSTM Baseline
            lstm_base = self.train_and_evaluate_lstm(X_train_scaled, y_train, None, None, X_test_scaled, y_test, 'baseline', seed)
            if lstm_base:
                results.append(lstm_base)
                
            # 4. LSTM Weighted
            lstm_weight = self.train_and_evaluate_lstm(X_train_scaled, y_train, X_retrain_scaled, y_retrain, X_test_scaled, y_test, 'weighted_retraining', seed)
            if lstm_weight:
                results.append(lstm_weight)
                
            logger.info(f"✅ Seed {seed} completed: {len(results)} results")
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
        
        return results

    def train_baseline_classical_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        models = {}
        try:
            models['XGBoost'] = xgb.XGBClassifier(**XGB_CONFIG).fit(X_train, y_train)
            logger.info(f"    ✅ Trained Baseline XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Baseline XGBoost unavailable: {e}")
            
        try:
            models['RandomForest'] = RandomForestClassifier(**RF_CONFIG).fit(X_train, y_train)
            logger.info(f"    ✅ Trained Baseline RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Baseline RandomForest unavailable: {e}")
            
        try:
            svm_model = LinearSVC(**SVM_CONFIG).fit(X_train, y_train)
            models['LinearSVM'] = svm_model
            logger.info(f"    ✅ Trained Baseline LinearSVM")
            if hasattr(svm_model, 'n_iter_'):
                n_iter = svm_model.n_iter_
                n_iter_val = int(n_iter[0]) if isinstance(n_iter, (list, np.ndarray)) else int(n_iter)
                logger.info(f"    ℹ️   Baseline LinearSVM iterations: {n_iter_val}")
        except Exception as e:
            logger.warning(f"    ⚠️  Baseline LinearSVM unavailable: {e}")
            
        return models

    def train_weighted_classical_models(self, X_train: np.ndarray, y_train: np.ndarray, sample_weights: np.ndarray) -> Dict[str, Any]:
        models = {}
        try:
            models['XGBoost'] = xgb.XGBClassifier(**XGB_CONFIG).fit(X_train, y_train, sample_weight=sample_weights)
            logger.info(f"    ✅ Trained Weighted XGBoost")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted XGBoost failed: {e}")
            
        try:
            models['RandomForest'] = RandomForestClassifier(**RF_CONFIG).fit(X_train, y_train, sample_weight=sample_weights)
            logger.info(f"    ✅ Trained Weighted RandomForest")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted RandomForest failed: {e}")
            
        try:
            svm_model = LinearSVC(**SVM_CONFIG).fit(X_train, y_train, sample_weight=sample_weights)
            models['LinearSVM'] = svm_model
            logger.info(f"    ✅ Trained Weighted LinearSVM")
            if hasattr(svm_model, 'n_iter_'):
                n_iter = svm_model.n_iter_
                n_iter_val = int(n_iter[0]) if isinstance(n_iter, (list, np.ndarray)) else int(n_iter)
                logger.info(f"    ℹ️   Weighted LinearSVM iterations: {n_iter_val}")
        except Exception as e:
            logger.warning(f"    ⚠️  Weighted LinearSVM failed: {e}")
            
        return models

    def evaluate_classical_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, model_type: str, seed: int) -> List[Dict]:
        results = []
        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
                
                results.append({
                    'seed': seed, 'model_name': model_name, 'model_type': model_type,
                    'evaluation_n': int(len(y_test)),
                    'accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                    'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                    'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
                    'false_positive': int(fp), 'false_negative': int(fn),
                    'status': 'success', 'epochs_trained': None
                })
                logger.info(f"    {model_name} ({model_type}): F1={results[-1]['f1_score']:.4f}, Recall={results[-1]['recall']:.4f}, FN={fn}")
            except Exception as e:
                logger.error(f"    ❌ Failed to evaluate {model_name}: {e}")
                results.append({'seed': seed, 'model_name': model_name, 'model_type': model_type, 'evaluation_n': 0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'false_positive': 0, 'false_negative': 0, 'status': 'failed', 'reason': str(e), 'epochs_trained': None})
        return results

    def train_and_evaluate_lstm(self, X_train, y_train, X_retrain, y_retrain, X_test, y_test, model_type: str, seed: int) -> Optional[Dict]:
        # Initialize cleanup variables to None before any operation that could fail
        model = None
        train_dataset = None
        val_dataset = None
        history = None
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("    ❌ LSTM unavailable: TensorFlow not installed")
            return {
                'seed': seed, 'model_name': 'LSTM', 'model_type': model_type, 'evaluation_n': 0,
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'false_positive': 0, 'false_negative': 0, 'status': 'failed', 'reason': 'TensorFlow not available', 'epochs_trained': None
            }
        
        try:
            n_features = X_train.shape[1]
            time_steps = 10
            
            # Test sequences (lazily generated)
            n_test_seq = len(X_test) - time_steps + 1
            logger.info(f"    ℹ️   LSTM Test sequences: {n_test_seq}")
            
            X_test_tf = tf.constant(X_test, dtype=tf.float32)
            y_test_tf = tf.constant(y_test, dtype=tf.float32)
            
            def test_map_fn(i):
                return X_test_tf[i : i + time_steps], y_test_tf[i + time_steps - 1]
            
            test_dataset = tf.data.Dataset.range(n_test_seq).map(test_map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(64).prefetch(tf.data.AUTOTUNE)
            
            if model_type == 'baseline':
                w_train = np.ones(len(y_train), dtype=np.float32)
                train_dataset, val_dataset, n_train_final = self._create_lstm_tf_dataset(
                    X_train, y_train, w_train, sequence_length=time_steps, batch_size=64, seed=seed, val_split=0.1
                )
                
                tf.keras.backend.clear_session()
                gc.collect()
                tf.keras.utils.set_random_seed(seed)
                
                model = self._build_lstm_model(n_features, time_steps)
                
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=50,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                    verbose=0
                )
                
            else:  # weighted_retraining
                w_train = np.ones(len(y_train), dtype=np.float32)
                w_train[y_train == 1] = 3.0
                
                w_retrain = np.ones(len(y_retrain), dtype=np.float32)
                w_retrain[y_retrain == 1] = 3.0
                
                train_dataset, val_dataset, n_train_final = self._create_combined_lstm_tf_dataset(
                    X_train, y_train, w_train,
                    X_retrain, y_retrain, w_retrain,
                    sequence_length=time_steps, batch_size=64, seed=seed, val_split=0.1
                )
                
                tf.keras.backend.clear_session()
                gc.collect()
                tf.keras.utils.set_random_seed(seed)
                
                model = self._build_lstm_model(n_features, time_steps)
                
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=50,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                    verbose=0
                )
            
            epochs_trained = len(history.history['loss'])
            logger.info(f"    ℹ️   LSTM {model_type} trained for {epochs_trained} epochs")
            
            # Evaluate on test sequences
            y_pred_proba = model.predict(test_dataset, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            evaluation_n = n_test_seq
            
            # True labels correspond to endpoints starting from index time_steps - 1
            y_test_seq = y_test[time_steps-1:]
            
            cm = confusion_matrix(y_test_seq, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            
            # Create result dict BEFORE deleting anything
            result = {
                'seed': seed, 'model_name': 'LSTM', 'model_type': model_type,
                'evaluation_n': int(evaluation_n),
                'accuracy': round(accuracy_score(y_test_seq, y_pred), 4),
                'precision': round(precision_score(y_test_seq, y_pred, zero_division=0), 4),
                'recall': round(recall_score(y_test_seq, y_pred, zero_division=0), 4),
                'f1_score': round(f1_score(y_test_seq, y_pred, zero_division=0), 4),
                'false_positive': int(fp), 'false_negative': int(fn),
                'status': 'success', 'epochs_trained': epochs_trained
            }
            
        except Exception as e:
            logger.error(f"    ❌ LSTM {model_type} failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'seed': seed, 'model_name': 'LSTM', 'model_type': model_type, 'evaluation_n': 0,
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                'false_positive': 0, 'false_negative': 0, 'status': 'failed', 'reason': str(e), 'epochs_trained': None
            }
            
        finally:
            # ALWAYS cleanup after LSTM (success or failure)
            # Wrapped in try/except to ensure cleanup errors do not mask the original exception
            try:
                if model is not None:
                    del model
                if train_dataset is not None:
                    del train_dataset
                if val_dataset is not None:
                    del val_dataset
                tf.keras.backend.clear_session()
                gc.collect()
            except Exception as cleanup_err:
                logger.warning(f"⚠️  Cleanup warning: {cleanup_err}")
        
        return result

    def _verify_completeness(self) -> Tuple[bool, List[str]]:
        missing_items = []
        if not self.results:
            missing_items.append("No results generated")
            return False, missing_items
        
        results_df = pd.DataFrame(self.results)
        successful = results_df[results_df['status'] == 'success']
        
        expected_combinations = {
            (model, seed, variant)
            for model in MODELS_OF_INTEREST
            for seed in SEEDS
            for variant in ['baseline', 'weighted_retraining']
        }
        
        actual_combinations = {
            (row['model_name'], row['seed'], row['model_type'])
            for _, row in successful.iterrows()
        }
        
        missing_combinations = expected_combinations - actual_combinations
        if missing_combinations:
            missing_items.append(f"Missing {len(missing_combinations)} expected combinations: {missing_combinations}")
            
        unexpected_combinations = actual_combinations - expected_combinations
        if unexpected_combinations:
            missing_items.append(f"Found {len(unexpected_combinations)} unexpected combinations: {unexpected_combinations}")
            
        if len(actual_combinations) != len(successful):
            missing_items.append("Duplicate model+seed+variant combinations found in successful results")
            
        if len(actual_combinations) != EXPECTED_RECORD_COUNT:
            missing_items.append(f"Expected {EXPECTED_RECORD_COUNT} successful unique records, got {len(actual_combinations)}")
            
        return len(missing_items) == 0 and len(actual_combinations) == EXPECTED_RECORD_COUNT, missing_items

    def compute_statistics(self) -> Dict:
        logger.info("\n📊 Computing statistics...")
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
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
                        'std_fn': round(type_data['false_negative'].std(), 2),
                        'mean_fp': round(type_data['false_positive'].mean(), 2),
                        'std_fp': round(type_data['false_positive'].std(), 2),
                        'mean_evaluation_n': round(type_data['evaluation_n'].mean(), 2)
                    }
            
            if 'baseline' in statistics[model_name] and 'weighted_retraining' in statistics[model_name]:
                baseline = statistics[model_name]['baseline']
                weighted = statistics[model_name]['weighted_retraining']
                statistics[model_name]['improvement'] = {
                    'f1_improvement': round(weighted['mean_f1'] - baseline['mean_f1'], 4),
                    'recall_improvement': round(weighted['mean_recall'] - baseline['mean_recall'], 4),
                    'fn_reduction': round(baseline['mean_fn'] - weighted['mean_fn'], 2),
                    'fp_change': round(weighted['mean_fp'] - baseline['mean_fp'], 2)
                }
                
        self.statistics = statistics
        return statistics

    def save_results(self) -> List[str]:
        logger.info("💾 Saving results...")
        saved_paths = []
        try:
            if self.results:
                csv_path = self.validation_dir / 'metrics' / 'multi_seed_four_model_results.csv'
                pd.DataFrame(self.results).to_csv(csv_path, index=False)
                saved_paths.append(str(csv_path))
                logger.info(f"   ✅ Saved CSV: {csv_path}")
            
            if self.statistics:
                summary = {
                    'validation_status': self.validation_status,
                    'tested_seeds': SEEDS,
                    'evaluated_models': list(self.statistics.keys()),
                    'expected_models': MODELS_OF_INTEREST,
                    'expected_record_count': EXPECTED_RECORD_COUNT,
                    'successful_record_count': len([r for r in self.results if r.get('status') == 'success']),
                    'normal_weight': NORMAL_WEIGHT,
                    'anomaly_weight': ANOMALY_WEIGHT,
                    'time_steps': 10,
                    'lstm_threshold': 0.5,
                    'statistics': self.statistics,
                    'total_experiments': len(self.results),
                    'experiment_scope': 'four_model_lstm_controlled_extension',
                    'scientific_interpretation': 'deep-learning comparator under the retained data representation',
                    'evaluation_population_note': 'Classical models use full test rows. LSTM uses valid 10-row sliding-window endpoints only. No padding is used. First 9 LSTM test rows are excluded from LSTM metrics.'
                }
                json_path = self.validation_dir / 'metrics' / 'multi_seed_four_model_summary.json'
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                saved_paths.append(str(json_path))
                logger.info(f"   ✅ Saved JSON: {json_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
        return saved_paths

    def run_full_validation(self) -> Dict:
        logger.info("\n" + "="*80)
        logger.info("🔢 RUNNING MULTI-SEED STABILITY ANALYSIS (FOUR-MODEL EXTENSION)")
        logger.info("="*80)
        
        results = {'status': 'pending', 'outputs_saved': [], 'validation_complete': False}
        
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        for seed in SEEDS:
            self.results.extend(self.run_single_seed(seed))
            
        if not self.results:
            results['status'] = 'failed'
            return results
        
        is_complete, missing_items = self._verify_completeness()
        if not is_complete:
            logger.error("❌ Completeness check FAILED:")
            for item in missing_items:
                logger.error(f"   - {item}")
            results['status'] = 'incomplete'
        else:
            logger.info("✅ Completeness check PASSED: 40 successful unique records")
            
        self.compute_statistics()
        
        if is_complete:
            self.validation_status = 'complete'
            results['status'] = 'complete'
            results['validation_complete'] = True
        else:
            self.validation_status = 'incomplete'
            results['status'] = 'incomplete'
            results['validation_complete'] = False
            
        results['outputs_saved'].extend(self.save_results())
        return results

# =============================================================================
# GLOBAL LOGGER & MAIN EXECUTION
# =============================================================================
logger = None

def main():
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'multi_seed_validation_four_model')
    
    logger.info("\n" + "="*80)
    logger.info("🔢 MULTI-SEED STABILITY ANALYSIS (FOUR-MODEL EXTENSION)")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    validation_dir = base_dir / 'adaptive' / 'final_validation_four_model'
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    validator = MultiSeedValidatorFourModel(base_dir=base_dir, validation_dir=validation_dir)
    results = validator.run_full_validation()
    
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🎉 MULTI-SEED VALIDATION (FOUR-MODEL) COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"✅ Validation complete: {results['validation_complete']}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)

if __name__ == '__main__':
    main()