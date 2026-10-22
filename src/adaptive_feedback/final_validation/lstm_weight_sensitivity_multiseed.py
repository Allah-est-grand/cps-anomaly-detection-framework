"""
LSTM Weight Sensitivity Multi-Seed Validation
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
Confirm the LSTM anomaly-weight candidate across 5 seeds using internal validation only,
while keeping every final test split untouched.

SEEDS: [42, 123, 2024, 2025, 3407]
ANOMALY WEIGHTS: [1.0, 1.5, 2.0, 2.5, 3.0]

IMPORTANT:
- Seed 42 is reused from existing pilot after strict validation.
- Final test set is completely preserved and NOT used for weight selection.
- This is a 'selected global candidate weight', NOT an 'optimal weight'.
- Checkpoint resume supported with strict validation.
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
import sys
import time
import logging
import json
import traceback
import gc
import hashlib
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
    logging.warning("⚠️  TensorFlow not available. LSTM multi-seed validation cannot run.")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 2024, 2025, 3407]
ANOMALY_WEIGHTS = [1.0, 1.5, 2.0, 2.5, 3.0]
NORMAL_WEIGHT = 1.0

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2
INTERNAL_VAL_RATIO = 0.3

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

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']
METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

# =============================================================================
# MULTI-SEED VALIDATION CLASS
# =============================================================================
class LSTMWeightSensitivityMultiseed:
    """
    Confirms LSTM anomaly-weight candidate across 5 seeds using internal validation only.
    """
    
    def __init__(self, base_dir: Path, multiseed_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.multiseed_dir = Path(multiseed_dir) if multiseed_dir else (
            self.base_dir / 'adaptive' / 'lstm_weight_sensitivity_multiseed'
        )
        
        (self.multiseed_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.multiseed_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.multiseed_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.all_results = []
        self.summary_df = None
        self.selected_global_weight = None
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        self.completed_seeds = []
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        self.pilot_dir = self.base_dir / 'adaptive' / 'lstm_weight_sensitivity_pilot'
        self.checkpoint_path = self.multiseed_dir / 'checkpoints' / 'multiseed_checkpoint.json'
        
        logger.info(f"✅ Initialized LSTMWeightSensitivityMultiseed")
        logger.info(f"📁 Multi-seed directory: {self.multiseed_dir}")
        logger.info(f"🔢 Testing seeds: {SEEDS}")
    
    def load_resources(self) -> bool:
        """Load dataset and compute fingerprints."""
        logger.info("📊 Loading resources...")
        if not self.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.dataset_path}")
            return False
        
        try:
            self.original_dataset = pd.read_csv(self.dataset_path)
            self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
            logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows, "
                       f"{len(self.original_dataset.columns)} columns")
            
            hasher = hashlib.sha256()
            with open(self.dataset_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
            self.dataset_fingerprint = hasher.hexdigest()[:32]
            
            config_dict = {
                'seeds': SEEDS,
                'anomaly_weights': ANOMALY_WEIGHTS,
                'time_steps': TIME_STEPS,
                'epochs': LSTM_EPOCHS,
                'batch_size': LSTM_BATCH_SIZE,
                'patience': PATIENCE,
                'imputation': 'mean',
                'scaler': 'StandardScaler'
            }
            config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
            self.config_fingerprint = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:32]
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def validate_seed_42_pilot(self) -> Tuple[bool, List[Dict]]:
        """Validate existing seed 42 pilot results."""
        logger.info("🔍 Validating existing seed 42 pilot results...")
        summary_path = self.pilot_dir / 'metrics' / 'weight_sensitivity_summary.json'
        
        if not summary_path.exists():
            logger.error("❌ Seed 42 pilot summary not found. Cannot proceed.")
            return False, []
            
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            validation_passed = True
            missing_fields = []
            
            if summary.get('seed') != 42:
                logger.error(f"❌ Seed 42 pilot has wrong seed: {summary.get('seed')}")
                validation_passed = False
                
            if summary.get('tested_anomaly_weights') != ANOMALY_WEIGHTS:
                logger.error(f"❌ Seed 42 pilot has wrong weights: {summary.get('tested_anomaly_weights')}")
                validation_passed = False
                
            if summary.get('dataset_fingerprint') != self.dataset_fingerprint:
                logger.error(f"❌ Seed 42 pilot has wrong dataset fingerprint: {summary.get('dataset_fingerprint')}")
                validation_passed = False
                
            if not summary.get('final_test_preserved', False):
                logger.error("❌ Seed 42 pilot did not preserve final test.")
                validation_passed = False
                
            weight_results = summary.get('weight_results', [])
            if len(weight_results) != 5:
                logger.error(f"❌ Seed 42 pilot has {len(weight_results)} weight records, expected 5.")
                validation_passed = False
                
            valid_count = sum(1 for w in weight_results if w.get('metrics_valid', True))
            if valid_count != 5:
                logger.error(f"❌ Seed 42 pilot has {valid_count} valid weight records, expected 5.")
                validation_passed = False
                
            meta = summary.get('split_metadata', {})
            expected_meta = {
                'time_steps': TIME_STEPS,
                'epochs': LSTM_EPOCHS,
                'batch_size': LSTM_BATCH_SIZE,
                'patience': PATIENCE,
                'imputation': 'mean',
                'scaler': 'StandardScaler',
                'train_ratio': TRAIN_RATIO,
                'retrain_ratio': RETRAIN_RATIO,
                'test_ratio': TEST_RATIO,
                'internal_val_ratio': INTERNAL_VAL_RATIO
            }
            
            for key, expected_val in expected_meta.items():
                if key in meta and meta[key] != expected_val:
                    logger.error(f"❌ Seed 42 pilot has wrong {key}: {meta.get(key)} (expected {expected_val})")
                    validation_passed = False
                elif key not in meta:
                    missing_fields.append(key)
                    
            if missing_fields:
                logger.warning(f"⚠️  Seed 42 pilot metadata missing fields: {missing_fields}. Validating only available fields.")
                
            if not validation_passed:
                return False, []
                
            logger.info("✅ Seed 42 pilot validation PASSED.")
            
            formatted_results = []
            for w in weight_results:
                formatted_results.append({
                    'seed': 42,
                    'anomaly_weight': w['anomaly_weight'],
                    'validation_accuracy': w['validation_accuracy'],
                    'validation_precision': w['validation_precision'],
                    'validation_recall': w['validation_recall'],
                    'validation_f1_score': w['validation_f1_score'],
                    'validation_false_positive': w['validation_false_positive'],
                    'validation_false_negative': w['validation_false_negative'],
                    'validation_true_positive': w['validation_true_positive'],
                    'validation_true_negative': w['validation_true_negative'],
                    'epochs_trained': w['epochs_trained'],
                    'final_test_used': False,
                    'source': 'existing_pilot',
                    'metrics_valid': True,
                    'split_metadata': meta
                })
            return True, formatted_results
            
        except Exception as e:
            logger.error(f"❌ Failed to validate seed 42 pilot: {e}")
            return False, []
    
    def load_checkpoint(self) -> Tuple[str, List[Dict], List[int]]:
        """
        Load and validate checkpoint if exists.
        
        Returns:
            Tuple of (status, results, completed_seeds)
            status: "missing", "valid", or "invalid"
        """
        logger.info("🔍 Checking for existing checkpoint...")
        
        if not self.checkpoint_path.exists():
            logger.info("ℹ️   No checkpoint found. Starting fresh.")
            return "missing", [], []
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # FIX #2: Validate BOTH fingerprints
            if checkpoint.get('dataset_fingerprint') != self.dataset_fingerprint:
                logger.error(f"❌ Checkpoint dataset fingerprint mismatch!")
                logger.error(f"   Checkpoint: {checkpoint.get('dataset_fingerprint')}")
                logger.error(f"   Current: {self.dataset_fingerprint}")
                return "invalid", [], []
            
            if checkpoint.get('config_fingerprint') != self.config_fingerprint:
                logger.error(f"❌ Checkpoint config fingerprint mismatch!")
                logger.error(f"   Checkpoint: {checkpoint.get('config_fingerprint')}")
                logger.error(f"   Current: {self.config_fingerprint}")
                return "invalid", [], []
            
            # Validate all stored records
            stored_results = checkpoint.get('results', [])
            completed_seeds = checkpoint.get('completed_seeds', [])
            
            # Verify seed 42 is present
            if 42 not in completed_seeds:
                logger.error("❌ Checkpoint missing seed 42!")
                return "invalid", [], []
            
            # Verify each completed seed has exactly 5 weights
            for seed in completed_seeds:
                seed_results = [r for r in stored_results if r.get('seed') == seed]
                if len(seed_results) != 5:
                    logger.error(f"❌ Seed {seed} has {len(seed_results)} records, expected 5!")
                    return "invalid", [], []
                
                # Verify all metrics_valid == True
                if not all(r.get('metrics_valid', False) for r in seed_results):
                    logger.error(f"❌ Seed {seed} has invalid metrics!")
                    return "invalid", [], []
                
                # Verify all final_test_used == False
                if not all(r.get('final_test_used', True) == False for r in seed_results):
                    logger.error(f"❌ Seed {seed} has final_test_used == True!")
                    return "invalid", [], []
                
                # Verify weights are exactly [1.0, 1.5, 2.0, 2.5, 3.0]
                seed_weights = sorted([r.get('anomaly_weight') for r in seed_results])
                if seed_weights != ANOMALY_WEIGHTS:
                    logger.error(f"❌ Seed {seed} has wrong weights: {seed_weights}")
                    return "invalid", [], []
            
            # Check for duplicate (seed, anomaly_weight) pairs
            pairs = [(r.get('seed'), r.get('anomaly_weight')) for r in stored_results]
            if len(pairs) != len(set(pairs)):
                logger.error("❌ Checkpoint has duplicate (seed, anomaly_weight) pairs!")
                return "invalid", [], []
            
            logger.info(f"✅ Checkpoint validated: {len(completed_seeds)} seeds completed")
            logger.info(f"   Completed seeds: {completed_seeds}")
            
            return "valid", stored_results, completed_seeds
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return "invalid", [], []
    
    def save_checkpoint(self):
        """Save checkpoint after each seed completes."""
        logger.info("💾 Saving checkpoint...")
        
        # FIX #3: Enforce validation before saving - raise ValueError if fails
        # Verify no duplicate (seed, anomaly_weight) pairs
        pairs = [(r['seed'], r['anomaly_weight']) for r in self.all_results]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Cannot save checkpoint: duplicate (seed, anomaly_weight) pairs detected!")
        
        # Verify each completed seed has exactly 5 records
        for seed in self.completed_seeds:
            seed_results = [r for r in self.all_results if r['seed'] == seed]
            if len(seed_results) != 5:
                raise ValueError(f"Cannot save checkpoint: seed {seed} has {len(seed_results)} records, expected 5!")
        
        # Verify all records are valid
        if not all(r.get('metrics_valid', False) for r in self.all_results):
            raise ValueError("Cannot save checkpoint: invalid metrics detected!")
        
        # Verify all final_test_used == False
        if not all(r.get('final_test_used', True) == False for r in self.all_results):
            raise ValueError("Cannot save checkpoint: final_test_used == True detected!")
        
        checkpoint_data = {
            'dataset_fingerprint': self.dataset_fingerprint,
            'config_fingerprint': self.config_fingerprint,
            'completed_seeds': self.completed_seeds,
            'results': self.all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.multiseed_dir / 'checkpoints' / 'multiseed_checkpoint.json'
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"✅ Checkpoint saved: {len(self.completed_seeds)} seeds")
    
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
        
        return df[feature_cols].values, feature_cols
    
    def _create_sequences_efficient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    def _create_lstm_model(self, n_features: int) -> Any:
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
        weights = np.ones(len(labels)) * NORMAL_WEIGHT
        weights[labels == 1] = anomaly_weight
        return weights
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics_valid = (
            np.isfinite(accuracy) and np.isfinite(precision) and 
            np.isfinite(recall) and np.isfinite(f1) and
            (0 <= accuracy <= 1) and (0 <= precision <= 1) and 
            (0 <= recall <= 1) and (0 <= f1 <= 1)
        )
        
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
    
    def run_single_seed(self, seed: int) -> Tuple[bool, List[Dict], Dict]:
        """Run weight sensitivity analysis for a single seed."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING SEED {seed}")
        logger.info(f"{'='*80}")
        
        seed_results = []
        split_metadata = {}
        try:
            set_random_seeds(seed)
            
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                raise ValueError("No label column found")
                
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                raise ValueError("No features found")
                
            n_features = len(feature_cols)
            
            all_indices = np.arange(len(self.original_dataset))
            
            # 1. Train/Test split with indices
            X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
                all_features, all_labels, all_indices,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=seed
            )
            
            # 2. Train/Retrain split with indices
            X_train, X_retrain, y_train, y_retrain, idx_train, idx_retrain = train_test_split(
                X_temp, y_temp, idx_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=seed
            )
            
            # Preprocessing on X_train ONLY
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            
            # 3. Retrain fit/val split with indices
            X_retrain_fit, X_retrain_val, y_retrain_fit, y_retrain_val, idx_retrain_fit, idx_retrain_val = train_test_split(
                X_retrain_scaled, y_retrain, idx_retrain,
                test_size=INTERNAL_VAL_RATIO,
                stratify=y_retrain,
                random_state=seed
            )
            
            # Extended disjointness verification
            train_set = set(idx_train)
            test_set = set(idx_test)
            retrain_fit_set = set(idx_retrain_fit)
            retrain_val_set = set(idx_retrain_val)
            retrain_set = set(idx_retrain)
            
            disjoint_pairs = [
                (train_set, retrain_fit_set, 'train vs retrain_fit'),
                (train_set, retrain_val_set, 'train vs retrain_val'),
                (train_set, test_set, 'train vs test'),
                (retrain_fit_set, retrain_val_set, 'retrain_fit vs retrain_val'),
                (retrain_fit_set, test_set, 'retrain_fit vs test'),
                (retrain_val_set, test_set, 'retrain_val vs test')
            ]
            
            for set1, set2, pair_name in disjoint_pairs:
                if set1.intersection(set2):
                    raise ValueError(f"{pair_name} indices overlap!")
            
            # Verify union(retrain_fit, retrain_val) == retrain
            if retrain_fit_set.union(retrain_val_set) != retrain_set:
                raise ValueError("union(retrain_fit, retrain_val) != retrain!")
            
            logger.info("✅ All disjointness checks passed.")
            
            # Create sequences separately
            X_train_seq, y_train_seq = self._create_sequences_efficient(X_train_scaled, y_train)
            X_retrain_fit_seq, y_retrain_fit_seq = self._create_sequences_efficient(X_retrain_fit, y_retrain_fit)
            X_retrain_val_seq, y_retrain_val_seq = self._create_sequences_efficient(X_retrain_val, y_retrain_val)
            
            # Combine train and retrain_fit
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
            
            # Save exact per-seed split metadata
            split_metadata = {
                'seed': seed,
                'train_samples': len(X_train),
                'retrain_samples': len(X_retrain),
                'retrain_fit_samples': len(X_retrain_fit),
                'retrain_val_samples': len(X_retrain_val),
                'test_samples': len(X_test),
                'train_normal': int(np.sum(y_train == 0)),
                'train_anomaly': int(np.sum(y_train == 1)),
                'retrain_fit_normal': int(np.sum(y_retrain_fit == 0)),
                'retrain_fit_anomaly': int(np.sum(y_retrain_fit == 1)),
                'retrain_val_normal': int(np.sum(y_retrain_val == 0)),
                'retrain_val_anomaly': int(np.sum(y_retrain_val == 1)),
                'combined_sequences': len(X_combined_seq),
                'val_sequences': len(X_retrain_val_seq),
                'n_features': n_features,
                'time_steps': TIME_STEPS,
                'epochs': LSTM_EPOCHS,
                'batch_size': LSTM_BATCH_SIZE,
                'patience': PATIENCE,
                'imputation': 'mean',
                'scaler': 'StandardScaler',
                'train_ratio': TRAIN_RATIO,
                'retrain_ratio': RETRAIN_RATIO,
                'test_ratio': TEST_RATIO,
                'internal_val_ratio': INTERNAL_VAL_RATIO
            }
            
            # Test each anomaly weight
            for anomaly_weight in ANOMALY_WEIGHTS:
                tf.keras.backend.clear_session()
                gc.collect()
                set_random_seeds(seed)
                tf.keras.utils.set_random_seed(seed)
                
                if anomaly_weight == 1.0:
                    sample_weights = np.ones(len(y_combined_seq))
                else:
                    sample_weights = self._create_sample_weights(y_combined_seq, anomaly_weight)
                    
                model = self._create_lstm_model(n_features)
                
                history = model.fit(
                    X_combined_seq, y_combined_seq,
                    sample_weight=sample_weights,
                    epochs=LSTM_EPOCHS,
                    batch_size=LSTM_BATCH_SIZE,
                    validation_split=0.1,
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)],
                    verbose=0
                )
                
                y_val_pred_proba = model.predict(X_retrain_val_seq, verbose=0).flatten()
                y_val_pred = (y_val_pred_proba > PREDICTION_THRESHOLD).astype(int)
                
                val_metrics = self._evaluate_predictions(y_retrain_val_seq, y_val_pred)
                
                if not val_metrics['metrics_valid']:
                    logger.warning(f"⚠️  Seed {seed} Weight {anomaly_weight}: Invalid metrics")
                    weight_result = {
                        'seed': seed,
                        'anomaly_weight': anomaly_weight,
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
                        'source': 'newly_executed',
                        'metrics_valid': False,
                        'split_metadata': split_metadata
                    }
                else:
                    weight_result = {
                        'seed': seed,
                        'anomaly_weight': anomaly_weight,
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
                        'source': 'newly_executed',
                        'metrics_valid': True,
                        'split_metadata': split_metadata
                    }
                seed_results.append(weight_result)
                
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                
            return True, seed_results, split_metadata
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
            return False, [], {}
    
    def aggregate_and_select(self):
        """Aggregate results across all 5 seeds and select global candidate."""
        logger.info("📊 Aggregating results and selecting global candidate...")
        
        valid_results = [r for r in self.all_results if r.get('metrics_valid', True)]
        
        summary_rows = []
        for weight in ANOMALY_WEIGHTS:
            weight_data = [r for r in valid_results if r['anomaly_weight'] == weight]
            
            if len(weight_data) != 5:
                logger.error(f"❌ Weight {weight} has {len(weight_data)} valid results, expected 5. STOPPING.")
                raise ValueError(f"Incomplete results for weight {weight}")
                
            f1_vals = [r['validation_f1_score'] for r in weight_data]
            prec_vals = [r['validation_precision'] for r in weight_data]
            rec_vals = [r['validation_recall'] for r in weight_data]
            fn_vals = [r['validation_false_negative'] for r in weight_data]
            fp_vals = [r['validation_false_positive'] for r in weight_data]
            
            summary_rows.append({
                'anomaly_weight': weight,
                'mean_f1': float(np.mean(f1_vals)),
                'std_f1': float(np.std(f1_vals, ddof=1)),
                'mean_precision': float(np.mean(prec_vals)),
                'std_precision': float(np.std(prec_vals, ddof=1)),
                'mean_recall': float(np.mean(rec_vals)),
                'std_recall': float(np.std(rec_vals, ddof=1)),
                'mean_fn': float(np.mean(fn_vals)),
                'std_fn': float(np.std(fn_vals, ddof=1)),
                'mean_fp': float(np.mean(fp_vals)),
                'std_fp': float(np.std(fp_vals, ddof=1)),
                'seeds_present': 5
            })
            
        self.summary_df = pd.DataFrame(summary_rows)
        
        # Global candidate selection
        best_row = max(
            summary_rows,
            key=lambda x: (x['mean_f1'], x['mean_recall'], -x['anomaly_weight'])
        )
        self.selected_global_weight = best_row['anomaly_weight']
        logger.info(f"✅ Selected global candidate weight: {self.selected_global_weight}")
    
    def save_results(self):
        """Save all outputs."""
        logger.info("💾 Saving results...")
        
        results_df = pd.DataFrame(self.all_results)
        results_csv = self.multiseed_dir / 'metrics' / 'lstm_weight_multiseed_results.csv'
        results_df.to_csv(results_csv, index=False)
        
        summary_csv = self.multiseed_dir / 'metrics' / 'lstm_weight_multiseed_summary.csv'
        self.summary_df.to_csv(summary_csv, index=False)
        
        summary_json = self.multiseed_dir / 'metrics' / 'lstm_weight_multiseed_summary.json'
        summary_dict = {
            'selected_global_candidate_weight': self.selected_global_weight,
            'aggregated_statistics': self.summary_df.to_dict(orient='records'),
            'dataset_fingerprint': self.dataset_fingerprint,
            'config_fingerprint': self.config_fingerprint,
            'scientific_note': 'Global candidate weight selected based on mean validation F1 across 5 seeds. Final test set was NOT used for selection.'
        }
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2)
            
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive report."""
        logger.info("📝 Generating report...")
        report = []
        report.append("="*80)
        report.append("LSTM WEIGHT SENSITIVITY MULTI-SEED VALIDATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("EXPERIMENTAL CONFIGURATION:")
        report.append(f"  - Seeds: {SEEDS}")
        report.append(f"  - Anomaly weights: {ANOMALY_WEIGHTS}")
        report.append(f"  - Normal weight: {NORMAL_WEIGHT}")
        report.append(f"  - Time steps: {TIME_STEPS}")
        report.append(f"  - Epochs: {LSTM_EPOCHS}, Batch size: {LSTM_BATCH_SIZE}, Patience: {PATIENCE}")
        report.append("")
        report.append("AGGREGATED RESULTS:")
        for _, row in self.summary_df.iterrows():
            report.append(f"  - Weight {row['anomaly_weight']}:")
            report.append(f"      Mean F1: {row['mean_f1']:.4f} ± {row['std_f1']:.4f}")
            report.append(f"      Mean Precision: {row['mean_precision']:.4f}")
            report.append(f"      Mean Recall: {row['mean_recall']:.4f}")
            report.append(f"      Mean FN: {row['mean_fn']:.2f} ± {row['std_fn']:.2f}")
            report.append(f"      Mean FP: {row['mean_fp']:.2f} ± {row['std_fp']:.2f}")
        report.append("")
        report.append(f"SELECTED GLOBAL CANDIDATE WEIGHT: {self.selected_global_weight}")
        report.append("")
        report.append("SCIENTIFIC NOTES:")
        report.append("  - Final test set was completely preserved and NOT used for weight selection.")
        report.append("  - Weight selection based on mean validation F1 across 5 seeds.")
        report.append("  - This is a 'selected global candidate weight', NOT an 'optimal weight'.")
        report.append("  - Final test evaluation will be performed in a separate experiment.")
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        report_path = self.multiseed_dir / 'reports' / 'lstm_weight_multiseed_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"💾 Saved report: {report_path}")
    
    def run_full_validation(self) -> Dict:
        """Run complete multi-seed validation with checkpoint resume support."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING LSTM WEIGHT SENSITIVITY MULTI-SEED VALIDATION")
        logger.info("="*80)
        
        if not self.load_resources():
            return {'status': 'failed', 'error': 'Failed to load resources'}
            
        # Validate seed 42 pilot
        valid_42, seed_42_results = self.validate_seed_42_pilot()
        if not valid_42:
            logger.error("❌ Seed 42 validation failed. STOPPING execution.")
            return {'status': 'failed', 'error': 'Seed 42 validation failed'}
            
        # FIX #1: Load checkpoint with three-state status
        checkpoint_status, checkpoint_results, checkpoint_completed_seeds = self.load_checkpoint()
        
        if checkpoint_status == "invalid":
            # FIX #1: STOP execution if checkpoint exists but is invalid
            logger.error("❌ Checkpoint exists but is INVALID. STOPPING execution.")
            logger.error("❌ Please remove or archive the invalid checkpoint before rerunning.")
            return {'status': 'failed', 'error': 'Invalid checkpoint detected'}
        
        elif checkpoint_status == "valid":
            # Restore from checkpoint
            logger.info("📥 Restoring from valid checkpoint...")
            self.all_results = checkpoint_results
            self.completed_seeds = checkpoint_completed_seeds
            
            # Verify seed 42 is not duplicated
            seed_42_in_checkpoint = any(r['seed'] == 42 for r in self.all_results)
            if not seed_42_in_checkpoint:
                logger.error("❌ Checkpoint missing seed 42 results!")
                return {'status': 'failed', 'error': 'Checkpoint missing seed 42'}
            
            logger.info(f"✅ Restored {len(self.completed_seeds)} completed seeds: {self.completed_seeds}")
            
            # Determine which seeds still need to run
            seeds_to_run = [s for s in SEEDS if s not in self.completed_seeds]
            logger.info(f"📊 Seeds remaining: {seeds_to_run}")
        else:
            # checkpoint_status == "missing": Start fresh with seed 42 from pilot
            self.all_results.extend(seed_42_results)
            self.completed_seeds = [42]
            seeds_to_run = [s for s in SEEDS if s != 42]
            logger.info(f"📊 Starting fresh. Seeds to run: {seeds_to_run}")
        
        # Run remaining seeds
        for seed in seeds_to_run:
            success, seed_results, split_meta = self.run_single_seed(seed)
            if not success or not seed_results:
                logger.error(f"❌ Seed {seed} failed to produce results. STOPPING.")
                return {'status': 'failed', 'error': f'Seed {seed} failed'}
                
            self.all_results.extend(seed_results)
            self.completed_seeds.append(seed)
            
            # FIX #3: Save checkpoint after each seed - raises ValueError if validation fails
            try:
                self.save_checkpoint()
            except ValueError as e:
                logger.error(f"❌ Checkpoint validation failed: {e}")
                logger.error("❌ STOPPING execution to prevent saving invalid checkpoint.")
                return {'status': 'failed', 'error': f'Checkpoint validation failed: {e}'}
        
        try:
            self.aggregate_and_select()
            self.save_results()
        except ValueError as e:
            logger.error(f"❌ Aggregation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
            
        print("\n" + "="*80)
        print("MULTI-SEED AGGREGATION SUMMARY")
        print("="*80)
        print(f"{'Weight':<8} | {'Mean F1 ± SD':<15} | {'Mean Precision':<15} | {'Mean Recall':<12} | {'Mean FN':<10} | {'Mean FP':<10}")
        print("-" * 85)
        for _, row in self.summary_df.iterrows():
            print(f"{row['anomaly_weight']:<8.1f} | {row['mean_f1']:.4f} ± {row['std_f1']:.4f} | {row['mean_precision']:.4f}        | {row['mean_recall']:.4f}       | {row['mean_fn']:<10.2f} | {row['mean_fp']:<10.2f}")
        print("="*80)
        print(f"Selected global candidate weight: {self.selected_global_weight}")
        print("="*80 + "\n")
        
        return {'status': 'complete', 'selected_weight': self.selected_global_weight}


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute multi-seed weight sensitivity validation"""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'lstm_weight_sensitivity_multiseed')
    
    logger.info("\n" + "="*80)
    logger.info("⚖️  LSTM WEIGHT SENSITIVITY MULTI-SEED VALIDATION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    multiseed_dir = base_dir / 'adaptive' / 'lstm_weight_sensitivity_multiseed'
    multiseed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Multi-seed directory: {multiseed_dir}")
    logger.info(f"⚠️  Final test set is PRESERVED (not used for weight selection)")
    logger.info(f"⚠️  Weight selection based on internal validation only")
    logger.info(f"⚠️  Checkpoint resume supported with strict validation")
    
    validator = LSTMWeightSensitivityMultiseed(base_dir=base_dir, multiseed_dir=multiseed_dir)
    results = validator.run_full_validation()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 MULTI-SEED VALIDATION COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results.get('status', 'unknown')}")
    logger.info(f"⚖️  Selected global candidate weight: {results.get('selected_weight', 'N/A')}")
    logger.info("="*80)


if __name__ == '__main__':
    main()