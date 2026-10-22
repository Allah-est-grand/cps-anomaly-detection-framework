"""
Four-Model Error Analysis Module - Sample-Level Error Transition Analysis
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module implements sample-level error transition analysis for FOUR models:
1. XGBoost
2. RandomForest
3. LinearSVM
4. LSTM

Analyzes error transitions between:
- BASELINE (train only)
- FINAL SELECTED ADAPTIVE RETRAINING CONFIGURATION

For LSTM, the selected adaptive configuration uses anomaly weight = 1.0
(as validated in the weight-sensitivity experiment).

OUTPUT FILES:
-------------
- results/adaptive/final_validation_four_model/error_analysis/metrics/four_model_error_analysis_results.csv
- results/adaptive/final_validation_four_model/error_analysis/metrics/four_model_error_transitions.csv
- results/adaptive/final_validation_four_model/error_analysis/metrics/four_model_error_analysis_summary.json
- results/adaptive/final_validation_four_model/error_analysis/figures/error_transition_summary.png
- results/adaptive/final_validation_four_model/error_analysis/figures/false_negative_correction.png
- results/adaptive/final_validation_four_model/error_analysis/figures/false_positive_tradeoff.png
- results/adaptive/final_validation_four_model/error_analysis/figures/persistent_errors_by_model.png
- results/adaptive/final_validation_four_model/error_analysis/reports/four_model_error_analysis_report.txt
- results/adaptive/final_validation_four_model/error_analysis/checkpoints/seed_{seed}_checkpoint.json

SCIENTIFIC NOTE:
----------------
For LSTM, the selected adaptive anomaly weight is 1.0, so no additional anomaly
up-weighting is applied. LSTM is a "deep-learning comparator under the retained
data representation" with evaluation population = 288335 endpoints (vs 288344 for
classical models).
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
import traceback
import gc
import hashlib
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("⚠️  TensorFlow not available. LSTM will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 123, 2024, 2025, 3407]
MODELS = ['XGBoost', 'RandomForest', 'LinearSVM', 'LSTM']

# Final selected anomaly weights (from validated experiments)
ANOMALY_WEIGHTS = {
    'XGBoost': 3.0,
    'RandomForest': 3.0,
    'LinearSVM': 3.0,
    'LSTM': 1.0  # Selected weight from weight-sensitivity experiment
}

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

# LSTM configuration (from validated four-model experiments)
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

# Model configurations (from validated four-model experiments)
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

CHECKPOINT_VERSION = '1.0'

# =============================================================================
# FOUR-MODEL ERROR ANALYSIS CLASS
# =============================================================================
class FourModelErrorAnalysis:
    """
    Implements sample-level error transition analysis for four models.
    """
    
    def __init__(self, base_dir: Path, error_analysis_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.error_analysis_dir = Path(error_analysis_dir) if error_analysis_dir else (
            self.base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis'
        )
        
        # Create output directories
        (self.error_analysis_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.error_analysis_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.error_analysis_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.error_analysis_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.seed_results = []
        self.transition_results = []
        self.multiseed_summary = {}
        self.data_source_description = ''
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        # Checkpoint management
        self.checkpoint_path = self.error_analysis_dir / 'checkpoints' / 'error_analysis_checkpoint.json'
        self.completed_seeds = []
        self.seed_results_cache = {}
        self.transition_results_cache = {}
        
        # FIX #4: Reference DataFrames for cross-validation
        self.four_model_ref_df = None
        self.ablation_ref_df = None
        
        # Compatibility validation paths
        self.four_model_results_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'metrics' / 'multi_seed_four_model_results.csv'
        self.four_model_ablation_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        logger.info(f"✅ Initialized FourModelErrorAnalysis")
        logger.info(f"📁 Error analysis directory: {self.error_analysis_dir}")
        logger.info(f"🔢 Models: {MODELS}")
        logger.info(f"🔢 Seeds: {SEEDS}")
        logger.info(f"⚖️  Anomaly weights: {ANOMALY_WEIGHTS}")
    
    def _log_library_versions(self):
        """FIX #4: Log library versions for reproducibility (diagnostic only)."""
        logger.info("📚 Library versions for reproducibility:")
        logger.info(f"   Python: {sys.version.split()[0]}")
        logger.info(f"   NumPy: {np.__version__}")
        try:
            import sklearn
            logger.info(f"   scikit-learn: {sklearn.__version__}")
        except:
            logger.info(f"   scikit-learn: unavailable")
        try:
            logger.info(f"   XGBoost: {xgb.__version__}")
        except:
            logger.info(f"   XGBoost: unavailable")
        try:
            import tensorflow
            logger.info(f"   TensorFlow: {tensorflow.__version__}")
        except:
            logger.info(f"   TensorFlow: unavailable")
    
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
            'models': MODELS,
            'anomaly_weights': ANOMALY_WEIGHTS,
            'train_ratio': TRAIN_RATIO,
            'retrain_ratio': RETRAIN_RATIO,
            'test_ratio': TEST_RATIO,
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
            'imputation': 'mean',
            'scaler': 'StandardScaler',
            'xgb_config': XGB_CONFIG,
            'rf_config': RF_CONFIG,
            'svm_config': SVM_CONFIG
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
    
    def _validate_seed_results(self, metric_results: List[Dict], transition_results: List[Dict], seed: int) -> Tuple[bool, str]:
        """
        Strict validation of seed results before checkpointing or restoring.
        Validates exact Cartesian product of models × configurations.
        """
        # Validate metric records (exactly 8 = 4 models × 2 configs)
        if len(metric_results) != 8:
            return False, f"Expected 8 metric records, got {len(metric_results)}"
        
        # Validate exact Cartesian product
        required_pairs = set()
        for model_name in MODELS:
            required_pairs.add((seed, model_name, 'baseline'))
            required_pairs.add((seed, model_name, 'selected_adaptive_retraining'))
        
        found_pairs = set()
        for result in metric_results:
            result_seed = result.get('seed')
            model_name = result.get('model_name')
            config = result.get('configuration')
            
            if result_seed != seed:
                return False, f"Metric record has wrong seed: {result_seed} (expected {seed})"
            
            if model_name not in MODELS:
                return False, f"Invalid model: {model_name}"
            
            if config not in ['baseline', 'selected_adaptive_retraining']:
                return False, f"Invalid configuration: {config}"
            
            pair = (result_seed, model_name, config)
            if pair in found_pairs:
                return False, f"Duplicate pair found: {pair}"
            
            found_pairs.add(pair)
            
            # Validate metrics are finite and in valid range
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                val = result.get(metric)
                if val is None or not np.isfinite(val):
                    return False, f"Non-finite {metric} for {model_name} {config}"
                if val < 0 or val > 1:
                    return False, f"{metric} outside [0,1] for {model_name} {config}"
            
            # Validate confusion counts are non-negative integers
            for cm_metric in ['false_positive', 'false_negative', 'true_positive', 'true_negative']:
                val = result.get(cm_metric)
                if val is None or not isinstance(val, (int, np.integer)) or val < 0:
                    return False, f"Invalid {cm_metric} for {model_name} {config}"
            
            # Validate evaluation population
            eval_samples = result.get('evaluation_samples')
            if model_name == 'LSTM':
                expected_eval = 288335  # Test size - TIME_STEPS + 1
            else:
                expected_eval = 288344  # Full test size
            
            if eval_samples is not None and eval_samples != expected_eval:
                return False, f"Wrong evaluation population for {model_name}: {eval_samples} (expected {expected_eval})"
            
            # Validate confusion matrix sum equals evaluation samples
            cm_sum = result.get('true_positive', 0) + result.get('true_negative', 0) + result.get('false_positive', 0) + result.get('false_negative', 0)
            if cm_sum != eval_samples:
                return False, f"Confusion matrix sum ({cm_sum}) != evaluation_samples ({eval_samples}) for {model_name} {config}"
        
        # Check all required pairs are present
        if found_pairs != required_pairs:
            missing = required_pairs - found_pairs
            return False, f"Missing pairs: {missing}"
        
        # Validate transition records (exactly 4 = one per model)
        if len(transition_results) != 4:
            return False, f"Expected 4 transition records, got {len(transition_results)}"
        
        found_transition_models = set()
        for trans in transition_results:
            trans_seed = trans.get('seed')
            model_name = trans.get('model_name')
            config = trans.get('configuration')
            
            if trans_seed != seed:
                return False, f"Transition record has wrong seed: {trans_seed}"
            
            if model_name not in MODELS:
                return False, f"Invalid model in transition: {model_name}"
            
            if config != 'selected_adaptive_retraining':
                return False, f"Transition config must be 'selected_adaptive_retraining', got {config}"
            
            if model_name in found_transition_models:
                return False, f"Duplicate transition model: {model_name}"
            
            found_transition_models.add(model_name)
            
            # Validate exact integer identities (no tolerance)
            baseline_fn = trans.get('baseline_fn', 0)
            adaptive_fn = trans.get('adaptive_fn', 0)
            corrected_fn = trans.get('corrected_fn', 0)
            persistent_fn = trans.get('persistent_fn', 0)
            newly_introduced_fn = trans.get('newly_introduced_fn', 0)
            
            # baseline_fn == corrected_fn + persistent_fn
            if baseline_fn != corrected_fn + persistent_fn:
                return False, f"FN identity failed for {model_name}: {baseline_fn} != {corrected_fn} + {persistent_fn}"
            
            # adaptive_fn == persistent_fn + newly_introduced_fn
            if adaptive_fn != persistent_fn + newly_introduced_fn:
                return False, f"Adaptive FN identity failed for {model_name}: {adaptive_fn} != {persistent_fn} + {newly_introduced_fn}"
            
            baseline_fp = trans.get('baseline_fp', 0)
            adaptive_fp = trans.get('adaptive_fp', 0)
            corrected_fp = trans.get('corrected_fp', 0)
            persistent_fp = trans.get('persistent_fp', 0)
            newly_introduced_fp = trans.get('newly_introduced_fp', 0)
            
            # baseline_fp == corrected_fp + persistent_fp
            if baseline_fp != corrected_fp + persistent_fp:
                return False, f"FP identity failed for {model_name}: {baseline_fp} != {corrected_fp} + {persistent_fp}"
            
            # adaptive_fp == persistent_fp + newly_introduced_fp
            if adaptive_fp != persistent_fp + newly_introduced_fp:
                return False, f"Adaptive FP identity failed for {model_name}: {adaptive_fp} != {persistent_fp} + {newly_introduced_fp}"
            
            # Additional consistency checks
            if (baseline_fn - adaptive_fn) != (corrected_fn - newly_introduced_fn):
                return False, f"FN reduction identity failed for {model_name}"
            
            if (adaptive_fp - baseline_fp) != (newly_introduced_fp - corrected_fp):
                return False, f"FP change identity failed for {model_name}"
        
        if found_transition_models != set(MODELS):
            missing = set(MODELS) - found_transition_models
            return False, f"Missing transition models: {missing}"
        
        return True, ""
    
    def _load_and_validate_checkpoints(self) -> str:
        """
        Load and validate checkpoints with explicit states.
        Returns: "missing", "valid", or "invalid"
        """
        logger.info("🔍 Checking for existing checkpoints...")
        
        if not self.checkpoint_path.exists():
            logger.info("ℹ️   No main checkpoint found. Starting fresh.")
            return "missing"
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                main_checkpoint = json.load(f)
            
            # Validate checkpoint version
            if main_checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
                logger.error(f"❌ Checkpoint version mismatch: {main_checkpoint.get('checkpoint_version')} vs {CHECKPOINT_VERSION}")
                return "invalid"
            
            # Validate fingerprints
            if main_checkpoint.get('config_fingerprint') != self.config_fingerprint:
                logger.error("❌ Main checkpoint config fingerprint mismatch")
                return "invalid"
            
            if main_checkpoint.get('dataset_fingerprint') != self.dataset_fingerprint:
                logger.error("❌ Main checkpoint dataset fingerprint mismatch")
                return "invalid"
            
            # Validate completed_seeds structure
            completed_seeds = main_checkpoint.get('completed_seeds', [])
            if not isinstance(completed_seeds, list):
                logger.error("❌ completed_seeds is not a list")
                return "invalid"
            
            # Check for duplicates and valid seeds
            if len(completed_seeds) != len(set(completed_seeds)):
                logger.error("❌ completed_seeds contains duplicates")
                return "invalid"
            
            for seed in completed_seeds:
                if not isinstance(seed, int):
                    logger.error(f"❌ completed_seeds contains non-integer: {seed}")
                    return "invalid"
                if seed not in SEEDS:
                    logger.error(f"❌ completed_seeds contains unexpected seed: {seed}")
                    return "invalid"
            
            # Validate each per-seed checkpoint
            for seed in completed_seeds:
                seed_checkpoint_path = self.error_analysis_dir / 'checkpoints' / f'seed_{seed}_checkpoint.json'
                if not seed_checkpoint_path.exists():
                    logger.error(f"❌ Per-seed checkpoint missing for seed {seed}")
                    return "invalid"
                
                try:
                    with open(seed_checkpoint_path, 'r', encoding='utf-8') as f:
                        seed_checkpoint = json.load(f)
                    
                    # Validate checkpoint version and seed identity
                    if seed_checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
                        logger.error(f"❌ Seed {seed} checkpoint version mismatch")
                        return "invalid"
                    
                    if seed_checkpoint.get('seed') != seed:
                        logger.error(f"❌ Seed {seed} checkpoint has wrong seed identity: {seed_checkpoint.get('seed')}")
                        return "invalid"
                    
                    # Validate fingerprints
                    if seed_checkpoint.get('config_fingerprint') != self.config_fingerprint:
                        logger.error(f"❌ Seed {seed} checkpoint config fingerprint mismatch")
                        return "invalid"
                    
                    if seed_checkpoint.get('dataset_fingerprint') != self.dataset_fingerprint:
                        logger.error(f"❌ Seed {seed} checkpoint dataset fingerprint mismatch")
                        return "invalid"
                    
                    # Apply strict validation
                    metric_results = seed_checkpoint.get('metric_results', [])
                    transition_results = seed_checkpoint.get('transition_results', [])
                    
                    is_valid, error_msg = self._validate_seed_results(metric_results, transition_results, seed)
                    if not is_valid:
                        logger.error(f"❌ Seed {seed} checkpoint failed strict validation: {error_msg}")
                        return "invalid"
                    
                    # Cache valid results
                    self.seed_results_cache[seed] = metric_results
                    self.transition_results_cache[seed] = transition_results
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load/validate seed {seed} checkpoint: {e}")
                    return "invalid"
            
            # Restore completed seeds
            self.completed_seeds = completed_seeds
            logger.info(f"✅ Restored {len(self.completed_seeds)} completed seeds from checkpoints")
            return "valid"
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Main checkpoint corrupted (JSON parse error): {e}")
            return "invalid"
        except Exception as e:
            logger.error(f"❌ Failed to load main checkpoint: {e}")
            return "invalid"
    
    def _save_main_checkpoint(self):
        """Save main checkpoint tracking completed seeds."""
        checkpoint = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'config_fingerprint': self.config_fingerprint,
            'dataset_fingerprint': self.dataset_fingerprint,
            'completed_seeds': self.completed_seeds,
            'saved_at': datetime.now().isoformat()
        }
        
        self._save_atomically(self.checkpoint_path, checkpoint)
    
    def _save_seed_checkpoint(self, seed: int, metric_results: List[Dict], transition_results: List[Dict]):
        """Save checkpoint after seed completes successfully."""
        checkpoint = {
            'seed': seed,
            'completed_at': datetime.now().isoformat(),
            'checkpoint_version': CHECKPOINT_VERSION,
            'config_fingerprint': self.config_fingerprint,
            'dataset_fingerprint': self.dataset_fingerprint,
            'metric_results': metric_results,
            'transition_results': transition_results
        }
        
        checkpoint_path = self.error_analysis_dir / 'checkpoints' / f'seed_{seed}_checkpoint.json'
        self._save_atomically(checkpoint_path, checkpoint)
        
        logger.info(f"💾 Saved seed {seed} checkpoint: {checkpoint_path}")
    
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
            self.data_source_description = str(self.dataset_path)
            
            # Compute fingerprints
            self.dataset_fingerprint = self._compute_dataset_fingerprint()
            self.config_fingerprint = self._get_config_hash()
            
            # FIX #4: Log library versions
            self._log_library_versions()
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def validate_compatibility(self) -> bool:
        """
        Real compatibility validation against existing four-model results.
        Fully validates BOTH reference files.
        """
        logger.info("🔍 Validating compatibility with existing four-model results...")
        
        # Required files must exist
        if not Path(self.four_model_results_path).exists():
            logger.error(f"❌ Four-model results not found: {self.four_model_results_path}")
            return False
        
        if not Path(self.four_model_ablation_path).exists():
            logger.error(f"❌ Four-model ablation results not found: {self.four_model_ablation_path}")
            return False
        
        try:
            # Load and validate multi_seed_four_model_results.csv
            self.four_model_ref_df = pd.read_csv(self.four_model_results_path)
            
            # Require exactly 40 records
            if len(self.four_model_ref_df) != 40:
                logger.error(f"❌ Four-model results should have 40 records, got {len(self.four_model_ref_df)}")
                return False
            
            # Validate exact models
            models_found = set(self.four_model_ref_df['model_name'].unique())
            if models_found != set(MODELS):
                logger.error(f"❌ Four-model should have models {MODELS}, got {models_found}")
                return False
            
            # Validate exact seeds
            seeds_found = set(self.four_model_ref_df['seed'].unique())
            if seeds_found != set(SEEDS):
                logger.error(f"❌ Four-model should have seeds {SEEDS}, got {seeds_found}")
                return False
            
            # Validate exact model types
            model_types_found = set(self.four_model_ref_df['model_type'].unique())
            if model_types_found != {'baseline', 'weighted_retraining'}:
                logger.error(f"❌ Four-model should have model types baseline/weighted_retraining, got {model_types_found}")
                return False
            
            # Check for duplicates
            self.four_model_ref_df['tuple_key'] = list(zip(self.four_model_ref_df['seed'], 
                                                            self.four_model_ref_df['model_name'], 
                                                            self.four_model_ref_df['model_type']))
            if len(self.four_model_ref_df['tuple_key']) != len(set(self.four_model_ref_df['tuple_key'])):
                logger.error("❌ Four-model results contain duplicate (seed, model_name, model_type)")
                return False
            
            # Load and validate four_model_ablation_results.csv
            self.ablation_ref_df = pd.read_csv(self.four_model_ablation_path)
            
            # Require exactly 60 records
            if len(self.ablation_ref_df) != 60:
                logger.error(f"❌ Ablation should have 60 records, got {len(self.ablation_ref_df)}")
                return False
            
            # Validate exact four models
            ablation_models = set(self.ablation_ref_df['model_name'].unique())
            if ablation_models != set(MODELS):
                logger.error(f"❌ Ablation should have models {MODELS}, got {ablation_models}")
                return False
            
            # Validate exact five seeds
            ablation_seeds = set(self.ablation_ref_df['seed'].unique())
            if ablation_seeds != set(SEEDS):
                logger.error(f"❌ Ablation should have seeds {SEEDS}, got {ablation_seeds}")
                return False
            
            # Validate exact three variants
            ablation_variants = set(self.ablation_ref_df['variant'].unique())
            if ablation_variants != {'baseline', 'retraining_without_weighting', 'weighted_retraining'}:
                logger.error(f"❌ Ablation should have variants baseline/retraining_without_weighting/weighted_retraining, got {ablation_variants}")
                return False
            
            # Check exactly three variants per (seed, model)
            for seed in SEEDS:
                for model_name in MODELS:
                    subset = self.ablation_ref_df[(self.ablation_ref_df['seed'] == seed) & 
                                                   (self.ablation_ref_df['model_name'] == model_name)]
                    if len(subset) != 3:
                        logger.error(f"❌ Ablation seed {seed} model {model_name} should have 3 variants, got {len(subset)}")
                        return False
            
            # Check for duplicates
            self.ablation_ref_df['tuple_key'] = list(zip(self.ablation_ref_df['seed'], 
                                                          self.ablation_ref_df['model_name'], 
                                                          self.ablation_ref_df['variant']))
            if len(self.ablation_ref_df['tuple_key']) != len(set(self.ablation_ref_df['tuple_key'])):
                logger.error("❌ Ablation results contain duplicate (seed, model_name, variant)")
                return False
            
            logger.info("✅ Compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Compatibility validation failed: {e}")
            return False
    
    def _cross_validate_model_results(self, seed: int, model_name: str, 
                                      baseline_metrics: Dict, adaptive_metrics: Dict) -> Tuple[bool, str]:
        """
        FIX #2 & #3: Cross-validate single model results against reference.
        FIX #2: Compare float metrics at 4-decimal precision.
        """
        if self.four_model_ref_df is None or self.ablation_ref_df is None:
            return True, ""  # Skip if references not loaded
        
        # Map configuration to reference model_type/variant
        if model_name in ['XGBoost', 'RandomForest', 'LinearSVM']:
            # Compare baseline against four-model baseline
            ref_record_baseline = self.four_model_ref_df[
                (self.four_model_ref_df['seed'] == seed) &
                (self.four_model_ref_df['model_name'] == model_name) &
                (self.four_model_ref_df['model_type'] == 'baseline')
            ]
            
            if len(ref_record_baseline) != 1:
                return False, f"Reference baseline record not found for {model_name} seed {seed}"
            
            ref_record_baseline = ref_record_baseline.iloc[0]
            
            # FIX #2: Compare float metrics at 4-decimal precision
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                generated_val = baseline_metrics.get(metric)
                reference_val = ref_record_baseline.get(metric)
                
                if generated_val is not None and reference_val is not None:
                    # FIX #2: Round both to 4 decimals before comparison
                    if round(float(generated_val), 4) != round(float(reference_val), 4):
                        return False, f"{model_name} baseline {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
            
            # Compare FP/FN exactly (integers)
            for cm_metric in ['false_positive', 'false_negative']:
                generated_val = baseline_metrics.get(cm_metric)
                reference_val = ref_record_baseline.get(cm_metric)
                
                if generated_val is not None and reference_val is not None:
                    if int(generated_val) != int(reference_val):
                        return False, f"{model_name} baseline {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
            
            # Compare adaptive against four-model weighted_retraining
            ref_record_adaptive = self.four_model_ref_df[
                (self.four_model_ref_df['seed'] == seed) &
                (self.four_model_ref_df['model_name'] == model_name) &
                (self.four_model_ref_df['model_type'] == 'weighted_retraining')
            ]
            
            if len(ref_record_adaptive) != 1:
                return False, f"Reference adaptive record not found for {model_name} seed {seed}"
            
            ref_record_adaptive = ref_record_adaptive.iloc[0]
            
            # FIX #2: Compare float metrics at 4-decimal precision
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                generated_val = adaptive_metrics.get(metric)
                reference_val = ref_record_adaptive.get(metric)
                
                if generated_val is not None and reference_val is not None:
                    # FIX #2: Round both to 4 decimals before comparison
                    if round(float(generated_val), 4) != round(float(reference_val), 4):
                        return False, f"{model_name} adaptive {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
            
            # Compare FP/FN exactly (integers)
            for cm_metric in ['false_positive', 'false_negative']:
                generated_val = adaptive_metrics.get(cm_metric)
                reference_val = ref_record_adaptive.get(cm_metric)
                
                if generated_val is not None and reference_val is not None:
                    if int(generated_val) != int(reference_val):
                        return False, f"{model_name} adaptive {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
        
        elif model_name == 'LSTM':
            # Compare baseline against four-model baseline
            ref_record_baseline = self.four_model_ref_df[
                (self.four_model_ref_df['seed'] == seed) &
                (self.four_model_ref_df['model_name'] == model_name) &
                (self.four_model_ref_df['model_type'] == 'baseline')
            ]
            
            if len(ref_record_baseline) != 1:
                return False, f"LSTM baseline reference record not found for seed {seed}"
            
            ref_record_baseline = ref_record_baseline.iloc[0]
            
            # FIX #2: Compare float metrics at 4-decimal precision
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                generated_val = baseline_metrics.get(metric)
                reference_val = ref_record_baseline.get(metric)
                
                if generated_val is not None and reference_val is not None:
                    # FIX #2: Round both to 4 decimals before comparison
                    if round(float(generated_val), 4) != round(float(reference_val), 4):
                        return False, f"LSTM baseline {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
            
            # Compare FP/FN exactly (integers)
            for cm_metric in ['false_positive', 'false_negative']:
                generated_val = baseline_metrics.get(cm_metric)
                reference_val = ref_record_baseline.get(cm_metric)
                
                if generated_val is not None and reference_val is not None:
                    if int(generated_val) != int(reference_val):
                        return False, f"LSTM baseline {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
            
            # FIX #3: LSTM adaptive weight=1.0 matches ablation retraining_without_weighting
            ref_record_adaptive = self.ablation_ref_df[
                (self.ablation_ref_df['seed'] == seed) &
                (self.ablation_ref_df['model_name'] == model_name) &
                (self.ablation_ref_df['variant'] == 'retraining_without_weighting')
            ]
            
            if len(ref_record_adaptive) != 1:
                return False, f"LSTM adaptive reference record not found for seed {seed}"
            
            ref_record_adaptive = ref_record_adaptive.iloc[0]
            
            # FIX #2: Compare float metrics at 4-decimal precision
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                generated_val = adaptive_metrics.get(metric)
                reference_val = ref_record_adaptive.get(metric)
                
                if generated_val is not None and reference_val is not None:
                    # FIX #2: Round both to 4 decimals before comparison
                    if round(float(generated_val), 4) != round(float(reference_val), 4):
                        return False, f"LSTM adaptive {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
            
            # Compare FP/FN exactly (integers)
            for cm_metric in ['false_positive', 'false_negative']:
                generated_val = adaptive_metrics.get(cm_metric)
                reference_val = ref_record_adaptive.get(cm_metric)
                
                if generated_val is not None and reference_val is not None:
                    if int(generated_val) != int(reference_val):
                        return False, f"LSTM adaptive {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
        
        return True, ""
    
    def _cross_validate_generated_seed(self, seed: int, metric_results: List[Dict]) -> Tuple[bool, str]:
        """
        Cross-check generated seed against reference files BEFORE checkpointing.
        FIX: Use 4-decimal precision comparison for float metrics (consistent with _cross_validate_model_results).
        """
        if self.four_model_ref_df is None or self.ablation_ref_df is None:
            return True, ""  # Skip if references not loaded
        
        for result in metric_results:
            model_name = result.get('model_name')
            config = result.get('configuration')
            
            # For XGBoost, RandomForest, LinearSVM
            if model_name in ['XGBoost', 'RandomForest', 'LinearSVM']:
                # Map configuration to reference model_type
                if config == 'baseline':
                    ref_model_type = 'baseline'
                elif config == 'selected_adaptive_retraining':
                    ref_model_type = 'weighted_retraining'
                else:
                    return False, f"Unknown configuration {config} for {model_name}"
                
                # Find reference record
                ref_record = self.four_model_ref_df[
                    (self.four_model_ref_df['seed'] == seed) &
                    (self.four_model_ref_df['model_name'] == model_name) &
                    (self.four_model_ref_df['model_type'] == ref_model_type)
                ]
                
                if len(ref_record) != 1:
                    return False, f"Reference record not found for {model_name} seed {seed} {ref_model_type}"
                
                ref_record = ref_record.iloc[0]
                
                # FIX: Compare float metrics at 4-decimal precision (consistent with _cross_validate_model_results)
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    generated_val = result.get(metric)
                    reference_val = ref_record.get(metric)
                    
                    if generated_val is not None and reference_val is not None:
                        # FIX: Round both to 4 decimals before comparison
                        if round(float(generated_val), 4) != round(float(reference_val), 4):
                            return False, f"{model_name} seed {seed} {config} {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
                
                # Compare FP/FN exactly (integers)
                for cm_metric in ['false_positive', 'false_negative']:
                    generated_val = result.get(cm_metric)
                    reference_val = ref_record.get(cm_metric)
                    
                    if generated_val is not None and reference_val is not None:
                        if int(generated_val) != int(reference_val):
                            return False, f"{model_name} seed {seed} {config} {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
            
            # For LSTM
            elif model_name == 'LSTM':
                if config == 'baseline':
                    # Compare against four-model baseline
                    ref_model_type = 'baseline'
                    ref_record = self.four_model_ref_df[
                        (self.four_model_ref_df['seed'] == seed) &
                        (self.four_model_ref_df['model_name'] == model_name) &
                        (self.four_model_ref_df['model_type'] == ref_model_type)
                    ]
                elif config == 'selected_adaptive_retraining':
                    # LSTM adaptive weight=1.0 matches ablation retraining_without_weighting
                    ref_record = self.ablation_ref_df[
                        (self.ablation_ref_df['seed'] == seed) &
                        (self.ablation_ref_df['model_name'] == model_name) &
                        (self.ablation_ref_df['variant'] == 'retraining_without_weighting')
                    ]
                else:
                    return False, f"Unknown configuration {config} for LSTM"
                
                if len(ref_record) != 1:
                    return False, f"Reference record not found for LSTM seed {seed} {config}"
                
                ref_record = ref_record.iloc[0]
                
                # FIX: Compare float metrics at 4-decimal precision (consistent with _cross_validate_model_results)
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    generated_val = result.get(metric)
                    reference_val = ref_record.get(metric)
                    
                    if generated_val is not None and reference_val is not None:
                        # FIX: Round both to 4 decimals before comparison
                        if round(float(generated_val), 4) != round(float(reference_val), 4):
                            return False, f"LSTM seed {seed} {config} {metric} mismatch at 4-decimal precision: generated={round(float(generated_val), 4)}, reference={round(float(reference_val), 4)}"
                
                # Compare FP/FN exactly (integers)
                for cm_metric in ['false_positive', 'false_negative']:
                    generated_val = result.get(cm_metric)
                    reference_val = ref_record.get(cm_metric)
                    
                    if generated_val is not None and reference_val is not None:
                        if int(generated_val) != int(reference_val):
                            return False, f"LSTM seed {seed} {config} {cm_metric} mismatch: generated={generated_val}, reference={reference_val}"
        
        return True, ""
    
    def _create_sample_weights(self, y, anomaly_weight):
        """
        Create anomaly-sensitive sample weights.
        Preserves exact scientific weighting protocol:
        - normal class = 1.0
        - anomaly class = selected anomaly weight
        """
        y = np.asarray(y)
        weights = np.ones(len(y), dtype=np.float32)
        weights[y == 1] = float(anomaly_weight)
        return weights
    
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
    
    def _create_sequences_efficient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input without crossing boundaries."""
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
    
    def _create_lstm_baseline_tf_dataset(self, X: np.ndarray, y: np.ndarray, 
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
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
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
        
        # Evaluation population
        if model_name == 'LSTM':
            eval_samples = 288335
        else:
            eval_samples = 288344
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'evaluation_samples': eval_samples
        }
    
    def _analyze_error_transitions(self, y_true: np.ndarray, 
                                   y_baseline: np.ndarray, 
                                   y_adaptive: np.ndarray,
                                   model_name: str) -> Dict:
        """
        Analyze sample-level error transitions between baseline and adaptive.
        """
        # Identify anomaly samples (true label = 1)
        anomaly_mask = (y_true == 1)
        n_anomalies = np.sum(anomaly_mask)
        
        # Identify normal samples (true label = 0)
        normal_mask = (y_true == 0)
        n_normals = np.sum(normal_mask)
        
        # False Negative transitions (among anomalies)
        baseline_fn_mask = anomaly_mask & (y_baseline == 0)
        adaptive_fn_mask = anomaly_mask & (y_adaptive == 0)
        
        corrected_fn = int(np.sum(baseline_fn_mask & (y_adaptive == 1)))
        persistent_fn = int(np.sum(baseline_fn_mask & adaptive_fn_mask))
        newly_introduced_fn = int(np.sum((y_baseline == 1) & adaptive_fn_mask))
        
        baseline_fn_count = int(np.sum(baseline_fn_mask))
        adaptive_fn_count = int(np.sum(adaptive_fn_mask))
        
        # False Positive transitions (among normals)
        baseline_fp_mask = normal_mask & (y_baseline == 1)
        adaptive_fp_mask = normal_mask & (y_adaptive == 1)
        
        corrected_fp = int(np.sum(baseline_fp_mask & (y_adaptive == 0)))
        persistent_fp = int(np.sum(baseline_fp_mask & adaptive_fp_mask))
        newly_introduced_fp = int(np.sum((y_baseline == 0) & adaptive_fp_mask))
        
        baseline_fp_count = int(np.sum(baseline_fp_mask))
        adaptive_fp_count = int(np.sum(adaptive_fp_mask))
        
        return {
            'model_name': model_name,
            'n_anomalies': int(n_anomalies),
            'n_normals': int(n_normals),
            'baseline_fn': baseline_fn_count,
            'adaptive_fn': adaptive_fn_count,
            'corrected_fn': corrected_fn,
            'persistent_fn': persistent_fn,
            'newly_introduced_fn': newly_introduced_fn,
            'baseline_fp': baseline_fp_count,
            'adaptive_fp': adaptive_fp_count,
            'corrected_fp': corrected_fp,
            'persistent_fp': persistent_fp,
            'newly_introduced_fp': newly_introduced_fp,
            # Validate transition identities
            'fn_identity_valid': bool(baseline_fn_count == corrected_fn + persistent_fn and
                                       adaptive_fn_count == persistent_fn + newly_introduced_fn),
            'fp_identity_valid': bool(baseline_fp_count == corrected_fp + persistent_fp and
                                       adaptive_fp_count == persistent_fp + newly_introduced_fp)
        }
    
    def run_single_seed(self, seed: int) -> Tuple[bool, List[Dict], List[Dict]]:
        """
        Run complete error analysis for a single seed.
        FIX #3: Cross-validate each model BEFORE training the next.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING ERROR ANALYSIS SEED {seed}")
        logger.info(f"{'='*80}")
        
        metric_results = []
        transition_results = []
        
        try:
            set_random_seeds(seed)
            
            # Identify label column
            label_col = self._identify_label_column(self.original_dataset)
            if label_col is None:
                logger.error("❌ No label column found")
                return False, [], []
            
            # Convert labels
            all_labels = self._convert_labels_to_binary(self.original_dataset, label_col)
            
            # Prepare features
            all_features, feature_cols = self._prepare_features(self.original_dataset, label_col)
            if all_features is None:
                logger.error("❌ No features found")
                return False, [], []
            
            n_features = len(feature_cols)
            
            # Split data (stratified, same seed for both stages)
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
            
            # FIX #3: Process models in order, validating each before next
            for model_idx, model_name in enumerate(MODELS):
                logger.info(f"   📊 Model {model_idx + 1}/{len(MODELS)}: {model_name}")
                
                anomaly_weight = ANOMALY_WEIGHTS[model_name]
                model_metric_results = []
                model_transition_result = None
                
                if model_name == 'LSTM':
                    # LSTM requires special handling
                    if not TENSORFLOW_AVAILABLE:
                        logger.warning(f"⚠️  TensorFlow not available - skipping LSTM")
                        continue
                    
                    # =========================================================
                    # LSTM BASELINE: Historical tf.data reference protocol
                    # =========================================================
                    logger.info(f"      LSTM baseline protocol: historical tf.data reference protocol")
                    
                    # Create baseline datasets using historical tf.data pipeline
                    baseline_train_dataset, baseline_val_dataset, n_train_baseline = self._create_lstm_baseline_tf_dataset(
                        X_train_scaled, y_train,
                        sequence_length=TIME_STEPS,
                        batch_size=LSTM_BATCH_SIZE,
                        seed=seed
                    )
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                    tf.keras.utils.set_random_seed(seed)
                    
                    model_baseline = self._create_lstm_model(n_features)
                    
                    history_baseline = model_baseline.fit(
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
                        verbose=0
                    )
                    
                    # Baseline test prediction using tf.data
                    X_test_tf = tf.constant(X_test_scaled, dtype=tf.float32)
                    y_test_tf = tf.constant(y_test, dtype=tf.float32)
                    
                    def test_map_fn(i):
                        return (
                            X_test_tf[i:i + TIME_STEPS],
                            y_test_tf[i + TIME_STEPS - 1]
                        )
                    
                    test_dataset = tf.data.Dataset.range(
                        len(X_test_scaled) - TIME_STEPS + 1
                    ).map(
                        test_map_fn,
                        num_parallel_calls=tf.data.AUTOTUNE
                    ).batch(LSTM_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                    
                    y_baseline_proba = model_baseline.predict(
                        test_dataset,
                        verbose=0
                    ).flatten()
                    
                    y_baseline = (y_baseline_proba > PREDICTION_THRESHOLD).astype(int)
                    
                    # y_true for baseline is y_test[TIME_STEPS - 1:]
                    y_baseline_true = y_test[TIME_STEPS - 1:]
                    
                    del model_baseline
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # =========================================================
                    # LSTM ADAPTIVE: Ablation retraining_without_weighting protocol
                    # =========================================================
                    logger.info(f"      LSTM adaptive protocol: ablation retraining_without_weighting protocol")
                    
                    # Create sequences separately (never cross boundaries)
                    X_train_seq, y_train_seq = self._create_sequences_efficient(X_train_scaled, y_train)
                    X_retrain_seq, y_retrain_seq = self._create_sequences_efficient(X_retrain_scaled, y_retrain)
                    X_test_seq, y_test_seq = self._create_sequences_efficient(X_test_scaled, y_test)
                    
                    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                        logger.warning(f"⚠️  LSTM: No valid sequences")
                        continue
                    
                    # Concatenate sequence arrays (not raw features)
                    X_combined_seq = np.concatenate([X_train_seq, X_retrain_seq], axis=0)
                    y_combined_seq = np.concatenate([y_train_seq, y_retrain_seq], axis=0)
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                    set_random_seeds(seed)
                    tf.keras.utils.set_random_seed(seed)
                    
                    model_adaptive = self._create_lstm_model(n_features)
                    
                    # Adaptive uses NumPy arrays with validation_split=0.1 (ablation-compatible)
                    # No anomaly up-weighting (weight = 1.0)
                    history_adaptive = model_adaptive.fit(
                        X_combined_seq, y_combined_seq,
                        epochs=LSTM_EPOCHS,
                        batch_size=LSTM_BATCH_SIZE,
                        validation_split=0.1,
                        callbacks=[callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=PATIENCE,
                            restore_best_weights=True
                        )],
                        verbose=0
                    )
                    
                    y_adaptive_proba = model_adaptive.predict(X_test_seq, verbose=0).flatten()
                    y_adaptive = (y_adaptive_proba > PREDICTION_THRESHOLD).astype(int)
                    
                    del model_adaptive
                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    # Evaluate baseline
                    baseline_metrics = self._evaluate_predictions(y_baseline_true, y_baseline, model_name)
                    baseline_metrics['model_name'] = model_name
                    baseline_metrics['configuration'] = 'baseline'
                    baseline_metrics['anomaly_weight'] = 1.0
                    baseline_metrics['seed'] = seed
                    baseline_metrics['evaluation_samples'] = int(len(y_baseline_true))
                    
                    # Evaluate adaptive
                    adaptive_metrics = self._evaluate_predictions(y_test_seq, y_adaptive, model_name)
                    adaptive_metrics['model_name'] = model_name
                    adaptive_metrics['configuration'] = 'selected_adaptive_retraining'
                    adaptive_metrics['anomaly_weight'] = 1.0
                    adaptive_metrics['additional_weighting_selected'] = False
                    adaptive_metrics['seed'] = seed
                    adaptive_metrics['evaluation_samples'] = int(len(y_test_seq))
                    
                    model_metric_results.append(baseline_metrics)
                    model_metric_results.append(adaptive_metrics)
                    
                    # Analyze transitions
                    model_transition_result = self._analyze_error_transitions(y_baseline_true, y_baseline, y_adaptive, model_name)
                    model_transition_result['seed'] = seed
                    model_transition_result['anomaly_weight'] = 1.0
                    model_transition_result['configuration'] = 'selected_adaptive_retraining'
                    model_transition_result['evaluation_samples'] = int(len(y_baseline_true))
                    
                else:
                    # Classical models (XGBoost, RandomForest, LinearSVM)
                    if model_name == 'XGBoost':
                        model_baseline = xgb.XGBClassifier(**XGB_CONFIG)
                        model_adaptive = xgb.XGBClassifier(**XGB_CONFIG)
                    elif model_name == 'RandomForest':
                        model_baseline = RandomForestClassifier(**RF_CONFIG)
                        model_adaptive = RandomForestClassifier(**RF_CONFIG)
                    elif model_name == 'LinearSVM':
                        model_baseline = LinearSVC(**SVM_CONFIG)
                        model_adaptive = LinearSVC(**SVM_CONFIG)
                    
                    # FIX #1: Baseline - NO sample_weight (matches validated four-model protocol)
                    logger.info(f"      Training baseline (NO sample_weight)...")
                    model_baseline.fit(X_train_scaled, y_train)
                    y_baseline = model_baseline.predict(X_test_scaled)
                    
                    # Adaptive: train + retrain with selected anomaly weight
                    X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
                    y_combined = np.concatenate([y_train, y_retrain])
                    sample_weights_adaptive = self._create_sample_weights(y_combined, anomaly_weight)
                    
                    logger.info(f"      Training adaptive (anomaly_weight={anomaly_weight})...")
                    model_adaptive.fit(X_combined, y_combined, sample_weight=sample_weights_adaptive)
                    y_adaptive = model_adaptive.predict(X_test_scaled)
                    
                    # Evaluate
                    baseline_metrics = self._evaluate_predictions(y_test, y_baseline, model_name)
                    adaptive_metrics = self._evaluate_predictions(y_test, y_adaptive, model_name)
                    
                    # Add model-specific metadata
                    baseline_metrics['model_name'] = model_name
                    baseline_metrics['configuration'] = 'baseline'
                    baseline_metrics['anomaly_weight'] = 1.0
                    baseline_metrics['seed'] = seed
                    
                    adaptive_metrics['model_name'] = model_name
                    adaptive_metrics['configuration'] = 'selected_adaptive_retraining'
                    adaptive_metrics['anomaly_weight'] = anomaly_weight
                    adaptive_metrics['additional_weighting_selected'] = (anomaly_weight != 1.0)
                    adaptive_metrics['seed'] = seed
                    
                    model_metric_results.append(baseline_metrics)
                    model_metric_results.append(adaptive_metrics)
                    
                    # Analyze transitions
                    model_transition_result = self._analyze_error_transitions(y_test, y_baseline, y_adaptive, model_name)
                    model_transition_result['seed'] = seed
                    model_transition_result['anomaly_weight'] = anomaly_weight
                    model_transition_result['configuration'] = 'selected_adaptive_retraining'
                    model_transition_result['evaluation_samples'] = int(len(y_test))
                
                # FIX #3: Cross-validate THIS model BEFORE training next
                is_valid, error_msg = self._cross_validate_model_results(seed, model_name, 
                                                                        model_metric_results[0], 
                                                                        model_metric_results[1])
                if not is_valid:
                    logger.error(f"❌ {model_name} failed reference cross-validation: {error_msg}")
                    return False, [], []
                
                logger.info(f"      ✅ {model_name} cross-validation passed")
                
                # Add to results
                metric_results.extend(model_metric_results)
                transition_results.append(model_transition_result)
            
            # Validate transition identities
            for trans in transition_results:
                if not trans.get('fn_identity_valid', False):
                    logger.warning(f"⚠️  Seed {seed} {trans['model_name']}: FN transition identity failed")
                if not trans.get('fp_identity_valid', False):
                    logger.warning(f"⚠️  Seed {seed} {trans['model_name']}: FP transition identity failed")
            
            return True, metric_results, transition_results
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
            return False, [], []
    
    def _compute_multiseed_summary(self):
        """Compute multi-seed summary statistics."""
        logger.info("📊 Computing multi-seed summary statistics...")
        
        # Aggregate all results
        all_metrics = []
        all_transitions = []
        
        for seed in self.completed_seeds:
            if seed in self.seed_results_cache:
                all_metrics.extend(self.seed_results_cache[seed])
            if seed in self.transition_results_cache:
                all_transitions.extend(self.transition_results_cache[seed])
        
        if not all_metrics or not all_transitions:
            self.multiseed_summary = {
                'status': 'no_results',
                'note': 'No results available'
            }
            return
        
        # Compute statistics per model
        metrics_df = pd.DataFrame(all_metrics)
        transitions_df = pd.DataFrame(all_transitions)
        
        summary_stats = {}
        
        for model_name in MODELS:
            model_metrics = metrics_df[metrics_df['model_name'] == model_name]
            model_transitions = transitions_df[transitions_df['model_name'] == model_name]
            
            if len(model_metrics) == 0:
                continue
            
            baseline = model_metrics[model_metrics['configuration'] == 'baseline']
            adaptive = model_metrics[model_metrics['configuration'] == 'selected_adaptive_retraining']
            
            def safe_mean_std(series):
                valid = series.dropna()
                if len(valid) == 0:
                    return None, None, 0
                mean = float(valid.mean())
                std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
                return mean, std, len(valid)
            
            # FIX: Add all missing std values
            summary_stats[model_name] = {
                'anomaly_weight': ANOMALY_WEIGHTS[model_name],
                'additional_weighting_selected': (ANOMALY_WEIGHTS[model_name] != 1.0),
                'baseline_fn_mean': safe_mean_std(baseline['false_negative'])[0],
                'baseline_fn_std': safe_mean_std(baseline['false_negative'])[1],
                'adaptive_fn_mean': safe_mean_std(adaptive['false_negative'])[0],
                'adaptive_fn_std': safe_mean_std(adaptive['false_negative'])[1],
                'fn_reduction_mean': safe_mean_std(baseline['false_negative'] - adaptive['false_negative'])[0],
                'fn_reduction_std': safe_mean_std(baseline['false_negative'] - adaptive['false_negative'])[1],
                'corrected_fn_mean': safe_mean_std(model_transitions['corrected_fn'])[0],
                'corrected_fn_std': safe_mean_std(model_transitions['corrected_fn'])[1],
                'persistent_fn_mean': safe_mean_std(model_transitions['persistent_fn'])[0],
                'persistent_fn_std': safe_mean_std(model_transitions['persistent_fn'])[1],
                'newly_introduced_fn_mean': safe_mean_std(model_transitions['newly_introduced_fn'])[0],
                'newly_introduced_fn_std': safe_mean_std(model_transitions['newly_introduced_fn'])[1],
                'baseline_fp_mean': safe_mean_std(baseline['false_positive'])[0],
                'baseline_fp_std': safe_mean_std(baseline['false_positive'])[1],
                'adaptive_fp_mean': safe_mean_std(adaptive['false_positive'])[0],
                'adaptive_fp_std': safe_mean_std(adaptive['false_positive'])[1],
                'fp_change_mean': safe_mean_std(adaptive['false_positive'] - baseline['false_positive'])[0],
                'fp_change_std': safe_mean_std(adaptive['false_positive'] - baseline['false_positive'])[1],
                'corrected_fp_mean': safe_mean_std(model_transitions['corrected_fp'])[0],
                'corrected_fp_std': safe_mean_std(model_transitions['corrected_fp'])[1],
                'persistent_fp_mean': safe_mean_std(model_transitions['persistent_fp'])[0],
                'persistent_fp_std': safe_mean_std(model_transitions['persistent_fp'])[1],
                'newly_introduced_fp_mean': safe_mean_std(model_transitions['newly_introduced_fp'])[0],
                'newly_introduced_fp_std': safe_mean_std(model_transitions['newly_introduced_fp'])[1],
                'f1_mean': safe_mean_std(adaptive['f1_score'])[0],
                'f1_std': safe_mean_std(adaptive['f1_score'])[1],
                'recall_mean': safe_mean_std(adaptive['recall'])[0],
                'recall_std': safe_mean_std(adaptive['recall'])[1],
                'precision_mean': safe_mean_std(adaptive['precision'])[0],
                'precision_std': safe_mean_std(adaptive['precision'])[1]
            }
        
        self.multiseed_summary = {
            'total_seeds': len(SEEDS),
            'completed_seeds': len(self.completed_seeds),
            'models': MODELS,
            'anomaly_weights': ANOMALY_WEIGHTS,
            'model_statistics': summary_stats,
            'scientific_note': 'For LSTM, the selected adaptive anomaly weight is 1.0, so no additional anomaly up-weighting is applied. LSTM is a deep-learning comparator under the retained data representation with evaluation population = 288335 endpoints (vs 288344 for classical models).'
        }
    
    def _generate_figures(self):
        """Generate error transition figures."""
        logger.info("🎨 Generating error transition figures...")
        
        # Aggregate all results
        all_transitions = []
        for seed in self.completed_seeds:
            if seed in self.transition_results_cache:
                all_transitions.extend(self.transition_results_cache[seed])
        
        if not all_transitions:
            logger.warning("⚠️  No transition results for figures")
            return
        
        transitions_df = pd.DataFrame(all_transitions)
        
        # FIGURE 1: Error Transition Summary (2x2 grid for 4 models)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for idx, model_name in enumerate(MODELS):
                model_data = transitions_df[transitions_df['model_name'] == model_name]
                
                if len(model_data) == 0:
                    continue
                
                ax = axes[idx]
                
                # FN transitions
                fn_corrected = model_data['corrected_fn'].mean()
                fn_persistent = model_data['persistent_fn'].mean()
                fn_newly = model_data['newly_introduced_fn'].mean()
                
                # FP transitions
                fp_corrected = model_data['corrected_fp'].mean()
                fp_persistent = model_data['persistent_fp'].mean()
                fp_newly = model_data['newly_introduced_fp'].mean()
                
                x = np.arange(6)
                values = [fn_corrected, fn_persistent, fn_newly, fp_corrected, fp_persistent, fp_newly]
                labels = ['Corrected\nFN', 'Persistent\nFN', 'Newly\nFN', 
                         'Corrected\nFP', 'Persistent\nFP', 'Newly\nFP']
                colors = ['#27ae60', '#e74c3c', '#f39c12', '#27ae60', '#e74c3c', '#f39c12']
                
                bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.7)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.0f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel('Mean Count per Seed', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name}\n(Weight={ANOMALY_WEIGHTS[model_name]})', 
                            fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, max(values) * 1.3)
            
            plt.tight_layout()
            fig_path = self.error_analysis_dir / 'figures' / 'error_transition_summary.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            logger.info(f"   ✅ Saved error transition summary: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate error transition summary: {e}")
        
        # FIGURE 2: False Negative Correction
        try:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            for idx, model_name in enumerate(MODELS):
                model_data = transitions_df[transitions_df['model_name'] == model_name]
                
                if len(model_data) == 0:
                    continue
                
                ax = axes[idx]
                
                baseline_fn = model_data['baseline_fn'].mean()
                adaptive_fn = model_data['adaptive_fn'].mean()
                corrected_fn = model_data['corrected_fn'].mean()
                newly_fn = model_data['newly_introduced_fn'].mean()
                
                x = np.arange(4)
                values = [baseline_fn, adaptive_fn, corrected_fn, newly_fn]
                labels = ['Baseline\nFN', 'Adaptive\nFN', 'Corrected\nFN', 'Newly\nFN']
                colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
                
                bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.7)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.0f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
                
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel('Mean Count per Seed', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name}\nFN Transitions', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, max(values) * 1.3)
            
            plt.tight_layout()
            fig_path = self.error_analysis_dir / 'figures' / 'false_negative_correction.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            logger.info(f"   ✅ Saved FN correction: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FN correction: {e}")
        
        # FIGURE 3: False Positive Tradeoff
        try:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            for idx, model_name in enumerate(MODELS):
                model_data = transitions_df[transitions_df['model_name'] == model_name]
                
                if len(model_data) == 0:
                    continue
                
                ax = axes[idx]
                
                baseline_fp = model_data['baseline_fp'].mean()
                adaptive_fp = model_data['adaptive_fp'].mean()
                corrected_fp = model_data['corrected_fp'].mean()
                newly_fp = model_data['newly_introduced_fp'].mean()
                
                x = np.arange(4)
                values = [baseline_fp, adaptive_fp, corrected_fp, newly_fp]
                labels = ['Baseline\nFP', 'Adaptive\nFP', 'Corrected\nFP', 'Newly\nFP']
                colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
                
                bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.7)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.0f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
                
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel('Mean Count per Seed', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name}\nFP Transitions', fontsize=12, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, max(values) * 1.3)
            
            plt.tight_layout()
            fig_path = self.error_analysis_dir / 'figures' / 'false_positive_tradeoff.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            logger.info(f"   ✅ Saved FP tradeoff: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate FP tradeoff: {e}")
        
        # FIGURE 4: Persistent Errors by Model
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            persistent_fn = [transitions_df[transitions_df['model_name'] == m]['persistent_fn'].mean() 
                            for m in MODELS]
            persistent_fp = [transitions_df[transitions_df['model_name'] == m]['persistent_fp'].mean() 
                            for m in MODELS]
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            bars1 = axes[0].bar(x - width/2, persistent_fn, width, 
                               label='Persistent FN', color='#e74c3c', edgecolor='black', alpha=0.7)
            bars2 = axes[0].bar(x + width/2, persistent_fp, width, 
                               label='Persistent FP', color='#f39c12', edgecolor='black', alpha=0.7)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    axes[0].annotate(f'{height:.0f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9)
            
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(MODELS, rotation=45, ha='right', fontsize=10)
            axes[0].set_ylabel('Mean Count per Seed', fontsize=11, fontweight='bold')
            axes[0].set_title('Persistent Errors by Model', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(axis='y', alpha=0.3)
            
            # FN reduction vs FP change
            fn_reduction = []
            fp_change = []
            for model_name in MODELS:
                model_data = transitions_df[transitions_df['model_name'] == model_name]
                fn_red = model_data['baseline_fn'].mean() - model_data['adaptive_fn'].mean()
                fp_ch = model_data['adaptive_fp'].mean() - model_data['baseline_fp'].mean()
                fn_reduction.append(fn_red)
                fp_change.append(fp_ch)
            
            colors = ['#27ae60' if val > 0 else '#e74c3c' for val in fn_reduction]
            axes[1].bar(x - width/2, fn_reduction, width, label='FN Reduction', 
                       color=colors, edgecolor='black', alpha=0.7)
            
            colors = ['#e74c3c' if val > 0 else '#27ae60' for val in fp_change]
            axes[1].bar(x + width/2, fp_change, width, label='FP Change', 
                       color=colors, edgecolor='black', alpha=0.7)
            
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(MODELS, rotation=45, ha='right', fontsize=10)
            axes[1].set_ylabel('Mean Change per Seed', fontsize=11, fontweight='bold')
            axes[1].set_title('Error Tradeoffs', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.error_analysis_dir / 'figures' / 'persistent_errors_by_model.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            
            logger.info(f"   ✅ Saved persistent errors: {fig_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate persistent errors: {e}")
    
    def _generate_report(self):
        """Generate comprehensive error analysis report."""
        logger.info("📝 Generating error analysis report...")
        
        report = []
        report.append("="*80)
        report.append("FOUR-MODEL ERROR ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 1.0.0")
        report.append(f"Thesis Project: CPS Anomaly Detection Framework")
        report.append("")
        
        report.append("EXPERIMENTAL CONFIGURATION:")
        report.append(f"  - Models: {MODELS}")
        report.append(f"  - Seeds: {SEEDS}")
        report.append(f"  - Completed seeds: {self.completed_seeds}")
        report.append(f"  - Train/Retrain/Test ratios: {TRAIN_RATIO}/{RETRAIN_RATIO}/{TEST_RATIO}")
        report.append(f"  - Anomaly weights: {ANOMALY_WEIGHTS}")
        report.append("")
        
        report.append("MODEL-SPECIFIC NOTES:")
        for model_name in MODELS:
            weight = ANOMALY_WEIGHTS[model_name]
            if weight == 1.0:
                report.append(f"  - {model_name}: Selected adaptive weight = {weight} (no additional up-weighting)")
            else:
                report.append(f"  - {model_name}: Selected adaptive weight = {weight}")
        report.append("")
        
        report.append("MULTI-SEED SUMMARY STATISTICS:")
        if 'model_statistics' in self.multiseed_summary:
            for model_name in MODELS:
                if model_name not in self.multiseed_summary['model_statistics']:
                    continue
                
                stats = self.multiseed_summary['model_statistics'][model_name]
                report.append(f"  {model_name}:")
                
                # Report all requested FN information
                report.append(f"    False Negatives:")
                report.append(f"      - Baseline FN: {stats.get('baseline_fn_mean', 'N/A'):.1f} ± {stats.get('baseline_fn_std', 0):.1f}")
                report.append(f"      - Adaptive FN: {stats.get('adaptive_fn_mean', 'N/A'):.1f} ± {stats.get('adaptive_fn_std', 0):.1f}")
                report.append(f"      - FN Reduction: {stats.get('fn_reduction_mean', 'N/A'):.1f} ± {stats.get('fn_reduction_std', 0):.1f}")
                report.append(f"      - Corrected FN: {stats.get('corrected_fn_mean', 'N/A'):.1f} ± {stats.get('corrected_fn_std', 0):.1f}")
                report.append(f"      - Persistent FN: {stats.get('persistent_fn_mean', 'N/A'):.1f} ± {stats.get('persistent_fn_std', 0):.1f}")
                report.append(f"      - Newly Introduced FN: {stats.get('newly_introduced_fn_mean', 'N/A'):.1f} ± {stats.get('newly_introduced_fn_std', 0):.1f}")
                
                # Report all requested FP information
                report.append(f"    False Positives:")
                report.append(f"      - Baseline FP: {stats.get('baseline_fp_mean', 'N/A'):.1f} ± {stats.get('baseline_fp_std', 0):.1f}")
                report.append(f"      - Adaptive FP: {stats.get('adaptive_fp_mean', 'N/A'):.1f} ± {stats.get('adaptive_fp_std', 0):.1f}")
                report.append(f"      - FP Change: {stats.get('fp_change_mean', 'N/A'):.1f} ± {stats.get('fp_change_std', 0):.1f}")
                report.append(f"      - Corrected FP: {stats.get('corrected_fp_mean', 'N/A'):.1f} ± {stats.get('corrected_fp_std', 0):.1f}")
                report.append(f"      - Persistent FP: {stats.get('persistent_fp_mean', 'N/A'):.1f} ± {stats.get('persistent_fp_std', 0):.1f}")
                report.append(f"      - Newly Introduced FP: {stats.get('newly_introduced_fp_mean', 'N/A'):.1f} ± {stats.get('newly_introduced_fp_std', 0):.1f}")
                
                # Report F1/recall/precision
                report.append(f"    Adaptive Performance:")
                report.append(f"      - F1: {stats.get('f1_mean', 'N/A'):.4f} ± {stats.get('f1_std', 0):.4f}")
                report.append(f"      - Recall: {stats.get('recall_mean', 'N/A'):.4f} ± {stats.get('recall_std', 0):.4f}")
                report.append(f"      - Precision: {stats.get('precision_mean', 'N/A'):.4f} ± {stats.get('precision_std', 0):.4f}")
                report.append("")
        
        report.append("SCIENTIFIC NOTES:")
        report.append(f"  - For LSTM, the selected adaptive anomaly weight is 1.0, so no additional anomaly up-weighting is applied.")
        report.append(f"  - LSTM is a deep-learning comparator under the retained data representation.")
        report.append(f"  - LSTM evaluation uses 288335 sequence endpoints (vs 288344 for classical models).")
        report.append(f"  - Error transitions are model-specific and should not be directly compared across models with different evaluation populations.")
        report.append(f"  - Positive FN reduction means improvement (fewer false negatives).")
        report.append(f"  - Positive FP change means degradation (more false positives).")
        report.append("")
        
        report.append("="*80)
        report.append("END OF FOUR-MODEL ERROR ANALYSIS REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        report_path = self.error_analysis_dir / 'reports' / 'four_model_error_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
    
    def _save_all_outputs(self):
        """Save all outputs."""
        logger.info("💾 Saving all outputs...")
        saved_paths = []
        
        try:
            # Aggregate all results
            all_metrics = []
            all_transitions = []
            
            for seed in self.completed_seeds:
                if seed in self.seed_results_cache:
                    all_metrics.extend(self.seed_results_cache[seed])
                if seed in self.transition_results_cache:
                    all_transitions.extend(self.transition_results_cache[seed])
            
            # Save metrics CSV
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                metrics_path = self.error_analysis_dir / 'metrics' / 'four_model_error_analysis_results.csv'
                metrics_df.to_csv(metrics_path, index=False)
                saved_paths.append(str(metrics_path))
                logger.info(f"   ✅ Saved metrics: {metrics_path}")
            
            # Save transitions CSV
            if all_transitions:
                transitions_df = pd.DataFrame(all_transitions)
                transitions_path = self.error_analysis_dir / 'metrics' / 'four_model_error_transitions.csv'
                transitions_df.to_csv(transitions_path, index=False)
                saved_paths.append(str(transitions_path))
                logger.info(f"   ✅ Saved transitions: {transitions_path}")
            
            # Save summary JSON
            if self.multiseed_summary:
                summary_path = self.error_analysis_dir / 'metrics' / 'four_model_error_analysis_summary.json'
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(self.multiseed_summary, f, indent=2)
                saved_paths.append(str(summary_path))
                logger.info(f"   ✅ Saved summary: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save outputs: {e}")
            raise
        
        return saved_paths
    
    def run_full_analysis(self) -> Dict:
        """
        Run complete four-model error analysis.
        FIX #3: Cross-validate each model before training next.
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING FOUR-MODEL ERROR ANALYSIS")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'analysis_complete': False
        }
        
        # Load resources
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        # Validate compatibility
        if not self.validate_compatibility():
            logger.error("❌ Compatibility validation failed")
            results['status'] = 'failed'
            return results
        
        # Load and validate checkpoints
        checkpoint_state = self._load_and_validate_checkpoints()
        
        if checkpoint_state == "invalid":
            logger.error("❌ Checkpoint is INVALID - STOPPING execution")
            logger.error("❌ Please archive or remove the invalid checkpoint before rerunning")
            results['status'] = 'failed'
            results['error'] = 'Invalid checkpoint'
            return results
        
        # Run analysis for missing seeds
        for seed in SEEDS:
            if seed in self.completed_seeds:
                logger.info(f"⏭️  Seed {seed} already completed - skipping")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📊 Processing seed {seed} ({SEEDS.index(seed) + 1}/{len(SEEDS)})")
            logger.info(f"{'='*80}")
            
            success, metric_results, transition_results = self.run_single_seed(seed)
            
            if success and metric_results and transition_results:
                # Strict validation
                is_valid, error_msg = self._validate_seed_results(metric_results, transition_results, seed)
                if not is_valid:
                    logger.error(f"❌ Seed {seed} failed strict validation: {error_msg}")
                    results['status'] = 'failed'
                    results['error'] = f'Seed {seed} validation failed: {error_msg}'
                    return results
                
                # Cross-validate generated seed
                is_valid, error_msg = self._cross_validate_generated_seed(seed, metric_results)
                if not is_valid:
                    logger.error(f"❌ Seed {seed} failed reference cross-validation: {error_msg}")
                    results['status'] = 'failed'
                    results['error'] = f'Seed {seed} cross-validation failed: {error_msg}'
                    return results
                
                # Cache results
                self.seed_results_cache[seed] = metric_results
                self.transition_results_cache[seed] = transition_results
                self.completed_seeds.append(seed)
                
                # Save seed checkpoint
                self._save_seed_checkpoint(seed, metric_results, transition_results)
                
                # Update main checkpoint
                self._save_main_checkpoint()
                
                logger.info(f"✅ Seed {seed} completed successfully")
            else:
                logger.error(f"❌ Seed {seed} failed")
                results['status'] = 'failed'
                return results
        
        # Compute summary
        self._compute_multiseed_summary()
        
        # Save all outputs
        saved_paths = self._save_all_outputs()
        results['outputs_saved'] = saved_paths
        
        # Generate figures
        self._generate_figures()
        results['outputs_saved'].extend([
            str(self.error_analysis_dir / 'figures' / 'error_transition_summary.png'),
            str(self.error_analysis_dir / 'figures' / 'false_negative_correction.png'),
            str(self.error_analysis_dir / 'figures' / 'false_positive_tradeoff.png'),
            str(self.error_analysis_dir / 'figures' / 'persistent_errors_by_model.png')
        ])
        
        # Generate report
        self._generate_report()
        results['outputs_saved'].append(
            str(self.error_analysis_dir / 'reports' / 'four_model_error_analysis_report.txt')
        )
        
        # Determine completion status
        if len(self.completed_seeds) == len(SEEDS):
            results['status'] = 'complete'
            results['analysis_complete'] = True
        elif len(self.completed_seeds) > 0:
            results['status'] = 'partial'
            results['analysis_complete'] = False
        else:
            results['status'] = 'failed'
            results['analysis_complete'] = False
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute four-model error analysis."""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'four_model_error_analysis')
    
    logger.info("\n" + "="*80)
    logger.info("🔬 FOUR-MODEL ERROR ANALYSIS")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    error_analysis_dir = base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis'
    error_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Error analysis directory: {error_analysis_dir}")
    logger.info(f"⚠️  Does NOT overwrite existing two-model error analysis outputs")
    logger.info(f"⚠️  LSTM uses selected weight 1.0 (no additional up-weighting)")
    logger.info(f"⚠️  Checkpoint/resume enabled for long-running analysis")
    
    analysis = FourModelErrorAnalysis(base_dir=base_dir, error_analysis_dir=error_analysis_dir)
    results = analysis.run_full_analysis()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 FOUR-MODEL ERROR ANALYSIS COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs saved to: {error_analysis_dir}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    logger.info("="*80)


if __name__ == '__main__':
    main()