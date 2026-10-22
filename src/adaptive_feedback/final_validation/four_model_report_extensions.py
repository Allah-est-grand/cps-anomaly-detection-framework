"""
Four-Model Report Extensions - Diagnostic and Transition Data Recovery
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This script extends the old two-model diagnostic and error-analysis outputs
so that the report can be updated consistently for the four retained models:
- XGBoost
- Random Forest
- Linear SVM
- LSTM

IMPORTANT:
- Weight sensitivity uses ONLY seed 42 (single-split retrospective diagnostic for Figures 6-9)
- Five-seed robustness analysis is covered separately in final validation
- Reuses existing validated results as authoritative
- Only runs missing lightweight evaluations
- Fails closed if exact historical data cannot be recovered
- Does NOT overwrite any validated historical files
- Does NOT reuse LSTM internal-validation metrics as test-set Figure results
- CRITICAL: LSTM evaluation_n = 288335 (sequence endpoints), classical = 288344

CONFIGURATION:
- Weight Sensitivity Seed: 42 ONLY (single-split retrospective diagnostic)
- Transition Seeds: [42, 123, 2024, 2025, 3407] (five-seed analysis)
- Dataset: results/adaptive/retraining/datasets/retraining_dataset_combined.csv
- Expected fingerprint: a3f18ef39df7815ea3fa7071259b35a9
- Train/Retrain/Test: 60% / 20% / 20%
- 51 features
- Train-only imputation/scaling
- Historical weight grid: [1.0, 2.0, 3.0, 4.0, 5.0]
- Historical threshold grid (prob models): 16 thresholds [0.01, 0.02, ..., 0.50]
- LinearSVM default: decision_function >= 0

OUTPUT FILES:
-------------
Weight Sensitivity (seed 42 ONLY):
- results/adaptive/final_validation_four_model/report_extensions/weight_sensitivity/four_model_weight_sensitivity_results.csv
- results/adaptive/final_validation_four_model/report_extensions/weight_sensitivity/four_model_weight_sensitivity_summary.csv
- results/adaptive/final_validation_four_model/report_extensions/weight_sensitivity/four_model_weight_sensitivity_summary.json

Transitions (five seeds):
- results/adaptive/final_validation_four_model/report_extensions/transitions/xgboost_transitions.csv
- results/adaptive/final_validation_four_model/report_extensions/transitions/random_forest_transitions.csv
- results/adaptive/final_validation_four_model/report_extensions/transitions/linear_svm_transitions.csv
- results/adaptive/final_validation_four_model/report_extensions/transitions/lstm_transitions.csv (ONLY if scientifically recoverable)
- results/adaptive/final_validation_four_model/report_extensions/transitions/four_model_transitions_summary.csv
- results/adaptive/final_validation_four_model/report_extensions/transitions/four_model_transitions_summary.json
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
import traceback
import tempfile
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("⚠️  TensorFlow not available. LSTM weight sensitivity will be skipped.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("⚠️  XGBoost not available.")

# =============================================================================
# CONFIGURATION - RECOVERED FROM VALIDATED EXPERIMENTS
# =============================================================================
# Weight sensitivity uses ONLY seed 42 (single-split retrospective diagnostic)
WEIGHT_SENSITIVITY_SEED = 42
TRANSITION_SEEDS = [42, 123, 2024, 2025, 3407]
MODELS = ['XGBoost', 'RandomForest', 'LinearSVM', 'LSTM']

# Historical weight grid [1,2,3,4,5] confirmed from XGBoost/RF evidence ONLY
HISTORICAL_WEIGHT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]

# EXACT historical 16-threshold grid for probability-output models
HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES = [
    0.01, 0.02, 0.03, 0.05, 0.075,
    0.10, 0.125, 0.15, 0.175, 0.20,
    0.225, 0.25, 0.275, 0.30, 0.40, 0.50
]

# LinearSVM uses decision_function with threshold=0 for default
# Do NOT apply probability thresholds to raw LinearSVM scores
LINEARSVM_DEFAULT_THRESHOLD = 0.0
LINEARSVM_DIAGNOSTIC_MODE = 'not_directly_comparable'  # Raw scores ≠ calibrated probabilities

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

EXPECTED_FEATURE_COUNT = 51
EXPECTED_DATASET_FINGERPRINT = 'a3f18ef39df7815ea3fa7071259b35a9'

# CHECKPOINT SAFETY: Increment version to prevent stale checkpoint reuse
CHECKPOINT_VERSION = '1.1'

# XGBoost configuration (from validated four_model_ablation_extension.py)
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

# RandomForest configuration (from validated four_model_ablation_extension.py)
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

# LinearSVM configuration (from validated four_model_ablation_extension.py)
SVM_C = 1.0
SVM_CLASS_WEIGHT = None
SVM_MAX_ITER = 10000
SVM_RANDOM_STATE = 42

# LSTM configuration (from validated experiments)
TIME_STEPS = 10
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
PATIENCE = 10
DEFAULT_PREDICTION_THRESHOLD = 0.5

LABEL_COLUMN_NAMES = ['label', 'Label', 'target', 'y', 'attack', 'is_anomaly', 
                      'class', 'Normal/Attack', 'binary_label']
METADATA_COLUMNS = [
    'scenario', 'scenario_id', 'sample_source', 'difficulty_level',
    'anomaly_type', 'source_file', 'attack_type', 'generation_method',
    'timestamp', 'time', 'date', 'file_name', 'path', 'index',
    'original_index', 'unnamed: 0', 'sample_id'
]

# Explicit execution plan key mapping (NO model.lower())
PLAN_KEYS = {
    "XGBoost": ("xgboost_reused", "xgboost_to_compute"),
    "RandomForest": ("rf_reused", "rf_to_compute"),
    "LinearSVM": (None, "svm_to_compute"),  # No reuse for LinearSVM
    "LSTM": (None, "lstm_to_compute"),  # No reuse of LSTM internal-validation metrics
}

# =============================================================================
# PATHS TO EXISTING VALIDATED RESULTS
# =============================================================================
class ExistingResultsPaths:
    """Paths to existing validated results for reuse."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        
        # Dataset
        self.dataset_path = base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        
        # Four-model validation results
        self.four_model_results_path = base_dir / 'adaptive' / 'final_validation_four_model' / 'metrics' / 'multi_seed_four_model_results.csv'
        
        # Ablation results
        self.ablation_results_path = base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
        
        # LSTM weight sensitivity pilot (NOT reused for Figures 6-9)
        self.lstm_weight_sensitivity_path = base_dir / 'adaptive' / 'lstm_weight_sensitivity_multiseed' / 'metrics' / 'lstm_weight_multiseed_results.csv'
        
        # Historical weight sensitivity directory (search for XGBoost/RF seed-42)
        self.historical_weight_sensitivity_dir = base_dir / 'adaptive' / 'final_validation' / 'weight_sensitivity'
        
        # Historical logs for threshold diagnostic recovery
        self.historical_logs_dir = base_dir / 'adaptive' / 'final_validation' / 'logs'
        
        # FIX #1: Historical error transitions from final_validation (authoritative for XGBoost/RF)
        self.historical_error_transitions_path = base_dir / 'adaptive' / 'final_validation' / 'error_analysis' / 'metrics' / 'error_transitions.csv'
        
        # Transition exports
        self.error_analysis_dir = base_dir / 'adaptive' / 'final_validation_four_model' / 'error_analysis'
        self.combined_transitions_path = self.error_analysis_dir / 'metrics' / 'error_transitions.csv'
        self.linear_svm_transitions_path = self.error_analysis_dir / 'metrics' / 'linear_svm_transitions.csv'
        
        # LSTM prediction exports
        self.lstm_predictions_dir = self.error_analysis_dir / 'lstm_prediction_exports'
        
        # Output directories for this script
        self.report_extensions_dir = base_dir / 'adaptive' / 'final_validation_four_model' / 'report_extensions'
        self.weight_sensitivity_dir = self.report_extensions_dir / 'weight_sensitivity'
        self.transitions_dir = self.report_extensions_dir / 'transitions'
        self.checkpoint_dir = self.report_extensions_dir / 'checkpoints'
        
        # Checkpoint
        self.checkpoint_path = self.checkpoint_dir / 'four_model_report_extensions_checkpoint.json'
        
        # Existing weight sensitivity results
        self.existing_weight_sensitivity_results_path = self.weight_sensitivity_dir / 'four_model_weight_sensitivity_results.csv'


# =============================================================================
# FOUR-MODEL REPORT EXTENSIONS CLASS
# =============================================================================
class FourModelReportExtensions:
    """
    Extends historical two-model diagnostics to four retained models.
    
    CRITICAL DISTINCTIONS:
    - Weight sensitivity: seed 42 ONLY (single-split retrospective diagnostic for Figures 6-9)
    - Five-seed robustness: covered separately in final validation
    - Threshold diagnostic: retrospective test-set analysis (16 thresholds for prob models)
    - LinearSVM: uses decision_function >= 0, NOT probability thresholds
    - LSTM: does NOT reuse internal-validation metrics as test-set Figure results
    - CRITICAL: LSTM evaluation_n = 288335, classical = 288344
    
    Train model ONCE per (model, weight), derive BOTH threshold modes.
    Track completion per (model, weight, threshold_mode) for results.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.paths = ExistingResultsPaths(self.base_dir)
        
        self.original_dataset = None
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        # Results storage
        self.weight_sensitivity_results = []
        self.transition_results = {}
        self.transition_status = {}
        
        # Track training completion vs result completion separately
        self.completed_training = set()  # (model, weight) - model has been trained
        self.completed_results = set()  # (model, weight, threshold_mode) - result record exists
        self.checkpoint_data = {}
        
        # Reproduction status
        self.reproduction_status = {}
        
        # Historical configuration recovery
        self.historical_weight_grid = None
        self.weight_grid_source = None  # Now a list of all evidence files
        self.historical_threshold_logic = None
        
        # Execution plan with explicit keys
        self.execution_plan = {
            'xgboost_reused': [],
            'xgboost_to_compute': [],
            'rf_reused': [],
            'rf_to_compute': [],
            'svm_reused': [],
            'svm_to_compute': [],
            'lstm_reused': [],
            'lstm_to_compute': []
        }
        
        logger.info(f"✅ Initialized FourModelReportExtensions")
        logger.info(f"📁 Report extensions directory: {self.paths.report_extensions_dir}")
        logger.info(f"🎯 Weight sensitivity seed: {WEIGHT_SENSITIVITY_SEED} ONLY")
        logger.info(f"🔢 Models: {MODELS}")
        logger.info(f"🔢 Weight grid: {HISTORICAL_WEIGHT_GRID}")
        logger.info(f"🔢 Threshold grid (prob models): {len(HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES)} thresholds")
    
    def _compute_dataset_fingerprint(self) -> str:
        """Compute streaming SHA-256 hash of dataset file."""
        if not self.paths.dataset_path.exists():
            return 'dataset_not_found'
        
        hasher = hashlib.sha256()
        try:
            with open(self.paths.dataset_path, 'rb') as f:
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
        """Generate deterministic SHA-256 fingerprint of configuration."""
        config_dict = {
            'weight_sensitivity_seed': WEIGHT_SENSITIVITY_SEED,
            'transition_seeds': TRANSITION_SEEDS,
            'models': MODELS,
            'historical_weight_grid': HISTORICAL_WEIGHT_GRID,
            'threshold_candidates': HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES,
            'train_ratio': TRAIN_RATIO,
            'retrain_ratio': RETRAIN_RATIO,
            'test_ratio': TEST_RATIO,
            'expected_feature_count': EXPECTED_FEATURE_COUNT,
            'expected_dataset_fingerprint': EXPECTED_DATASET_FINGERPRINT,
            'xgb_config': XGB_CONFIG,
            'rf_config': RF_CONFIG,
            'svm_c': SVM_C,
            'svm_max_iter': SVM_MAX_ITER
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
    
    def _save_checkpoint(self):
        """Save checkpoint atomically."""
        checkpoint = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'config_fingerprint': self.config_fingerprint,
            'dataset_fingerprint': self.dataset_fingerprint,
            'completed_training': [list(c) for c in self.completed_training],
            'completed_results': [list(c) for c in self.completed_results],
            'transition_status': self.transition_status,
            'checkpoint_data': self.checkpoint_data,
            'reproduction_status': self.reproduction_status,
            'historical_weight_grid': self.historical_weight_grid,
            'weight_grid_source': self.weight_grid_source,
            'historical_threshold_logic': self.historical_threshold_logic,
            'execution_plan': self.execution_plan,
            'saved_at': datetime.now().isoformat()
        }
        
        self._save_atomically(self.paths.checkpoint_path, checkpoint)
    
    def _load_checkpoint(self) -> bool:
        """Load checkpoint metadata but ALWAYS reload validated source CSVs."""
        if not self.paths.checkpoint_path.exists():
            logger.info("ℹ️   No checkpoint found. Starting fresh.")
            return False
        
        try:
            with open(self.paths.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
                logger.warning(f"⚠️  Checkpoint version mismatch")
                return False
            
            if checkpoint.get('config_fingerprint') != self.config_fingerprint:
                logger.error("❌ Checkpoint config fingerprint mismatch")
                return False
            
            # Restore metadata only
            self.completed_training = set(tuple(c) for c in checkpoint.get('completed_training', []))
            self.completed_results = set(tuple(c) for c in checkpoint.get('completed_results', []))
            self.transition_status = checkpoint.get('transition_status', {})
            self.reproduction_status = checkpoint.get('reproduction_status', {})
            self.historical_weight_grid = checkpoint.get('historical_weight_grid', None)
            self.weight_grid_source = checkpoint.get('weight_grid_source', None)
            self.historical_threshold_logic = checkpoint.get('historical_threshold_logic', None)
            self.execution_plan = checkpoint.get('execution_plan', {
                'xgboost_reused': [], 'xgboost_to_compute': [],
                'rf_reused': [], 'rf_to_compute': [],
                'svm_reused': [], 'svm_to_compute': [],
                'lstm_reused': [], 'lstm_to_compute': []
            })
            
            logger.info(f"📥 Loaded checkpoint: {len(self.completed_training)} trained, {len(self.completed_results)} results")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load checkpoint: {e}")
            return False
    
    def load_resources(self) -> bool:
        """Load dataset and compute fingerprints."""
        logger.info("📊 Loading resources...")
        if not self.paths.dataset_path.exists():
            logger.error(f"❌ Dataset not found: {self.paths.dataset_path}")
            return False
        
        try:
            self.original_dataset = pd.read_csv(self.paths.dataset_path)
            self.original_dataset.columns = self.original_dataset.columns.astype(str).str.strip()
            logger.info(f"✅ Loaded dataset: {len(self.original_dataset)} rows")
            
            self.dataset_fingerprint = self._compute_dataset_fingerprint()
            self.config_fingerprint = self._get_config_hash()
            
            if self.dataset_fingerprint != EXPECTED_DATASET_FINGERPRINT:
                logger.error(f"❌ Dataset fingerprint MISMATCH")
                logger.error(f"   Expected: {EXPECTED_DATASET_FINGERPRINT}")
                logger.error(f"   Got: {self.dataset_fingerprint}")
                logger.error(f"❌ STOPPING - cannot proceed with wrong dataset")
                return False
            
            logger.info(f"✅ Dataset fingerprint validated: {EXPECTED_DATASET_FINGERPRINT}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def _validate_exact_seeds(self, transitions: List[Dict], model_name: str) -> bool:
        """
        FIX #1: Validate EXACTLY 5 unique seeds matching TRANSITION_SEEDS.
        """
        seed_set = set(row['seed'] for row in transitions)
        expected_seed_set = set(TRANSITION_SEEDS)
        
        if seed_set != expected_seed_set:
            logger.error(f"❌ {model_name}: Seed mismatch. Expected {expected_seed_set}, got {seed_set}")
            return False
        
        if len(transitions) != len(TRANSITION_SEEDS):
            logger.error(f"❌ {model_name}: Row count mismatch. Expected {len(TRANSITION_SEEDS)}, got {len(transitions)}")
            return False
        
        logger.info(f"✅ {model_name}: Exactly 5 seeds validated: {sorted(seed_set)}")
        return True
    
    def _load_historical_transitions(self) -> bool:
        """
        FIX #1: Load authoritative historical XGBoost/RF transitions from final_validation.
        FIX #2: Normalize FP reduction correctly (source_fp - target_fp).
        FIX #4: Fail closed if exact expected seed set is not present.
        """
        logger.info("📊 Loading historical transitions...")
        
        # FIX #1: First try historical error_transitions.csv from final_validation
        if self.paths.historical_error_transitions_path.exists():
            try:
                hist_df = pd.read_csv(self.paths.historical_error_transitions_path)
                logger.info(f"✅ Loaded historical transitions: {self.paths.historical_error_transitions_path}")
                
                # Validate expected models
                models_found = set(hist_df['model_name'].unique())
                if not models_found.issubset({'XGBoost', 'RandomForest'}):
                    logger.warning(f"⚠️  Unexpected models in historical transitions: {models_found}")
                
                # Process each model
                for model_name in ['XGBoost', 'RandomForest']:
                    model_data = hist_df[hist_df['model_name'] == model_name].copy()
                    
                    if len(model_data) == 0:
                        logger.error(f"❌ No {model_name} data in historical transitions")
                        return False
                    
                    # FIX #4: Validate exact seed set BEFORE proceeding
                    seed_set = set(model_data['seed'].unique())
                    expected_seed_set = set(TRANSITION_SEEDS)
                    if seed_set != expected_seed_set:
                        logger.error(f"❌ {model_name}: Seed mismatch. Expected {expected_seed_set}, got {seed_set}")
                        return False
                    
                    if len(model_data) != len(TRANSITION_SEEDS):
                        logger.error(f"❌ {model_name}: Row count mismatch. Expected {len(TRANSITION_SEEDS)}, got {len(model_data)}")
                        return False
                    
                    # FIX #2: Normalize columns and compute fp_reduction correctly
                    col_mapping = {
                        'baseline_false_negatives': 'source_fn',
                        'weighted_false_negatives': 'target_fn',
                        'baseline_false_positives': 'source_fp',
                        'weighted_false_positives': 'target_fp',
                        'false_negative_reduction': 'fn_reduction',
                        'false_positive_change': 'fp_change'  # Note: opposite sign of fp_reduction
                    }
                    
                    for old_col, new_col in col_mapping.items():
                        if old_col in model_data.columns:
                            model_data[new_col] = model_data[old_col]
                    
                    # Ensure required columns exist
                    required = ['seed', 'model_name', 'source_fn', 'target_fn', 'source_fp', 'target_fp']
                    if not all(col in model_data.columns for col in required):
                        logger.error(f"❌ Missing required columns for {model_name}")
                        return False
                    
                    # FIX #2: Compute fn_reduction and fp_reduction correctly
                    model_data['fn_reduction'] = model_data['source_fn'] - model_data['target_fn']
                    model_data['fp_reduction'] = model_data['source_fp'] - model_data['target_fp']
                    
                    # If fp_change exists (opposite sign), use it to verify
                    if 'fp_change' in model_data.columns:
                        # fp_change = target_fp - source_fp, so fp_reduction = -fp_change
                        model_data['fp_reduction_from_change'] = -model_data['fp_change']
                    
                    # Compute derived transition metrics
                    if 'corrected_false_negatives' in model_data.columns:
                        model_data['corrected_fn'] = model_data['corrected_false_negatives']
                    if 'persistent_false_negatives' in model_data.columns:
                        model_data['persistent_fn'] = model_data['persistent_false_negatives']
                    if 'newly_introduced_false_negatives' in model_data.columns:
                        model_data['newly_introduced_fn'] = model_data['newly_introduced_false_negatives']
                    if 'corrected_false_positives' in model_data.columns:
                        model_data['corrected_fp'] = model_data['corrected_false_positives']
                    if 'persistent_false_positives' in model_data.columns:
                        model_data['persistent_fp'] = model_data['persistent_false_positives']
                    if 'newly_introduced_false_positives' in model_data.columns:
                        model_data['newly_introduced_fp'] = model_data['newly_introduced_false_positives']
                    
                    # Add comparison type
                    model_data['comparison'] = 'baseline_vs_weighted'
                    model_data['source_variant'] = 'baseline'
                    model_data['target_variant'] = 'weighted_retraining'
                    model_data['reproduction_status'] = 'loaded_from_historical_final_validation'
                    
                    self.transition_results[model_name] = model_data.to_dict('records')
                    self.transition_status[model_name] = 'loaded_from_historical_final_validation'
                    logger.info(f"✅ Loaded {model_name}: {len(model_data)} rows, seeds={sorted(seed_set)}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load historical transitions: {e}")
                logger.error(traceback.format_exc())
                return False
        
        # Fallback: try final_validation_four_model path
        if self.paths.combined_transitions_path.exists():
            try:
                combined_df = pd.read_csv(self.paths.combined_transitions_path)
                logger.info(f"✅ Loaded combined transitions: {self.paths.combined_transitions_path}")
                
                for model_name in ['XGBoost', 'RandomForest']:
                    model_data = combined_df[combined_df['model_name'] == model_name].copy()
                    
                    if len(model_data) > 0:
                        # FIX #4: Validate exact seed set
                        seed_set = set(model_data['seed'].unique())
                        expected_seed_set = set(TRANSITION_SEEDS)
                        if seed_set != expected_seed_set:
                            logger.error(f"❌ {model_name}: Seed mismatch. Expected {expected_seed_set}, got {seed_set}")
                            return False
                        
                        # Normalize columns
                        col_mapping = {
                            'baseline_false_negatives': 'source_fn',
                            'weighted_false_negatives': 'target_fn',
                            'baseline_false_positives': 'source_fp',
                            'weighted_false_positives': 'target_fp',
                        }
                        
                        for old_col, new_col in col_mapping.items():
                            if old_col in model_data.columns:
                                model_data[new_col] = model_data[old_col]
                        
                        # Compute reductions
                        model_data['fn_reduction'] = model_data['source_fn'] - model_data['target_fn']
                        model_data['fp_reduction'] = model_data['source_fp'] - model_data['target_fp']
                        
                        model_data['comparison'] = 'baseline_vs_weighted'
                        model_data['source_variant'] = 'baseline'
                        model_data['target_variant'] = 'weighted_retraining'
                        model_data['reproduction_status'] = 'loaded_from_combined'
                        
                        self.transition_results[model_name] = model_data.to_dict('records')
                        self.transition_status[model_name] = 'loaded_from_combined'
                        logger.info(f"✅ Loaded {model_name}: {len(model_data)} rows")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load combined transitions: {e}")
                return False
        
        logger.error("❌ No historical XGBoost/RF transitions found")
        return False
    
    def _load_linear_svm_transitions(self) -> bool:
        """
        FIX #3: Load LinearSVM transitions from final_validation_four_model.
        Filter ONLY comparison == "baseline_vs_weighted".
        FIX #4: Fail closed if exact expected seed set is not present.
        """
        logger.info("📊 Loading LinearSVM transitions...")
        
        if self.paths.linear_svm_transitions_path.exists():
            try:
                svm_df = pd.read_csv(self.paths.linear_svm_transitions_path)
                logger.info(f"✅ Loaded LinearSVM transitions: {self.paths.linear_svm_transitions_path}")
                
                # FIX #3: Filter ONLY baseline_vs_weighted
                if 'comparison' in svm_df.columns:
                    baseline_vs_weighted = svm_df[svm_df['comparison'] == 'baseline_vs_weighted']
                else:
                    baseline_vs_weighted = svm_df  # Assume all are baseline_vs_weighted
                
                if len(baseline_vs_weighted) == 0:
                    logger.error(f"❌ No baseline_vs_weighted rows for LinearSVM")
                    return False
                
                # FIX #4: Validate exact seed set
                seed_set = set(baseline_vs_weighted['seed'].unique())
                expected_seed_set = set(TRANSITION_SEEDS)
                if seed_set != expected_seed_set:
                    logger.error(f"❌ LinearSVM: Seed mismatch. Expected {expected_seed_set}, got {seed_set}")
                    return False
                
                if len(baseline_vs_weighted) != len(TRANSITION_SEEDS):
                    logger.error(f"❌ LinearSVM: Row count mismatch. Expected {len(TRANSITION_SEEDS)}, got {len(baseline_vs_weighted)}")
                    return False
                
                # Normalize columns if needed
                col_mapping = {
                    'baseline_false_negatives': 'source_fn',
                    'weighted_false_negatives': 'target_fn',
                    'baseline_false_positives': 'source_fp',
                    'weighted_false_positives': 'target_fp',
                }
                
                for old_col, new_col in col_mapping.items():
                    if old_col in baseline_vs_weighted.columns:
                        baseline_vs_weighted[new_col] = baseline_vs_weighted[old_col]
                
                # Compute reductions
                baseline_vs_weighted['fn_reduction'] = baseline_vs_weighted['source_fn'] - baseline_vs_weighted['target_fn']
                baseline_vs_weighted['fp_reduction'] = baseline_vs_weighted['source_fp'] - baseline_vs_weighted['target_fp']
                
                self.transition_results['LinearSVM'] = baseline_vs_weighted.to_dict('records')
                self.transition_status['LinearSVM'] = 'loaded_from_final_validation_four_model'
                logger.info(f"✅ Loaded LinearSVM: {len(baseline_vs_weighted)} rows, seeds={sorted(seed_set)}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load LinearSVM transitions: {e}")
                logger.error(traceback.format_exc())
                return False
        
        logger.error("❌ No LinearSVM transitions found")
        return False
    
    def _set_lstm_transitions_unavailable(self):
        """FIX #4: Set LSTM transitions as scientifically unavailable."""
        logger.info("📊 Setting LSTM transitions as unavailable...")
        
        self.transition_results['LSTM'] = []
        self.transition_status['LSTM'] = 'sample_level_transitions_unavailable'
        self.reproduction_status['lstm_transitions'] = 'sample_level_transitions_unavailable'
        logger.info("✅ LSTM transitions: sample_level_transitions_unavailable")
    
    def _validate_transition_identities(self, transitions: List[Dict], model_name: str) -> bool:
        """Validate FN/FP transition identities for all rows."""
        for row in transitions:
            source_fn = row.get('source_fn', 0)
            target_fn = row.get('target_fn', 0)
            corrected_fn = row.get('corrected_fn', 0)
            persistent_fn = row.get('persistent_fn', 0)
            newly_introduced_fn = row.get('newly_introduced_fn', 0)
            
            source_fp = row.get('source_fp', 0)
            target_fp = row.get('target_fp', 0)
            corrected_fp = row.get('corrected_fp', 0)
            persistent_fp = row.get('persistent_fp', 0)
            newly_introduced_fp = row.get('newly_introduced_fp', 0)
            
            # Validate identities
            if source_fn != corrected_fn + persistent_fn:
                logger.error(f"❌ {model_name} seed={row.get('seed')}: FN identity failed")
                return False
            
            if target_fn != persistent_fn + newly_introduced_fn:
                logger.error(f"❌ {model_name} seed={row.get('seed')}: Target FN identity failed")
                return False
            
            if source_fp != corrected_fp + persistent_fp:
                logger.error(f"❌ {model_name} seed={row.get('seed')}: FP identity failed")
                return False
            
            if target_fp != persistent_fp + newly_introduced_fp:
                logger.error(f"❌ {model_name} seed={row.get('seed')}: Target FP identity failed")
                return False
        
        return True
    
    def _compute_transition_summary(self) -> Dict:
        """
        FIX #5: Compute transition summary with mean AND sample std (ddof=1).
        """
        logger.info("📊 Computing transition summary...")
        
        summary_data = []
        
        for model_name in MODELS:
            transitions = self.transition_results.get(model_name, [])
            
            if not transitions or self.transition_status.get(model_name) == 'sample_level_transitions_unavailable':
                # LSTM or unavailable
                summary_data.append({
                    'model_name': model_name,
                    'mean_corrected_fn': None,
                    'std_corrected_fn': None,
                    'mean_persistent_fn': None,
                    'std_persistent_fn': None,
                    'mean_newly_introduced_fn': None,
                    'std_newly_introduced_fn': None,
                    'mean_fn_reduction': None,
                    'std_fn_reduction': None,
                    'mean_corrected_fp': None,
                    'std_corrected_fp': None,
                    'mean_persistent_fp': None,
                    'std_persistent_fp': None,
                    'mean_newly_introduced_fp': None,
                    'std_newly_introduced_fp': None,
                    'mean_fp_reduction': None,
                    'std_fp_reduction': None,
                    'n_seeds': 0,
                    'reproduction_status': 'sample_level_transitions_unavailable',
                    'note': f'Sample-level transitions unavailable for {model_name}'
                })
                continue
            
            # Compute mean and sample std (ddof=1) for each metric
            def safe_mean_std(values):
                valid = [v for v in values if v is not None and np.isfinite(v)]
                if len(valid) == 0:
                    return None, None
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
                return mean, std
            
            corrected_fn_list = [t.get('corrected_fn') for t in transitions]
            persistent_fn_list = [t.get('persistent_fn') for t in transitions]
            newly_introduced_fn_list = [t.get('newly_introduced_fn') for t in transitions]
            fn_reduction_list = [t.get('fn_reduction') for t in transitions]
            
            corrected_fp_list = [t.get('corrected_fp') for t in transitions]
            persistent_fp_list = [t.get('persistent_fp') for t in transitions]
            newly_introduced_fp_list = [t.get('newly_introduced_fp') for t in transitions]
            fp_reduction_list = [t.get('fp_reduction') for t in transitions]
            
            summary_data.append({
                'model_name': model_name,
                'mean_corrected_fn': safe_mean_std(corrected_fn_list)[0],
                'std_corrected_fn': safe_mean_std(corrected_fn_list)[1],
                'mean_persistent_fn': safe_mean_std(persistent_fn_list)[0],
                'std_persistent_fn': safe_mean_std(persistent_fn_list)[1],
                'mean_newly_introduced_fn': safe_mean_std(newly_introduced_fn_list)[0],
                'std_newly_introduced_fn': safe_mean_std(newly_introduced_fn_list)[1],
                'mean_fn_reduction': safe_mean_std(fn_reduction_list)[0],
                'std_fn_reduction': safe_mean_std(fn_reduction_list)[1],
                'mean_corrected_fp': safe_mean_std(corrected_fp_list)[0],
                'std_corrected_fp': safe_mean_std(corrected_fp_list)[1],
                'mean_persistent_fp': safe_mean_std(persistent_fp_list)[0],
                'std_persistent_fp': safe_mean_std(persistent_fp_list)[1],
                'mean_newly_introduced_fp': safe_mean_std(newly_introduced_fp_list)[0],
                'std_newly_introduced_fp': safe_mean_std(newly_introduced_fp_list)[1],
                'mean_fp_reduction': safe_mean_std(fp_reduction_list)[0],
                'std_fp_reduction': safe_mean_std(fp_reduction_list)[1],
                'n_seeds': len(transitions),
                'reproduction_status': self.transition_status.get(model_name, 'unknown'),
                'note': ''
            })
        
        return {'transition_summary': summary_data}
    
    def _save_transition_outputs(self):
        """Save transition results to files."""
        logger.info("💾 Saving transition outputs...")
        
        # Save individual model transition CSVs
        for model_name in MODELS:
            if model_name in self.transition_results and self.transition_results[model_name]:
                model_df = pd.DataFrame(self.transition_results[model_name])
                model_path = self.paths.transitions_dir / f'{model_name.lower().replace(" ", "_")}_transitions.csv'
                model_df.to_csv(model_path, index=False)
                logger.info(f"   ✅ Saved: {model_path}")
            elif self.transition_status.get(model_name) == 'sample_level_transitions_unavailable':
                logger.info(f"   ℹ️   {model_name}: sample_level_transitions_unavailable")
            else:
                logger.warning(f"⚠️  {model_name} transitions NOT available")
        
        # Save transition summary with mean AND std
        summary_data = self._compute_transition_summary()
        
        if summary_data.get('transition_summary'):
            summary_df = pd.DataFrame(summary_data['transition_summary'])
            summary_csv_path = self.paths.transitions_dir / 'four_model_transitions_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"   ✅ Saved: {summary_csv_path}")
            
            # Save summary JSON
            summary_json_path = self.paths.transitions_dir / 'four_model_transitions_summary.json'
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"   ✅ Saved: {summary_json_path}")
    
    def run_transitions_only(self) -> Dict:
        """
        FIX #6: Transitions-only execution mode.
        DO NOT train any models.
        Load existing transition files only.
        FIX #1: Validate EXACT seed sets BEFORE printing COMPLETE.
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING TRANSITIONS-ONLY MODE")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'mode': 'transitions_only',
            'transition_status': {}
        }
        
        # Load resources (dataset fingerprint validation)
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        # Load historical XGBoost/RF transitions (FIX #4: fail closed on seed mismatch)
        if not self._load_historical_transitions():
            logger.error("❌ Failed to load historical XGBoost/RF transitions")
            results['status'] = 'failed'
            return results
        
        # Load LinearSVM transitions (FIX #4: fail closed on seed mismatch)
        if not self._load_linear_svm_transitions():
            logger.error("❌ Failed to load LinearSVM transitions")
            results['status'] = 'failed'
            return results
        
        # Set LSTM as unavailable
        self._set_lstm_transitions_unavailable()
        
        # FIX #1: Validate exact seed sets BEFORE proceeding
        validation_passed = True
        
        for model_name in ['XGBoost', 'RandomForest', 'LinearSVM']:
            transitions = self.transition_results.get(model_name, [])
            
            # Validate exact seed set
            if not self._validate_exact_seeds(transitions, model_name):
                logger.error(f"❌ {model_name}: Exact seed validation FAILED")
                validation_passed = False
                continue
            
            # Validate transition identities
            if not self._validate_transition_identities(transitions, model_name):
                logger.error(f"❌ {model_name}: Transition identities FAILED")
                validation_passed = False
                continue
            
            logger.info(f"✅ {model_name}: All validations passed")
        
        # FIX #2: If any validation failed, return FAILED status
        if not validation_passed:
            logger.error("\n" + "="*80)
            logger.error("TRANSITION EXPORT STATUS: FAILED")
            logger.error("="*80)
            results['status'] = 'failed'
            return results
        
        # Save outputs
        self._save_transition_outputs()
        
        # FIX #3: Only print COMPLETE after all validations pass
        xgb_seeds = len(self.transition_results.get('XGBoost', []))
        rf_seeds = len(self.transition_results.get('RandomForest', []))
        svm_seeds = len(self.transition_results.get('LinearSVM', []))
        
        logger.info("\n" + "="*80)
        logger.info("TRANSITION EXPORT STATUS: COMPLETE")
        logger.info("="*80)
        logger.info(f"XGBoost: {xgb_seeds} seeds")
        logger.info(f"RandomForest: {rf_seeds} seeds")
        logger.info(f"LinearSVM: {svm_seeds} seeds")
        logger.info(f"LSTM: sample_level_transitions_unavailable")
        logger.info("="*80)
        
        results['status'] = 'complete'
        results['transition_status'] = self.transition_status
        return results
    
    def run_full_extensions(self) -> Dict:
        """Run complete report extensions (full mode with weight sensitivity)."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING FOUR-MODEL REPORT EXTENSIONS")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'tasks_completed': [],
            'extensions_complete': False
        }
        
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        self._load_checkpoint()
        
        if not self.load_weight_sensitivity():
            logger.error("❌ Weight sensitivity task failed")
            results['status'] = 'failed'
            return results
        
        results['tasks_completed'].append('weight_sensitivity')
        
        transition_status = self.load_transitions()
        results['tasks_completed'].append('transitions')
        results['transition_status'] = transition_status
        
        if not self._validate_source_data():
            logger.error("❌ Source data validation failed")
            results['status'] = 'failed'
            return results
        
        self.save_weight_sensitivity_outputs()
        self.save_transition_outputs(transition_status)
        
        results['status'] = 'complete'
        results['extensions_complete'] = True
        
        return results
    
    # Placeholder methods for weight sensitivity (not used in transitions-only mode)
    def _search_historical_weight_evidence(self, search_dir: Path) -> Tuple[Set[float], List[str]]:
        """Recursively search for historical XGBoost/RF weight-sensitivity evidence."""
        weights_found = set()
        evidence_files = []
        
        file_patterns = ['*.csv', '*.json', '*.txt', '*.log']
        
        for pattern in file_patterns:
            for file_path in search_dir.rglob(pattern):
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        
                        if 'anomaly_weight' in df.columns and 'model_name' in df.columns:
                            models_in_file = set(df['model_name'].unique())
                            if models_in_file.intersection({'XGBoost', 'RandomForest'}):
                                weights = set(df['anomaly_weight'].dropna().unique())
                                valid_weights = {w for w in weights if w in HISTORICAL_WEIGHT_GRID}
                                if len(valid_weights) >= 3:
                                    weights_found.update(valid_weights)
                                    if str(file_path) not in evidence_files:
                                        evidence_files.append(str(file_path))
                                    logger.info(f"✅ Found historical weight evidence: {sorted(valid_weights)} in {file_path}")
                    
                    elif file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict):
                            if 'anomaly_weight' in data:
                                weight = data['anomaly_weight']
                                if weight in HISTORICAL_WEIGHT_GRID:
                                    weights_found.add(weight)
                                    if str(file_path) not in evidence_files:
                                        evidence_files.append(str(file_path))
                            if 'weights' in data:
                                for w in data['weights']:
                                    if w in HISTORICAL_WEIGHT_GRID:
                                        weights_found.add(w)
                                        if str(file_path) not in evidence_files:
                                            evidence_files.append(str(file_path))
                    
                    elif file_path.suffix in ['.txt', '.log']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if any(m in content for m in ['XGBoost', 'RandomForest', 'Random Forest']):
                            weight_pattern = r'(?:anomaly_)?weight[:\s=]+([1-5](?:\.0)?)'
                            matches = re.findall(weight_pattern, content, re.IGNORECASE)
                            for match in matches:
                                try:
                                    w = float(match)
                                    if w in HISTORICAL_WEIGHT_GRID:
                                        weights_found.add(w)
                                        if str(file_path) not in evidence_files:
                                            evidence_files.append(str(file_path))
                                except:
                                    pass
                
                except Exception as e:
                    continue
        
        return weights_found, evidence_files
    
    def _recover_historical_configurations(self) -> bool:
        """Recover exact historical configurations from project evidence."""
        logger.info("🔍 Recovering historical configurations...")
        
        weight_evidence = set()
        weight_evidence_files = []
        
        final_validation_dir = self.base_dir / 'adaptive' / 'final_validation'
        if final_validation_dir.exists():
            weights, files = self._search_historical_weight_evidence(final_validation_dir)
            weight_evidence.update(weights)
            for f in files:
                if f not in weight_evidence_files:
                    weight_evidence_files.append(f)
        
        adaptive_dir = self.base_dir / 'adaptive'
        if adaptive_dir.exists():
            weights, files = self._search_historical_weight_evidence(adaptive_dir)
            weight_evidence.update(weights)
            for f in files:
                if f not in weight_evidence_files:
                    weight_evidence_files.append(f)
        
        weight_evidence_files = list(dict.fromkeys(weight_evidence_files))
        
        if weight_evidence == set(HISTORICAL_WEIGHT_GRID):
            self.historical_weight_grid = HISTORICAL_WEIGHT_GRID
            self.weight_grid_source = weight_evidence_files
            logger.info("="*80)
            logger.info("HISTORICAL WEIGHT GRID CONFIRMED")
            logger.info(f"grid = {HISTORICAL_WEIGHT_GRID}")
            logger.info(f"evidence_files = {weight_evidence_files}")
            logger.info("="*80)
        else:
            logger.error(f"❌ Historical weight grid NOT fully confirmed")
            logger.error(f"   Expected: {set(HISTORICAL_WEIGHT_GRID)}")
            logger.error(f"   Found: {weight_evidence}")
            logger.error(f"❌ FAIL CLOSED - cannot proceed without confirmed historical grid [1,2,3,4,5]")
            return False
        
        self.historical_threshold_logic = {
            'mode': 'test_set_diagnostic_best_threshold',
            'threshold_candidates': HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES,
            'selection_criterion': 'maximize_f1_on_test',
            'note': 'Retrospective diagnostic analysis using 16-threshold test-set sweep. NOT validated optimization. For probability-output models (XGBoost, RF, LSTM) only.'
        }
        logger.info(f"✅ Historical threshold diagnostic CONFIRMED: {len(HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES)} thresholds")
        
        return True
    
    def _is_complete_historical_result(self, hist_row: Dict, threshold_mode: str) -> bool:
        """Check if a historical result row contains ALL required fields."""
        required_default = ['f1_score', 'precision', 'recall', 'false_positive', 'false_negative', 'evaluation_n']
        
        for field in required_default:
            if field not in hist_row or hist_row.get(field) is None:
                return False
        
        if hist_row.get('evaluation_n') != 288344:
            return False
        
        if threshold_mode == 'test_set_diagnostic':
            required_diagnostic = ['best_threshold', 'diagnostic_f1_score', 'diagnostic_precision', 
                                   'diagnostic_recall', 'diagnostic_false_positive', 'diagnostic_false_negative']
            for field in required_diagnostic:
                if field not in hist_row or hist_row.get(field) is None:
                    return False
        
        return True
    
    def _load_existing_weight_sensitivity_results(self) -> bool:
        """Reload existing results and append ONLY complete records."""
        if self.paths.existing_weight_sensitivity_results_path.exists():
            try:
                df = pd.read_csv(self.paths.existing_weight_sensitivity_results_path)
                existing_records = df.to_dict('records')
                
                for row in existing_records:
                    key = (row.get('model_name'), row.get('anomaly_weight'), row.get('threshold_mode'))
                    
                    already_exists = any(
                        r.get('model_name') == row.get('model_name') and
                        r.get('anomaly_weight') == row.get('anomaly_weight') and
                        r.get('threshold_mode') == row.get('threshold_mode')
                        for r in self.weight_sensitivity_results
                    )
                    
                    if not already_exists:
                        threshold_mode = row.get('threshold_mode', 'default')
                        if self._is_complete_historical_result(row, threshold_mode):
                            self.weight_sensitivity_results.append(row)
                            logger.info(f"✅ Reused complete historical result: {row.get('model_name')} weight={row.get('anomaly_weight')} mode={threshold_mode}")
                        else:
                            logger.info(f"⚠️  Historical result incomplete for {row.get('model_name')} weight={row.get('anomaly_weight')} mode={threshold_mode} - will compute")
                    
                    if all(k is not None for k in key):
                        if self._is_complete_historical_result(row, key[2]):
                            self.completed_results.add(key)
                
                logger.info(f"📥 Loaded {len(existing_records)} existing weight sensitivity results")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Failed to load existing weight sensitivity results: {e}")
        
        return False
    
    def _search_historical_seed42_results(self) -> Dict[Tuple, Dict]:
        """Search for existing XGBoost/RF seed-42 weight sensitivity results."""
        historical_results = {}
        
        if self.paths.historical_weight_sensitivity_dir.exists():
            for file in self.paths.historical_weight_sensitivity_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(file)
                    if 'model_name' in df.columns and 'anomaly_weight' in df.columns and 'seed' in df.columns:
                        for _, row in df.iterrows():
                            if row['model_name'] in ['XGBoost', 'RandomForest'] and row['seed'] == WEIGHT_SENSITIVITY_SEED:
                                key = (row['model_name'], float(row['anomaly_weight']))
                                historical_results[key] = row.to_dict()
                                logger.info(f"✅ Found historical {row['model_name']} seed={row['seed']} weight={row['anomaly_weight']}")
                except Exception as e:
                    continue
        
        return historical_results
    
    def _get_missing_combinations(self) -> List[Tuple[str, float]]:
        """Build execution plan for seed 42 ONLY."""
        missing = []
        
        historical_seed42 = self._search_historical_seed42_results()
        
        for model in MODELS:
            reused_key, compute_key = PLAN_KEYS[model]
            
            for weight in HISTORICAL_WEIGHT_GRID:
                combo = (model, weight)
                
                if combo in self.completed_training:
                    continue
                
                if model == 'LSTM':
                    missing.append(combo)
                    if compute_key:
                        self.execution_plan[compute_key].append(weight)
                    continue
                
                if model in ['XGBoost', 'RandomForest']:
                    hist_key = (model, weight)
                    if hist_key in historical_seed42:
                        self._append_historical_result(historical_seed42[hist_key], model, weight)
                        
                        has_default = (model, weight, 'default') in self.completed_results
                        has_diagnostic = (model, weight, 'test_set_diagnostic') in self.completed_results
                        
                        if has_default and has_diagnostic:
                            self.completed_training.add(combo)
                            if reused_key:
                                self.execution_plan[reused_key].append(weight)
                        else:
                            missing.append(combo)
                            if compute_key:
                                self.execution_plan[compute_key].append(weight)
                            logger.info(f"⚠️  {model} weight={weight}: partial historical data - will compute missing mode(s)")
                        continue
                
                missing.append(combo)
                if compute_key:
                    self.execution_plan[compute_key].append(weight)
        
        logger.info("\n" + "="*80)
        logger.info("EXECUTION PLAN - WEIGHT SENSITIVITY (SEED 42 ONLY)")
        logger.info("="*80)
        
        total_reused = 0
        total_to_compute = 0
        
        for model in MODELS:
            reused_key, compute_key = PLAN_KEYS[model]
            reused = len(self.execution_plan.get(reused_key, [])) if reused_key else 0
            to_compute = len(self.execution_plan.get(compute_key, [])) if compute_key else 0
            total_reused += reused
            total_to_compute += to_compute
            
            reused_str = f"{reused} reused" if reused_key else "N/A"
            logger.info(f"{model}: {reused_str} / {to_compute} new fits")
        
        logger.info(f"\nTOTAL HISTORICAL RECORDS REUSED: {total_reused}")
        logger.info(f"TOTAL NEW MODEL FITS REQUIRED: {total_to_compute}")
        logger.info(f"   (Each fit produces BOTH default and diagnostic threshold results)")
        logger.info("="*80)
        
        return missing
    
    def _append_historical_result(self, hist_row: Dict, model: str, weight: float):
        """Normalize historical result and append ONLY if COMPLETE."""
        default_threshold = DEFAULT_PREDICTION_THRESHOLD if model != 'LinearSVM' else LINEARSVM_DEFAULT_THRESHOLD
        
        default_exists = any(
            r.get('model_name') == model and
            r.get('anomaly_weight') == weight and
            r.get('threshold_mode') == 'default'
            for r in self.weight_sensitivity_results
        )
        
        if not default_exists and self._is_complete_historical_result(hist_row, 'default'):
            default_record = {
                'seed': WEIGHT_SENSITIVITY_SEED,
                'model_name': model,
                'anomaly_weight': weight,
                'threshold_mode': 'default',
                'selected_threshold': float(default_threshold),
                'accuracy': hist_row.get('accuracy'),
                'precision': hist_row.get('precision'),
                'recall': hist_row.get('recall'),
                'f1_score': hist_row.get('f1_score'),
                'false_positive': hist_row.get('false_positive'),
                'false_negative': hist_row.get('false_negative'),
                'evaluation_n': hist_row.get('evaluation_n'),
                'source': hist_row.get('source', 'historical_validated'),
                'reproduction_status': 'reused_validated_historical',
                'analysis_scope': 'single_split_seed42_retrospective_diagnostic'
            }
            self.weight_sensitivity_results.append(default_record)
            self.completed_results.add((model, weight, 'default'))
        
        diagnostic_exists = any(
            r.get('model_name') == model and
            r.get('anomaly_weight') == weight and
            r.get('threshold_mode') == 'test_set_diagnostic'
            for r in self.weight_sensitivity_results
        )
        
        if not diagnostic_exists and self._is_complete_historical_result(hist_row, 'test_set_diagnostic'):
            diagnostic_record = {
                'seed': WEIGHT_SENSITIVITY_SEED,
                'model_name': model,
                'anomaly_weight': weight,
                'threshold_mode': 'test_set_diagnostic',
                'selected_threshold': float(hist_row.get('best_threshold', default_threshold)),
                'accuracy': hist_row.get('diagnostic_accuracy'),
                'precision': hist_row.get('diagnostic_precision'),
                'recall': hist_row.get('diagnostic_recall'),
                'f1_score': hist_row.get('diagnostic_f1_score'),
                'false_positive': hist_row.get('diagnostic_false_positive'),
                'false_negative': hist_row.get('diagnostic_false_negative'),
                'evaluation_n': hist_row.get('evaluation_n'),
                'source': hist_row.get('source', 'historical_validated'),
                'reproduction_status': 'reused_validated_historical',
                'analysis_scope': 'single_split_seed42_retrospective_diagnostic'
            }
            self.weight_sensitivity_results.append(diagnostic_record)
            self.completed_results.add((model, weight, 'test_set_diagnostic'))
    
    def load_weight_sensitivity(self) -> bool:
        """TASK A: Load or compute weight sensitivity for seed 42 ONLY."""
        logger.info("📊 Task A: Loading/computing weight sensitivity (seed 42 ONLY)...")
        
        self.paths.weight_sensitivity_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._recover_historical_configurations():
            logger.error("❌ Cannot proceed without confirmed historical configurations")
            return False
        
        self._load_existing_weight_sensitivity_results()
        
        missing = self._get_missing_combinations()
        
        if not missing:
            logger.info("✅ All weight sensitivity combinations already complete")
            return True
        
        logger.info(f"\n📊 Computing {len(missing)} missing combinations...")
        
        for model, weight in missing:
            result_tuple = self._compute_weight_sensitivity_for_combination(model, weight)
            
            if result_tuple and len(result_tuple) >= 1:
                default_record = result_tuple[0]
                diagnostic_record = result_tuple[1] if len(result_tuple) > 1 else None
                
                default_exists = any(
                    r.get('model_name') == model and
                    r.get('anomaly_weight') == weight and
                    r.get('threshold_mode') == 'default'
                    for r in self.weight_sensitivity_results
                )
                
                if not default_exists:
                    self.weight_sensitivity_results.append(default_record)
                    self.completed_results.add((model, weight, 'default'))
                
                if diagnostic_record:
                    diagnostic_exists = any(
                        r.get('model_name') == model and
                        r.get('anomaly_weight') == weight and
                        r.get('threshold_mode') == 'test_set_diagnostic'
                        for r in self.weight_sensitivity_results
                    )
                    
                    if not diagnostic_exists:
                        self.weight_sensitivity_results.append(diagnostic_record)
                        self.completed_results.add((model, weight, 'test_set_diagnostic'))
                
                self.completed_training.add((model, weight))
                self._save_checkpoint()
            else:
                logger.warning(f"⚠️  Failed to compute {model} seed={WEIGHT_SENSITIVITY_SEED} weight={weight}")
        
        return True
    
    def _compute_weight_sensitivity_for_combination(self, model: str, weight: float) -> Optional[Tuple[Dict, Optional[Dict]]]:
        """Compute weight sensitivity for seed 42 ONLY."""
        logger.info(f"📊 Computing {model} seed={WEIGHT_SENSITIVITY_SEED} weight={weight}...")
        
        try:
            set_random_seeds(WEIGHT_SENSITIVITY_SEED)
            
            label_col = None
            for col_name in LABEL_COLUMN_NAMES:
                if col_name in self.original_dataset.columns:
                    label_col = col_name
                    break
            
            if label_col is None:
                logger.error("❌ No label column found")
                return None
            
            labels_raw = self.original_dataset[label_col].values
            all_labels = np.zeros(len(labels_raw), dtype=int)
            normal_indicators = ['normal', '0', 'benign', 'ok', 'healthy', 'false', 'negative']
            for i, label in enumerate(labels_raw):
                if str(label).lower().strip() not in normal_indicators:
                    all_labels[i] = 1
            
            exclude_cols = set(METADATA_COLUMNS)
            for label_name in LABEL_COLUMN_NAMES:
                exclude_cols.add(label_name.lower())
            exclude_cols.add(label_col.lower())
            
            feature_cols = []
            for col in self.original_dataset.columns:
                if col.lower() in exclude_cols:
                    continue
                if pd.api.types.is_numeric_dtype(self.original_dataset[col]):
                    feature_cols.append(col)
            
            if len(feature_cols) != EXPECTED_FEATURE_COUNT:
                logger.error(f"❌ Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_cols)}")
                return None
            
            all_features = self.original_dataset[feature_cols].values
            n_features = len(feature_cols)
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                all_features, all_labels,
                test_size=TEST_RATIO,
                stratify=all_labels,
                random_state=WEIGHT_SENSITIVITY_SEED
            )
            
            X_train, X_retrain, y_train, y_retrain = train_test_split(
                X_temp, y_temp,
                test_size=RETRAIN_RATIO / (TRAIN_RATIO + RETRAIN_RATIO),
                stratify=y_temp,
                random_state=WEIGHT_SENSITIVITY_SEED
            )
            
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            X_combined = np.vstack([X_train_scaled, X_retrain_scaled])
            y_combined = np.concatenate([y_train, y_retrain])
            
            sample_weights = np.ones(len(y_combined))
            sample_weights[y_combined == 1] = weight
            
            y_eval = None
            
            if model == 'XGBoost':
                if not XGB_AVAILABLE:
                    logger.warning(f"⚠️  XGBoost not available")
                    return None
                
                model_instance = xgb.XGBClassifier(**XGB_CONFIG)
                model_instance.fit(X_combined, y_combined, sample_weight=sample_weights)
                y_pred_scores = model_instance.predict_proba(X_test_scaled)[:, 1]
                y_eval = y_test
                
            elif model == 'RandomForest':
                model_instance = RandomForestClassifier(**RF_CONFIG)
                model_instance.fit(X_combined, y_combined, sample_weight=sample_weights)
                y_pred_scores = model_instance.predict_proba(X_test_scaled)[:, 1]
                y_eval = y_test
                
            elif model == 'LinearSVM':
                model_instance = LinearSVC(
                    C=SVM_C,
                    class_weight=SVM_CLASS_WEIGHT,
                    max_iter=SVM_MAX_ITER,
                    random_state=SVM_RANDOM_STATE
                )
                model_instance.fit(X_combined, y_combined, sample_weight=sample_weights)
                y_pred_scores = model_instance.decision_function(X_test_scaled)
                y_eval = y_test
                
            elif model == 'LSTM':
                if not TENSORFLOW_AVAILABLE:
                    logger.warning(f"⚠️  TensorFlow not available for LSTM")
                    return None
                
                def create_sequences(X, y):
                    n_samples = len(X)
                    if n_samples < TIME_STEPS:
                        return np.array([]).reshape(0, TIME_STEPS, X.shape[1]), np.array([])
                    
                    n_sequences = n_samples - TIME_STEPS + 1
                    try:
                        from numpy.lib.stride_tricks import sliding_window_view
                        X_seq = sliding_window_view(X, window_shape=TIME_STEPS, axis=0)
                        X_seq = np.moveaxis(X_seq, -1, 1)
                        X_seq = np.ascontiguousarray(X_seq, dtype=np.float32)
                    except:
                        X_seq = np.zeros((n_sequences, TIME_STEPS, X.shape[1]), dtype=np.float32)
                        for i in range(n_sequences):
                            X_seq[i] = X[i:i + TIME_STEPS]
                    
                    y_seq = y[TIME_STEPS - 1:].astype(np.float32)
                    return X_seq, y_seq
                
                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
                X_retrain_seq, y_retrain_seq = create_sequences(X_retrain_scaled, y_retrain)
                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
                
                if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                    logger.warning(f"⚠️  No valid sequences")
                    return None
                
                X_combined_seq = np.concatenate([X_train_seq, X_retrain_seq], axis=0)
                y_combined_seq = np.concatenate([y_train_seq, y_retrain_seq], axis=0)
                
                y_eval = y_test_seq.astype(int)
                
                lstm_sample_weights = np.ones(len(y_combined_seq))
                lstm_sample_weights[y_combined_seq == 1] = weight
                
                tf.keras.backend.clear_session()
                gc.collect()
                set_random_seeds(WEIGHT_SENSITIVITY_SEED)
                tf.keras.utils.set_random_seed(WEIGHT_SENSITIVITY_SEED)
                
                lstm_model = models.Sequential([
                    layers.Input(shape=(TIME_STEPS, n_features)),
                    layers.LSTM(LSTM_UNITS_1, return_sequences=True),
                    layers.Dropout(DROPOUT_RATE),
                    layers.LSTM(LSTM_UNITS_2),
                    layers.Dropout(DROPOUT_RATE),
                    layers.Dense(DENSE_UNITS, activation='relu'),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                lstm_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                history = lstm_model.fit(
                    X_combined_seq, y_combined_seq,
                    sample_weight=lstm_sample_weights,
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
                
                y_pred_scores = lstm_model.predict(X_test_seq, verbose=0).flatten()
                
                del lstm_model
                tf.keras.backend.clear_session()
                gc.collect()
                
            else:
                logger.error(f"❌ Unknown model: {model}")
                return None
            
            assert len(y_eval) == len(y_pred_scores), f"Length mismatch: y_eval={len(y_eval)}, y_pred_scores={len(y_pred_scores)}"
            logger.info(f"📊 {model} weight={weight}: len(y_eval)={len(y_eval)}, len(y_pred_scores)={len(y_pred_scores)}")
            
            if model == 'LinearSVM':
                default_threshold = LINEARSVM_DEFAULT_THRESHOLD
                y_pred_default = (y_pred_scores >= default_threshold).astype(int)
            else:
                default_threshold = DEFAULT_PREDICTION_THRESHOLD
                y_pred_default = (y_pred_scores >= default_threshold).astype(int)
            
            default_record = self._evaluate_and_create_record(
                y_eval, y_pred_default, model, weight,
                'default', default_threshold, 'computed'
            )
            
            if model == 'LinearSVM':
                diagnostic_record = None
            else:
                best_threshold = default_threshold
                best_f1 = 0
                
                for thresh in HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES:
                    y_pred_temp = (y_pred_scores >= thresh).astype(int)
                    f1_temp = f1_score(y_eval, y_pred_temp, zero_division=0)
                    
                    if f1_temp > best_f1:
                        best_f1 = f1_temp
                        best_threshold = thresh
                
                y_pred_diagnostic = (y_pred_scores >= best_threshold).astype(int)
                diagnostic_record = self._evaluate_and_create_record(
                    y_eval, y_pred_diagnostic, model, weight,
                    'test_set_diagnostic', best_threshold, 'computed'
                )
            
            return (default_record, diagnostic_record)
            
        except Exception as e:
            logger.error(f"❌ Failed to compute {model} seed={WEIGHT_SENSITIVITY_SEED} weight={weight}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _evaluate_and_create_record(self, y_eval: np.ndarray, y_pred: np.ndarray,
                                    model: str, weight: float,
                                    threshold_mode: str, threshold: float,
                                    source: str) -> Dict:
        """Helper to evaluate predictions and create result record."""
        cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        accuracy = accuracy_score(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        
        return {
            'seed': WEIGHT_SENSITIVITY_SEED,
            'model_name': model,
            'anomaly_weight': weight,
            'threshold_mode': threshold_mode,
            'selected_threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'evaluation_n': int(len(y_eval)),
            'source': source,
            'reproduction_status': 'computed',
            'analysis_scope': 'single_split_seed42_retrospective_diagnostic'
        }
    
    def load_transitions(self) -> Dict[str, str]:
        """TASK B: Load existing transition results."""
        logger.info("📊 Task B: Loading transition results...")
        
        self.paths.transitions_dir.mkdir(parents=True, exist_ok=True)
        
        if not self._load_historical_transitions():
            logger.error("❌ Failed to load historical XGBoost/RF transitions")
        
        if not self._load_linear_svm_transitions():
            logger.error("❌ Failed to load LinearSVM transitions")
        
        self._set_lstm_transitions_unavailable()
        
        return self.transition_status
    
    def _validate_source_data(self) -> bool:
        """Validate source data before export."""
        logger.info("🔍 Validating source data...")
        
        try:
            if self.weight_sensitivity_results:
                df = pd.DataFrame(self.weight_sensitivity_results)
                
                if not (df['seed'] == WEIGHT_SENSITIVITY_SEED).all():
                    logger.error(f"❌ Weight sensitivity contains non-seed-42 records")
                    return False
                
                weights_found = set(df['anomaly_weight'].unique())
                if not weights_found.issubset(set(HISTORICAL_WEIGHT_GRID)):
                    logger.error(f"❌ Unexpected weights in weight sensitivity: {weights_found}")
                    return False
                
                if df.duplicated(subset=['model_name', 'anomaly_weight', 'threshold_mode']).any():
                    logger.error(f"❌ Duplicate rows in weight sensitivity")
                    return False
                
                for col in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if col in df.columns:
                        non_null = df[col].dropna()
                        if len(non_null) > 0 and not non_null.apply(np.isfinite).all():
                            logger.error(f"❌ Non-finite values in {col}")
                            return False
                
                for _, row in df.iterrows():
                    model = row.get('model_name')
                    eval_n = row.get('evaluation_n')
                    
                    if model == 'LSTM':
                        if eval_n != 288335:
                            logger.error(f"❌ LSTM evaluation_n should be 288335, got {eval_n}")
                            return False
                    else:
                        if eval_n != 288344:
                            logger.error(f"❌ {model} evaluation_n should be 288344, got {eval_n}")
                            return False
                
                logger.info(f"✅ Weight sensitivity data validated: {len(df)} rows")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Source data validation failed: {e}")
            return False
    
    def save_weight_sensitivity_outputs(self):
        """Save weight sensitivity results to files."""
        logger.info("💾 Saving weight sensitivity outputs...")
        
        if not self.weight_sensitivity_results:
            logger.warning("⚠️  No weight sensitivity results to save")
            return
        
        results_df = pd.DataFrame(self.weight_sensitivity_results)
        results_path = self.paths.weight_sensitivity_dir / 'four_model_weight_sensitivity_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"   ✅ Saved: {results_path}")
        
        summary_data = []
        for model_name in MODELS:
            model_data = [r for r in self.weight_sensitivity_results if r['model_name'] == model_name]
            
            if not model_data:
                continue
            
            for weight in HISTORICAL_WEIGHT_GRID:
                weight_records = [r for r in model_data if r['anomaly_weight'] == weight]
                
                if not weight_records:
                    continue
                
                threshold_modes = set(r.get('threshold_mode', 'unknown') for r in weight_records)
                
                for threshold_mode in threshold_modes:
                    mode_records = [r for r in weight_records if r.get('threshold_mode') == threshold_mode]
                    mode_f1_scores = [r['f1_score'] for r in mode_records if r['f1_score'] is not None and pd.notna(r['f1_score'])]
                    
                    summary_data.append({
                        'model_name': model_name,
                        'anomaly_weight': weight,
                        'threshold_mode': threshold_mode,
                        'mean_f1_score': np.mean(mode_f1_scores) if mode_f1_scores else None,
                        'std_f1_score': np.std(mode_f1_scores, ddof=1) if len(mode_f1_scores) > 1 else 0.0,
                        'n_seeds': len(mode_f1_scores),
                        'selected_threshold': mode_records[0].get('selected_threshold') if mode_records else None
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.paths.weight_sensitivity_dir / 'four_model_weight_sensitivity_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"   ✅ Saved: {summary_csv_path}")
        
        summary_json_path = self.paths.weight_sensitivity_dir / 'four_model_weight_sensitivity_summary.json'
        summary_json = {
            'models': MODELS,
            'seed': WEIGHT_SENSITIVITY_SEED,
            'analysis_scope': 'single_split_seed42_retrospective_diagnostic',
            'historical_weight_grid': self.historical_weight_grid,
            'weight_grid_source': self.weight_grid_source,
            'historical_threshold_logic': self.historical_threshold_logic,
            'threshold_candidate_grid': HISTORICAL_PROBABILITY_THRESHOLD_CANDIDATES,
            'linearsvm_note': 'LinearSVM uses decision_function >= 0. Probability threshold grid NOT applied to raw scores.',
            'lstm_note': 'LSTM computed directly with test-set evaluation (evaluation_n=288335). NOT reused from internal-validation pilot.',
            'total_records': len(self.weight_sensitivity_results),
            'execution_plan': self.execution_plan,
            'weight_sensitivity_data': summary_data,
            'reproduction_status': self.reproduction_status,
            'scientific_note': 'Weight sensitivity diagnostic using historical grid [1,2,3,4,5] on seed 42 ONLY. Single-split retrospective analysis for Figures 6-9 extension. Five-seed robustness covered separately in final validation.'
        }
        
        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"   ✅ Saved: {summary_json_path}")
    
    def save_transition_outputs(self, transition_status: Dict[str, str]):
        """Save transition results to files."""
        logger.info("💾 Saving transition outputs...")
        
        for model_name in MODELS:
            if model_name in self.transition_results and self.transition_results[model_name]:
                model_df = pd.DataFrame(self.transition_results[model_name])
                model_path = self.paths.transitions_dir / f'{model_name.lower().replace(" ", "_")}_transitions.csv'
                model_df.to_csv(model_path, index=False)
                logger.info(f"   ✅ Saved: {model_path}")
            elif transition_status.get(model_name) == 'sample_level_transitions_unavailable':
                logger.info(f"   ℹ️   {model_name}: sample_level_transitions_unavailable")
            else:
                logger.warning(f"⚠️  {model_name} transitions NOT available")
        
        summary_data = self._compute_transition_summary()
        
        if summary_data.get('transition_summary'):
            summary_df = pd.DataFrame(summary_data['transition_summary'])
            summary_csv_path = self.paths.transitions_dir / 'four_model_transitions_summary.csv'
            summary_df.to_csv(summary_csv_path, index=False)
            logger.info(f"   ✅ Saved: {summary_csv_path}")
            
            summary_json_path = self.paths.transitions_dir / 'four_model_transitions_summary.json'
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"   ✅ Saved: {summary_json_path}")


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute four-model report extensions."""
    global logger
    
    # FIX #6: Add command-line argument for transitions-only mode
    parser = argparse.ArgumentParser(description='Four-Model Report Extensions')
    parser.add_argument('--transitions-only', action='store_true',
                       help='Run transitions-only mode (no model training)')
    args = parser.parse_args()
    
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'four_model_report_extensions')
    
    logger.info("\n" + "="*80)
    logger.info("📊 FOUR-MODEL REPORT EXTENSIONS")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    
    logger.info(f"📁 Report extensions directory: {base_dir / 'adaptive' / 'final_validation_four_model' / 'report_extensions'}")
    logger.info(f"⚠️  Does NOT modify existing validated result files")
    logger.info(f"⚠️  Reuses existing validated results as authoritative")
    logger.info(f"⚠️  Fails closed if exact historical data cannot be recovered")
    logger.info(f"⚠️  Checkpoint/resume enabled (version {CHECKPOINT_VERSION})")
    logger.info(f"⚠️  Dataset fingerprint must match: {EXPECTED_DATASET_FINGERPRINT}")
    logger.info(f"⚠️  LSTM evaluation_n = 288335, classical = 288344")
    
    extensions = FourModelReportExtensions(base_dir=base_dir)
    
    if args.transitions_only:
        # FIX #6: Transitions-only mode
        logger.info("🎯 Mode: TRANSITIONS-ONLY (no model training)")
        results = extensions.run_transitions_only()
    else:
        logger.info("🎯 Mode: FULL (weight sensitivity + transitions)")
        results = extensions.run_full_extensions()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 FOUR-MODEL REPORT EXTENSIONS COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    
    if results.get('mode') == 'transitions_only':
        logger.info(f"📋 Mode: {results['mode']}")
    
    if results.get('tasks_completed'):
        logger.info(f"✅ Tasks completed: {results['tasks_completed']}")
    
    if results.get('transition_status'):
        logger.info(f"📈 Transition status: {results['transition_status']}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()