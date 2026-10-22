"""
Four-Model Ablation Extension - Component Contribution Analysis
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module extends the existing component-contribution ablation from
XGBoost/RandomForest to include LinearSVM and LSTM.

IMPORTANT:
- Does NOT modify existing ablation results
- Loads and validates existing XGBoost/RandomForest ablation results
- Reuses them ONLY if compatibility is PROVEN via cross-validation
- Otherwise stops and reports incompatibility
- Implements robust per-seed checkpointing to prevent data loss

MODELS:
-------
1. XGBoost (anomaly weight = 3.0)
2. RandomForest (anomaly weight = 3.0)
3. LinearSVM (anomaly weight = 3.0)
4. LSTM (anomaly weight = 1.0 - no anomaly up-weighting selected)

SEEDS:
------
[42, 123, 2024, 2025, 3407]

ABLATION VARIANTS:
------------------
1. baseline = train only
2. retraining_without_weighting = train + retrain with anomaly weight 1.0
3. weighted_retraining = train + retrain with anomaly weighting

CONTRIBUTIONS:
--------------
- retraining_contribution = retraining_without_weighting - baseline
- weighting_contribution = weighted_retraining - retraining_without_weighting
- total_contribution = weighted_retraining - baseline

For false negatives: positive reduction means improvement.

IMPORTANT FOR LSTM:
-------------------
For LSTM, both retraining_without_weighting and weighted_retraining use
anomaly weight = 1.0 because the validated global candidate weight is 1.0.
No additional anomaly up-weighting was selected for LSTM.
This is explicitly marked in all outputs.

OUTPUT DIRECTORY:
-----------------
results/adaptive/final_validation_four_model/ablation/

OUTPUT FILES:
-------------
- metrics/four_model_ablation_results.csv
- metrics/four_model_ablation_summary.csv
- metrics/four_model_ablation_summary.json
- reports/four_model_ablation_report.txt
- checkpoints/four_model_ablation_checkpoint.json (per-seed checkpoint)
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
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

# Anomaly weights per model
ANOMALY_WEIGHTS = {
    'XGBoost': 3.0,
    'RandomForest': 3.0,
    'LinearSVM': 3.0,
    'LSTM': 1.0  # No anomaly up-weighting selected
}

TRAIN_RATIO = 0.6
RETRAIN_RATIO = 0.2
TEST_RATIO = 0.2

ABLATION_VARIANTS = ['baseline', 'retraining_without_weighting', 'weighted_retraining']

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

# Model configurations
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
# FOUR-MODEL ABLATION EXTENSION CLASS
# =============================================================================
class FourModelAblationExtension:
    """
    Extends component-contribution ablation to four models.
    """
    
    def __init__(self, base_dir: Path, ablation_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir)
        self.ablation_dir = Path(ablation_dir) if ablation_dir else (
            self.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation'
        )
        
        (self.ablation_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.ablation_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.ablation_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        self.original_dataset = None
        self.all_results = []
        self.summary_df = None
        self.dataset_fingerprint = None
        self.config_fingerprint = None
        
        # Existing ablation results storage
        self.existing_ablation_results = {}
        self.existing_ablation_compatible = False
        self.existing_ablation_validation_errors = []
        
        # Checkpoint management
        self.checkpoint_path = self.ablation_dir / 'checkpoints' / 'four_model_ablation_checkpoint.json'
        self.completed_seeds = []
        self.seed_results_cache = {}
        
        # Cross-validation file path
        self.four_model_results_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'metrics' / 'multi_seed_four_model_results.csv'
        
        self.dataset_path = self.base_dir / 'adaptive' / 'retraining' / 'datasets' / 'retraining_dataset_combined.csv'
        self.existing_ablation_path = self.base_dir / 'adaptive' / 'final_validation' / 'ablation' / 'metrics' / 'ablation_results.csv'
        self.existing_ablation_summary_path = self.base_dir / 'adaptive' / 'final_validation' / 'ablation' / 'metrics' / 'ablation_summary.json'
        
        logger.info(f"✅ Initialized FourModelAblationExtension")
        logger.info(f"📁 Ablation directory: {self.ablation_dir}")
        logger.info(f"🔢 Models: {MODELS}")
        logger.info(f"🔢 Seeds: {SEEDS}")
    
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
            
            # Compute dataset fingerprint
            hasher = hashlib.sha256()
            with open(self.dataset_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
            self.dataset_fingerprint = hasher.hexdigest()[:32]
            
            # Compute COMPLETE config fingerprint covering ALL scientific settings
            config_dict = {
                'seeds': SEEDS,
                'models': MODELS,
                'anomaly_weights': ANOMALY_WEIGHTS,
                'train_ratio': TRAIN_RATIO,
                'retrain_ratio': RETRAIN_RATIO,
                'test_ratio': TEST_RATIO,
                'ablation_variants': ABLATION_VARIANTS,
                'prediction_threshold': PREDICTION_THRESHOLD,
                'time_steps': TIME_STEPS,
                'lstm_units_1': LSTM_UNITS_1,
                'lstm_units_2': LSTM_UNITS_2,
                'dense_units': DENSE_UNITS,
                'dropout_rate': DROPOUT_RATE,
                'learning_rate': LEARNING_RATE,
                'lstm_epochs': LSTM_EPOCHS,
                'lstm_batch_size': LSTM_BATCH_SIZE,
                'patience': PATIENCE,
                'imputation': 'mean',
                'scaler': 'StandardScaler',
                'xgb_config': XGB_CONFIG,
                'rf_config': RF_CONFIG,
                'svm_config': SVM_CONFIG
            }
            config_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
            self.config_fingerprint = hashlib.sha256(config_json.encode('utf-8')).hexdigest()[:32]
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return False
    
    def validate_existing_ablation(self) -> bool:
        """Validate existing XGBoost/RandomForest ablation results via cross-validation."""
        logger.info("🔍 Validating existing ablation results via cross-validation...")
        
        self.existing_ablation_validation_errors = []
        
        if not Path(self.existing_ablation_path).exists():
            logger.error("❌ Historical ablation results file not found.")
            self.existing_ablation_validation_errors.append(
                "Historical ablation results file not found"
            )
            return False
        
        if not Path(self.existing_ablation_summary_path).exists():
            logger.error("❌ Historical ablation summary file not found.")
            self.existing_ablation_validation_errors.append(
                "Historical ablation summary file not found"
            )
            return False
        
        if not Path(self.four_model_results_path).exists():
            logger.error("❌ Four-model results file not found for cross-validation.")
            self.existing_ablation_validation_errors.append(
                "Four-model results file not found"
            )
            return False
        
        try:
            existing_df = pd.read_csv(self.existing_ablation_path)
            with open(self.existing_ablation_summary_path, 'r', encoding='utf-8') as f:
                existing_summary = json.load(f)
            
            logger.info(f"📊 Loaded existing ablation: {len(existing_df)} records")
            
            # Schema normalization: configuration -> variant
            if 'variant' not in existing_df.columns:
                if 'configuration' in existing_df.columns:
                    logger.info("📝 Normalizing historical 'configuration' column to 'variant'...")
                    config_to_variant = {
                        'Baseline': 'baseline',
                        'Retraining_No_Weighting': 'retraining_without_weighting',
                        'Weighted_Retraining': 'weighted_retraining'
                    }
                    existing_df['variant'] = existing_df['configuration'].map(config_to_variant)
                    unmapped_mask = existing_df['variant'].isna()
                    if unmapped_mask.any():
                        unmapped_values = existing_df.loc[unmapped_mask, 'configuration'].unique().tolist()
                        logger.error(f"❌ Unknown configuration values: {unmapped_values}")
                        self.existing_ablation_validation_errors.append(
                            f"Unknown configuration values: {unmapped_values}"
                        )
                        return False
                else:
                    logger.error("❌ Neither 'variant' nor 'configuration' column found")
                    self.existing_ablation_validation_errors.append(
                        "Missing required column: neither 'variant' nor 'configuration' found"
                    )
                    return False
            
            # Validate structure
            if len(existing_df) != 30:
                self.existing_ablation_validation_errors.append(
                    f"Expected exactly 30 records, got {len(existing_df)}"
                )
                return False
            
            existing_models = set(existing_df['model_name'].unique())
            if existing_models != {'XGBoost', 'RandomForest'}:
                self.existing_ablation_validation_errors.append(
                    f"Expected models XGBoost/RandomForest, got {existing_models}"
                )
                return False
            
            existing_seeds = set(existing_df['seed'].unique())
            if existing_seeds != set(SEEDS):
                self.existing_ablation_validation_errors.append(
                    f"Expected seeds {SEEDS}, got {existing_seeds}"
                )
                return False
            
            existing_variants = set(existing_df['variant'].unique())
            if existing_variants != set(ABLATION_VARIANTS):
                self.existing_ablation_validation_errors.append(
                    f"Expected variants {ABLATION_VARIANTS}, got {existing_variants}"
                )
                return False
            
            # Check for duplicates
            existing_df['tuple_key'] = list(zip(existing_df['model_name'], 
                                                 existing_df['seed'], 
                                                 existing_df['variant']))
            if len(existing_df['tuple_key']) != len(set(existing_df['tuple_key'])):
                self.existing_ablation_validation_errors.append(
                    "Duplicate (model_name, seed, variant) tuples found"
                )
                return False
            
            # Required columns
            required_cols = ['seed', 'model_name', 'variant', 'accuracy', 'precision', 
                           'recall', 'f1_score', 'false_positive', 'false_negative']
            missing_cols = [col for col in required_cols if col not in existing_df.columns]
            if missing_cols:
                self.existing_ablation_validation_errors.append(
                    f"Missing required columns: {missing_cols}"
                )
                return False
            
            # Load four-model results for REAL cross-validation
            try:
                four_model_df = pd.read_csv(self.four_model_results_path)
                logger.info(f"📊 Loaded four-model results: {len(four_model_df)} records")
            except Exception as e:
                self.existing_ablation_validation_errors.append(
                    f"Failed to load four-model results: {str(e)}"
                )
                return False
            
            # Validate four-model structure
            if len(four_model_df) != 40:
                self.existing_ablation_validation_errors.append(
                    f"Four-model results should have 40 records, got {len(four_model_df)}"
                )
                return False
            
            four_model_models = set(four_model_df['model_name'].unique())
            if four_model_models != set(MODELS):
                self.existing_ablation_validation_errors.append(
                    f"Four-model should have models {MODELS}, got {four_model_models}"
                )
                return False
            
            four_model_seeds = set(four_model_df['seed'].unique())
            if four_model_seeds != set(SEEDS):
                self.existing_ablation_validation_errors.append(
                    f"Four-model should have seeds {SEEDS}, got {four_model_seeds}"
                )
                return False
            
            four_model_variants = set(four_model_df['model_type'].unique())
            if four_model_variants != {'baseline', 'weighted_retraining'}:
                self.existing_ablation_validation_errors.append(
                    f"Four-model should have baseline/weighted_retraining, got {four_model_variants}"
                )
                return False
            
            # Check for duplicates in four-model
            four_model_df['tuple_key'] = list(zip(four_model_df['model_name'], 
                                                   four_model_df['seed'], 
                                                   four_model_df['model_type']))
            if len(four_model_df['tuple_key']) != len(set(four_model_df['tuple_key'])):
                self.existing_ablation_validation_errors.append(
                    "Duplicate (model_name, seed, model_type) tuples in four-model results"
                )
                return False
            
            # REAL cross-validation - compare historical vs four-model for XGBoost/RF
            float_tolerance = 5e-5  # Appropriate for 4-decimal metrics
            
            for model_name in ['XGBoost', 'RandomForest']:
                for seed in SEEDS:
                    for variant, model_type in [('baseline', 'baseline'), 
                                                 ('weighted_retraining', 'weighted_retraining')]:
                        historical_record = existing_df[
                            (existing_df['model_name'] == model_name) & 
                            (existing_df['seed'] == seed) & 
                            (existing_df['variant'] == variant)
                        ]
                        
                        if len(historical_record) != 1:
                            self.existing_ablation_validation_errors.append(
                                f"Missing historical record for {model_name} seed {seed} variant {variant}"
                            )
                            continue
                        
                        four_model_record = four_model_df[
                            (four_model_df['model_name'] == model_name) & 
                            (four_model_df['seed'] == seed) & 
                            (four_model_df['model_type'] == model_type)
                        ]
                        
                        if len(four_model_record) != 1:
                            self.existing_ablation_validation_errors.append(
                                f"Missing four-model record for {model_name} seed {seed} variant {model_type}"
                            )
                            continue
                        
                        # Compare float metrics with tolerance
                        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                            historical_val = historical_record[metric].values[0]
                            four_model_val = four_model_record[metric].values[0]
                            
                            historical_rounded = round(float(historical_val), 4)
                            four_model_rounded = round(float(four_model_val), 4)
                            
                            if historical_rounded != four_model_rounded:
                                if not np.isclose(historical_val, four_model_val, atol=float_tolerance):
                                    self.existing_ablation_validation_errors.append(
                                        f"{model_name} seed {seed} {variant} {metric} mismatch: "
                                        f"historical={historical_val}, four_model={four_model_val}"
                                    )
                        
                        # Compare integer confusion matrix values exactly
                        for cm_metric in ['false_positive', 'false_negative']:
                            historical_val = int(historical_record[cm_metric].values[0])
                            four_model_val = int(four_model_record[cm_metric].values[0])
                            
                            if historical_val != four_model_val:
                                self.existing_ablation_validation_errors.append(
                                    f"{model_name} seed {seed} {variant} {cm_metric} mismatch: "
                                    f"historical={historical_val}, four_model={four_model_val}"
                                )
            
            if self.existing_ablation_validation_errors:
                logger.error("❌ Existing ablation cross-validation FAILED:")
                for error in self.existing_ablation_validation_errors:
                    logger.error(f"   - {error}")
                return False
            else:
                logger.info("✅ Existing ablation compatibility PROVEN via cross-validation")
                
                for model in ['XGBoost', 'RandomForest']:
                    model_data = existing_df[existing_df['model_name'] == model]
                    self.existing_ablation_results[model] = model_data.to_dict('records')
                
                self.existing_ablation_compatible = True
                return True
            
        except Exception as e:
            self.existing_ablation_validation_errors.append(
                f"Failed to validate existing ablation: {str(e)}"
            )
            logger.error(f"❌ Existing ablation validation failed: {e}")
            return False
    
    def validate_seed_results(self, seed: int, seed_results: List[Dict]) -> bool:
        """
        Strict per-seed checkpoint validation before saving or loading.
        """
        # Check exactly 12 records
        if len(seed_results) != 12:
            logger.error(f"❌ Seed {seed} has {len(seed_results)} records, expected 12")
            return False
        
        # Check exact seed value in every record
        for record in seed_results:
            if record.get('seed') != seed:
                logger.error(f"❌ Seed {seed} record has wrong seed value: {record.get('seed')}")
                return False
        
        # Check exact models
        models_found = set(r['model_name'] for r in seed_results)
        if models_found != set(MODELS):
            logger.error(f"❌ Seed {seed} has wrong models: {models_found}")
            return False
        
        # Check exactly 3 variants for each model
        for model_name in MODELS:
            model_records = [r for r in seed_results if r['model_name'] == model_name]
            if len(model_records) != 3:
                logger.error(f"❌ Seed {seed} model {model_name} has {len(model_records)} variants, expected 3")
                return False
            
            variants_found = set(r['variant'] for r in model_records)
            if variants_found != set(ABLATION_VARIANTS):
                logger.error(f"❌ Seed {seed} model {model_name} has wrong variants: {variants_found}")
                return False
        
        # Check no duplicate (seed, model_name, variant)
        tuples = [(r['seed'], r['model_name'], r['variant']) for r in seed_results]
        if len(tuples) != len(set(tuples)):
            logger.error(f"❌ Seed {seed} has duplicate (seed, model_name, variant) tuples")
            return False
        
        # Validate metrics
        for record in seed_results:
            # accuracy/precision/recall/f1_score: non-null, finite, within [0,1]
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                val = record.get(metric)
                if val is None:
                    logger.error(f"❌ Seed {seed} record has null {metric}")
                    return False
                if not np.isfinite(val):
                    logger.error(f"❌ Seed {seed} record has non-finite {metric}")
                    return False
                if val < 0 or val > 1:
                    logger.error(f"❌ Seed {seed} record has {metric} outside [0,1]: {val}")
                    return False
            
            # false_positive/false_negative: non-null, non-negative integers
            for cm_metric in ['false_positive', 'false_negative']:
                val = record.get(cm_metric)
                if val is None:
                    logger.error(f"❌ Seed {seed} record has null {cm_metric}")
                    return False
                if not isinstance(val, (int, np.integer)) or val < 0:
                    logger.error(f"❌ Seed {seed} record has invalid {cm_metric}: {val}")
                    return False
            
            # TP/TN optional, but if present must be non-negative integers
            for cm_metric in ['true_positive', 'true_negative']:
                val = record.get(cm_metric)
                if val is not None:
                    if not isinstance(val, (int, np.integer)) or val < 0:
                        logger.error(f"❌ Seed {seed} record has invalid {cm_metric}: {val}")
                        return False
        
        return True
    
    def load_and_validate_checkpoint(self) -> str:
        """
        Load and validate checkpoint with STRICT seed validation.
        Returns state: "missing", "valid", or "invalid"
        """
        logger.info("🔍 Checking for checkpoint...")
        
        if not self.checkpoint_path.exists():
            logger.info("ℹ️   No checkpoint found. Starting fresh.")
            return "missing"
        
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # Validate checkpoint version and fingerprints
            if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
                logger.error(f"❌ Checkpoint version mismatch: {checkpoint.get('checkpoint_version')} vs {CHECKPOINT_VERSION}")
                return "invalid"
            
            if checkpoint.get('dataset_fingerprint') != self.dataset_fingerprint:
                logger.error("❌ Checkpoint dataset fingerprint mismatch")
                return "invalid"
            
            if checkpoint.get('config_fingerprint') != self.config_fingerprint:
                logger.error("❌ Checkpoint config fingerprint mismatch")
                return "invalid"
            
            # Validate completed_seeds structure
            completed_seeds = checkpoint.get('completed_seeds', [])
            if not isinstance(completed_seeds, list):
                logger.error("❌ completed_seeds is not a list")
                return "invalid"
            
            # Check no duplicates in completed_seeds
            if len(completed_seeds) != len(set(completed_seeds)):
                logger.error("❌ completed_seeds contains duplicates")
                return "invalid"
            
            # Check all completed_seeds are integers and belong to SEEDS
            for seed in completed_seeds:
                if not isinstance(seed, int):
                    logger.error(f"❌ completed_seeds contains non-integer: {seed}")
                    return "invalid"
                if seed not in SEEDS:
                    logger.error(f"❌ completed_seeds contains unexpected seed: {seed}")
                    return "invalid"
            
            # Validate seed_results
            seed_results = checkpoint.get('seed_results', {})
            if not isinstance(seed_results, dict):
                logger.error("❌ seed_results is not a dict")
                return "invalid"
            
            # Check no unexpected seed_results keys outside completed_seeds
            for seed_key in seed_results.keys():
                try:
                    seed_int = int(seed_key)
                    if seed_int not in completed_seeds:
                        logger.error(f"❌ seed_results has key {seed_key} not in completed_seeds")
                        return "invalid"
                except (ValueError, TypeError):
                    logger.error(f"❌ seed_results has invalid key: {seed_key}")
                    return "invalid"
            
            # STRICT validation - call validate_seed_results for each seed
            for seed in completed_seeds:
                seed_key = str(seed)
                seed_data = seed_results.get(seed_key)
                
                if seed_data is None:
                    logger.error(f"❌ Completed seed {seed} missing from seed_results")
                    return "invalid"
                
                # Call validate_seed_results - same validation as for newly completed seeds
                if not self.validate_seed_results(seed, seed_data):
                    logger.error(f"❌ Completed seed {seed} failed strict validation")
                    return "invalid"
                
                logger.info(f"✅ Validated seed {seed}: 12 records complete")
            
            # ONLY restore after ENTIRE checkpoint is proven valid
            self.completed_seeds = completed_seeds
            for seed in completed_seeds:
                seed_key = str(seed)
                self.seed_results_cache[seed_key] = seed_results[seed_key]
                # Add to all_results exactly once
                self.all_results.extend(seed_results[seed_key])
            
            # Verify exact record count
            expected_count = 12 * len(self.completed_seeds)
            if len(self.all_results) != expected_count:
                logger.error(f"❌ all_results has {len(self.all_results)} records, expected {expected_count}")
                return "invalid"
            
            logger.info(f"✅ Restored {len(self.completed_seeds)} completed seeds from checkpoint: {self.completed_seeds}")
            return "valid"
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.warning("⚠️   Checkpoint is INVALID - STOPPING execution")
            return "invalid"
    
    def save_checkpoint(self, seed: int, seed_results: List[Dict]):
        """Save checkpoint after seed completes successfully. Uses atomic write."""
        logger.info(f"💾 Saving checkpoint for seed {seed}...")
        
        # Validate seed results before saving
        if not self.validate_seed_results(seed, seed_results):
            logger.error(f"❌ Seed {seed} results failed validation - NOT saving checkpoint")
            raise ValueError(f"Seed {seed} results invalid")
        
        # Add seed results to cache
        seed_key = str(seed)
        self.seed_results_cache[seed_key] = seed_results
        if seed not in self.completed_seeds:
            self.completed_seeds.append(seed)
        
        # Add to all_results (exactly once)
        existing_count = sum(1 for r in self.all_results if r.get('seed') == seed)
        if existing_count == 0:
            self.all_results.extend(seed_results)
        elif existing_count != 12:
            logger.error(f"❌ Seed {seed} already has {existing_count} records in all_results, expected 12")
            raise ValueError(f"Seed {seed} duplicate detection failed")
        
        # Build checkpoint data
        checkpoint_data = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'dataset_fingerprint': self.dataset_fingerprint,
            'config_fingerprint': self.config_fingerprint,
            'completed_seeds': self.completed_seeds,
            'seed_results': self.seed_results_cache,
            'saved_at': datetime.now().isoformat()
        }
        
        # Write to temp file first, then rename (atomic)
        fd, temp_path = tempfile.mkstemp(suffix='.json', dir=self.checkpoint_path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(temp_path, self.checkpoint_path)
            logger.info(f"✅ Checkpoint saved: {len(self.completed_seeds)} seeds completed")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
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
        weights = np.ones(len(labels))
        weights[labels == 1] = anomaly_weight
        return weights
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             expected_population: Optional[int] = None) -> Dict:
        """
        Evaluate predictions and return metrics.
        FIX #1: Ensure all values are native Python scalar types for JSON serialization.
        """
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
        
        cm_sum = tn + fp + fn + tp
        cm_valid = True
        if expected_population is not None:
            if cm_sum != expected_population:
                cm_valid = False
                logger.warning(f"⚠️  Confusion matrix sum ({cm_sum}) != expected population ({expected_population})")
        
        cm_values_valid = (
            tn >= 0 and fp >= 0 and fn >= 0 and tp >= 0 and
            isinstance(tn, (int, np.integer)) and
            isinstance(fp, (int, np.integer)) and
            isinstance(fn, (int, np.integer)) and
            isinstance(tp, (int, np.integer))
        )
        
        # FIX #1: Ensure all values are native Python scalar types for JSON serialization
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'metrics_valid': bool(metrics_valid and cm_valid and cm_values_valid),
            'confusion_matrix_sum': int(cm_sum),
            'expected_population': int(expected_population) if expected_population is not None else None
        }
    
    def run_single_seed_ablation(self, seed: int, test_size: int) -> List[Dict]:
        """Run ablation for all 4 models for a single seed."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 RUNNING ABLATION SEED {seed}")
        logger.info(f"{'='*80}")
        
        results = []
        
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
            
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)
            X_retrain_imp = imputer.transform(X_retrain)
            X_test_imp = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_retrain_scaled = scaler.transform(X_retrain_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            logger.info(f"📊 Split sizes: Train={len(X_train)}, Retrain={len(X_retrain)}, Test={len(X_test)}")
            
            for model_name in MODELS:
                logger.info(f"   📊 Model: {model_name}")
                
                # Reuse existing XGBoost/RandomForest if compatible
                if model_name in ['XGBoost', 'RandomForest'] and self.existing_ablation_compatible:
                    existing_model_results = [
                        r for r in self.existing_ablation_results.get(model_name, []) 
                        if r.get('seed') == seed
                    ]
                    if len(existing_model_results) == 3:
                        logger.info(f"      ✅ Reusing existing results for {model_name}")
                        for r in existing_model_results:
                            r['source'] = 'existing_ablation'
                            results.append(r)
                        continue
                
                for variant in ABLATION_VARIANTS:
                    logger.info(f"      Variant: {variant}")
                    
                    if model_name == 'LSTM':
                        if variant == 'baseline':
                            X_train_seq, y_train_seq = self._create_sequences_efficient(
                                X_train_scaled, y_train
                            )
                            anomaly_weight = 1.0
                            
                        elif variant == 'retraining_without_weighting':
                            X_train_seq_part, y_train_seq_part = self._create_sequences_efficient(
                                X_train_scaled, y_train
                            )
                            X_retrain_seq_part, y_retrain_seq_part = self._create_sequences_efficient(
                                X_retrain_scaled, y_retrain
                            )
                            
                            if len(X_train_seq_part) > 0 and len(X_retrain_seq_part) > 0:
                                X_train_seq = np.concatenate([X_train_seq_part, X_retrain_seq_part], axis=0)
                                y_train_seq = np.concatenate([y_train_seq_part, y_retrain_seq_part], axis=0)
                            elif len(X_train_seq_part) > 0:
                                X_train_seq = X_train_seq_part
                                y_train_seq = y_train_seq_part
                            else:
                                X_train_seq = X_retrain_seq_part
                                y_train_seq = y_retrain_seq_part
                            
                            anomaly_weight = 1.0
                            
                        else:  # weighted_retraining
                            continue
                        
                        if len(X_train_seq) == 0:
                            logger.warning(f"      ⚠️  LSTM: No valid sequences")
                            result = {
                                'seed': seed,
                                'model_name': model_name,
                                'variant': variant,
                                'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
                                'false_positive': None, 'false_negative': None,
                                'true_positive': None, 'true_negative': None,
                                'anomaly_weight': anomaly_weight,
                                'lstm_weight_note': 'No anomaly up-weighting selected (weight=1.0)',
                                'source': 'computed',
                                'metrics_valid': False
                            }
                            results.append(result)
                            continue
                        
                        X_test_seq, y_test_seq = self._create_sequences_efficient(
                            X_test_scaled, y_test
                        )
                        
                        if len(X_test_seq) == 0:
                            logger.warning(f"      ⚠️  LSTM: No valid test sequences")
                            result = {
                                'seed': seed,
                                'model_name': model_name,
                                'variant': variant,
                                'accuracy': None, 'precision': None, 'recall': None, 'f1_score': None,
                                'false_positive': None, 'false_negative': None,
                                'true_positive': None, 'true_negative': None,
                                'anomaly_weight': anomaly_weight,
                                'lstm_weight_note': 'No anomaly up-weighting selected (weight=1.0)',
                                'source': 'computed',
                                'metrics_valid': False
                            }
                            results.append(result)
                            continue
                        
                        expected_population = len(X_test_seq)
                        
                        tf.keras.backend.clear_session()
                        gc.collect()
                        set_random_seeds(seed)
                        tf.keras.utils.set_random_seed(seed)
                        
                        model = self._create_lstm_model(n_features)
                        
                        history = model.fit(
                            X_train_seq, y_train_seq,
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
                        
                        y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
                        y_pred = (y_pred_proba > PREDICTION_THRESHOLD).astype(int)
                        metrics = self._evaluate_predictions(y_test_seq, y_pred, expected_population)
                        
                        del model
                        tf.keras.backend.clear_session()
                        gc.collect()
                        
                        result = {
                            'seed': seed,
                            'model_name': model_name,
                            'variant': variant,
                            **metrics,
                            'anomaly_weight': anomaly_weight,
                            'lstm_weight_note': 'No anomaly up-weighting selected (weight=1.0)',
                            'source': 'computed'
                        }
                        results.append(result)
                    
                    else:
                        if variant == 'baseline':
                            X_train_var = X_train_scaled
                            y_train_var = y_train
                            anomaly_weight = 1.0
                        elif variant == 'retraining_without_weighting':
                            X_train_var = np.vstack([X_train_scaled, X_retrain_scaled])
                            y_train_var = np.concatenate([y_train, y_retrain])
                            anomaly_weight = 1.0
                        else:  # weighted_retraining
                            X_train_var = np.vstack([X_train_scaled, X_retrain_scaled])
                            y_train_var = np.concatenate([y_train, y_retrain])
                            anomaly_weight = ANOMALY_WEIGHTS[model_name]
                        
                        expected_population = len(X_test_scaled)
                        
                        if model_name == 'XGBoost':
                            model = xgb.XGBClassifier(**XGB_CONFIG)
                            sample_weights = self._create_sample_weights(y_train_var, anomaly_weight)
                            model.fit(X_train_var, y_train_var, sample_weight=sample_weights)
                        elif model_name == 'RandomForest':
                            model = RandomForestClassifier(**RF_CONFIG)
                            sample_weights = self._create_sample_weights(y_train_var, anomaly_weight)
                            model.fit(X_train_var, y_train_var, sample_weight=sample_weights)
                        elif model_name == 'LinearSVM':
                            model = LinearSVC(**SVM_CONFIG)
                            sample_weights = self._create_sample_weights(y_train_var, anomaly_weight)
                            model.fit(X_train_var, y_train_var, sample_weight=sample_weights)
                        
                        y_pred = model.predict(X_test_scaled)
                        metrics = self._evaluate_predictions(y_test, y_pred, expected_population)
                        
                        result = {
                            'seed': seed,
                            'model_name': model_name,
                            'variant': variant,
                            **metrics,
                            'anomaly_weight': anomaly_weight,
                            'lstm_weight_note': None,
                            'source': 'computed'
                        }
                        results.append(result)
                
                # LSTM weighted_retraining derived from retraining_without_weighting
                if model_name == 'LSTM':
                    retrain_no_weight_result = next(
                        (r for r in results if r['seed'] == seed and 
                         r['model_name'] == 'LSTM' and 
                         r['variant'] == 'retraining_without_weighting'),
                        None
                    )
                    if retrain_no_weight_result:
                        weighted_result = retrain_no_weight_result.copy()
                        weighted_result['variant'] = 'weighted_retraining'
                        weighted_result['source'] = 'derived_same_configuration'
                        weighted_result['additional_weighting_selected'] = False
                        results.append(weighted_result)
                        logger.info(f"      ✅ Created weighted_retraining record by copying (LSTM weight=1.0)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def compute_contributions(self) -> pd.DataFrame:
        """
        Compute retraining, weighting, and total contributions.
        FIX #2: Remove truthiness bug - compute differences directly since metrics are already validated.
        """
        logger.info("📊 Computing contributions...")
        
        results_df = pd.DataFrame(self.all_results)
        
        expected_records = len(MODELS) * len(SEEDS) * len(ABLATION_VARIANTS)
        if len(results_df) != expected_records:
            logger.error(f"❌ Expected {expected_records} records, got {len(results_df)}")
            raise ValueError(f"Incomplete results: expected {expected_records} records")
        
        results_df['tuple_key'] = list(zip(results_df['model_name'], 
                                           results_df['seed'], 
                                           results_df['variant']))
        if len(results_df['tuple_key']) != len(set(results_df['tuple_key'])):
            logger.error("❌ Duplicate (model_name, seed, variant) tuples found")
            raise ValueError("Duplicate records found")
        
        for model_name in MODELS:
            for seed in SEEDS:
                model_seed_data = results_df[
                    (results_df['seed'] == seed) & 
                    (results_df['model_name'] == model_name)
                ]
                if len(model_seed_data) != 3:
                    logger.error(f"❌ Model {model_name} seed {seed} has {len(model_seed_data)} variants, expected 3")
                    raise ValueError(f"Incomplete variants for {model_name} seed {seed}")
                
                variants = set(model_seed_data['variant'].unique())
                if variants != set(ABLATION_VARIANTS):
                    logger.error(f"❌ Model {model_name} seed {seed} has wrong variants: {variants}")
                    raise ValueError(f"Wrong variants for {model_name} seed {seed}")
        
        # Mandatory metrics validation (FP/FN required, TP/TN optional)
        metric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
        for col in metric_cols:
            if col in results_df.columns:
                null_count = results_df[col].isna().sum()
                if null_count > 0:
                    logger.error(f"❌ {null_count} None/NaN values in {col}")
                    raise ValueError(f"None/NaN values in {col}")
                
                finite_mask = results_df[col].apply(lambda x: x is None or np.isfinite(x))
                if not finite_mask.all():
                    logger.error(f"❌ Inf values in {col}")
                    raise ValueError(f"Inf values in {col}")
        
        cm_cols_mandatory = ['false_positive', 'false_negative']
        for col in cm_cols_mandatory:
            if col in results_df.columns:
                null_count = results_df[col].isna().sum()
                if null_count > 0:
                    logger.error(f"❌ {null_count} None values in mandatory {col}")
                    raise ValueError(f"None values in mandatory {col}")
                
                negative_count = (results_df[col] < 0).sum()
                if negative_count > 0:
                    logger.error(f"❌ {negative_count} negative values in {col}")
                    raise ValueError(f"Negative values in {col}")
        
        cm_cols_optional = ['true_positive', 'true_negative']
        for col in cm_cols_optional:
            if col in results_df.columns:
                null_count = results_df[col].isna().sum()
                if null_count > 0:
                    logger.info(f"ℹ️   {null_count} records have unavailable {col} (historical XGBoost/RandomForest)")
        
        contributions = []
        
        for seed in SEEDS:
            for model_name in MODELS:
                model_data = results_df[
                    (results_df['seed'] == seed) & 
                    (results_df['model_name'] == model_name)
                ]
                
                if len(model_data) < 3:
                    logger.warning(f"⚠️  Missing variants for {model_name} seed {seed}")
                    continue
                
                baseline = model_data[model_data['variant'] == 'baseline'].iloc[0]
                retrain = model_data[model_data['variant'] == 'retraining_without_weighting'].iloc[0]
                weighted = model_data[model_data['variant'] == 'weighted_retraining'].iloc[0]
                
                # FIX #2: Compute contributions directly since metrics are already validated
                # Do NOT use truthiness checks (if retrain['f1_score']) as 0.0 is valid
                contribution = {
                    'seed': seed,
                    'model_name': model_name,
                    'retraining_contribution_f1': retrain['f1_score'] - baseline['f1_score'],
                    'weighting_contribution_f1': weighted['f1_score'] - retrain['f1_score'],
                    'total_contribution_f1': weighted['f1_score'] - baseline['f1_score'],
                    'retraining_contribution_recall': retrain['recall'] - baseline['recall'],
                    'weighting_contribution_recall': weighted['recall'] - retrain['recall'],
                    'total_contribution_recall': weighted['recall'] - baseline['recall'],
                    'retraining_fn_reduction': baseline['false_negative'] - retrain['false_negative'] if baseline['false_negative'] is not None and retrain['false_negative'] is not None else None,
                    'weighting_fn_reduction': retrain['false_negative'] - weighted['false_negative'] if retrain['false_negative'] is not None and weighted['false_negative'] is not None else None,
                    'total_fn_reduction': baseline['false_negative'] - weighted['false_negative'] if baseline['false_negative'] is not None and weighted['false_negative'] is not None else None,
                    'anomaly_weight': ANOMALY_WEIGHTS[model_name],
                    'lstm_note': 'No anomaly up-weighting selected' if model_name == 'LSTM' else None
                }
                contributions.append(contribution)
        
        return pd.DataFrame(contributions)
    
    def save_results(self, contributions_df: pd.DataFrame):
        """Save all outputs."""
        logger.info("💾 Saving results...")
        saved_paths = []
        
        try:
            if self.all_results:
                results_df = pd.DataFrame(self.all_results)
                results_csv = self.ablation_dir / 'metrics' / 'four_model_ablation_results.csv'
                results_df.to_csv(results_csv, index=False)
                saved_paths.append(str(results_csv))
                logger.info(f"   ✅ Saved four_model_ablation_results.csv")
            
            if contributions_df is not None and len(contributions_df) > 0:
                summary_csv = self.ablation_dir / 'metrics' / 'four_model_ablation_summary.csv'
                contributions_df.to_csv(summary_csv, index=False)
                saved_paths.append(str(summary_csv))
                logger.info(f"   ✅ Saved four_model_ablation_summary.csv")
            
            summary_json_path = self.ablation_dir / 'metrics' / 'four_model_ablation_summary.json'
            summary_dict = {
                'models': MODELS,
                'seeds': SEEDS,
                'variants': ABLATION_VARIANTS,
                'anomaly_weights': ANOMALY_WEIGHTS,
                'total_records': len(self.all_results) if self.all_results else 0,
                'existing_ablation_reused': self.existing_ablation_compatible,
                'existing_ablation_validation_errors': self.existing_ablation_validation_errors,
                'dataset_fingerprint': self.dataset_fingerprint,
                'config_fingerprint': self.config_fingerprint,
                'scientific_note': 'For LSTM, both retraining variants use weight=1.0 (no anomaly up-weighting selected).'
            }
            with open(summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, indent=2)
            saved_paths.append(str(summary_json_path))
            logger.info(f"   ✅ Saved four_model_ablation_summary.json")
            
            report_path = self.generate_report(contributions_df)
            saved_paths.append(str(report_path))
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise
        
        return saved_paths
    
    def generate_report(self, contributions_df: pd.DataFrame) -> str:
        """Generate comprehensive report."""
        logger.info("📝 Generating report...")
        
        report = []
        report.append("="*80)
        report.append("FOUR-MODEL ABLATION EXTENSION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 1.0.0")
        report.append(f"Thesis Project: CPS Anomaly Detection Framework")
        report.append("")
        report.append("EXPERIMENTAL CONFIGURATION:")
        report.append(f"  - Models: {MODELS}")
        report.append(f"  - Seeds: {SEEDS}")
        report.append(f"  - Variants: {ABLATION_VARIANTS}")
        report.append(f"  - Anomaly weights: {ANOMALY_WEIGHTS}")
        report.append("")
        report.append("EXISTING ABLATION REUSE:")
        report.append(f"  - Compatible existing results: {self.existing_ablation_compatible}")
        if self.existing_ablation_compatible:
            report.append(f"  - Reused models: XGBoost, RandomForest")
        if self.existing_ablation_validation_errors:
            report.append(f"  - Validation errors: {self.existing_ablation_validation_errors}")
        report.append("")
        report.append("CONTRIBUTION ANALYSIS:")
        report.append("")
        
        if contributions_df is not None and len(contributions_df) > 0:
            for model in MODELS:
                model_data = contributions_df[contributions_df['model_name'] == model]
                if len(model_data) > 0:
                    report.append(f"  {model} (weight={ANOMALY_WEIGHTS[model]}):")
                    avg_total_f1 = model_data['total_contribution_f1'].mean()
                    avg_total_fn = model_data['total_fn_reduction'].mean()
                    report.append(f"    - Total F1 contribution: {avg_total_f1:+.4f}")
                    report.append(f"    - Total FN reduction: {avg_total_fn:+.1f}")
                    if model == 'LSTM':
                        report.append(f"    - NOTE: No anomaly up-weighting selected (weight=1.0)")
                    report.append("")
        else:
            report.append("  No contribution data available")
            report.append("")
        
        report.append("SCIENTIFIC NOTES:")
        report.append(f"  - Positive FN reduction means improvement (fewer false negatives)")
        report.append(f"  - For LSTM, both retraining variants use weight=1.0")
        report.append(f"  - This reflects the validated global candidate weight from weight sensitivity pilot")
        report.append(f"  - No artificial weighting benefit is fabricated for LSTM")
        report.append("")
        report.append("="*80)
        report.append("END OF FOUR-MODEL ABLATION EXTENSION REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        report_path = self.ablation_dir / 'reports' / 'four_model_ablation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return str(report_path)
    
    def run_full_ablation(self) -> Dict:
        """Run complete four-model ablation extension with checkpoint support."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING FOUR-MODEL ABLATION EXTENSION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'ablation_complete': False
        }
        
        if not self.load_resources():
            results['status'] = 'failed'
            return results
        
        # Validate historical ablation FIRST
        self.validate_existing_ablation()
        
        if not self.existing_ablation_compatible and self.existing_ablation_validation_errors:
            logger.error("❌ Existing XGBoost/RandomForest ablation compatibility cannot be verified.")
            logger.error("❌ STOPPING execution. Please resolve compatibility issues.")
            for error in self.existing_ablation_validation_errors:
                logger.error(f"   - {error}")
            results['status'] = 'failed'
            results['error'] = 'Existing ablation incompatible'
            return results
        
        # Load and validate checkpoint
        checkpoint_state = self.load_and_validate_checkpoint()
        
        if checkpoint_state == "invalid":
            logger.error("❌ Checkpoint is INVALID - STOPPING execution")
            logger.error("❌ Please archive or remove the invalid checkpoint before rerunning")
            results['status'] = 'failed'
            results['error'] = 'Invalid checkpoint'
            return results
        
        # Get test size for confusion matrix validation
        test_size = int(len(self.original_dataset) * TEST_RATIO)
        
        # Run ablation for seeds not yet completed
        for seed in SEEDS:
            if seed in self.completed_seeds:
                logger.info(f"⏭️  Seed {seed} already completed - skipping")
                continue
            
            seed_results = self.run_single_seed_ablation(seed, test_size)
            if not seed_results:
                logger.error(f"❌ Seed {seed} failed to produce results")
                results['status'] = 'failed'
                return results
            
            # Validate seed results before saving
            if not self.validate_seed_results(seed, seed_results):
                logger.error(f"❌ Seed {seed} results failed validation")
                results['status'] = 'failed'
                return results
            
            # Add to all_results BEFORE checkpoint
            existing_count = sum(1 for r in self.all_results if r.get('seed') == seed)
            if existing_count == 0:
                self.all_results.extend(seed_results)
            else:
                logger.error(f"❌ Seed {seed} already has {existing_count} records in all_results")
                results['status'] = 'failed'
                return results
            
            # Save checkpoint after seed completes
            try:
                self.save_checkpoint(seed, seed_results)
            except Exception as e:
                logger.error(f"❌ Failed to save checkpoint for seed {seed}: {e}")
                results['status'] = 'failed'
                return results
            
            logger.info(f"✅ Seed {seed} completed and checkpointed")
        
        # All seeds complete - save final results
        try:
            contributions_df = self.compute_contributions()
            self.save_results(contributions_df)
            results['outputs_saved'] = [
                str(self.ablation_dir / 'metrics' / 'four_model_ablation_results.csv'),
                str(self.ablation_dir / 'metrics' / 'four_model_ablation_summary.csv'),
                str(self.ablation_dir / 'metrics' / 'four_model_ablation_summary.json'),
                str(self.ablation_dir / 'reports' / 'four_model_ablation_report.txt')
            ]
            results['status'] = 'complete'
            results['ablation_complete'] = True
            
            # Clean up checkpoint after successful completion
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                logger.info("🗑️  Cleaned up checkpoint after successful completion")
            
        except Exception as e:
            logger.error(f"❌ Final post-processing failed: {e}")
            logger.error("⚠️   Checkpoint preserved for resume capability")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute four-model ablation extension"""
    global logger
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'four_model_ablation_extension')
    
    logger.info("\n" + "="*80)
    logger.info("🔬 FOUR-MODEL ABLATION EXTENSION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    ablation_dir = base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation'
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Ablation directory: {ablation_dir}")
    logger.info(f"⚠️  Does NOT modify existing ablation results")
    logger.info(f"⚠️  LSTM uses weight=1.0 (no anomaly up-weighting selected)")
    logger.info(f"⚠️  Checkpoint system enabled for crash recovery")
    
    ablation = FourModelAblationExtension(base_dir=base_dir, ablation_dir=ablation_dir)
    results = ablation.run_full_ablation()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 FOUR-MODEL ABLATION EXTENSION COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"💾 Outputs saved to: {ablation_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()