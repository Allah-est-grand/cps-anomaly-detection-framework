"""
Main Execution Script - CPS Anomaly Detection Framework
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

IMPORTANT: Scientifically Correct Preprocessing Workflow (No Data Leakage)
---------------------------------------------------------------------------
This script implements a SCIENTIFICALLY CORRECT pipeline:
1. Load raw dataset (no preprocessing)
2. Extract features and labels (no fitting)
3. Split into train/test FIRST (on raw data)
4. Fit imputer and scaler on TRAIN ONLY
5. Transform both train and test using fitted preprocessors
6. Save preprocessors (fitted on train only)
7. Train models on preprocessed train data
8. Evaluate on preprocessed test data

This prevents DATA LEAKAGE where test data influences training preprocessing.
The saved preprocessors are REQUIRED for run_robustness_evaluation.py.

COMPATIBILITY NOTE:
-------------------
This main.py is designed to work with the REAL src/models.py which contains:
- register_model(name, model)
- get_model(name)
- initialize_all_models(X_train, y_train)
- train_model(name, X_train, y_train, X_val=None, y_val=None)
- predict(name, X)
- get_feature_importance(name, feature_names)
- save_model(name, filepath)
- load_model(name, filepath)

DATASET COMPATIBILITY:
----------------------
This script is configured for SWaT-style datasets with:
- Label column: 'Normal/Attack' (NOT 'label')
- Timestamp column: ' Timestamp' (SWaT format with leading space)
- Encoding: 'Normal' -> 0, 'Attack' -> 1

AUTOENCODER THRESHOLD:
----------------------
This script computes and saves the Autoencoder anomaly threshold during training.
The threshold is computed as the 95th percentile of reconstruction error on
NORMAL TRAINING SAMPLES ONLY (y_train == 0).

CRITICAL EXECUTION ORDER FOR AUTOENCODER:
-----------------------------------------
1. Train Autoencoder
2. Compute threshold from NORMAL training samples only
3. Assign threshold to model_registry.autoencoder_threshold
4. THEN predict on X_test (uses training-derived threshold)

This prevents data leakage where Autoencoder prediction would otherwise
compute threshold from X_test data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import FrameworkConfig
from src.data_loader import CPSDataLoader
from src.models import ModelRegistry
from src.visualization import CPSVisualizer
from src.utils import setup_logging, set_random_seeds
import time
import logging
import json
import joblib

logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for CPS anomaly detection framework
    
    Scientific Workflow (No Data Leakage):
    --------------------------------------
    1. Load raw dataset (no preprocessing)
    2. Extract features and labels (no fitting)
    3. Split into train/test FIRST (on raw data)
    4. Fit preprocessors on TRAIN ONLY
    5. Transform both train and test
    6. Save preprocessors (fitted on train only)
    7. Initialize models with initialize_all_models(X_train, y_train)
    8. For Autoencoder (SPECIAL ORDER):
       - train_model()
       - Compute threshold on NORMAL samples only
       - Assign threshold to model_registry.autoencoder_threshold
       - THEN predict() (uses training-derived threshold)
    9. For other models (standard order):
       - train_model()
       - predict()
       - Compute metrics
    10. Compute feature importance
    11. Generate visualizations
    12. Save all results
    
    The saved preprocessors + trained models + Autoencoder threshold are then used by:
    → run_robustness_evaluation.py for robustness analysis
    """
    start_total = time.time()
    set_random_seeds(42)
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'cps_anomaly_detection')
    
    logger.info("\n" + "="*80)
    logger.info("🎓 CPS ANOMALY DETECTION FRAMEWORK")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # =========================================================================
    # PHASE 1: LOAD RAW DATASET (NO PREPROCESSING)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: LOAD RAW DATASET (NO PREPROCESSING)")
    logger.info("="*80)
    
    data_loader = CPSDataLoader(seed=42)
    
    # Load dataset (RAW, no preprocessing)
    df = data_loader.load_dataset(config.data.dataset_path)
    
    # =========================================================================
    # PHASE 2: EXTRACT FEATURES AND LABELS (NO FITTING)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: EXTRACT FEATURES AND LABELS (NO FITTING)")
    logger.info("="*80)
    
    # Extract features and labels (RAW, no scaling/imputation yet)
    X, y = data_loader.extract_features_labels(
        df=df,
        label_column='Normal/Attack',
        timestamp_column=' Timestamp'
    )
    
    logger.info(f"✅ Features extracted: X shape={X.shape}, y shape={y.shape}")
    logger.info(f"✅ Class distribution: Normal={len(y)-y.sum()}, Attack={y.sum()}")
    
    # =========================================================================
    # PHASE 3: SPLIT INTO TRAIN/TEST FIRST (ON RAW DATA)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: SPLIT INTO TRAIN/TEST FIRST (ON RAW DATA)")
    logger.info("="*80)
    
    # Split RAW data BEFORE any preprocessing (CRITICAL for no data leakage)
    X_train_raw, X_test_raw, y_train, y_test = data_loader.split_train_test(
        X=X,
        y=y,
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"✅ Train set: {len(X_train_raw)} samples (RAW)")
    logger.info(f"✅ Test set: {len(X_test_raw)} samples (RAW)")
    logger.info(f"✅ Feature names: {data_loader.feature_columns[:10]}...")
    
    # =========================================================================
    # PHASE 4: FIT PREPROCESSORS ON TRAIN ONLY (NO DATA LEAKAGE)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: FIT PREPROCESSORS ON TRAIN ONLY (NO DATA LEAKAGE)")
    logger.info("="*80)
    
    # Fit imputer and scaler on TRAINING DATA ONLY
    X_train = data_loader.fit_preprocessors(X_train_raw)
    
    logger.info(f"✅ Preprocessors fitted on training data only")
    logger.info(f"✅ X_train shape after preprocessing: {X_train.shape}")
    
    # Transform test data using fitted preprocessors (DO NOT REFIT)
    X_test = data_loader.transform_data(X_test_raw, is_training=False)
    
    logger.info(f"✅ X_test transformed using fitted preprocessors: {X_test.shape}")
    
    # ========================================================================
    # PHASE 5: SAVE PREPROCESSORS (FITTED ON TRAIN ONLY)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: SAVE PREPROCESSORS (FITTED ON TRAIN ONLY)")
    logger.info("="*80)
    
    preprocessing_paths = data_loader.save_preprocessors(str(config.output.models_dir))
    
    logger.info(f"💾 Preprocessing objects saved:")
    for key, path in preprocessing_paths.items():
        logger.info(f"   - {key}: {path}")
    
    # Verify preprocessing was saved
    if 'scaler' not in preprocessing_paths:
        logger.error("❌ CRITICAL: Scaler not saved! Evaluation will be unreliable!")
        logger.error("❌ Please check fit_preprocessors() completed successfully")
    
    # =========================================================================
    # PHASE 6: MODEL INITIALIZATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: MODEL INITIALIZATION")
    logger.info("="*80)
    
    # CORRECTION: ModelRegistry.__init__() does NOT accept seed parameter
    model_registry = ModelRegistry()
    
    # Initialize all models (compatible with REAL models.py)
    model_registry.initialize_all_models(X_train, y_train)
    
    logger.info(f"✅ Models initialized")
    
    # =========================================================================
    # PHASE 7: MODEL TRAINING AND EVALUATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 7: MODEL TRAINING AND EVALUATION")
    logger.info("="*80)
    
    results = {}
    autoencoder_threshold = None  # Will be computed after Autoencoder training
    
    # =========================================================================
    # SPECIAL HANDLING FOR AUTOENCODER (MUST COMPUTE THRESHOLD BEFORE PREDICTION)
    # =========================================================================
    # Train Autoencoder first (if in model list)
    if 'Autoencoder' in config.model.models:
        logger.info(f"\n🔧 Training Autoencoder (SPECIAL ORDER: threshold before prediction)...")
        
        start_time = time.time()
        
        # Train Autoencoder
        model_registry.train_model(
            name='Autoencoder',
            X_train=X_train,
            y_train=y_train,
            X_val=None,
            y_val=None
        )
        
        train_time_ae = time.time() - start_time
        
        # ====================================================================
        # COMPUTE AUTOENCODER THRESHOLD BEFORE PREDICTION (SCIENTIFICALLY VALID)
        # ====================================================================
        logger.info(f"\n  🔬 Computing Autoencoder anomaly threshold BEFORE prediction...")
        logger.info(f"  📊 SCIENTIFIC APPROACH: Using NORMAL training samples only (y_train == 0)")
        
        autoencoder = model_registry.trained_models.get('Autoencoder')
        
        if autoencoder is not None:
            # Select normal training samples (y_train == 0)
            normal_mask = (y_train == 0)
            X_train_normal = X_train[normal_mask]
            n_normal = np.sum(normal_mask)
            n_attack = len(y_train) - n_normal
            
            logger.info(f"  📊 Training data: {len(X_train)} total samples")
            logger.info(f"  📊 Normal samples: {n_normal} (used for threshold)")
            logger.info(f"  📊 Attack samples: {n_attack} (excluded from threshold)")
            
            if n_normal > 0:
                # Get reconstructions for NORMAL training data only
                logger.info(f"  📊 Computing reconstruction errors on {n_normal} normal training samples...")
                reconstructions = autoencoder.predict(X_train_normal, verbose=0)
                
                # Compute reconstruction errors (MSE per sample)
                mse_normal = np.mean(np.power(X_train_normal - reconstructions, 2), axis=1)
                
                # Set threshold as 95th percentile of NORMAL reconstruction errors
                autoencoder_threshold = np.percentile(mse_normal, 95)
                
                logger.info(f"  ✅ Autoencoder threshold computed: {autoencoder_threshold:.6f}")
                logger.info(f"  📊 Normal MSE - Min: {mse_normal.min():.6f}, Max: {mse_normal.max():.6f}, Mean: {mse_normal.mean():.6f}")
                logger.info(f"  📊 Threshold: 95th percentile = {autoencoder_threshold:.6f}")
                
                # CRITICAL: Assign threshold to model_registry BEFORE prediction
                # This ensures prediction uses training-derived threshold (NOT computed from X_test)
                model_registry.autoencoder_threshold = autoencoder_threshold
                logger.info(f"  ✅ Assigned threshold to model_registry.autoencoder_threshold")
                logger.info(f"  ✅ Autoencoder prediction will now use training-derived threshold")
                
                # Save threshold for robustness evaluation
                threshold_path = config.output.models_dir / 'autoencoder_threshold.pkl'
                joblib.dump(autoencoder_threshold, threshold_path)
                logger.info(f"  💾 Saved Autoencoder threshold to: {threshold_path}")
                
                # Save MSE statistics
                mse_stats = {
                    'threshold': float(autoencoder_threshold),
                    'percentile': 95,
                    'mse_min': float(mse_normal.min()),
                    'mse_max': float(mse_normal.max()),
                    'mse_mean': float(mse_normal.mean()),
                    'mse_std': float(mse_normal.std()),
                    'computed_on': 'normal_training_samples_only',
                    'n_normal_samples': int(n_normal),
                    'n_attack_samples': int(n_attack),
                    'total_training_samples': int(len(X_train))
                }
                mse_stats_path = config.output.metrics_dir / 'autoencoder_mse_stats.json'
                with open(mse_stats_path, 'w') as f:
                    json.dump(mse_stats, f, indent=2)
                logger.info(f"  💾 Saved MSE statistics to: {mse_stats_path}")
            else:
                logger.warning(f"  ⚠️  No normal samples found in training data! Cannot compute threshold.")
        else:
            logger.warning(f"  ⚠️  Autoencoder model not found in trained_models!")
        
        # ====================================================================
        # NOW PREDICT (USES TRAINING-DERIVED THRESHOLD)
        # ====================================================================
        logger.info(f"\n  📊 Predicting with Autoencoder (using training-derived threshold)...")
        y_pred_ae = model_registry.predict(name='Autoencoder', X=X_test)
        
        # Compute metrics for Autoencoder
        try:
            accuracy_ae = accuracy_score(y_test, y_pred_ae)
            precision_ae = precision_score(y_test, y_pred_ae, zero_division=0)
            recall_ae = recall_score(y_test, y_pred_ae, zero_division=0)
            f1_ae = f1_score(y_test, y_pred_ae, zero_division=0)
            cm_ae = confusion_matrix(y_test, y_pred_ae)
            
            results['Autoencoder'] = {
                'accuracy': accuracy_ae,
                'precision': precision_ae,
                'recall': recall_ae,
                'f1_score': f1_ae,
                'roc_auc': None,  # Autoencoder doesn't have predict_proba
                'train_time': train_time_ae,
                'confusion_matrix': cm_ae.tolist()
            }
            
            logger.info(f"  ✅ Autoencoder: F1={f1_ae*100:.2f}%, Accuracy={accuracy_ae*100:.2f}%, Time={train_time_ae:.2f}s")
            logger.info(f"  📊 Confusion Matrix: {cm_ae.tolist()}")
            
        except Exception as e:
            logger.error(f"  ❌ Autoencoder: Evaluation failed - {e}")
            results['Autoencoder'] = {'error': str(e)}
        
        # Save Autoencoder model
        model_path_ae = config.output.models_dir / 'Autoencoder.keras'
        model_registry.save_model(name='Autoencoder', filepath=str(model_path_ae))
        logger.info(f"  💾 Saved model: {model_path_ae}")
    
    # =========================================================================
    # STANDARD HANDLING FOR OTHER MODELS
    # =========================================================================
    for model_name in config.model.models:
        # Skip Autoencoder (already handled above)
        if model_name == 'Autoencoder':
            continue
        
        logger.info(f"\n🔧 Training {model_name}...")
        
        start_time = time.time()
        
        # Train model (compatible method)
        model_registry.train_model(
            name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=None,
            y_val=None
        )
        
        train_time = time.time() - start_time
        
        # Predict on test set (compatible method)
        y_pred = model_registry.predict(name=model_name, X=X_test)
        
        # Compute metrics manually (since evaluate_model doesn't exist)
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Compute confusion matrix (for visualization dashboard)
            cm = confusion_matrix(y_test, y_pred)
            
            # Try ROC-AUC (may fail for some models)
            roc_auc = None
            try:
                trained_model = model_registry.trained_models.get(model_name)
                
                if trained_model is not None and hasattr(trained_model, 'predict_proba'):
                    y_proba = trained_model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                    logger.debug(f"  📊 {model_name}: ROC-AUC computed from trained model")
                else:
                    logger.debug(f"  ⚠️  {model_name}: No predict_proba method or model not found")
            except Exception as e:
                logger.debug(f"  ⚠️  {model_name}: ROC-AUC computation failed - {e}")
                roc_auc = None
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'train_time': train_time,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"  ✅ {model_name}: F1={f1*100:.2f}%, Accuracy={accuracy*100:.2f}%, Time={train_time:.2f}s")
            logger.info(f"  📊 Confusion Matrix: {cm.tolist()}")
            
        except Exception as e:
            logger.error(f"  ❌ {model_name}: Evaluation failed - {e}")
            results[model_name] = {'error': str(e)}
        
        # Save trained model (compatible method)
        model_path = config.output.models_dir / f'{model_name}.pkl' if model_name not in ['LSTM'] else config.output.models_dir / f'{model_name}.keras'
        model_registry.save_model(name=model_name, filepath=str(model_path))
        logger.info(f"  💾 Saved model: {model_path}")
    
    # =========================================================================
    # PHASE 8: FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 8: FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    
    feature_importance = {}
    
    # Compute feature importance for tree-based models (compatible method)
    for model_name in ['RandomForest', 'XGBoost']:
        if model_name in config.model.models:
            try:
                importance = model_registry.get_feature_importance(
                    name=model_name,
                    feature_names=data_loader.feature_columns
                )
                if importance is not None:
                    feature_importance[model_name] = importance
                    logger.info(f"✅ Feature importance computed for {model_name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not compute feature importance for {model_name}: {e}")
    
    # =========================================================================
    # PHASE 9: VISUALIZATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 9: VISUALIZATION")
    logger.info("="*80)
    
    visualizer = CPSVisualizer(config.output.figures_dir)
    
    # Generate visualizations (only if we have valid results)
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        visualizer.plot_model_comparison(valid_results, metric='f1_score', save_name='f1_comparison.png')
        visualizer.plot_model_comparison(valid_results, metric='accuracy', save_name='accuracy_comparison.png')
        visualizer.plot_training_time(valid_results, save_name='training_time.png')
        logger.info(f"✅ Generated model comparison plots")
    
    # Feature importance plot
    if feature_importance and 'RandomForest' in feature_importance:
        visualizer.plot_feature_importance(
            feature_importance['RandomForest'],
            top_n=10,
            save_name='feature_importance.png'
        )
        logger.info(f"✅ Generated feature importance plot")
    
    # Comprehensive dashboard (now with real confusion matrices)
    rf_importance = feature_importance.get('RandomForest', {})
    visualizer.create_comprehensive_dashboard(valid_results, rf_importance)
    logger.info(f"✅ Generated comprehensive dashboard")
    
    # =========================================================================
    # PHASE 10: SAVING RESULTS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 10: SAVING RESULTS")
    logger.info("="*80)
    
    # Save metrics (convert numpy types for JSON)
    results_serializable = {}
    for model_name, metrics in results.items():
        results_serializable[model_name] = {}
        for key, value in metrics.items():
            if isinstance(value, (np.floating, np.integer)):
                results_serializable[model_name][key] = float(value)
            elif isinstance(value, np.ndarray):
                results_serializable[model_name][key] = value.tolist()
            else:
                results_serializable[model_name][key] = value
    
    metrics_path = config.output.metrics_dir / 'all_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"💾 Saved metrics: {metrics_path}")
    
    # Save feature importance
    if feature_importance:
        fi_serializable = {}
        for model_name, importance in feature_importance.items():
            if isinstance(importance, dict):
                fi_serializable[model_name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                              for k, v in importance.items()}
        
        fi_path = config.output.metrics_dir / 'feature_importance.json'
        with open(fi_path, 'w') as f:
            json.dump(fi_serializable, f, indent=2)
        logger.info(f"💾 Saved feature importance: {fi_path}")
    
    # Save preprocessing info
    preprocessing_info = {
        'feature_columns': data_loader.feature_columns,
        'label_column': data_loader.label_column,
        'timestamp_column': data_loader.timestamp_column,
        'imputer_fitted': data_loader.imputer is not None,
        'scaler_fitted': data_loader.scaler is not None,
        'fitted_on': 'training_data_only'
    }
    preprocessing_info_path = config.output.metrics_dir / 'preprocessing_info.json'
    with open(preprocessing_info_path, 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    logger.info(f"💾 Saved preprocessing info: {preprocessing_info_path}")
    
    # Save Autoencoder threshold info
    if autoencoder_threshold is not None:
        ae_threshold_str = f"{autoencoder_threshold:.6f}"
        ae_threshold_info = {
            'threshold': float(autoencoder_threshold),
            'percentile': 95,
            'computed_on': 'normal_training_samples_only',
            'saved_path': str(config.output.models_dir / 'autoencoder_threshold.pkl'),
            'assigned_to_model_registry': True,
            'used_for_prediction_in_main': True
        }
        ae_threshold_path = config.output.metrics_dir / 'autoencoder_threshold_info.json'
        with open(ae_threshold_path, 'w') as f:
            json.dump(ae_threshold_info, f, indent=2)
        logger.info(f"💾 Saved Autoencoder threshold info: {ae_threshold_path}")
    else:
        ae_threshold_str = "Not computed"
    
    # Save experiment report
    total_time = time.time() - start_total
    best_model = max(valid_results, key=lambda x: valid_results[x].get('f1_score', 0)) if valid_results else None
    
    report = f"""
================================================================================
CPS ANOMALY DETECTION - EXPERIMENT REPORT
================================================================================
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Author: BERETE MAMADY
Institution: Dalian University of Technology

DATASET:
  - Path: {config.data.dataset_path}
  - Samples: {len(df)}
  - Features: {len(data_loader.feature_columns)}
  - Label Column: {data_loader.label_column}
  - Timestamp Column: {data_loader.timestamp_column}
  - Train/Test Split: {len(X_train)}/{len(X_test)}
  - Class Distribution: Normal={len(y)-y.sum()}, Attack={y.sum()}

PREPROCESSING (NO DATA LEAKAGE):
  - Imputer: Fitted on TRAINING DATA ONLY (strategy=mean)
  - Scaler: Fitted on TRAINING DATA ONLY (StandardScaler)
  - Test Data: Transformed using fitted preprocessors (NOT refit)
  - Saved: {list(preprocessing_paths.keys())}

AUTOENCODER THRESHOLD (SCIENTIFICALLY VALID):
  - Threshold: {ae_threshold_str}
  - Percentile: 95th
  - Computed on: Normal training samples only (prevents data leakage)
  - Saved: results/models/autoencoder_threshold.pkl
  - Assigned to model_registry: Yes
  - Used for prediction in main.py: Yes (BEFORE predict() call)

MODELS TRAINED: {len(valid_results)}
  {', '.join(valid_results.keys())}

RESULTS SUMMARY:
"""
    for model_name, metrics in valid_results.items():
        report += f"  - {model_name}: F1={metrics.get('f1_score', 0)*100:.2f}%, Accuracy={metrics.get('accuracy', 0)*100:.2f}%\n"
    
    if best_model and best_model in valid_results:
        report += f"""
BEST MODEL: {best_model}
  - F1-Score: {valid_results[best_model].get('f1_score', 0)*100:.2f}%
  - Accuracy: {valid_results[best_model].get('accuracy', 0)*100:.2f}%
  - Training Time: {valid_results[best_model].get('train_time', 0):.2f} sec
"""
    
    report += f"""
TOTAL EXECUTION TIME: {total_time/60:.2f} minutes

OUTPUT FILES:
  - Models: {config.output.models_dir}
  - Figures: {config.output.figures_dir}
  - Metrics: {config.output.metrics_dir}
  - Logs: {config.output.logs_dir}
  - Preprocessing: {list(preprocessing_paths.values())}

================================================================================
END OF REPORT
================================================================================
"""
    
    report_path = config.output.logs_dir / f'experiment_report_{time.strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"💾 Saved experiment report: {report_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\n📊 Total Execution Time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    
    if best_model and best_model in valid_results:
        logger.info(f"🏆 Best Model: {best_model}")
        logger.info(f"📈 Best F1-Score: {valid_results[best_model].get('f1_score', 0)*100:.2f}%")
        logger.info(f"📈 Best Accuracy: {valid_results[best_model].get('accuracy', 0)*100:.2f}%")
    
    logger.info(f"\n💾 Results saved to: {config.output.base_dir}")
    logger.info(f"   - Models: {config.output.models_dir}")
    logger.info(f"   - Figures: {config.output.figures_dir}")
    logger.info(f"   - Metrics: {config.output.metrics_dir}")
    logger.info(f"   - Logs: {config.output.logs_dir}")
    logger.info(f"   - Preprocessing: {list(preprocessing_paths.values())}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ SCIENTIFIC VALIDITY CONFIRMED:")
    logger.info("✅ Preprocessors fitted on TRAINING DATA ONLY (no data leakage)")
    logger.info("✅ Test data transformed using fitted preprocessors (not refit)")
    logger.info("✅ Confusion matrices computed for all models (dashboard ready)")
    if autoencoder_threshold is not None:
        logger.info("✅ Autoencoder threshold computed on NORMAL training samples only")
        logger.info(f"✅ Autoencoder threshold saved: {autoencoder_threshold:.6f}")
        logger.info("✅ Autoencoder threshold assigned to model_registry BEFORE prediction")
        logger.info("✅ Autoencoder prediction used training-derived threshold (NOT X_test)")
        logger.info("✅ This ensures NO data leakage in Autoencoder evaluation")
    logger.info("⚠️  Preprocessing objects (imputer, scaler) have been saved.")
    logger.info("⚠️  Trained models have been saved to results/models/")
    logger.info("⚠️  Run 'python run_robustness_evaluation.py' for robustness analysis.")
    logger.info("⚠️  DO NOT delete files in results/models/ before evaluation!")
    logger.info("="*80)

if __name__ == '__main__':
    main()