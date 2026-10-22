"""
CPS Model Robustness Evaluator
Evaluates trained models on original and anomalous data
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

IMPORTANT: Scientific Consistency with anomaly_generator.py and robustness_evaluator.py
-----------------------------------------------------------------------------------------
This evaluator is designed to work seamlessly with the CPSAnomalyGenerator module.
Key design principles:

1. LABEL DISTINCTION:
   - original_label: Ground-truth from the dataset (e.g., 'Normal/Attack' for SWaT)
   - generated_anomaly_label: Ground-truth from simulator (injected anomalies)
   
   This distinction is CRITICAL for simulation-based evaluation because:
   - Original labels measure standard detection performance
   - Generated labels measure robustness to simulated anomalies
   - Separating them prevents evaluation bias and data leakage

2. THRESHOLD HANDLING (Autoencoder):
   - Anomaly thresholds MUST be learned from training/validation data
   - Computing thresholds from test/evaluation data causes DATA LEAKAGE
   - This file accepts pre-computed thresholds as parameters

3. SEQUENCE ALIGNMENT (LSTM):
   - LSTM requires careful handling of time-step sequences
   - Labels must be properly aligned with sequence outputs
   - This file ensures consistency with training procedure

4. ROBUSTNESS METRICS:
   - Performance on original data = baseline capability
   - Performance on generated anomalies = robustness capability
   - Performance drop = vulnerability to specific anomaly types

5. LABEL TYPE CONSISTENCY:
   - All labels are converted to numeric (0=normal, 1=anomaly) before metric computation
   - Supports both string labels ('Normal'/'Attack') and numeric labels (0/1)
   - Ensures sklearn metrics work correctly without TypeError

Research Framework Contribution:
--------------------------------
This evaluator enables scientifically valid robustness assessment by:
- Maintaining separation between original and generated ground-truth
- Preventing data leakage in threshold-based models
- Ensuring temporal consistency in sequence models
- Providing reproducible robustness scores for thesis validation
- Handling label type conversions automatically for metric computation

SWaT Dataset Compatibility:
---------------------------
- Original label: 'Normal/Attack' (primary) or 'label' (fallback)
- Generated label: 'generated_anomaly_label'
- Label encoding: Automatic conversion to numeric (0=normal, 1=anomaly)
"""

import numpy as np
import pandas as pd
import time
import joblib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)

class CPSRobustnessEvaluator:
    """
    Evaluates robustness of trained models on anomalous data
    
    Research Application:
    ---------------------
    This class provides scientifically rigorous evaluation of anomaly detection
    models under simulated CPS anomaly conditions. It supports:
    
    1. Baseline Evaluation: Performance on original dataset (standard metrics)
    2. Robustness Evaluation: Performance on injected anomalies (resilience metrics)
    3. Comparative Analysis: Model ranking by robustness scores
    4. Threshold Management: Pre-computed thresholds for unsupervised models
    5. Label Type Conversion: Automatic conversion of string labels to numeric
    
    The generated_anomaly_label column from anomaly_generator.py serves as
    ground-truth for robustness evaluation, enabling precise measurement of
    model degradation under controlled anomaly conditions.
    
    SWaT Compatibility:
    -------------------
    - Original label: 'Normal/Attack' (primary) or 'label' (fallback)
    - Generated label: 'generated_anomaly_label'
    - Label encoding: Automatic (string 'Normal'->0, 'Attack'->1)
    """
    
    # Column names for label handling (SWaT-compatible)
    ORIGINAL_LABEL_COLUMN = 'Normal/Attack'  # Primary: SWaT-style label
    ORIGINAL_LABEL_COLUMN_FALLBACK = 'label'  # Fallback: standard format
    GENERATED_LABEL_COLUMN = 'generated_anomaly_label'
    
    def __init__(self, models_dir: Path, autoencoder_threshold: Optional[float] = None):
        """
        Initialize evaluator
        
        Args:
            models_dir: Directory containing saved models
            autoencoder_threshold: Pre-computed threshold for Autoencoder
                                 (CRITICAL: Must be from training/validation, NOT test data)
        
        Research Note:
        --------------
        The autoencoder_threshold parameter is essential for scientific validity.
        Computing the reconstruction error threshold from evaluation data introduces
        data leakage, as the threshold becomes optimized for the test distribution.
        
        Proper practice:
        1. Train autoencoder on normal training data
        2. Compute threshold on validation set (e.g., 95th percentile of reconstruction error)
        3. Use this FIXED threshold for all subsequent evaluations
        4. Never recompute threshold from test/evaluation data
        
        This mirrors real-world deployment where thresholds are fixed after training.
        """
        self.models_dir = Path(models_dir)
        self.trained_models = {}
        self.results = {}
        
        # Autoencoder threshold (must be pre-computed from training/validation)
        self.autoencoder_threshold = autoencoder_threshold
        if autoencoder_threshold is None:
            logger.warning("⚠️  No autoencoder_threshold provided. Will use default 95th percentile.")
            logger.warning("⚠️  WARNING: This may introduce data leakage in evaluation!")
            logger.warning("⚠️  Recommended: Compute threshold from validation data and pass as parameter.")
        
        # LSTM configuration (must match training configuration)
        self.lstm_time_steps = 10  # Must match the time_steps used during LSTM training
        
    def load_all_models(self, model_names: List[str]):
        """
        Load all trained models from disk
        
        Args:
            model_names: List of model names to load
            
        Research Note:
        --------------
        Models are loaded from their saved state (.pkl for sklearn, .keras for deep learning).
        This ensures evaluation uses the exact same model weights/parameters as training,
        maintaining scientific reproducibility.
        """
        logger.info(f"📥 Loading {len(model_names)} trained models...")
        
        for model_name in model_names:
            try:
                # Determine file extension based on model type
                if model_name in ['Autoencoder', 'LSTM']:
                    filepath = self.models_dir / f'{model_name}.keras'
                    from tensorflow import keras
                    model = keras.models.load_model(str(filepath))
                    logger.info(f"✅ Loaded {model_name} (Keras model)")
                else:
                    filepath = self.models_dir / f'{model_name}.pkl'
                    model = joblib.load(str(filepath))
                    logger.info(f"✅ Loaded {model_name} (sklearn model)")
                
                self.trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
        
        logger.info(f"✅ Loaded {len(self.trained_models)}/{len(model_names)} models")
    
    def set_autoencoder_threshold(self, threshold: float, source: str = "validation"):
        """
        Set pre-computed Autoencoder threshold
        
        Args:
            threshold: Reconstruction error threshold
            source: Where threshold was computed from (e.g., "validation", "training")
            
        Research Note:
        --------------
        This method should be called BEFORE evaluation with a threshold computed from
        training or validation data. Example:
        
        # During training phase:
        train_recon_errors = autoencoder.predict(X_train_normal)
        threshold = np.percentile(train_recon_errors, 95)  # 95th percentile
        
        # During evaluation phase:
        evaluator.set_autoencoder_threshold(threshold, source="training_95th_percentile")
        
        This prevents data leakage and ensures scientific validity.
        """
        self.autoencoder_threshold = threshold
        logger.info(f"✅ Autoencoder threshold set to {threshold:.6f} (from {source})")
    
    def predict_with_model(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with loaded model
        
        Args:
            model_name: Name of the model
            X: Features
            
        Returns:
            (predictions, probabilities)
            
        Research Note:
        --------------
        This method handles model-specific prediction logic:
        - Deep learning models (LSTM, Autoencoder): Special sequence/threshold handling
        - Unsupervised models (IsolationForest, OneClassSVM): Convert -1/1 to 0/1
        - Supervised models (RandomForest, XGBoost, LinearSVM): Standard predict/proba
        
        The label distinction (original vs. generated_anomaly_label) is handled at the
        evaluation level (evaluate_model, evaluate_all_models), not at prediction level.
        This separation of concerns ensures clean, maintainable code.
        """
        model = self.trained_models[model_name]
        
        try:
            if model_name in ['Autoencoder', 'LSTM']:
                # Deep learning models require special handling
                if model_name == 'LSTM':
                    y_pred, y_proba = self._predict_lstm(model, X)
                    
                elif model_name == 'Autoencoder':
                    y_pred, y_proba = self._predict_autoencoder(model, X)
                    
            elif model_name in ['IsolationForest', 'OneClassSVM']:
                # Unsupervised anomaly detection models
                y_pred, y_proba = self._predict_unsupervised(model, X, model_name)
                
            else:
                # Standard supervised classifiers (RandomForest, XGBoost, LinearSVM)
                y_pred = model.predict(X)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]
                else:
                    y_proba = None
            
            return y_pred, y_proba
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {model_name}: {e}")
            return None, None
    
    def _predict_lstm(self, model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM prediction with proper sequence handling
        
        Args:
            model: Trained LSTM model
            X: Input features
            
        Returns:
            (predictions, probabilities)
            
        Research Note:
        --------------
        LSTM models require input sequences of shape (samples, time_steps, features).
        Critical considerations:
        
        1. SEQUENCE CREATION: Must match training procedure exactly
           - Same time_steps value
           - Same padding/trimming strategy
           - Same normalization (if applied)
        
        2. LABEL ALIGNMENT: LSTM outputs one prediction per sequence
           - If sequences overlap, labels must be aligned accordingly
           - First (time_steps - 1) samples may not have valid predictions
           - This implementation pads predictions to match original length
        
        3. THRESHOLD: Binary classification threshold (default 0.5)
           - Can be adjusted based on validation performance
           - Should be consistent with training threshold
        """
        time_steps = self.lstm_time_steps
        
        # Create sequences matching training procedure
        X_seq = self._create_sequences(X, time_steps)
        
        # Get predictions from LSTM
        y_proba = model.predict(X_seq, verbose=0)
        
        # Handle output shape
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            # Multi-class output, take probability of anomaly class
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba.flatten()
        
        # Pad predictions to match original length
        # First (time_steps - 1) samples don't have valid sequence predictions
        pad_width = time_steps - 1
        y_proba_padded = np.concatenate([np.zeros(pad_width), y_proba])
        y_proba_padded = y_proba_padded[:len(X)]  # Trim to original length
        
        # Convert probabilities to binary predictions
        y_pred = (y_proba_padded > 0.5).astype(int)
        
        logger.debug(f"📊 LSTM: Input shape={X.shape}, Sequence shape={X_seq.shape}, Output length={len(y_pred)}")
        
        return y_pred, y_proba_padded
    
    def _predict_autoencoder(self, model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoencoder prediction with fixed threshold
        
        Args:
            model: Trained Autoencoder model
            X: Input features
            
        Returns:
            (predictions, probabilities)
            
        Research Note:
        --------------
        Autoencoders detect anomalies via reconstruction error:
        1. Normal samples → Low reconstruction error
        2. Anomalous samples → High reconstruction error
        
        CRITICAL: The threshold MUST be pre-computed from training/validation data.
        Computing threshold from evaluation data causes DATA LEAKAGE because:
        - Threshold becomes optimized for test distribution
        - Metrics become overly optimistic
        - Results are not reproducible in real deployment
        
        Proper threshold computation (done during training):
        ```
        train_recon = autoencoder.predict(X_train_normal)
        train_mse = np.mean(np.power(X_train_normal - train_recon, 2), axis=1)
        threshold = np.percentile(train_mse, 95)  # 95th percentile
        ```
        
        This evaluator uses the pre-computed threshold stored in self.autoencoder_threshold.
        """
        # Get reconstruction from autoencoder
        reconstructions = model.predict(X, verbose=0)
        
        # Compute reconstruction error (MSE per sample)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Use pre-computed threshold (CRITICAL for scientific validity)
        if self.autoencoder_threshold is not None:
            threshold = self.autoencoder_threshold
            logger.debug(f"📊 Autoencoder: Using pre-computed threshold={threshold:.6f}")
        else:
            # Fallback: compute from current data (NOT RECOMMENDED - data leakage risk!)
            threshold = np.percentile(mse, 95)
            logger.warning(f"⚠️  Autoencoder: Computing threshold from evaluation data (data leakage risk!)")
            logger.warning(f"⚠️  Threshold={threshold:.6f}")
        
        # Convert reconstruction error to anomaly probability (normalized)
        # Higher MSE = higher probability of anomaly
        mse_max = mse.max() if mse.max() > 0 else 1.0
        y_proba = mse / mse_max  # Normalize to [0, 1]
        
        # Binary predictions using threshold
        y_pred = (mse > threshold).astype(int)
        
        logger.debug(f"📊 Autoencoder: MSE range=[{mse.min():.6f}, {mse.max():.6f}], Threshold={threshold:.6f}")
        
        return y_pred, y_proba
    
    def _predict_unsupervised(self, model, X: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unsupervised model prediction (IsolationForest, OneClassSVM)
        
        Args:
            model: Trained unsupervised model
            X: Input features
            model_name: Name of the model
            
        Returns:
            (predictions, probabilities)
            
        Research Note:
        --------------
        Unsupervised models output:
        - 1 for normal samples
        - -1 for anomalies
        
        We convert to binary (0=normal, 1=anomaly) for consistency with generated_anomaly_label.
        
        Note: These models don't provide true probabilities, only anomaly scores.
        For ROC-AUC computation, we use decision_function or score_samples as proxy.
        """
        # Get raw predictions (-1 for anomaly, 1 for normal)
        raw_pred = model.predict(X)
        
        # Convert to binary (0=normal, 1=anomaly)
        y_pred = np.where(raw_pred == -1, 1, 0)
        
        # Get anomaly scores for probability estimation (if available)
        y_proba = None
        try:
            if hasattr(model, 'decision_function'):
                # OneClassSVM: decision_function
                scores = model.decision_function(X)
                # Convert to probability-like scores (higher = more anomalous)
                y_proba = -scores  # Negate because higher decision_function = more normal
                # Normalize to [0, 1]
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
            elif hasattr(model, 'score_samples'):
                # IsolationForest: score_samples
                scores = model.score_samples(X)
                # Convert to probability-like scores
                y_proba = -scores
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
        except Exception as e:
            logger.debug(f"⚠️  Could not compute anomaly scores for {model_name}: {e}")
        
        logger.debug(f"📊 {model_name}: Predicted {y_pred.sum()} anomalies out of {len(y_pred)} samples")
        
        return y_pred, y_proba
    
    def _create_sequences(self, X: np.ndarray, time_steps: int) -> np.ndarray:
        """
        Create sequences for LSTM input
        
        Args:
            X: Input features
            time_steps: Number of time steps per sequence
            
        Returns:
            Sequences of shape (n_sequences, time_steps, n_features)
            
        Research Note:
        --------------
        This method MUST match the sequence creation used during LSTM training.
        Any mismatch will cause prediction errors or degraded performance.
        
        Sequence creation strategy:
        - Sliding window approach
        - Each sequence overlaps with previous by (time_steps - 1) samples
        - Number of sequences = len(X) - time_steps + 1
        """
        sequences = []
        for i in range(len(X) - time_steps + 1):
            sequences.append(X[i:i + time_steps])
        
        X_seq = np.array(sequences)
        logger.debug(f"📊 Created {len(X_seq)} sequences with shape {X_seq.shape}")
        
        return X_seq
    
    def _convert_labels_to_numeric(self, y: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
        """
        Convert labels to numeric format (0=normal, 1=anomaly)
        
        Args:
            y: Labels (can be string or numeric)
            
        Returns:
            Numeric labels (0=normal, 1=anomaly)
            
        Research Note:
        --------------
        This method ensures label type consistency for sklearn metrics.
        It handles:
        - String labels: 'Normal'->0, 'Attack'->1, 'normal'->0, 'attack'->1
        - Numeric labels: 0->0, 1->1 (unchanged)
        - Mixed types: Converted consistently
        
        This prevents TypeError when y_true and y_pred have different types.
        """
        if y is None:
            return None
        
        # Convert to numpy array for consistent handling
        y_array = np.asarray(y)
        
        # Check if already numeric
        if np.issubdtype(y_array.dtype, np.number):
            # Already numeric, ensure it's integer type
            return y_array.astype(int)
        
        # String labels - convert to numeric
        # Convert to lowercase for case-insensitive comparison
        y_lower = np.char.lower(y_array.astype(str))
        
        # Create numeric array
        y_numeric = np.zeros(len(y_array), dtype=int)
        
        # Map anomaly/attack labels to 1
        anomaly_indicators = ['attack', 'anomaly', 'intrusion', '1', 'true', 'abnormal']
        for indicator in anomaly_indicators:
            y_numeric[y_lower == indicator] = 1
        
        # Normal labels remain 0
        # (already initialized to 0)
        
        logger.debug(f"📊 Converted {len(y_array)} labels to numeric: {np.sum(y_numeric == 0)} normal, {np.sum(y_numeric == 1)} anomaly")
        
        return y_numeric
    
    def evaluate_model(self, 
                      model_name: str, 
                      X: np.ndarray, 
                      y_true: Union[np.ndarray, pd.Series, list],
                      label_type: str = 'generated') -> Dict:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            X: Features
            y_true: True labels (either original or generated)
            label_type: 'generated' or 'original' - specifies which label column was used
            
        Returns:
            Dictionary with evaluation metrics
            
        Research Note:
        --------------
        The label_type parameter documents which ground-truth was used for evaluation:
        
        - label_type='generated': Evaluating robustness to injected anomalies
          (y_true = generated_anomaly_label from anomaly_generator.py)
        
        - label_type='original': Evaluating standard detection performance
          (y_true = original label column from dataset, e.g., 'Normal/Attack' for SWaT)
        
        This distinction is CRITICAL for thesis research because:
        1. Generated labels provide controlled ground-truth for robustness testing
        2. Original labels measure real-world detection capability
        3. Comparing both reveals model strengths and vulnerabilities
        
        LABEL TYPE CONSISTENCY:
        -----------------------
        All labels are converted to numeric (0=normal, 1=anomaly) before metric computation.
        This ensures sklearn metrics work correctly regardless of input label type.
        
        The prediction logic (predict_with_model) is independent of label type.
        Label distinction is handled HERE at the evaluation level, not at prediction level.
        This ensures clean separation of concerns.
        """
        logger.info(f"📊 Evaluating {model_name} (label_type={label_type})...")
        
        start_time = time.time()
        y_pred, y_proba = self.predict_with_model(model_name, X)
        inference_time = time.time() - start_time
        
        if y_pred is None:
            return {'error': 'Prediction failed'}
        
        # CRITICAL FIX: Convert y_true to numeric for consistent metric computation
        y_true_numeric = self._convert_labels_to_numeric(y_true)
        
        # Ensure y_pred is also numeric (should already be, but ensure consistency)
        y_pred_numeric = self._convert_labels_to_numeric(y_pred)
        
        # Compute standard metrics
        metrics = {
            'accuracy': accuracy_score(y_true_numeric, y_pred_numeric),
            'precision': precision_score(y_true_numeric, y_pred_numeric, zero_division=0),
            'recall': recall_score(y_true_numeric, y_pred_numeric, zero_division=0),
            'f1_score': f1_score(y_true_numeric, y_pred_numeric, zero_division=0),
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(X),
            'confusion_matrix': confusion_matrix(y_true_numeric, y_pred_numeric).tolist(),
            'label_type': label_type,
            'n_samples': len(X),
            'n_anomalies': int(y_true_numeric.sum())
        }
        
        # Compute ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                # Check if we have both classes in y_true
                if len(np.unique(y_true_numeric)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true_numeric, y_proba)
                else:
                    metrics['roc_auc'] = None
                    logger.debug(f"⚠️  ROC-AUC not computed: only one class present in labels")
            except Exception as e:
                metrics['roc_auc'] = None
                logger.debug(f"⚠️  ROC-AUC computation failed: {e}")
        else:
            metrics['roc_auc'] = None
        
        logger.info(f"  Accuracy={metrics['accuracy']*100:.2f}%, F1={metrics['f1_score']*100:.2f}%, "
                   f"Time={inference_time:.3f}s ({metrics['inference_time_per_sample']*1000:.2f}ms/sample)")
        
        return metrics
    
    def evaluate_all_models(self, 
                           X_original: np.ndarray, 
                           y_original: Union[np.ndarray, pd.Series, list],
                           X_anomalous: np.ndarray,
                           y_anomalous: Union[np.ndarray, pd.Series, list],
                           anomaly_name: str = "generated",
                           use_generated_label: bool = True) -> Dict:
        """
        Evaluate all models on both original and anomalous data
        
        Args:
            X_original: Original features
            y_original: Original labels (from dataset label column, e.g., 'Normal/Attack')
            X_anomalous: Anomalous features (with injected anomalies)
            y_anomalous: Anomalous labels (generated_anomaly_label from anomaly_generator.py)
            anomaly_name: Name of the anomaly type
            use_generated_label: If True, use generated_anomaly_label for robustness evaluation
            
        Returns:
            Dictionary with all evaluation results
            
        Research Note:
        --------------
        This method performs COMPARATIVE evaluation:
        
        1. BASELINE (original data):
           - Measures standard detection performance
           - Uses original label column as ground-truth (e.g., 'Normal/Attack' for SWaT)
           - Answers: "How well does the model detect real anomalies?"
        
        2. ROBUSTNESS (anomalous data):
           - Measures resilience to injected anomalies
           - Uses generated_anomaly_label as ground-truth
           - Answers: "How well does the model maintain performance under simulated attacks?"
        
        3. PERFORMANCE DROP:
           - Quantifies degradation from baseline to robustness
           - Higher drop = more vulnerable to that anomaly type
           - Used to compute robustness scores
        
        The use_generated_label parameter controls which labels are used for anomalous 
        - True (recommended): Use generated_anomaly_label for precise robustness measurement
        - False: Use original labels (less precise for injected anomaly evaluation)
        
        Label distinction is handled HERE, not in predict_with_model().
        This ensures clean separation: predictions are label-agnostic, evaluation is label-aware.
        
        LABEL TYPE CONSISTENCY:
        -----------------------
        All labels are automatically converted to numeric (0=normal, 1=anomaly) in
        evaluate_model() before metric computation. This handles:
        - String labels: 'Normal'/'Attack' -> 0/1
        - Numeric labels: 0/1 -> 0/1 (unchanged)
        - Ensures sklearn metrics work without TypeError
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 EVALUATING MODELS ON {anomaly_name.upper()} DATA")
        logger.info(f"{'='*80}")
        logger.info(f"📊 Original samples: {len(X_original)}, Anomalous samples: {len(X_anomalous)}")
        logger.info(f"📊 Using generated_anomaly_label: {use_generated_label}")
        
        results = {
            'anomaly_type': anomaly_name,
            'models': {}
        }
        
        for model_name in self.trained_models.keys():
            # Evaluate on original data (baseline performance)
            metrics_original = self.evaluate_model(
                model_name, X_original, y_original, 
                label_type='original'
            )
            
            # Evaluate on anomalous data (robustness performance)
            # Use generated_anomaly_label as ground-truth for injected anomalies
            metrics_anomalous = self.evaluate_model(
                model_name, X_anomalous, y_anomalous,
                label_type='generated' if use_generated_label else 'original'
            )
            
            # Calculate performance drop (key robustness metric)
            performance_drop = {}
            if 'error' not in metrics_original and 'error' not in metrics_anomalous:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                    if metric in metrics_original and metric in metrics_anomalous:
                        if metrics_original[metric] is not None and metrics_anomalous[metric] is not None:
                            drop = metrics_original[metric] - metrics_anomalous[metric]
                            performance_drop[f'{metric}_drop'] = drop
                            performance_drop[metric] = metrics_anomalous[metric]  # Also store anomalous value
            
            results['models'][model_name] = {
                'original': metrics_original,
                'anomalous': metrics_anomalous,
                'performance_drop': performance_drop
            }
            
            # Log summary for this model
            if 'error' not in metrics_original and 'error' not in metrics_anomalous:
                f1_drop = performance_drop.get('f1_score_drop', 0)
                logger.info(f"  {model_name}: F1 Original={metrics_original['f1_score']*100:.2f}%, "
                           f"Anomalous={metrics_anomalous['f1_score']*100:.2f}%, "
                           f"Drop={f1_drop*100:.2f}%")
        
        self.results[anomaly_name] = results
        
        return results
    
    def compute_robustness_scores(self) -> pd.DataFrame:
        """
        Compute robustness scores for all models
        
        Returns:
            DataFrame with robustness scores
            
        Research Note:
        --------------
        Robustness Score Formula:
        -------------------------
        robustness_score = 1 - (performance_drop / original_performance)
        
        Interpretation:
        - Score = 1.0: Perfect robustness (no performance degradation)
        - Score = 0.5: 50% performance retention (moderate vulnerability)
        - Score = 0.0: Complete failure (model cannot handle anomaly)
        - Score < 0.0: Performance worse than random (severe vulnerability)
        
        This normalized score enables:
        1. Fair comparison across models with different baseline performance
        2. Aggregation across multiple anomaly types
        3. Ranking models by overall robustness
        4. Statistical analysis of robustness patterns
        
        The score is computed using F1-Score as the primary metric because:
        - F1 balances precision and recall (important for imbalanced CPS data)
        - F1 is standard in anomaly detection research
        - F1 is more informative than accuracy for imbalanced datasets
        """
        logger.info("📊 Computing robustness scores...")
        
        robustness_data = []
        
        for anomaly_name, results in self.results.items():
            for model_name, metrics in results['models'].items():
                if 'error' not in metrics['original'] and 'error' not in metrics['anomalous']:
                    # Compute robustness score based on F1-Score
                    f1_drop = metrics['performance_drop'].get('f1_score_drop', 0)
                    f1_original = metrics['original']['f1_score']
                    
                    # Robustness = 1 - (drop / original)
                    # This normalizes the drop relative to baseline performance
                    robustness_score = 1 - (f1_drop / f1_original) if f1_original > 0 else 0
                    
                    # Also compute accuracy-based robustness for comparison
                    acc_drop = metrics['performance_drop'].get('accuracy_drop', 0)
                    acc_original = metrics['original']['accuracy']
                    robustness_score_acc = 1 - (acc_drop / acc_original) if acc_original > 0 else 0
                    
                    robustness_data.append({
                        'anomaly_type': anomaly_name,
                        'model': model_name,
                        'f1_original': f1_original,
                        'f1_anomalous': metrics['anomalous']['f1_score'],
                        'f1_drop': f1_drop,
                        'robustness_score_f1': robustness_score,
                        'accuracy_original': acc_original,
                        'accuracy_anomalous': metrics['anomalous']['accuracy'],
                        'accuracy_drop': acc_drop,
                        'robustness_score_acc': robustness_score_acc,
                        'precision_drop': metrics['performance_drop'].get('precision_drop', 0),
                        'recall_drop': metrics['performance_drop'].get('recall_drop', 0),
                        'roc_auc_drop': metrics['performance_drop'].get('roc_auc_drop', 0),
                        'inference_time_original': metrics['original']['inference_time'],
                        'inference_time_anomalous': metrics['anomalous']['inference_time']
                    })
        
        df_robustness = pd.DataFrame(robustness_data)
        
        # Log summary statistics
        if len(df_robustness) > 0:
            avg_robustness = df_robustness.groupby('model')['robustness_score_f1'].mean()
            logger.info(f"📈 Average Robustness Scores by Model:")
            for model, score in avg_robustness.sort_values(ascending=False).items():
                logger.info(f"   {model}: {score:.4f}")
        
        return df_robustness
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to CSV
        
        Args:
            filepath: Path to save results
        """
        df_robustness = self.compute_robustness_scores()
        df_robustness.to_csv(filepath, index=False)
        logger.info(f"💾 Robustness results saved to: {filepath}")
    
    def generate_robustness_report(self) -> str:
        """
        Generate text report of robustness analysis
        
        Returns:
            Formatted report string
            
        Research Note:
        --------------
        This report provides a comprehensive summary suitable for thesis documentation.
        It includes:
        1. Overall best performing model (by average robustness)
        2. Hardest anomaly type to detect (by average performance drop)
        3. Model ranking by robustness
        4. Anomaly difficulty ranking
        5. Key insights for research discussion
        """
        report = []
        report.append("\n" + "="*80)
        report.append("📊 CPS MODEL ROBUSTNESS ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Evaluated: {len(self.trained_models)}")
        report.append(f"Anomaly Types Tested: {len(self.results)}")
        report.append("="*80)
        
        df_robustness = self.compute_robustness_scores()
        
        if len(df_robustness) == 0:
            report.append("\n⚠️  No robustness data available.")
            return "\n".join(report)
        
        # Best performing model (by average robustness score)
        model_robustness = df_robustness.groupby('model')['robustness_score_f1'].mean()
        best_model = model_robustness.idxmax()
        best_score = model_robustness.max()
        
        report.append(f"\n🏆 MOST ROBUST MODEL: {best_model}")
        report.append(f"   Average Robustness Score (F1): {best_score:.4f}")
        report.append(f"   Interpretation: {(1-best_score)*100:.1f}% average performance degradation")
        
        # Worst performing model
        worst_model = model_robustness.idxmin()
        worst_score = model_robustness.min()
        report.append(f"\n⚠️  LEAST ROBUST MODEL: {worst_model}")
        report.append(f"   Average Robustness Score (F1): {worst_score:.4f}")
        report.append(f"   Interpretation: {(1-worst_score)*100:.1f}% average performance degradation")
        
        # Hardest anomaly to detect (by average F1 drop)
        anomaly_drops = df_robustness.groupby('anomaly_type')['f1_drop'].mean()
        hardest_anomaly = anomaly_drops.idxmax()
        hardest_drop = anomaly_drops.max()
        
        report.append(f"\n⚠️  HARDEST ANOMALY TO DETECT: {hardest_anomaly}")
        report.append(f"   Average F1-Score Drop: {hardest_drop:.4f} ({hardest_drop*100:.2f}%)")
        
        # Easiest anomaly to detect
        easiest_anomaly = anomaly_drops.idxmin()
        easiest_drop = anomaly_drops.min()
        report.append(f"\n✅ EASIEST ANOMALY TO DETECT: {easiest_anomaly}")
        report.append(f"   Average F1-Score Drop: {easiest_drop:.4f} ({easiest_drop*100:.2f}%)")
        
        # Model ranking
        report.append(f"\n📈 MODEL ROBUSTNESS RANKING (Most to Least Robust):")
        report.append("-" * 60)
        model_ranking = model_robustness.sort_values(ascending=False)
        for i, (model, score) in enumerate(model_ranking.items(), 1):
            degradation = (1 - score) * 100
            report.append(f"   {i}. {model:20s}: {score:.4f} ({degradation:.1f}% degradation)")
        
        # Anomaly difficulty ranking
        report.append(f"\n🎯 ANOMALY DIFFICULTY RANKING (Most to Least Difficult):")
        report.append("-" * 60)
        anomaly_ranking = anomaly_drops.sort_values(ascending=False)
        for i, (anomaly, drop) in enumerate(anomaly_ranking.items(), 1):
            report.append(f"   {i}. {anomaly:20s}: {drop:.4f} F1-drop ({drop*100:.2f}%)")
        
        # Per-model, per-anomaly breakdown (top 3 models)
        report.append(f"\n📋 DETAILED BREAKDOWN (Top 3 Models):")
        report.append("-" * 60)
        top_models = model_ranking.head(3).index.tolist()
        for model in top_models:
            report.append(f"\n   {model}:")
            model_data = df_robustness[df_robustness['model'] == model]
            for _, row in model_data.iterrows():
                report.append(f"      {row['anomaly_type']:20s}: F1={row['f1_original']:.3f}→{row['f1_anomalous']:.3f} "
                             f"(drop={row['f1_drop']:.3f}, robust={row['robustness_score_f1']:.3f})")
        
        # Research insights
        report.append(f"\n💡 RESEARCH INSIGHTS:")
        report.append("-" * 60)
        
        # Insight 1: Best model for deployment
        report.append(f"   1. Recommended for Deployment: {best_model}")
        report.append(f"      Reason: Lowest average performance degradation across all anomaly types")
        
        # Insight 2: Most critical vulnerability
        report.append(f"   2. Most Critical Vulnerability: {hardest_anomaly}")
        report.append(f"      Reason: Causes highest average performance degradation across models")
        
        # Insight 3: Robustness variance
        robustness_std = df_robustness.groupby('model')['robustness_score_f1'].std().mean()
        report.append(f"   3. Robustness Variance Across Models: {robustness_std:.4f}")
        if robustness_std > 0.1:
            report.append(f"      Interpretation: HIGH variance - model selection is CRITICAL")
        else:
            report.append(f"      Interpretation: LOW variance - most models perform similarly")
        
        report.append("\n" + "="*80)
        report.append("END OF ROBUSTNESS ANALYSIS REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        logger.info(report_text)
        
        return report_text
    
    def get_model_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table suitable for thesis tables
        
        Returns:
            DataFrame with model comparison metrics
            
        Research Note:
        --------------
        This method generates a publication-ready comparison table that can be
        directly exported to LaTeX or included in thesis documents.
        """
        df_robustness = self.compute_robustness_scores()
        
        # Aggregate by model
        comparison = df_robustness.groupby('model').agg({
            'f1_original': 'mean',
            'f1_anomalous': 'mean',
            'f1_drop': 'mean',
            'robustness_score_f1': 'mean',
            'accuracy_original': 'mean',
            'accuracy_anomalous': 'mean',
            'inference_time_original': 'mean',
            'inference_time_anomalous': 'mean'
        }).round(4)
        
        # Rename columns for clarity
        comparison.columns = [
            'F1_Original_Mean', 'F1_Anomalous_Mean', 'F1_Drop_Mean',
            'Robustness_Score', 'Accuracy_Original_Mean', 'Accuracy_Anomalous_Mean',
            'Inference_Time_Original_s', 'Inference_Time_Anomalous_s'
        ]
        
        # Sort by robustness score
        comparison = comparison.sort_values('Robustness_Score', ascending=False)
        
        return comparison