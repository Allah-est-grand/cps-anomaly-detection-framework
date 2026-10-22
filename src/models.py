"""
Model Registry Module - Manages Multiple Anomaly Detection Models
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import joblib
from datetime import datetime

# Scikit-learn models
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Deep learning models
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("⚠️  TensorFlow not available. Deep learning models will be skipped.")

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Central registry for managing multiple anomaly detection models
    
    Research Application:
    ---------------------
    This class provides a unified interface for:
    1. Model initialization with consistent random seeds
    2. Training with optional validation
    3. Prediction with model-specific post-processing
    4. Feature importance extraction (for tree-based models)
    5. Model persistence (save/load)
    
    Supported Models:
    -----------------
    - RandomForest: Ensemble tree-based classifier
    - XGBoost: Gradient boosting classifier
    - LinearSVM: Linear support vector machine
    - LSTM: Long short-term memory neural network
    - Autoencoder: Reconstruction-based anomaly detector
    - IsolationForest: Tree-based anomaly detector
    - OneClassSVM: Kernel-based anomaly detector
    """
    
    def __init__(self):
        """
        Initialize model registry
        
        Note: Does NOT accept seed parameter (uses global seed from utils)
        """
        self.models = {}
        self.trained_models = {}
        self.model_configs = {}
        
        # Autoencoder threshold (computed from normal training data)
        # This is CRITICAL for scientific validity - prevents data leakage
        self.autoencoder_threshold = None
        
        logger.info("🔧 ModelRegistry initialized")
    
    def register_model(self, name: str, model):
        """
        Register a model in the registry
        
        Args:
            name: Model name (e.g., 'RandomForest', 'LSTM')
            model: Model instance
        """
        self.models[name] = model
        self.trained_models[name] = None
        logger.debug(f"📝 Registered model: {name}")
    
    def get_model(self, name: str):
        """
        Get a model from the registry
        
        Args:
            name: Model name
            
        Returns:
            Model instance or None
        """
        return self.models.get(name)
    
    def initialize_all_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize all supported models with appropriate configurations
        
        Args:
            X_train: Training features (for shape inference)
            y_train: Training labels (for class inference)
            
        Research Note:
        --------------
        Model hyperparameters are set based on CPS anomaly detection literature:
        - RandomForest: 100 trees, balanced class weights
        - XGBoost: Learning rate 0.1, max depth 6
        - LSTM: 2 layers, 64 units, dropout 0.2
        - Autoencoder: 3 layers, reconstruction loss
        """
        logger.info("🔧 Initializing all models...")
        
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        
        # 1. RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.register_model('RandomForest', rf_model)
        
        # 2. XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.register_model('XGBoost', xgb_model)
        
        # 3. LinearSVM
        svm_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=10000,
            random_state=42
        )
        self.register_model('LinearSVM', svm_model)
        
        # 4. IsolationForest
        if_model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.register_model('IsolationForest', if_model)
        
        # 5. OneClassSVM
        ocsvm_model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.register_model('OneClassSVM', ocsvm_model)
        
        # 6. LSTM (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            lstm_model = self._create_lstm_model(n_features)
            self.register_model('LSTM', lstm_model)
        else:
            logger.warning("⚠️  LSTM skipped (TensorFlow not available)")
        
        # 7. Autoencoder (if TensorFlow available)
        if TENSORFLOW_AVAILABLE:
            ae_model = self._create_autoencoder_model(n_features)
            self.register_model('Autoencoder', ae_model)
        else:
            logger.warning("⚠️  Autoencoder skipped (TensorFlow not available)")
        
        logger.info(f"✅ Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def _create_lstm_model(self, n_features: int, time_steps: int = 10) -> keras.Model:
        """
        Create LSTM model for time-series anomaly detection
        
        Args:
            n_features: Number of input features
            time_steps: Number of time steps
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(time_steps, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.debug(f"📊 LSTM model created: {model.count_params()} parameters")
        
        return model
    
    def _create_autoencoder_model(self, n_features: int) -> keras.Model:
        """
        Create Autoencoder model for reconstruction-based anomaly detection
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
            
        Research Note:
        --------------
        Autoencoder architecture:
        - Encoder: Compresses input to lower-dimensional representation
        - Decoder: Reconstructs input from compressed representation
        - Anomaly detection: High reconstruction error = anomaly
        """
        # Encoder
        encoder_input = layers.Input(shape=(n_features,))
        x = layers.Dense(64, activation='relu')(encoder_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(16, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        decoded = layers.Dense(n_features, activation='sigmoid')(x)
        
        # Autoencoder
        autoencoder = models.Model(encoder_input, decoded)
        
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        logger.debug(f"📊 Autoencoder model created: {autoencoder.count_params()} parameters")
        
        return autoencoder
    
    def train_model(self, name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train a specific model
        
        Args:
            name: Model name
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history (for deep learning models)
        """
        logger.info(f"🔧 Training {name}...")
        
        model = self.models.get(name)
        if model is None:
            logger.error(f"❌ Model {name} not found in registry")
            return None
        
        try:
            if name in ['LSTM', 'Autoencoder']:
                # Deep learning models require special handling
                if not TENSORFLOW_AVAILABLE:
                    logger.error(f"❌ Cannot train {name}: TensorFlow not available")
                    return None
                
                if name == 'LSTM':
                    # Prepare sequences for LSTM
                    time_steps = 10
                    X_train_seq = self._create_sequences(X_train, time_steps)
                    y_train_seq = y_train[time_steps-1:]
                    
                    # Training callbacks
                    early_stop = callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    # Train
                    history = model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=64,
                        validation_split=0.1,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                elif name == 'Autoencoder':
                    # Train autoencoder on ALL training data
                    # (threshold will be computed separately on normal samples only)
                    history = model.fit(
                        X_train, X_train,
                        epochs=100,
                        batch_size=256,
                        validation_split=0.1,
                        verbose=0
                    )
                
                self.trained_models[name] = model
                logger.info(f"✅ {name} trained successfully")
                return history
                
            else:
                # Standard sklearn models
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                logger.info(f"✅ {name} trained successfully")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to train {name}: {e}")
            return None
    
    def _create_sequences(self, X: np.ndarray, time_steps: int) -> np.ndarray:
        """
        Create sequences for LSTM input
        
        Args:
            X: Input features
            time_steps: Number of time steps per sequence
            
        Returns:
            Sequences of shape (n_sequences, time_steps, n_features)
        """
        sequences = []
        for i in range(len(X) - time_steps + 1):
            sequences.append(X[i:i + time_steps])
        return np.array(sequences)
    
    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a trained model
        
        Args:
            name: Model name
            X: Input features
            
        Returns:
            Predicted labels (0=normal, 1=anomaly)
            
        Research Note:
        --------------
        Model-specific prediction logic:
        - Classifiers: Direct predict()
        - IsolationForest/OneClassSVM: Convert -1/1 to 0/1
        - Autoencoder: Reconstruction error threshold
        - LSTM: Probability threshold (0.5)
        
        AUTOENCODER THRESHOLD:
        ----------------------
        For scientific validity, the Autoencoder threshold should be:
        1. Computed from NORMAL training samples only (in main.py)
        2. Stored in self.autoencoder_threshold
        3. Used during prediction (NOT recomputed from X)
        
        If self.autoencoder_threshold is None, falls back to computing
        from X with a warning (not recommended for evaluation).
        """
        model = self.trained_models.get(name)
        if model is None:
            logger.error(f"❌ Model {name} not trained or not found")
            return None
        
        try:
            if name == 'Autoencoder':
                # SCIENTIFICALLY CORRECT AUTOENCODER PREDICTION
                # Use pre-computed threshold from normal training data
                
                # Get reconstruction error
                reconstructions = model.predict(X, verbose=0)
                mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                
                # Check if we have a pre-computed threshold from training
                if self.autoencoder_threshold is not None:
                    # Use the training-derived threshold (SCIENTIFICALLY VALID)
                    threshold = self.autoencoder_threshold
                    logger.debug(f"📊 Autoencoder: Using training-derived threshold={threshold:.6f}")
                else:
                    # Fallback: compute from current data (NOT RECOMMENDED - data leakage risk!)
                    threshold = np.percentile(mse, 95)
                    logger.warning(f"⚠️  Autoencoder: No training threshold available. Computing from evaluation data (data leakage risk!)")
                    logger.warning(f"⚠️  Threshold={threshold:.6f}")
                
                # Binary predictions using threshold
                y_pred = (mse > threshold).astype(int)
                
                logger.debug(f"📊 Autoencoder: MSE range=[{mse.min():.6f}, {mse.max():.6f}], Threshold={threshold:.6f}")
                
                return y_pred
                
            elif name == 'LSTM':
                if not TENSORFLOW_AVAILABLE:
                    logger.error(f"❌ Cannot predict with LSTM: TensorFlow not available")
                    return None
                
                # Prepare sequences
                time_steps = 10
                X_seq = self._create_sequences(X, time_steps)
                
                # Get predictions
                y_proba = model.predict(X_seq, verbose=0).flatten()
                
                # Pad to match original length
                pad_width = time_steps - 1
                y_proba_padded = np.concatenate([np.zeros(pad_width), y_proba])
                y_proba_padded = y_proba_padded[:len(X)]
                
                # Binary predictions
                y_pred = (y_proba_padded > 0.5).astype(int)
                
                return y_pred
                
            elif name in ['IsolationForest', 'OneClassSVM']:
                # Unsupervised models: convert -1/1 to 0/1
                raw_pred = model.predict(X)
                y_pred = np.where(raw_pred == -1, 1, 0)
                return y_pred
                
            else:
                # Standard classifiers
                y_pred = model.predict(X)
                return y_pred
                
        except Exception as e:
            logger.error(f"❌ Prediction failed for {name}: {e}")
            return None
    
    def get_feature_importance(self, name: str, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Get feature importance for tree-based models
        
        Args:
            name: Model name
            feature_names: List of feature names
            
        Returns:
            Dictionary {feature_name: importance_score} or None
        """
        model = self.trained_models.get(name)
        if model is None:
            logger.warning(f"⚠️  Model {name} not trained")
            return None
        
        try:
            if name in ['RandomForest', 'XGBoost']:
                importances = model.feature_importances_
                importance_dict = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
                logger.debug(f"📊 {name}: Feature importance computed for {len(feature_names)} features")
                return importance_dict
            else:
                logger.debug(f"⚠️  {name}: Feature importance not available")
                return None
        except Exception as e:
            logger.warning(f"⚠️  Failed to get feature importance for {name}: {e}")
            return None
    
    def save_model(self, name: str, filepath: str):
        """
        Save a trained model to disk
        
        Args:
            name: Model name
            filepath: Path to save model
        """
        model = self.trained_models.get(name)
        if model is None:
            logger.error(f"❌ Model {name} not trained")
            return
        
        try:
            if name in ['LSTM', 'Autoencoder']:
                if TENSORFLOW_AVAILABLE:
                    model.save(filepath)
                    logger.info(f"💾 Saved {name} to: {filepath}")
                else:
                    logger.error(f"❌ Cannot save {name}: TensorFlow not available")
            else:
                joblib.dump(model, filepath)
                logger.info(f"💾 Saved {name} to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save {name}: {e}")
    
    def load_model(self, name: str, filepath: str):
        """
        Load a trained model from disk
        
        Args:
            name: Model name
            filepath: Path to load model from
        """
        try:
            if name in ['LSTM', 'Autoencoder']:
                if TENSORFLOW_AVAILABLE:
                    from tensorflow import keras
                    model = keras.models.load_model(filepath)
                    self.trained_models[name] = model
                    logger.info(f"📥 Loaded {name} from: {filepath}")
                else:
                    logger.error(f"❌ Cannot load {name}: TensorFlow not available")
            else:
                model = joblib.load(filepath)
                self.trained_models[name] = model
                logger.info(f"📥 Loaded {name} from: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """
        Perform cross-validation on all trained models
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results per model
        """
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"🔍 Performing {n_splits}-fold cross-validation...")
        
        cv_results = {}
        
        for name, model in self.trained_models.items():
            if model is None:
                continue
            
            try:
                # Skip deep learning models for CV (too slow)
                if name in ['LSTM', 'Autoencoder']:
                    logger.debug(f"⚠️  Skipping CV for {name} (deep learning)")
                    continue
                
                # Compute CV scores
                cv_accuracy = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
                cv_f1 = cross_val_score(model, X, y, cv=n_splits, scoring='f1')
                
                cv_results[name] = {
                    'cv_accuracy': float(cv_accuracy.mean()),
                    'cv_accuracy_std': float(cv_accuracy.std()),
                    'cv_f1': float(cv_f1.mean()),
                    'cv_f1_std': float(cv_f1.std())
                }
                
                logger.info(f"  {name}: CV Accuracy={cv_accuracy.mean()*100:.2f}%, CV F1={cv_f1.mean()*100:.2f}%")
                
            except Exception as e:
                logger.warning(f"⚠️  CV failed for {name}: {e}")
                cv_results[name] = {'error': str(e)}
        
        return cv_results
    
    def evaluate_robustness(self, X: np.ndarray, y: np.ndarray, 
                           baseline_results: Dict) -> Dict:
        """
        Quick robustness check (placeholder - full evaluation in run_robustness_evaluation.py)
        
        Args:
            X: Features
            y: Labels
            baseline_results: Baseline performance results
            
        Returns:
            Robustness metrics dictionary
        """
        logger.info("🔍 Running quick robustness check...")
        
        # Placeholder - full robustness evaluation is in run_robustness_evaluation.py
        return {'note': 'Full robustness evaluation in run_robustness_evaluation.py'}