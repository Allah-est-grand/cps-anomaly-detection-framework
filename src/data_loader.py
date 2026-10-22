"""
Data Loading and Preprocessing Module
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

IMPORTANT: Scientific Preprocessing Workflow (No Data Leakage)
----------------------------------------------------------------
This module implements a SCIENTIFICALLY CORRECT preprocessing pipeline:
1. Load raw dataset (no preprocessing)
2. Split into train/test FIRST (on raw data)
3. Fit imputer and scaler on TRAIN ONLY
4. Transform both train and test using fitted preprocessors
5. Save preprocessors fitted on train only

This prevents DATA LEAKAGE where test data influences training preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import joblib
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class CPSDataLoader:
    """
    Data loading and preprocessing for CPS anomaly detection
    
    Scientific Workflow:
    --------------------
    1. load_dataset() - Load raw data (no preprocessing)
    2. extract_features_labels() - Identify columns (no fitting)
    3. split_train_test() - Split RAW data (before any fitting)
    4. fit_preprocessors() - Fit imputer/scaler on TRAIN ONLY
    5. transform_data() - Transform both train and test using fitted preprocessors
    6. save_preprocessors() - Save preprocessors fitted on train only
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize data loader
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Preprocessing objects (fitted on TRAIN ONLY)
        self.imputer = None
        self.scaler = None
        
        # Dataset metadata
        self.feature_columns = []
        self.label_column = None
        self.timestamp_column = None
        
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load dataset from CSV file (RAW, no preprocessing)
        
        Args:
            filepath: Path to dataset CSV file
            
        Returns:
            Loaded DataFrame (raw, unprocessed)
        """
        logger.info(f"📥 Loading dataset from: {filepath}")
        
        df = pd.read_csv(filepath)
        
        logger.info(f"✅ Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
        
        return df
    
    def extract_features_labels(self, 
                               df: pd.DataFrame,
                               label_column: str = 'Normal/Attack',
                               timestamp_column: Optional[str] = ' Timestamp') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from dataset (NO preprocessing, NO fitting)
        
        Args:
            df: Input DataFrame (raw)
            label_column: Name of label column (default: 'Normal/Attack' for SWaT)
            timestamp_column: Name of timestamp column (default: ' Timestamp' for SWaT)
            
        Returns:
            Tuple of (X, y) raw features and encoded labels
            
        Note: This method does NOT fit any preprocessors.
        """
        logger.info("🔧 Extracting features and labels...")
        
        # Store column information
        self.label_column = label_column
        self.timestamp_column = timestamp_column
        
        # Identify feature columns (exclude label and timestamp)
        exclude_cols = [label_column]
        if timestamp_column:
            exclude_cols.append(timestamp_column)
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"📊 Feature columns: {len(self.feature_columns)}")
        
        # Extract features (RAW, no scaling/imputation yet)
        X = df[self.feature_columns].values
        
        # Extract and encode labels
        if label_column in df.columns:
            logger.info(f"📊 Found label column: '{label_column}'")
            y = self._encode_labels(df[label_column])
            logger.info(f"📊 Encoded labels: Normal=0, Attack=1")
            logger.info(f"📊 Class distribution: Normal={len(y)-y.sum()}, Attack={y.sum()}")
        else:
            logger.error(f"❌ Label column '{label_column}' not found in dataset!")
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        logger.info(f"📊 Features shape: {X.shape}")
        logger.info(f"📊 Labels shape: {y.shape}")
        
        return X, y
    
    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Encode SWaT-style labels to binary format
        
        Args:
            labels: Series with 'Normal'/'Attack' values
            
        Returns:
            Numpy array with 0 (Normal) and 1 (Attack)
        """
        # Convert to string for consistent comparison
        labels_str = labels.astype(str).str.lower().str.strip()
        
        # Create encoded array
        y = np.zeros(len(labels), dtype=int)
        
        # Mark attacks as 1
        attack_indicators = ['attack', '1', 'true', 'anomaly', 'intrusion']
        for indicator in attack_indicators:
            y[labels_str == indicator] = 1
        
        logger.info(f"📊 Label encoding: {len(y)-y.sum()} Normal (0), {y.sum()} Attack (1)")
        
        return y
    
    def split_train_test(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets (BEFORE fitting preprocessors)
        
        Args:
            X: Raw features (unprocessed)
            y: Encoded labels
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Research Note:
        --------------
        CRITICAL: This split happens on RAW data BEFORE any preprocessing.
        This prevents data leakage where test data influences training preprocessing.
        """
        logger.info(f"🔀 Splitting data: train={(1-test_size)*100:.0f}%, test={test_size*100:.0f}%")
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"✅ Train set: {len(X_train)} samples")
        logger.info(f"✅ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def fit_preprocessors(self, X_train: np.ndarray):
        """
        Fit imputer and scaler on TRAINING DATA ONLY
        
        Args:
            X_train: Training features (RAW, unprocessed)
            
        Returns:
            X_train_imputed_scaled: Training data after imputation and scaling
            
        Research Note:
        --------------
        CRITICAL: Preprocessors are fit on TRAIN ONLY to prevent data leakage.
        Test data must NEVER influence the fitting of imputer or scaler.
        
        This mirrors real-world deployment where:
        1. Preprocessing is fitted on historical training data
        2. New data (test/production) is transformed using fitted preprocessors
        3. Test data statistics never leak into training
        """
        logger.info("🔧 Fitting preprocessors on TRAINING DATA ONLY...")
        
        # Check for missing values in training data
        missing_count = np.isnan(X_train).sum()
        if missing_count > 0:
            logger.info(f"⚠️  Found {missing_count} missing values in training data")
        
        # Fit imputer on TRAIN ONLY
        self.imputer = SimpleImputer(strategy='mean')
        X_train_imputed = self.imputer.fit_transform(X_train)
        logger.info(f"✅ Fitted imputer on training data (strategy=mean)")
        
        # Fit scaler on TRAIN ONLY
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        logger.info(f"✅ Fitted StandardScaler on training data (mean=0, std=1)")
        
        logger.info(f"✅ Preprocessors fitted: X_train shape={X_train_scaled.shape}")
        
        return X_train_scaled
    
    def transform_data(self, X: np.ndarray, is_training: bool = False) -> np.ndarray:
        """
        Transform data using FITTED preprocessors
        
        Args:
            X: Raw features to transform
            is_training: If True, this is training data (should use fit_preprocessors instead)
            
        Returns:
            X_transformed: Preprocessed data
            
        Research Note:
        --------------
        This method uses TRANSFORM only (NOT fit_transform) to ensure:
        1. No data leakage from test/evaluation data
        2. Consistent preprocessing with training phase
        3. Valid performance comparison
        
        CRITICAL: Call fit_preprocessors(X_train) BEFORE using this method.
        """
        if self.imputer is None or self.scaler is None:
            logger.error("❌ Preprocessors not fitted! Call fit_preprocessors(X_train) first.")
            raise ValueError("Preprocessors not fitted. Must fit on training data first.")
        
        logger.info("🔧 Transforming data using fitted preprocessors...")
        
        # Apply imputer (transform only, DO NOT REFIT)
        X_imputed = self.imputer.transform(X)
        
        # Apply scaler (transform only, DO NOT REFIT)
        X_scaled = self.scaler.transform(X_imputed)
        
        logger.info(f"✅ Transformation complete: X shape={X_scaled.shape}")
        
        return X_scaled
    
    def save_preprocessors(self, output_dir: str):
        """
        Save fitted preprocessing objects to disk
        
        Args:
            output_dir: Directory to save preprocessors
            
        Returns:
            Dictionary with saved file paths
            
        Research Note:
        --------------
        These saved preprocessors are CRITICAL for robustness evaluation.
        They were fitted on TRAINING DATA ONLY to prevent data leakage.
        """
        logger.info(f"💾 Saving preprocessing objects to: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save imputer (fitted on train only)
        if self.imputer is not None:
            imputer_path = output_dir / 'preprocessing_imputer.pkl'
            joblib.dump(self.imputer, imputer_path)
            saved_files['imputer'] = str(imputer_path)
            logger.info(f"✅ Saved imputer: {imputer_path}")
        else:
            logger.error("❌ No imputer to save (not fitted yet)")
        
        # Save scaler (fitted on train only)
        if self.scaler is not None:
            scaler_path = output_dir / 'preprocessing_scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            saved_files['scaler'] = str(scaler_path)
            logger.info(f"✅ Saved scaler: {scaler_path}")
        else:
            logger.error("❌ No scaler to save (not fitted yet)")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'label_column': self.label_column,
            'timestamp_column': self.timestamp_column,
            'fitted_on': 'training_data_only'
        }
        
        import json
        metadata_path = output_dir / 'preprocessing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_path)
        logger.info(f"✅ Saved metadata: {metadata_path}")
        
        return saved_files
    
    def load_preprocessors(self, models_dir: str):
        """
        Load fitted preprocessing objects from disk
        
        Args:
            models_dir: Directory containing saved preprocessors
            
        Returns:
            Dictionary with loaded file paths
        """
        logger.info(f"📥 Loading preprocessing objects from: {models_dir}")
        
        models_dir = Path(models_dir)
        loaded_files = {}
        
        # Load imputer
        imputer_path = models_dir / 'preprocessing_imputer.pkl'
        if imputer_path.exists():
            self.imputer = joblib.load(imputer_path)
            loaded_files['imputer'] = str(imputer_path)
            logger.info(f"✅ Loaded imputer: {imputer_path}")
        
        # Load scaler
        scaler_path = models_dir / 'preprocessing_scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            loaded_files['scaler'] = str(scaler_path)
            logger.info(f"✅ Loaded scaler: {scaler_path}")
        
        # Load metadata
        metadata_path = models_dir / 'preprocessing_metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.feature_columns = metadata.get('feature_columns', [])
            self.label_column = metadata.get('label_column', None)
            self.timestamp_column = metadata.get('timestamp_column', None)
            loaded_files['metadata'] = str(metadata_path)
            logger.info(f"✅ Loaded metadata: {metadata_path}")
        
        return loaded_files