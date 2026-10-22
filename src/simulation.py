"""
CPS Anomaly Simulation Module
Master's Thesis Research - Core Contribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of CPS anomalies to simulate"""
    SENSOR_NOISE = "sensor_noise"
    SENSOR_DRIFT = "sensor_drift"
    COMMUNICATION_DELAY = "communication_delay"
    MISSING_SIGNAL = "missing_signal"
    CYBER_ATTACK = "cyber_attack"
    ACTUATOR_FAILURE = "actuator_failure"

class CPSSimulator:
    """
    Simulation module for generating and injecting CPS anomalies
    Core component of the simulation-based framework
    """
    
    def __init__(self, config):
        """
        Initialize simulator with configuration
        
        Args:
            config: SimulationConfig object
        """
        self.config = config
        self.anomaly_log = []
        
    def inject_sensor_noise(self, X: np.ndarray, 
                           noise_level: float,
                           feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Inject Gaussian noise into sensor readings
        
        Args:
            X: Feature matrix
            noise_level: Standard deviation of noise (as fraction of feature std)
            feature_indices: Which features to perturb (None = all)
            
        Returns:
            Noisy feature matrix
        """
        logger.info(f"🔊 Injecting sensor noise (level={noise_level:.3f})...")
        
        X_noisy = X.copy()
        
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        for idx in feature_indices:
            feature_std = np.std(X[:, idx])
            noise = np.random.normal(0, noise_level * feature_std, len(X))
            X_noisy[:, idx] += noise
        
        self.anomaly_log.append({
            'type': AnomalyType.SENSOR_NOISE.value,
            'noise_level': noise_level,
            'features_affected': len(feature_indices)
        })
        
        return X_noisy
    
    def inject_sensor_drift(self, X: np.ndarray,
                           drift_magnitude: float,
                           feature_indices: Optional[List[int]] = None,
                           drift_type: str = 'linear') -> np.ndarray:
        """
        Inject sensor drift (systematic deviation over time)
        
        Args:
            X: Feature matrix (assumed time-ordered)
            drift_magnitude: Maximum drift as fraction of feature range
            feature_indices: Which features to drift
            drift_type: 'linear', 'exponential', or 'random_walk'
            
        Returns:
            Feature matrix with drift
        """
        logger.info(f"📈 Injecting sensor drift (magnitude={drift_magnitude:.3f}, type={drift_type})...")
        
        X_drifted = X.copy()
        n_samples = len(X)
        
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        for idx in feature_indices:
            feature_range = np.max(X[:, idx]) - np.min(X[:, idx])
            
            if drift_type == 'linear':
                drift = np.linspace(0, drift_magnitude * feature_range, n_samples)
            elif drift_type == 'exponential':
                drift = (np.exp(np.linspace(0, 3, n_samples)) - 1) / (np.exp(3) - 1)
                drift *= drift_magnitude * feature_range
            elif drift_type == 'random_walk':
                drift = np.cumsum(np.random.normal(0, 0.01, n_samples))
                drift *= drift_magnitude * feature_range / np.max(np.abs(drift))
            else:
                raise ValueError(f"Unknown drift_type: {drift_type}")
            
            X_drifted[:, idx] += drift
        
        self.anomaly_log.append({
            'type': AnomalyType.SENSOR_DRIFT.value,
            'magnitude': drift_magnitude,
            'drift_type': drift_type,
            'features_affected': len(feature_indices)
        })
        
        return X_drifted
    
    def inject_communication_delay(self, X: np.ndarray,
                                  delay_samples: int,
                                  feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Simulate communication delays in sensor data
        
        Args:
            X: Feature matrix
            delay_samples: Number of samples to delay
            feature_indices: Which features to delay
            
        Returns:
            Feature matrix with delays
        """
        logger.info(f"⏱️ Injecting communication delay ({delay_samples} samples)...")
        
        X_delayed = X.copy()
        
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        for idx in feature_indices:
            X_delayed[delay_samples:, idx] = X[:-delay_samples, idx]
            X_delayed[:delay_samples, idx] = np.nan  # Mark delayed values
        
        # Fill NaN with mean
        col_means = np.nanmean(X_delayed, axis=0)
        inds = np.where(np.isnan(X_delayed))
        X_delayed[inds] = np.take(col_means, inds[1])
        
        self.anomaly_log.append({
            'type': AnomalyType.COMMUNICATION_DELAY.value,
            'delay_samples': delay_samples,
            'features_affected': len(feature_indices)
        })
        
        return X_delayed
    
    def inject_missing_signals(self, X: np.ndarray,
                              corruption_rate: float,
                              feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Simulate missing sensor signals (packet loss)
        
        Args:
            X: Feature matrix
            corruption_rate: Fraction of values to corrupt (0-1)
            feature_indices: Which features to corrupt
            
        Returns:
            Feature matrix with missing values
        """
        logger.info(f"📡 Injecting missing signals (rate={corruption_rate:.2%})...")
        
        X_corrupted = X.copy()
        
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        
        for idx in feature_indices:
            n_missing = int(len(X) * corruption_rate)
            missing_indices = np.random.choice(len(X), n_missing, replace=False)
            X_corrupted[missing_indices, idx] = np.nan
        
        # Impute missing values (simulating real-world handling)
        col_means = np.nanmean(X_corrupted, axis=0)
        inds = np.where(np.isnan(X_corrupted))
        X_corrupted[inds] = np.take(col_means, inds[1])
        
        self.anomaly_log.append({
            'type': AnomalyType.MISSING_SIGNAL.value,
            'corruption_rate': corruption_rate,
            'features_affected': len(feature_indices)
        })
        
        return X_corrupted
    
    def inject_cyber_attack(self, X: np.ndarray,
                           y: np.ndarray,
                           attack_type: str = 'scaling',
                           intensity: float = 2.0,
                           feature_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate cyber attack patterns on sensor data
        
        Args:
            X: Feature matrix
            y: Labels
            attack_type: 'scaling', 'min_max', 'constant', or 'random'
            intensity: Attack intensity multiplier
            feature_indices: Which features to attack
            
        Returns:
            Tuple of (attacked features, updated labels)
        """
        logger.info(f"🔓 Injecting cyber attack (type={attack_type}, intensity={intensity})...")
        
        X_attacked = X.copy()
        y_attacked = y.copy()
        
        if feature_indices is None:
            # Attack random 20% of features
            n_attack_features = max(1, int(X.shape[1] * 0.2))
            feature_indices = np.random.choice(X.shape[1], n_attack_features, replace=False)
        
        # Select attack samples (normally labeled as normal)
        normal_indices = np.where(y == 0)[0]
        n_attack_samples = max(1, int(len(normal_indices) * 0.1))
        attack_sample_indices = np.random.choice(normal_indices, n_attack_samples, replace=False)
        
        for idx in feature_indices:
            feature_mean = np.mean(X[:, idx])
            feature_std = np.std(X[:, idx])
            
            if attack_type == 'scaling':
                X_attacked[attack_sample_indices, idx] *= intensity
            elif attack_type == 'min_max':
                X_attacked[attack_sample_indices, idx] = np.random.choice(
                    [np.min(X[:, idx]), np.max(X[:, idx])], 
                    len(attack_sample_indices)
                )
            elif attack_type == 'constant':
                X_attacked[attack_sample_indices, idx] = feature_mean + intensity * feature_std
            elif attack_type == 'random':
                X_attacked[attack_sample_indices, idx] = np.random.normal(
                    feature_mean, intensity * feature_std, len(attack_sample_indices)
                )
        
        # Mark attacked samples as anomalies
        y_attacked[attack_sample_indices] = 1
        
        self.anomaly_log.append({
            'type': AnomalyType.CYBER_ATTACK.value,
            'attack_type': attack_type,
            'intensity': intensity,
            'samples_attacked': len(attack_sample_indices),
            'features_affected': len(feature_indices)
        })
        
        return X_attacked, y_attacked
    
    def run_robustness_test(self, X: np.ndarray, y: np.ndarray,
                           model, scenario_name: str) -> Dict:
        """
        Run comprehensive robustness test across multiple anomaly scenarios
        
        Args:
            X: Test features
            y: Test labels
            model: Trained model
            scenario_name: Name for this test scenario
            
        Returns:
            Dictionary with robustness metrics
        """
        logger.info(f"\n🧪 Running robustness test: {scenario_name}")
        logger.info("="*60)
        
        results = {
            'scenario': scenario_name,
            'baseline_accuracy': None,
            'noise_tests': [],
            'drift_tests': [],
            'delay_tests': [],
            'corruption_tests': [],
            'attack_tests': []
        }
        
        # Baseline (no anomalies)
        y_pred = model.predict(X)
        baseline_acc = np.mean(y_pred == y)
        results['baseline_accuracy'] = baseline_acc
        logger.info(f"📊 Baseline Accuracy: {baseline_acc*100:.2f}%")
        
        # Test noise levels
        logger.info("\n🔊 Testing noise robustness...")
        for noise_level in self.config.noise_levels:
            X_noisy = self.inject_sensor_noise(X, noise_level)
            y_pred = model.predict(X_noisy)
            acc = np.mean(y_pred == y)
            results['noise_tests'].append({
                'noise_level': noise_level,
                'accuracy': acc,
                'degradation': baseline_acc - acc
            })
            logger.info(f"  Noise {noise_level:.2f}: {acc*100:.2f}% (Δ{-100*(baseline_acc-acc):.2f}%)")
        
        # Test drift
        logger.info("\n📈 Testing drift robustness...")
        for drift_mag in [0.05, 0.1, 0.2, 0.3]:
            X_drifted = self.inject_sensor_drift(X, drift_mag)
            y_pred = model.predict(X_drifted)
            acc = np.mean(y_pred == y)
            results['drift_tests'].append({
                'drift_magnitude': drift_mag,
                'accuracy': acc,
                'degradation': baseline_acc - acc
            })
            logger.info(f"  Drift {drift_mag:.2f}: {acc*100:.2f}% (Δ{-100*(baseline_acc-acc):.2f}%)")
        
        # Test delays
        logger.info("\n⏱️ Testing delay robustness...")
        for delay in [1, 5, 10, 20]:
            X_delayed = self.inject_communication_delay(X, delay)
            y_pred = model.predict(X_delayed)
            acc = np.mean(y_pred == y)
            results['delay_tests'].append({
                'delay_samples': delay,
                'accuracy': acc,
                'degradation': baseline_acc - acc
            })
            logger.info(f"  Delay {delay}: {acc*100:.2f}% (Δ{-100*(baseline_acc-acc):.2f}%)")
        
        # Test corruption
        logger.info("\n📡 Testing corruption robustness...")
        for corr_rate in [0.01, 0.05, 0.1, 0.2]:
            X_corrupted = self.inject_missing_signals(X, corr_rate)
            y_pred = model.predict(X_corrupted)
            acc = np.mean(y_pred == y)
            results['corruption_tests'].append({
                'corruption_rate': corr_rate,
                'accuracy': acc,
                'degradation': baseline_acc - acc
            })
            logger.info(f"  Corruption {corr_rate:.2f}: {acc*100:.2f}% (Δ{-100*(baseline_acc-acc):.2f}%)")
        
        logger.info("="*60)
        
        return results
    
    def clear_log(self):
        """Clear anomaly injection log"""
        self.anomaly_log = []
    
    def get_log(self) -> List[Dict]:
        """Get anomaly injection log"""
        return self.anomaly_log.copy()
    
    def save_log(self, filepath: str):
        """Save anomaly log to JSON"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.anomaly_log, f, indent=2)
        logger.info(f"💾 Anomaly log saved to: {filepath}")