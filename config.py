"""
Configuration Management for CPS Anomaly Detection Framework
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json

@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_path: str = 'merged.csv'
    timestamp_col: str = 'Timestamp'
    label_col: str = 'Normal/Attack'
    test_size: float = 0.3
    random_state: int = 42
    use_imputation: bool = True
    imputation_strategy: str = 'mean'
    scale_features: bool = True
    feature_selection: bool = False
    max_features: int = None

@dataclass
class ModelConfig:
    """Model configuration"""
    models: List[str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                'IsolationForest',
                'RandomForest',
                'XGBoost',
                'LinearSVM',
                'OneClassSVM',
                'Autoencoder',
                'LSTM'
            ]

@dataclass
class ValidationConfig:
    """Cross-validation configuration"""
    k_folds: int = 5
    use_stratified: bool = True
    use_time_series: bool = False
    n_repeats: int = 3

@dataclass
class SimulationConfig:
    """CPS anomaly simulation configuration"""
    enable_simulation: bool = True
    noise_levels: List[float] = None
    anomaly_frequencies: List[float] = None
    drift_magnitude: float = 0.1
    delay_samples: int = 5
    corruption_rate: float = 0.05
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        if self.anomaly_frequencies is None:
            self.anomaly_frequencies = [0.01, 0.03, 0.05, 0.1, 0.15]

@dataclass
class OptimizationConfig:
    """Hyperparameter optimization configuration"""
    enable_optimization: bool = True
    method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    cv_folds: int = 3
    n_iterations: int = 50  # for Bayesian optimization

@dataclass
class OutputConfig:
    """Output paths configuration"""
    base_dir: Path = Path('results')
    models_dir: Path = None
    figures_dir: Path = None
    logs_dir: Path = None
    metrics_dir: Path = None
    
    def __post_init__(self):
        self.models_dir = self.base_dir / 'models'
        self.figures_dir = self.base_dir / 'figures'
        self.logs_dir = self.base_dir / 'logs'
        self.metrics_dir = self.base_dir / 'metrics'
        
        # Create directories
        for dir_path in [self.models_dir, self.figures_dir, 
                         self.logs_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class FrameworkConfig:
    """Main framework configuration"""
    data: DataConfig = None
    model: ModelConfig = None
    validation: ValidationConfig = None
    simulation: SimulationConfig = None
    optimization: OptimizationConfig = None
    output: OutputConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.output is None:
            self.output = OutputConfig()
    
    def save(self, path: str):
        """Save configuration to JSON"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'validation': self.validation.__dict__,
            'simulation': self.simulation.__dict__,
            'optimization': self.optimization.__dict__,
            'output': {
                'base_dir': str(self.output.base_dir),
                'models_dir': str(self.output.models_dir),
                'figures_dir': str(self.output.figures_dir),
                'logs_dir': str(self.output.logs_dir),
                'metrics_dir': str(self.output.metrics_dir)
            }
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Reconstruct config object
        config = cls()
        # TODO: Add reconstruction logic
        return config

# Default configuration
DEFAULT_CONFIG = FrameworkConfig()