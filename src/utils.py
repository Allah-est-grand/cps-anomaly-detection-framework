"""
Utility Functions - Logging, Reproducibility, and File Management
"""
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime
import json
import pickle

def setup_logging(log_dir: Path, experiment_name: str = 'experiment') -> logging.Logger:
    """
    Setup comprehensive logging for experiment
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('CPS_Framework')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler avec encodage UTF-8
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{experiment_name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')  # ✅ LIGNE CORRIGÉE
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info("="*80)
    logger.info(f"🚀 Experiment Started: {experiment_name}")
    logger.info(f"📅 Timestamp: {timestamp}")
    logger.info(f"📁 Log file: {log_file}")
    logger.info("="*80)
    
    return logger

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"✅ Random seeds set to {seed} for reproducibility")

def save_experiment_metadata(metadata: Dict, output_dir: Path):
    """
    Save experiment metadata for reproducibility
    
    Args:
        metadata: Dictionary with experiment metadata
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_dir / f'metadata_{timestamp}.json'
    
    with open(filepath, 'w', encoding='utf-8') as f:  # ✅ Ajout encoding ici aussi
        json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
    
    logging.info(f"💾 Metadata saved to: {filepath}")

def save_model(model, model_name: str, output_dir: Path):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f'{model_name}.pkl'
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"💾 Model {model_name} saved to: {filepath}")

def load_model(model_name: str, input_dir: Path):
    """
    Load trained model from disk
    
    Args:
        model_name: Name of the model
        input_dir: Input directory
        
    Returns:
        Loaded model
    """
    filepath = input_dir / f'{model_name}.pkl'
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logging.info(f"📥 Model {model_name} loaded from: {filepath}")
    return model

def create_experiment_report(results: Dict, output_dir: Path) -> str:
    """
    Create comprehensive experiment report
    
    Args:
        results: Dictionary with all experiment results
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'experiment_report_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:  # ✅ Ajout encoding ici aussi
        f.write("="*80 + "\n")
        f.write("CPS ANOMALY DETECTION - EXPERIMENT REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'time' in metric_name.lower():
                        f.write(f"  {metric_name}: {value/60:.2f} min\n")
                    elif 'score' in metric_name.lower() or 'accuracy' in metric_name.lower():
                        f.write(f"  {metric_name}: {value*100:.2f}%\n")
                    else:
                        f.write(f"  {metric_name}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logging.info(f"💾 Experiment report saved to: {report_path}")
    return str(report_path)