"""
Adaptive Feedback Module for CPS Anomaly Detection Framework
Master's Thesis Innovation Layer - Dalian University of Technology
Author: BERETE MAMADY

This module extends the existing baseline framework with:
- Stage 1: Failure Case Analysis + Detector Optimization
- Stage 2: Difficulty-Aware Anomaly Generation + Retraining Rules
- Stage 3: Overall Framework Evaluation

All components integrate with existing:
- src/anomaly_generator.py
- src/robustness_evaluator.py
- src/models.py
"""

from .failure_case_analyzer import FailureCaseAnalyzer
from .detector_optimizer import DetectorOptimizer
from .difficulty_aware_generator import DifficultyAwareGenerator
from .retraining_manager import RetrainingManager
from .overall_framework_evaluator import OverallFrameworkEvaluator

__all__ = [
    'FailureCaseAnalyzer',
    'DetectorOptimizer',
    'DifficultyAwareGenerator',
    'RetrainingManager',
    'OverallFrameworkEvaluator'
]