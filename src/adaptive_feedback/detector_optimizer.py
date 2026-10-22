"""
Detector Optimizer - Stage 1 Component
Optimizes detectors based on failure analysis

Master's Thesis Research - Dalian University of Technology
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

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DetectorOptimizer:
    """
    Optimizes anomaly detectors based on failure analysis
    
    Research Purpose:
    -----------------
    This component:
    1. Loads failure analysis from FailureCaseAnalyzer
    2. Creates optimization plans for weak models
    3. Applies hyperparameter tuning or ensemble methods
    4. Saves optimized detectors for Stage 2 evaluation
    
    Key Innovation: Closes the loop from analysis → action
    
    Input: Failure analysis report
    Output: Optimized detectors + optimization plan
    """
    
    def __init__(self, 
                 models_dir: Path,
                 results_dir: Path,
                 adaptive_dir: Optional[Path] = None):
        """
        Initialize Detector Optimizer
        
        Args:
            models_dir: Path to results/models directory (trained models)
            results_dir: Path to the directory containing robustness-related result files
                         (e.g., stress_test_robustness_scores.csv, attack_mimic_aligned_results.json,
                          feature_importance.json, model_predictions.csv)
            adaptive_dir: Path to results/adaptive/stage1 directory
        """
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.adaptive_dir = Path(adaptive_dir) if adaptive_dir else (
            Path(results_dir).parent / 'adaptive' / 'stage1'
        )
        
        self.optimization_plan = {}
        self.optimized_models = {}
        self.model_weakness_scores = pd.DataFrame()  # Fix 2: Initialize DataFrame
        
        logger.info(f"✅ Initialized DetectorOptimizer")
        logger.info(f"📁 Adaptive directory: {self.adaptive_dir}")
        
    def load_failure_analysis(self, 
                             failure_report_path: Optional[str] = None) -> Dict:
        """
        Import results from FailureCaseAnalyzer
        
        Args:
            failure_report_path: Path to failure_analysis_report.json
            
        Returns:
            Dictionary with failure analysis
        """
        logger.info("📊 Loading failure analysis report...")
        
        if failure_report_path is None:
            # Fix 1: Update path to reports subdirectory
            failure_report_path = self.adaptive_dir / 'reports' / 'failure_analysis_report.json'
        
        if Path(failure_report_path).exists():
            try:
                with open(failure_report_path, 'r', encoding='utf-8') as f:
                    failure_analysis = json.load(f)
                
                logger.info(f"✅ Loaded failure analysis: {failure_analysis.get('summary', {}).get('total_models_analyzed', 0)} models")
                logger.info(f"   High priority optimizations: {failure_analysis.get('summary', {}).get('models_with_high_priority', 0)}")
                
                # Fix 3: Load weakness_scores into DataFrame if available
                if 'weakness_scores' in failure_analysis and failure_analysis['weakness_scores']:
                    self.model_weakness_scores = pd.DataFrame(failure_analysis['weakness_scores'])
                    logger.info(f"   Loaded weakness scores: {len(self.model_weakness_scores)} models")
                
                return failure_analysis
            except Exception as e:
                logger.error(f"❌ Failed to load failure analysis: {e}")
                return {}
        else:
            logger.error(f"❌ Failure report not found: {failure_report_path}")
            return {}
    
    def create_optimization_plan(self, 
                                 failure_analysis: Dict,
                                 max_models_to_optimize: int = 3) -> Dict:
        """
        Generate optimization plan for weak models
        
        Args:
            failure_analysis: Output from load_failure_analysis()
            max_models_to_optimize: Maximum number of models to optimize (default: 3)
            
        Returns:
            Dictionary with optimization plan
        """
        logger.info("📋 Creating optimization plan...")
        
        optimization_plan = {
            'models_to_optimize': [],
            'optimization_strategies': {},
            'expected_improvements': {},
            'total_models': 0,
            'selected_models': 0
        }
        
        # Get models sorted by weakness score
        if 'weakness_scores' in failure_analysis:
            weakness_df = pd.DataFrame(failure_analysis['weakness_scores'])
            weakness_df = weakness_df.sort_values('weakness_score', ascending=False)
            
            # Select top N models to optimize
            models_to_optimize = weakness_df.head(max_models_to_optimize)['model'].tolist()
            
            optimization_plan['models_to_optimize'] = models_to_optimize
            optimization_plan['total_models'] = len(weakness_df)
            optimization_plan['selected_models'] = len(models_to_optimize)
            
            # Create optimization strategy for each selected model
            for model_name in models_to_optimize:
                model_data = weakness_df[weakness_df['model'] == model_name].iloc[0]
                model_failures = failure_analysis.get('model_failures', {}).get(model_name, {})
                
                strategy = self._determine_optimization_strategy(
                    model_name=model_name,
                    weakness_score=model_data['weakness_score'],
                    weakest_anomaly=model_data.get('weakest_anomaly', 'unknown'),
                    model_failures=model_failures
                )
                
                optimization_plan['optimization_strategies'][model_name] = strategy
                
                # Estimate expected improvement
                expected_improvement = self._estimate_improvement(
                    current_f1_drop=model_data.get('max_f1_drop', 0.0),
                    strategy=strategy['action']
                )
                
                optimization_plan['expected_improvements'][model_name] = expected_improvement
                
                logger.info(f"  {model_name}: {strategy['action']} "
                           f"(expected F1 improvement: +{expected_improvement:.3f})")
        
        self.optimization_plan = optimization_plan
        
        return optimization_plan
    
    def _determine_optimization_strategy(self, 
                                         model_name: str,
                                         weakness_score: float,
                                         weakest_anomaly: str,
                                         model_failures: Dict) -> Dict:
        """
        Determine specific optimization strategy for a model
        
        Args:
            model_name: Name of the model
            weakness_score: Computed weakness score
            weakest_anomaly: Anomaly type causing most failures
            model_failures: Detailed failure patterns
            
        Returns:
            Dictionary with optimization strategy
        """
        strategy = {
            'model': model_name,
            'action': '',
            'params_to_tune': [],
            'target_anomalies': [],
            'priority': 'MEDIUM'
        }
        
        # Determine action based on model type and weakness
        if weakness_score > 0.60:
            strategy['action'] = 'hyperparameter_tuning_ensemble'
            strategy['priority'] = 'HIGH'
        elif weakness_score > 0.40:
            strategy['action'] = 'hyperparameter_tuning'
            strategy['priority'] = 'HIGH'
        elif model_failures.get('generalization_issue', False):
            strategy['action'] = 'retrain_with_hard_samples'
            strategy['priority'] = 'MEDIUM'
        else:
            strategy['action'] = 'feature_engineering'
            strategy['priority'] = 'LOW'
        
        # Set parameters to tune based on model type
        if 'RandomForest' in model_name or 'XGBoost' in model_name:
            strategy['params_to_tune'] = [
                'n_estimators', 'max_depth', 'min_samples_split', 
                'learning_rate' if 'XGBoost' in model_name else None
            ]
            strategy['params_to_tune'] = [p for p in strategy['params_to_tune'] if p is not None]
        
        elif 'LSTM' in model_name or 'Autoencoder' in model_name:
            strategy['params_to_tune'] = [
                'time_steps', 'hidden_units', 'dropout_rate', 'learning_rate'
            ]
        
        elif 'SVM' in model_name:
            strategy['params_to_tune'] = ['C', 'kernel', 'gamma']
        
        elif 'IsolationForest' in model_name or 'OneClassSVM' in model_name:
            strategy['params_to_tune'] = ['contamination', 'n_estimators']
        
        # Set target anomalies
        if 'severe_anomalies' in model_failures:
            strategy['target_anomalies'] = model_failures['severe_anomalies']
        elif weakest_anomaly != 'N/A':
            strategy['target_anomalies'] = [weakest_anomaly]
        
        return strategy
    
    def _estimate_improvement(self, 
                             current_f1_drop: float,
                             strategy: str) -> float:
        """
        Estimate expected F1 improvement from optimization
        
        Args:
            current_f1_drop: Current F1 drop on weak anomalies
            strategy: Optimization strategy name
            
        Returns:
            Expected F1 improvement (0.0-0.3 range)
        """
        # Conservative estimates based on literature
        improvement_estimates = {
            'retrain_with_hard_samples': 0.25,
            'hyperparameter_tuning_ensemble': 0.20,
            'hyperparameter_tuning': 0.15,
            'feature_engineering': 0.10
        }
        
        base_improvement = improvement_estimates.get(strategy, 0.10)
        
        # Scale by current weakness (more room for improvement if very weak)
        scaling_factor = min(current_f1_drop / 0.50, 1.0)
        
        return base_improvement * scaling_factor
    
    def apply_optimization(self, 
                          optimization_plan: Optional[Dict] = None,
                          save_optimized_models: bool = True) -> Dict:
        """
        Apply optimization to selected models
        
        NOTE: This is a placeholder for Stage 1.
        Actual hyperparameter tuning happens in Stage 2 after retraining.
        For Stage 1, we create the optimization plan and prepare configurations.
        
        Args:
            optimization_plan: Output from create_optimization_plan()
            save_optimized_models: Whether to save optimization configs
            
        Returns:
            Dictionary with optimization results
        """
        logger.info("⚙️  Applying optimization (Stage 1 preparation)...")
        
        if optimization_plan is None:
            optimization_plan = self.optimization_plan
        
        optimization_results = {
            'models_optimized': 0,
            'configurations_saved': 0,
            'optimization_configs': {}
        }
        
        # For Stage 1: Save optimization configurations for Stage 2
        for model_name, strategy in optimization_plan.get('optimization_strategies', {}).items():
            config = {
                'model_name': model_name,
                'strategy': strategy,
                'expected_improvement': optimization_plan['expected_improvements'].get(model_name, 0.0),
                'status': 'CONFIGURED',  # Will be 'OPTIMIZED' after Stage 2
                'stage': 1
            }
            
            optimization_results['optimization_configs'][model_name] = config
            optimization_results['configurations_saved'] += 1
            
            logger.info(f"  {model_name}: Configuration saved ({strategy['action']})")
        
        # Save optimization plan
        if save_optimized_models:
            self._save_optimization_configs(optimization_results)
        
        optimization_results['models_optimized'] = len(optimization_plan.get('models_to_optimize', []))
        
        return optimization_results
    
    def _save_optimization_configs(self, optimization_results: Dict):
        """Save optimization configurations for Stage 2"""
        output_path = self.adaptive_dir / 'optimization_plan.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine with optimization plan
        full_plan = {
            'stage': 1,
            'optimization_plan': self.optimization_plan,
            'optimization_results': optimization_results,
            'next_stage': 2,
            'ready_for_stage2': True
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_plan, f, indent=2)
        
        logger.info(f"💾 Saved optimization plan: {output_path}")
        
        # Fix 4: Save weakness scores only if DataFrame is not empty
        if isinstance(self.model_weakness_scores, pd.DataFrame) and len(self.model_weakness_scores) > 0:
            weakness_csv_path = self.adaptive_dir / 'failure_analysis' / 'model_weakness_scores.csv'
            weakness_csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_weakness_scores.to_csv(weakness_csv_path, index=False)
            logger.info(f"💾 Saved model weakness scores: {weakness_csv_path}")
    
    def generate_optimizer_report(self, 
                                  failure_analysis: Dict,
                                  optimization_results: Dict,
                                  output_path: Optional[str] = None) -> str:
        """
        Create interpretable report for thesis
        
        Args:
            failure_analysis: Output from FailureCaseAnalyzer
            optimization_results: Output from apply_optimization()
            output_path: Path to save report
            
        Returns:
            Formatted report string
        """
        logger.info("📝 Generating optimizer report...")
        
        report = []
        report.append("="*80)
        report.append("STAGE 1: FAILURE ANALYSIS + DETECTOR OPTIMIZATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append(f"  - Models analyzed: {failure_analysis.get('summary', {}).get('total_models_analyzed', 0)}")
        report.append(f"  - Models selected for optimization: {optimization_results.get('models_optimized', 0)}")
        report.append(f"  - Configurations saved: {optimization_results.get('configurations_saved', 0)}")
        
        # Fix 5: Read from self.optimization_plan instead of optimization_results
        expected_improvements = self.optimization_plan.get('expected_improvements', {})
        avg_improvement = np.mean(list(expected_improvements.values())) if expected_improvements else 0.0
        
        report.append(f"  - Average expected F1 improvement: +{avg_improvement:.3f}")
        report.append("")
        
        # Optimization details
        report.append("OPTIMIZATION PLAN:")
        for model_name, strategy in self.optimization_plan.get('optimization_strategies', {}).items():
            report.append(f"  {model_name}:")
            report.append(f"    Action: {strategy['action']}")
            report.append(f"    Priority: {strategy['priority']}")
            report.append(f"    Target anomalies: {', '.join(strategy['target_anomalies'])}")
            report.append(f"    Params to tune: {', '.join(strategy['params_to_tune'])}")
            expected_imp = self.optimization_plan['expected_improvements'].get(model_name, 0.0)
            report.append(f"    Expected F1 improvement: +{expected_imp:.3f}")
            report.append("")
        
        # Recommendations for Stage 2
        report.append("RECOMMENDATIONS FOR STAGE 2:")
        report.append("  1. Generate difficulty-aware anomalies targeting weak anomalies")
        # Fix 6: Replace overly specific wording with general wording
        report.append("  2. Retrain selected models with updated data informed by hard and weak anomaly cases")
        report.append("  3. Validate improvement on held-out test set")
        report.append("  4. Compare Stage 1 vs. Stage 2 F1 scores")
        report.append("")
        
        report.append("="*80)
        report.append("END OF STAGE 1 REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path is None:
            output_path = self.adaptive_dir / 'reports' / 'stage1_optimizer_report.txt'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved optimizer report: {output_path}")
        
        return report_text