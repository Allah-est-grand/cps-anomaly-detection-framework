"""
Overall Framework Evaluator - Stage 3 Component
Evaluates the complete adaptive framework by comparing baseline vs adaptive workflow

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

This module extends the existing baseline framework with:
- Stage 1: Failure Case Analysis + Detector Optimization
- Stage 2: Difficulty-Aware Anomaly Generation + Retraining Rules
- Stage 3: Overall Framework Evaluation (THIS MODULE)

All components integrate with existing:
- src/anomaly_generator.py
- src/robustness_evaluator.py
- src/models.py

Scientific Role:
----------------
This module evaluates the overall adaptive framework AFTER Stage 1 and Stage 2 
have been completed. It compares the baseline framework with the adaptive framework
to assess whether the adaptive feedback workflow improves performance.

This is Stage 3 - Overall Framework Evaluation.
It does NOT redesign the baseline pipeline or rebuild Stage 1/Stage 2.
It focuses on evaluation, comparison, metrics aggregation, and reporting.

Key Functions:
--------------
1. Load baseline results and Stage 1/Stage 2 outputs
2. Compare baseline vs adaptive framework
3. Summarize expected or observed improvements
4. Prepare final evaluation tables and metadata
5. Generate comparison reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class OverallFrameworkEvaluator:
    """
    Evaluates the complete adaptive framework by comparing baseline vs adaptive workflow
    
    Research Purpose:
    -----------------
    This component:
    1. Loads baseline results from robustness evaluation
    2. Loads Stage 1 failure analysis and optimization plans
    3. Loads Stage 2 scenarios and retraining configurations
    4. Compares baseline framework with adaptive framework
    5. Summarizes expected improvements based on adaptive workflow
    6. Prepares final evaluation reports for thesis documentation
    
    Key Innovation: Provides comprehensive framework-level comparison
    showing how adaptive feedback improves upon baseline anomaly detection
    
    This is Stage 3 - Overall Framework Evaluation.
    It distinguishes between observed baseline metrics and planned adaptive improvements.
    
    Input: Baseline results + Stage 1 outputs + Stage 2 outputs
    Output: Comparison tables, metrics, and evaluation reports
    """
    
    def __init__(self,
                 base_dir: Path,
                 baseline_dir: Optional[Path] = None,
                 adaptive_dir: Optional[Path] = None,
                 stage3_dir: Optional[Path] = None):
        """
        Initialize Overall Framework Evaluator
        
        Args:
            base_dir: Path to results directory
            baseline_dir: Path to results/metrics directory (baseline results)
            adaptive_dir: Path to results/adaptive directory
            stage3_dir: Path to results/adaptive/stage3 directory
        """
        self.base_dir = Path(base_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir else (
            self.base_dir / 'metrics'  # Changed from 'robustness' to 'metrics'
        )
        self.adaptive_dir = Path(adaptive_dir) if adaptive_dir else (
            self.base_dir / 'adaptive'
        )
        self.stage3_dir = Path(stage3_dir) if stage3_dir else (
            self.adaptive_dir / 'stage3'
        )
        
        # Create Stage 3 output directories
        self.comparisons_dir = self.stage3_dir / 'comparisons'
        self.metrics_dir = self.stage3_dir / 'metrics'
        self.reports_dir = self.stage3_dir / 'reports'
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.baseline_results = {}
        self.stage1_outputs = {}
        self.stage2_outputs = {}
        self.comparison_tables = {}
        self.evaluation_metadata = {}
        
        logger.info(f"✅ Initialized OverallFrameworkEvaluator")
        logger.info(f"📁 Baseline directory: {self.baseline_dir}")
        logger.info(f"📁 Adaptive directory: {self.adaptive_dir}")
        logger.info(f"📁 Stage 3 directory: {self.stage3_dir}")
        
    def load_baseline_results(self,
                             stress_test_path: Optional[str] = None,
                             attack_mimic_path: Optional[str] = None) -> Dict:
        """
        Load baseline framework results from robustness evaluation
        
        Args:
            stress_test_path: Path to stress_test_robustness_scores.csv
            attack_mimic_path: Path to attack_mimic_aligned_results.json
            
        Returns:
            Dictionary with baseline results and loading status
        """
        logger.info("📊 Loading baseline framework results...")
        
        loading_status = {
            'stress_test_loaded': False,
            'attack_mimic_loaded': False,
            'status': 'pending'
        }
        
        # Load stress-test robustness scores
        if stress_test_path is None:
            stress_test_path = self.baseline_dir / 'stress_test_robustness_scores.csv'
        
        if Path(stress_test_path).exists():
            try:
                self.baseline_results['stress_test'] = pd.read_csv(stress_test_path)
                loading_status['stress_test_loaded'] = True
                logger.info(f"✅ Loaded stress-test results: {len(self.baseline_results['stress_test'])} rows")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load stress-test results: {e}")
        else:
            logger.warning(f"⚠️  Stress-test results not found: {stress_test_path}")
        
        # Load attack-mimic aligned results
        if attack_mimic_path is None:
            attack_mimic_path = self.baseline_dir / 'attack_mimic_aligned_results.json'
        
        if Path(attack_mimic_path).exists():
            try:
                with open(attack_mimic_path, 'r', encoding='utf-8') as f:
                    self.baseline_results['attack_mimic'] = json.load(f)
                loading_status['attack_mimic_loaded'] = True
                logger.info(f"✅ Loaded attack-mimic results: {len(self.baseline_results['attack_mimic'])} models")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load attack-mimic results: {e}")
        else:
            logger.warning(f"⚠️  Attack-mimic results not found: {attack_mimic_path}")
        
        # Determine overall status
        if loading_status['stress_test_loaded'] and loading_status['attack_mimic_loaded']:
            loading_status['status'] = 'complete'
        elif loading_status['stress_test_loaded'] or loading_status['attack_mimic_loaded']:
            loading_status['status'] = 'partial'
        else:
            loading_status['status'] = 'missing'
        
        logger.info(f"📊 Baseline loading status: {loading_status['status']}")
        
        return loading_status
    
    def load_stage1_outputs(self,
                           failure_analysis_path: Optional[str] = None,
                           optimization_plan_path: Optional[str] = None) -> Dict:
        """
        Load Stage 1 outputs (failure analysis and optimization plans)
        
        Args:
            failure_analysis_path: Path to failure_analysis_report.json
            optimization_plan_path: Path to optimization_plan.json
            
        Returns:
            Dictionary with Stage 1 outputs and loading status
        """
        logger.info("📊 Loading Stage 1 outputs...")
        
        loading_status = {
            'failure_analysis_loaded': False,
            'optimization_plan_loaded': False,
            'status': 'pending'
        }
        
        # Load failure analysis report
        if failure_analysis_path is None:
            failure_analysis_path = self.adaptive_dir / 'stage1' / 'reports' / 'failure_analysis_report.json'
        
        if Path(failure_analysis_path).exists():
            try:
                with open(failure_analysis_path, 'r', encoding='utf-8') as f:
                    self.stage1_outputs['failure_analysis'] = json.load(f)
                loading_status['failure_analysis_loaded'] = True
                logger.info(f"✅ Loaded failure analysis: {self.stage1_outputs['failure_analysis'].get('summary', {}).get('total_models_analyzed', 0)} models")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load failure analysis: {e}")
        else:
            logger.warning(f"⚠️  Failure analysis not found: {failure_analysis_path}")
        
        # Load optimization plan
        if optimization_plan_path is None:
            optimization_plan_path = self.adaptive_dir / 'stage1' / 'optimization_plan.json'
        
        if Path(optimization_plan_path).exists():
            try:
                with open(optimization_plan_path, 'r', encoding='utf-8') as f:
                    self.stage1_outputs['optimization_plan'] = json.load(f)
                loading_status['optimization_plan_loaded'] = True
                logger.info(f"✅ Loaded optimization plan: {self.stage1_outputs['optimization_plan'].get('optimization_plan', {}).get('selected_models', 0)} models selected")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load optimization plan: {e}")
        else:
            logger.warning(f"⚠️  Optimization plan not found: {optimization_plan_path}")
        
        # Determine overall status
        if loading_status['failure_analysis_loaded'] and loading_status['optimization_plan_loaded']:
            loading_status['status'] = 'complete'
        elif loading_status['failure_analysis_loaded'] or loading_status['optimization_plan_loaded']:
            loading_status['status'] = 'partial'
        else:
            loading_status['status'] = 'missing'
        
        logger.info(f"📊 Stage 1 loading status: {loading_status['status']}")
        
        return loading_status
    
    def load_stage2_outputs(self,
                           scenarios_path: Optional[str] = None,
                           retraining_plans_path: Optional[str] = None,
                           retraining_summary_path: Optional[str] = None) -> Dict:
        """
        Load Stage 2 outputs (scenarios and retraining configurations)
        
        Args:
            scenarios_path: Path to difficulty_aware_scenarios.json
            retraining_plans_path: Path to retraining_plans.json
            retraining_summary_path: Path to retraining_summary.csv
            
        Returns:
            Dictionary with Stage 2 outputs and loading status
        """
        logger.info("📊 Loading Stage 2 outputs...")
        
        loading_status = {
            'scenarios_loaded': False,
            'retraining_plans_loaded': False,
            'retraining_summary_loaded': False,
            'status': 'pending'
        }
        
        # Load difficulty-aware scenarios
        if scenarios_path is None:
            scenarios_path = self.adaptive_dir / 'stage2' / 'generated_scenarios' / 'difficulty_aware_scenarios.json'
        
        if Path(scenarios_path).exists():
            try:
                with open(scenarios_path, 'r', encoding='utf-8') as f:
                    scenarios_data = json.load(f)
                self.stage2_outputs['scenarios'] = scenarios_data.get('scenarios', [])
                loading_status['scenarios_loaded'] = True
                logger.info(f"✅ Loaded scenarios: {len(self.stage2_outputs['scenarios'])} scenarios")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load scenarios: {e}")
        else:
            logger.warning(f"⚠️  Scenarios not found: {scenarios_path}")
        
        # Load retraining plans
        if retraining_plans_path is None:
            retraining_plans_path = self.adaptive_dir / 'stage2' / 'retrained_models' / 'retraining_plans.json'
        
        if Path(retraining_plans_path).exists():
            try:
                with open(retraining_plans_path, 'r', encoding='utf-8') as f:
                    self.stage2_outputs['retraining_plans'] = json.load(f)
                loading_status['retraining_plans_loaded'] = True
                logger.info(f"✅ Loaded retraining plans: {len(self.stage2_outputs['retraining_plans'].get('models', {}))} models")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load retraining plans: {e}")
        else:
            logger.warning(f"⚠️  Retraining plans not found: {retraining_plans_path}")
        
        # Load retraining summary
        if retraining_summary_path is None:
            retraining_summary_path = self.adaptive_dir / 'stage2' / 'retrained_models' / 'retraining_summary.csv'
        
        if Path(retraining_summary_path).exists():
            try:
                self.stage2_outputs['retraining_summary'] = pd.read_csv(retraining_summary_path)
                loading_status['retraining_summary_loaded'] = True
                logger.info(f"✅ Loaded retraining summary: {len(self.stage2_outputs['retraining_summary'])} rows")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load retraining summary: {e}")
        else:
            logger.warning(f"⚠️  Retraining summary not found: {retraining_summary_path}")
        
        # Determine overall status
        loaded_count = sum([
            loading_status['scenarios_loaded'],
            loading_status['retraining_plans_loaded'],
            loading_status['retraining_summary_loaded']
        ])
        
        if loaded_count == 3:
            loading_status['status'] = 'complete'
        elif loaded_count > 0:
            loading_status['status'] = 'partial'
        else:
            loading_status['status'] = 'missing'
        
        logger.info(f"📊 Stage 2 loading status: {loading_status['status']}")
        
        return loading_status
    
    def build_comparison_tables(self) -> Dict:
        """
        Build comparison tables between baseline and adaptive framework
        
        Returns:
            Dictionary with comparison tables
        """
        logger.info("📊 Building comparison tables...")
        
        comparison_tables = {
            'model_level_comparison': None,
            'framework_level_comparison': None,
            'weakness_coverage': None,
            'scenario_coverage': None
        }
        
        # Model-level comparison (baseline vs adaptive plans)
        model_comparison_data = []
        
        # Get baseline attack-mimic F1 scores
        baseline_f1 = {}
        if 'attack_mimic' in self.baseline_results:
            for model_name, metrics in self.baseline_results['attack_mimic'].items():
                baseline_f1[model_name] = metrics.get('f1_score', 0.0)
        
        # Get adaptive retraining plans
        adaptive_plans = {}
        if 'retraining_plans' in self.stage2_outputs:
            retraining_models = self.stage2_outputs['retraining_plans'].get('models', {})
            for model_name, config in retraining_models.items():
                adaptive_plans[model_name] = {
                    'retraining_required': config.get('retraining_required', False),
                    'priority': config.get('priority', 'N/A'),
                    'expected_improvement': config.get('expected_outcomes', {}).get('target_f1_improvement', 0.0)
                }
        
        # Build comparison rows
        all_models = set(baseline_f1.keys()) | set(adaptive_plans.keys())
        for model_name in sorted(all_models):
            row = {
                'model_name': model_name,
                'baseline_f1_score': baseline_f1.get(model_name, 0.0),
                'retraining_required': adaptive_plans.get(model_name, {}).get('retraining_required', False),
                'retraining_priority': adaptive_plans.get(model_name, {}).get('priority', 'N/A'),
                'expected_f1_improvement': adaptive_plans.get(model_name, {}).get('expected_improvement', 0.0),
                'adaptive_workflow_applied': model_name in adaptive_plans
            }
            model_comparison_data.append(row)
        
        if model_comparison_data:
            comparison_tables['model_level_comparison'] = pd.DataFrame(model_comparison_data)
            logger.info(f"  Created model-level comparison: {len(model_comparison_data)} models")
        
        # Framework-level comparison
        framework_comparison = {
            'metric': ['Baseline Framework', 'Adaptive Framework'],
            'evaluation_protocols': [
                'Stress-Test + Attack-Mimic Aligned',
                'Stress-Test + Attack-Mimic Aligned + Adaptive Feedback'
            ],
            'weakness_identification': [
                'Manual analysis required',
                f"Automated (Stage 1: {self.stage1_outputs.get('failure_analysis', {}).get('summary', {}).get('total_models_analyzed', 0)} models analyzed)"
            ],
            'scenario_generation': [
                'Fixed anomaly types',
                f"Difficulty-aware ({len(self.stage2_outputs.get('scenarios', []))} scenarios)"
            ],
            'retraining_configuration': [
                'Not applicable',
                f"Planned for {len(adaptive_plans)} models"
            ],
            'framework_readiness': [
                'Complete',
                'Ready for experimental validation'
            ]
        }
        comparison_tables['framework_level_comparison'] = pd.DataFrame(framework_comparison)
        logger.info("  Created framework-level comparison")
        
        # Weakness coverage analysis
        if 'failure_analysis' in self.stage1_outputs:
            weakness_data = []
            failure_summary = self.stage1_outputs['failure_analysis'].get('summary', {})
            weakness_data.append({
                'weakness_type': 'High-priority models',
                'count': failure_summary.get('models_with_high_priority', 0),
                'addressed_in_stage2': len(adaptive_plans) > 0
            })
            weakness_data.append({
                'weakness_type': 'Most common weak anomaly',
                'count': 1,
                'addressed_in_stage2': True,
                'anomaly_name': failure_summary.get('most_common_weak_anomaly', 'N/A')
            })
            comparison_tables['weakness_coverage'] = pd.DataFrame(weakness_data)
            logger.info("  Created weakness coverage analysis")
        
        # Scenario coverage analysis
        if 'scenarios' in self.stage2_outputs:
            scenarios = self.stage2_outputs['scenarios']
            scenario_counts = {}
            for scenario in scenarios:
                difficulty = scenario.get('difficulty_level', 'UNKNOWN')
                scenario_counts[difficulty] = scenario_counts.get(difficulty, 0) + 1
            
            scenario_data = []
            for difficulty, count in scenario_counts.items():
                scenario_data.append({
                    'difficulty_level': difficulty,
                    'scenario_count': count,
                    'percentage': count / len(scenarios) * 100 if scenarios else 0
                })
            comparison_tables['scenario_coverage'] = pd.DataFrame(scenario_data)
            logger.info("  Created scenario coverage analysis")
        
        self.comparison_tables = comparison_tables
        
        return comparison_tables
    
    def summarize_model_differences(self) -> Dict:
        """
        Summarize model-level differences between baseline and adaptive
        
        Returns:
            Dictionary with model-level summary
        """
        logger.info("📊 Summarizing model-level differences...")
        
        summary = {
            'total_models_evaluated': 0,
            'models_requiring_retraining': 0,
            'high_priority_models': 0,
            'medium_priority_models': 0,
            'low_priority_models': 0,
            'average_baseline_f1': 0.0,
            'total_expected_improvement': 0.0,
            'model_details': []
        }
        
        # Get baseline F1 scores
        baseline_f1 = {}
        if 'attack_mimic' in self.baseline_results:
            for model_name, metrics in self.baseline_results['attack_mimic'].items():
                baseline_f1[model_name] = metrics.get('f1_score', 0.0)
        
        # Get adaptive plans
        adaptive_plans = {}
        if 'retraining_plans' in self.stage2_outputs:
            retraining_models = self.stage2_outputs['retraining_plans'].get('models', {})
            for model_name, config in retraining_models.items():
                adaptive_plans[model_name] = config
        
        # Calculate summary statistics
        all_models = set(baseline_f1.keys()) | set(adaptive_plans.keys())
        summary['total_models_evaluated'] = len(all_models)
        
        f1_scores = list(baseline_f1.values())
        if f1_scores:
            summary['average_baseline_f1'] = np.mean(f1_scores)
        
        for model_name in sorted(all_models):
            model_detail = {
                'model_name': model_name,
                'baseline_f1': baseline_f1.get(model_name, 0.0),
                'retraining_required': False,
                'priority': 'N/A',
                'expected_improvement': 0.0
            }
            
            if model_name in adaptive_plans:
                config = adaptive_plans[model_name]
                model_detail['retraining_required'] = config.get('retraining_required', False)
                model_detail['priority'] = config.get('priority', 'N/A')
                model_detail['expected_improvement'] = config.get('expected_outcomes', {}).get('target_f1_improvement', 0.0)
                
                if model_detail['retraining_required']:
                    summary['models_requiring_retraining'] += 1
                    summary['total_expected_improvement'] += model_detail['expected_improvement']
                    
                    priority = model_detail['priority']
                    if priority == 'HIGH':
                        summary['high_priority_models'] += 1
                    elif priority == 'MEDIUM':
                        summary['medium_priority_models'] += 1
                    else:
                        summary['low_priority_models'] += 1
            
            summary['model_details'].append(model_detail)
        
        logger.info(f"  Total models: {summary['total_models_evaluated']}")
        logger.info(f"  Models requiring retraining: {summary['models_requiring_retraining']}")
        logger.info(f"  Average baseline F1: {summary['average_baseline_f1']:.3f}")
        logger.info(f"  Total expected improvement: {summary['total_expected_improvement']:.3f}")
        
        return summary
    
    def summarize_framework_differences(self) -> Dict:
        """
        Summarize framework-level differences between baseline and adaptive
        
        Returns:
            Dictionary with framework-level summary
        """
        logger.info("📊 Summarizing framework-level differences...")
        
        summary = {
            'baseline_framework': {
                'evaluation_protocols': ['Stress-Test', 'Attack-Mimic Aligned'],
                'weakness_identification': 'Manual',
                'scenario_generation': 'Fixed anomaly types',
                'retraining_support': 'Not applicable',
                'adaptivity': 'None'
            },
            'adaptive_framework': {
                'evaluation_protocols': ['Stress-Test', 'Attack-Mimic Aligned', 'Adaptive Feedback'],
                'weakness_identification': f"Automated (Stage 1)",
                'scenario_generation': f"Difficulty-aware (Stage 2: {len(self.stage2_outputs.get('scenarios', []))} scenarios)",
                'retraining_support': f"Configured for {len(self.stage2_outputs.get('retraining_plans', {}).get('models', {}))} models",
                'adaptivity': 'Weakness-driven optimization'
            },
            'key_improvements': [
                'Automated weakness identification through failure analysis',
                'Difficulty-aware scenario generation targeting weak points',
                'Prioritized retraining based on empirical failure patterns',
                'Comprehensive framework-level comparison and evaluation'
            ],
            'framework_readiness': {
                'stage1_completed': 'failure_analysis' in self.stage1_outputs,
                'stage2_completed': 'scenarios' in self.stage2_outputs and 'retraining_plans' in self.stage2_outputs,
                'stage3_evaluation': 'Comparison/reporting complete, experimental validation pending',
                'overall_status': 'Preparation complete, awaiting retraining execution'
            }
        }
        
        logger.info("  Framework-level summary completed")
        
        return summary
    
    def prepare_final_metrics_report(self) -> Dict:
        """
        Prepare final metrics report combining all evaluation results
        
        Returns:
            Dictionary with comprehensive metrics report
        """
        logger.info("📊 Preparing final metrics report...")
        
        # Gather all summaries
        model_summary = self.summarize_model_differences()
        framework_summary = self.summarize_framework_differences()
        
        # Build comprehensive report
        metrics_report = {
            'metadata': {
                'evaluation_timestamp': datetime.now().isoformat(),
                'framework_version': '3.0.0',
                'stage': '3_overall_framework_evaluation',
                'thesis_project': 'CPS Anomaly Detection Framework'
            },
            'baseline_performance': {
                'total_models': model_summary['total_models_evaluated'],
                'average_f1_score': model_summary['average_baseline_f1'],
                'evaluation_protocols': ['Stress-Test', 'Attack-Mimic Aligned'],
                'best_performing_model': max(model_summary['model_details'], key=lambda x: x['baseline_f1'])['model_name'] if model_summary['model_details'] else 'N/A'
            },
            'adaptive_workflow': {
                'stage1_weaknesses_identified': model_summary['models_requiring_retraining'],
                'stage2_scenarios_generated': len(self.stage2_outputs.get('scenarios', [])),
                'retraining_configurations': len(self.stage2_outputs.get('retraining_plans', {}).get('models', {})),
                'high_priority_models': model_summary['high_priority_models'],
                'medium_priority_models': model_summary['medium_priority_models'],
                'low_priority_models': model_summary['low_priority_models']
            },
            'expected_improvements': {
                'total_expected_f1_improvement': model_summary['total_expected_improvement'],
                'average_expected_improvement_per_model': (
                    model_summary['total_expected_improvement'] / model_summary['models_requiring_retraining']
                    if model_summary['models_requiring_retraining'] > 0 else 0.0
                ),
                'improvement_basis': 'Planned retraining configurations (Stage 2)',
                'experimental_validation': 'Pending execution'
            },
            'framework_comparison': framework_summary,
            'model_level_details': model_summary['model_details'],
            'evaluation_readiness': {
                'baseline_evaluated': True,
                'stage1_completed': 'failure_analysis' in self.stage1_outputs,
                'stage2_completed': 'scenarios' in self.stage2_outputs and 'retraining_plans' in self.stage2_outputs,
                'stage3_evaluation': 'Comparison/reporting complete, experimental validation pending',
                'ready_for_experimental_validation': True,
                'remaining_work': 'Execute retraining and measure actual performance improvements'
            }
        }
        
        logger.info("  Final metrics report prepared")
        
        return metrics_report
    
    def save_comparison_outputs(self,
                               comparison_tables: Optional[Dict] = None,
                               metrics_report: Optional[Dict] = None) -> List[str]:
        """
        Save all comparison outputs to Stage 3 directories
        
        Args:
            comparison_tables: Comparison tables from build_comparison_tables()
            metrics_report: Metrics report from prepare_final_metrics_report()
            
        Returns:
            List of saved output paths
        """
        logger.info("💾 Saving comparison outputs...")
        
        if comparison_tables is None:
            comparison_tables = self.comparison_tables
        if metrics_report is None:
            metrics_report = self.prepare_final_metrics_report()
        
        saved_outputs = []
        
        # Save comparison tables as CSV
        for table_name, table_df in comparison_tables.items():
            if table_df is not None and isinstance(table_df, pd.DataFrame):
                csv_path = self.comparisons_dir / f'{table_name}.csv'
                table_df.to_csv(csv_path, index=False)
                saved_outputs.append(str(csv_path))
                logger.info(f"  💾 Saved: {table_name}.csv")
        
        # Save metrics report as JSON
        metrics_json_path = self.metrics_dir / 'framework_evaluation_metrics.json'
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_report, f, indent=2)
        saved_outputs.append(str(metrics_json_path))
        logger.info(f"  💾 Saved: framework_evaluation_metrics.json")
        
        # Save summary metrics as CSV
        if 'model_level_comparison' in comparison_tables and comparison_tables['model_level_comparison'] is not None:
            summary_csv_path = self.metrics_dir / 'model_comparison_summary.csv'
            comparison_tables['model_level_comparison'].to_csv(summary_csv_path, index=False)
            saved_outputs.append(str(summary_csv_path))
            logger.info(f"  💾 Saved: model_comparison_summary.csv")
        
        # Save evaluation metadata
        metadata_path = self.metrics_dir / 'evaluation_metadata.json'
        self.evaluation_metadata = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'baseline_dir': str(self.baseline_dir),
            'adaptive_dir': str(self.adaptive_dir),
            'stage3_dir': str(self.stage3_dir),
            'outputs_saved': saved_outputs,
            'framework_readiness': metrics_report.get('evaluation_readiness', {})
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_metadata, f, indent=2)
        saved_outputs.append(str(metadata_path))
        logger.info(f"  💾 Saved: evaluation_metadata.json")
        
        logger.info(f"📊 Total outputs saved: {len(saved_outputs)}")
        
        return saved_outputs
    
    def generate_stage3_evaluation_report(self,
                                         metrics_report: Optional[Dict] = None,
                                         output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Stage 3 evaluation report
        
        Args:
            metrics_report: Metrics report from prepare_final_metrics_report()
            output_path: Path to save report
            
        Returns:
            Formatted report string
        """
        logger.info("📝 Generating Stage 3 evaluation report...")
        
        if metrics_report is None:
            metrics_report = self.prepare_final_metrics_report()
        
        report = []
        report.append("="*80)
        report.append("STAGE 3: OVERALL FRAMEWORK EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {metrics_report['metadata']['evaluation_timestamp']}")
        report.append(f"Framework Version: {metrics_report['metadata']['framework_version']}")
        report.append(f"Thesis Project: {metrics_report['metadata']['thesis_project']}")
        report.append("")
        
        # Baseline Performance Summary
        report.append("BASELINE FRAMEWORK PERFORMANCE:")
        baseline = metrics_report['baseline_performance']
        report.append(f"  - Total models evaluated: {baseline['total_models']}")
        report.append(f"  - Average F1-score: {baseline['average_f1_score']:.3f}")
        report.append(f"  - Best performing model: {baseline['best_performing_model']}")
        report.append(f"  - Evaluation protocols: {', '.join(baseline['evaluation_protocols'])}")
        report.append("")
        
        # Adaptive Workflow Summary
        report.append("ADAPTIVE WORKFLOW SUMMARY:")
        adaptive = metrics_report['adaptive_workflow']
        report.append(f"  - Stage 1 weaknesses identified: {adaptive['stage1_weaknesses_identified']}")
        report.append(f"  - Stage 2 scenarios generated: {adaptive['stage2_scenarios_generated']}")
        report.append(f"  - Retraining configurations: {adaptive['retraining_configurations']}")
        report.append(f"  - High priority models: {adaptive['high_priority_models']}")
        report.append(f"  - Medium priority models: {adaptive['medium_priority_models']}")
        report.append(f"  - Low priority models: {adaptive['low_priority_models']}")
        report.append("")
        
        # Expected Improvements
        report.append("EXPECTED IMPROVEMENTS (BASED ON ADAPTIVE WORKFLOW):")
        improvements = metrics_report['expected_improvements']
        report.append(f"  - Total expected F1 improvement: {improvements['total_expected_f1_improvement']:.3f}")
        report.append(f"  - Average expected improvement per model: {improvements['average_expected_improvement_per_model']:.3f}")
        report.append(f"  - Improvement basis: {improvements['improvement_basis']}")
        report.append(f"  - Experimental validation: {improvements['experimental_validation']}")
        report.append("")
        
        # Framework Comparison
        report.append("FRAMEWORK COMPARISON:")
        framework = metrics_report['framework_comparison']
        report.append("  Baseline Framework:")
        for key, value in framework['baseline_framework'].items():
            report.append(f"    - {key.replace('_', ' ').title()}: {value}")
        report.append("  Adaptive Framework:")
        for key, value in framework['adaptive_framework'].items():
            report.append(f"    - {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Key Improvements
        report.append("KEY IMPROVEMENTS OF ADAPTIVE FRAMEWORK:")
        for i, improvement in enumerate(framework['key_improvements'], 1):
            report.append(f"  {i}. {improvement}")
        report.append("")
        
        # Evaluation Readiness
        report.append("EVALUATION READINESS:")
        readiness = metrics_report['evaluation_readiness']
        for key, value in readiness.items():
            status = "✅" if value else "⏳"
            report.append(f"  {status} {key.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Scientific Notes
        report.append("SCIENTIFIC NOTES:")
        report.append("  - This evaluation compares baseline framework with adaptive workflow preparations")
        report.append("  - Expected improvements are based on Stage 2 retraining configurations")
        report.append("  - Actual performance improvements require experimental retraining execution")
        report.append("  - Stage 3 provides comprehensive framework-level comparison and evaluation readiness")
        report.append("  - All outputs are saved for thesis documentation and experimental validation")
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS:")
        report.append("  1. Execute retraining configurations prepared in Stage 2")
        report.append("  2. Measure actual performance improvements after retraining")
        report.append("  3. Compare expected vs. observed improvements")
        report.append("  4. Document final thesis results with experimental validation")
        report.append("  5. Prepare comprehensive thesis chapters on adaptive framework")
        report.append("")
        
        report.append("="*80)
        report.append("END OF STAGE 3 EVALUATION REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path is None:
            output_path = self.reports_dir / 'stage3_framework_evaluation_report.txt'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved Stage 3 evaluation report: {output_path}")
        
        return report_text
    
    def execute_stage3_evaluation(self) -> Dict:
        """
        Execute complete Stage 3 evaluation workflow
        
        Returns:
            Dictionary with execution results
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 EXECUTING STAGE 3: OVERALL FRAMEWORK EVALUATION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'evaluation_complete': False
        }
        
        # Step 1: Load baseline results
        baseline_status = self.load_baseline_results()
        
        # Step 2: Load Stage 1 outputs
        stage1_status = self.load_stage1_outputs()
        
        # Step 3: Load Stage 2 outputs
        stage2_status = self.load_stage2_outputs()
        
        # Step 4: Build comparison tables
        comparison_tables = self.build_comparison_tables()
        
        # Step 5: Prepare final metrics report
        metrics_report = self.prepare_final_metrics_report()
        
        # Step 6: Save comparison outputs
        saved_outputs = self.save_comparison_outputs(comparison_tables, metrics_report)
        results['outputs_saved'] = saved_outputs
        
        # Step 7: Generate Stage 3 evaluation report
        self.generate_stage3_evaluation_report(metrics_report)
        results['outputs_saved'].append(str(self.reports_dir / 'stage3_framework_evaluation_report.txt'))
        
        # Determine overall status
        if (baseline_status['status'] != 'missing' and
            stage1_status['status'] != 'missing' and
            stage2_status['status'] != 'missing'):
            results['status'] = 'complete'
            results['evaluation_complete'] = True
        else:
            results['status'] = 'partial'
            results['evaluation_complete'] = False
        
        logger.info("\n" + "="*80)
        logger.info("✅ STAGE 3 EVALUATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Evaluation status: {results['status']}")
        logger.info(f"📊 Outputs saved: {len(results['outputs_saved'])}")
        logger.info(f"💾 Stage 3 directory: {self.stage3_dir}")
        for output in results['outputs_saved']:
            logger.info(f"   - {output}")
        logger.info("="*80)
        
        return results