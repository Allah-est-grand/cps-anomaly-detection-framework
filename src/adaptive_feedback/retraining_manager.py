"""
Retraining Manager - Stage 2 Component (Continuation)
Prepares retraining logic based on difficulty-aware scenarios and Stage 1 failure analysis

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

This module extends the existing baseline framework with:
- Stage 1: Failure Case Analysis + Detector Optimization
- Stage 2: Difficulty-Aware Anomaly Generation + Retraining Rules (THIS MODULE)
- Stage 3: Overall Framework Evaluation

All components integrate with existing:
- src/anomaly_generator.py
- src/robustness_evaluator.py
- src/models.py

Scientific Role:
----------------
This module is part of Stage 2 continuation. It prepares retraining logic based on 
difficulty-aware scenarios and Stage 1 failure analysis. It does NOT execute full 
retraining but prepares all configurations, rules, and metadata for later execution.
Stage 3 will be the overall framework evaluation AFTER retraining is executed.

This is Stage 2 continuation - retraining preparation and configuration.
Stage 3 will be the overall framework evaluation AFTER retraining is executed.

Key Functions:
--------------
1. Load Stage 1 optimization outputs and Stage 2 scenarios
2. Identify target models for retraining
3. Define retraining rules based on optimization priority, difficulty levels, weak anomalies
4. Prepare updated retraining configurations and scenario assignments
5. Save retraining plans and metadata for Stage 2 continuation and subsequent Stage 3 evaluation
6. Generate Stage 2 continuation report
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class RetrainingManager:
    """
    Manages retraining preparation based on Stage 1 failure analysis and Stage 2 scenarios
    
    Research Purpose:
    -----------------
    This component:
    1. Loads optimization plans from Stage 1 and scenarios from Stage 2
    2. Identifies which models need retraining based on weakness analysis
    3. Defines retraining rules prioritizing weak models and hard anomalies
    4. Prepares retraining configurations without executing full retraining
    5. Saves all metadata for Stage 2 continuation and subsequent Stage 3 evaluation
    
    Key Innovation: Adaptive retraining rules based on empirical failure patterns
    rather than generic retraining strategies
    
    This is Stage 2 continuation - retraining preparation and configuration.
    Stage 3 will evaluate the overall framework AFTER retraining is executed.
    
    Input: Stage 1 optimization plan + Stage 2 difficulty-aware scenarios
    Output: Retraining plans, rules, and metadata for Stage 2 continuation
    """
    
    def __init__(self,
                 adaptive_dir: Path,
                 stage1_dir: Optional[Path] = None,
                 stage2_dir: Optional[Path] = None):
        """
        Initialize Retraining Manager
        
        Args:
            adaptive_dir: Path to results/adaptive directory
            stage1_dir: Path to results/adaptive/stage1 directory
            stage2_dir: Path to results/adaptive/stage2 directory
        """
        self.adaptive_dir = Path(adaptive_dir)
        self.stage1_dir = Path(stage1_dir) if stage1_dir else (
            self.adaptive_dir / 'stage1'
        )
        self.stage2_dir = Path(stage2_dir) if stage2_dir else (
            self.adaptive_dir / 'stage2'
        )
        
        # Create Stage 2 output directories
        self.retrained_models_dir = self.stage2_dir / 'retrained_models'
        self.reports_dir = self.stage2_dir / 'reports'
        self.retrained_models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.stage1_optimization_plan = {}
        self.stage1_failure_analysis = {}
        self.stage2_scenarios = []
        self.retraining_targets = []
        self.retraining_rules = {}
        self.retraining_metadata = {}
        
        logger.info(f"✅ Initialized RetrainingManager")
        logger.info(f"📁 Stage 1 directory: {self.stage1_dir}")
        logger.info(f"📁 Stage 2 directory: {self.stage2_dir}")
        
    def load_stage_artifacts(self,
                            optimization_plan_path: Optional[str] = None,
                            failure_analysis_path: Optional[str] = None,
                            scenarios_path: Optional[str] = None) -> Dict:
        """
        Load Stage 1 optimization outputs and Stage 2 scenarios
        
        Args:
            optimization_plan_path: Path to optimization_plan.json
            failure_analysis_path: Path to failure_analysis_report.json
            scenarios_path: Path to difficulty_aware_scenarios.json
            
        Returns:
            Dictionary with loading status
        """
        logger.info("📊 Loading Stage 1 and Stage 2 artifacts...")
        
        loading_status = {
            'optimization_plan_loaded': False,
            'failure_analysis_loaded': False,
            'scenarios_loaded': False,
            'status': 'pending'
        }
        
        # Load Stage 1 optimization plan
        if optimization_plan_path is None:
            optimization_plan_path = self.stage1_dir / 'optimization_plan.json'
        
        if Path(optimization_plan_path).exists():
            try:
                with open(optimization_plan_path, 'r', encoding='utf-8') as f:
                    self.stage1_optimization_plan = json.load(f)
                loading_status['optimization_plan_loaded'] = True
                logger.info(f"✅ Loaded optimization plan: {self.stage1_optimization_plan.get('optimization_plan', {}).get('selected_models', 0)} models selected")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load optimization plan: {e}")
        else:
            logger.warning(f"⚠️  Optimization plan not found: {optimization_plan_path}")
        
        # Load Stage 1 failure analysis
        if failure_analysis_path is None:
            failure_analysis_path = self.stage1_dir / 'reports' / 'failure_analysis_report.json'
        
        if Path(failure_analysis_path).exists():
            try:
                with open(failure_analysis_path, 'r', encoding='utf-8') as f:
                    self.stage1_failure_analysis = json.load(f)
                loading_status['failure_analysis_loaded'] = True
                logger.info(f"✅ Loaded failure analysis: {self.stage1_failure_analysis.get('summary', {}).get('total_models_analyzed', 0)} models analyzed")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load failure analysis: {e}")
        else:
            logger.warning(f"⚠️  Failure analysis not found: {failure_analysis_path}")
        
        # Load Stage 2 scenarios
        if scenarios_path is None:
            scenarios_path = self.stage2_dir / 'generated_scenarios' / 'difficulty_aware_scenarios.json'
        
        if Path(scenarios_path).exists():
            try:
                with open(scenarios_path, 'r', encoding='utf-8') as f:
                    scenarios_data = json.load(f)
                self.stage2_scenarios = scenarios_data.get('scenarios', [])
                loading_status['scenarios_loaded'] = True
                logger.info(f"✅ Loaded scenarios: {len(self.stage2_scenarios)} difficulty-aware scenarios")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load scenarios: {e}")
        else:
            logger.warning(f"⚠️  Scenarios not found: {scenarios_path}")
        
        # Determine overall status
        if loading_status['optimization_plan_loaded'] and loading_status['scenarios_loaded']:
            loading_status['status'] = 'complete'
        elif loading_status['optimization_plan_loaded'] or loading_status['scenarios_loaded']:
            loading_status['status'] = 'partial'
        else:
            loading_status['status'] = 'missing'
        
        logger.info(f"📊 Artifact loading status: {loading_status['status']}")
        
        # Save loading metadata
        metadata_path = self.reports_dir / 'artifact_loading_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(loading_status, f, indent=2)
        logger.info(f"💾 Saved loading metadata: {metadata_path}")
        
        return loading_status
    
    def identify_retraining_targets(self) -> List[Dict]:
        """
        Identify which models should be retrained based on Stage 1 analysis
        
        Returns:
            List of retraining target specifications
        """
        logger.info("🎯 Identifying retraining targets...")
        
        targets = []
        
        # Get models selected for optimization from Stage 1
        optimization_plan = self.stage1_optimization_plan.get('optimization_plan', {})
        models_to_optimize = optimization_plan.get('models_to_optimize', [])
        optimization_strategies = optimization_plan.get('optimization_strategies', {})
        
        # Also check failure analysis for additional weak models
        model_failures = self.stage1_failure_analysis.get('model_failures', {})
        
        for model_name in models_to_optimize:
            strategy = optimization_strategies.get(model_name, {})
            failure_info = model_failures.get(model_name, {})
            
            # Determine retraining priority
            priority = strategy.get('priority', 'MEDIUM')
            if failure_info.get('generalization_issue', False):
                priority = 'HIGH'
            
            # Determine retraining reason
            reason = strategy.get('action', 'performance_improvement')
            if failure_info.get('weakest_anomaly'):
                reason += f"_for_{failure_info['weakest_anomaly']}"
            
            target = {
                'model_name': model_name,
                'priority': priority,
                'reason': reason,
                'target_anomalies': strategy.get('target_anomalies', []),
                'params_to_tune': strategy.get('params_to_tune', []),
                'expected_improvement': optimization_plan.get('expected_improvements', {}).get(model_name, 0.0),
                'stage1_weakness_score': failure_info.get('max_f1_drop', 0.0),
                'retraining_required': priority in ['HIGH', 'MEDIUM']
            }
            
            targets.append(target)
            logger.info(f"  {model_name}: Priority={priority}, Reason={reason}, Required={target['retraining_required']}")
        
        # If no targets from optimization plan, check for models with severe failures
        if len(targets) == 0:
            for model_name, failure_info in model_failures.items():
                if model_name == '_summary':
                    continue
                if failure_info.get('max_f1_drop', 0.0) > 0.40:
                    target = {
                        'model_name': model_name,
                        'priority': 'MEDIUM',
                        'reason': 'high_f1_drop_detected',
                        'target_anomalies': failure_info.get('severe_anomalies', []),
                        'params_to_tune': [],
                        'expected_improvement': 0.10,
                        'stage1_weakness_score': failure_info.get('max_f1_drop', 0.0),
                        'retraining_required': True
                    }
                    targets.append(target)
                    logger.info(f"  {model_name}: Added due to high F1 drop ({target['stage1_weakness_score']:.3f})")
        
        self.retraining_targets = targets
        
        logger.info(f"📊 Identified {len(targets)} retraining targets")
        logger.info(f"   High priority: {sum(1 for t in targets if t['priority'] == 'HIGH')}")
        logger.info(f"   Medium priority: {sum(1 for t in targets if t['priority'] == 'MEDIUM')}")
        
        return targets
    
    def create_retraining_rules(self,
                               targets: Optional[List[Dict]] = None) -> Dict:
        """
        Create retraining rules based on targets and scenarios
        
        Args:
            targets: Retraining targets from identify_retraining_targets()
            
        Returns:
            Dictionary with retraining rules
        """
        logger.info("📋 Creating retraining rules...")
        
        if targets is None:
            targets = self.retraining_targets
        
        rules = {
            'retraining_priorities': {},
            'scenario_emphasis': {},
            'difficulty_adaptation': {},
            'feature_focus': {},
            'stopping_criteria': {}
        }
        
        # Define retraining priorities based on target priority
        priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        sorted_targets = sorted(targets, key=lambda x: priority_order.get(x['priority'], 3))
        
        for i, target in enumerate(sorted_targets):
            model_name = target['model_name']
            
            # Rule 1: Prioritize weak models
            rules['retraining_priorities'][model_name] = {
                'order': i + 1,
                'priority': target['priority'],
                'reason': target['reason'],
                'weakness_score': target['stage1_weakness_score']
            }
            
            # Rule 2: Emphasize scenarios targeting weak anomalies
            target_anomalies = target['target_anomalies']
            if target_anomalies:
                # Find scenarios that match target anomalies
                matching_scenarios = [
                    s for s in self.stage2_scenarios
                    if s.get('anomaly_type') in target_anomalies
                ]
                
                # Prioritize HARD difficulty scenarios for weak models
                hard_scenarios = [
                    s for s in matching_scenarios
                    if s.get('difficulty_level') == 'HARD'
                ]
                
                rules['scenario_emphasis'][model_name] = {
                    'target_anomalies': target_anomalies,
                    'matching_scenarios_count': len(matching_scenarios),
                    'hard_scenarios_count': len(hard_scenarios),
                    'emphasis_ratio': len(hard_scenarios) / max(len(matching_scenarios), 1)
                }
                
                # Rule 3: Adapt difficulty based on model weakness
                if target['stage1_weakness_score'] > 0.50:
                    # Very weak model: start with EASY scenarios, progress to HARD
                    rules['difficulty_adaptation'][model_name] = {
                        'strategy': 'progressive',
                        'start_level': 'EASY',
                        'progression': ['EASY', 'MEDIUM', 'HARD'],
                        'rationale': 'Model shows severe weakness, gradual difficulty increase recommended'
                    }
                elif target['stage1_weakness_score'] > 0.30:
                    # Moderately weak: focus on MEDIUM and HARD
                    rules['difficulty_adaptation'][model_name] = {
                        'strategy': 'focused',
                        'start_level': 'MEDIUM',
                        'progression': ['MEDIUM', 'HARD'],
                        'rationale': 'Model shows moderate weakness, focus on challenging scenarios'
                    }
                else:
                    # Relatively strong: focus on HARD for fine-tuning
                    rules['difficulty_adaptation'][model_name] = {
                        'strategy': 'refinement',
                        'start_level': 'HARD',
                        'progression': ['HARD'],
                        'rationale': 'Model is relatively strong, fine-tune on hardest scenarios'
                    }
            
            # Rule 4: Focus on critical features if available
            critical_features = self.stage1_failure_analysis.get('feature_correlations', {}).get(model_name, {}).get('top_features', [])
            if critical_features:
                rules['feature_focus'][model_name] = {
                    'critical_features': critical_features[:5],  # Top 5 features
                    'feature_engineering': 'emphasis_on_critical',
                    'rationale': f"Model decisions dominated by {len(critical_features)} features"
                }
            
            # Rule 5: Define stopping criteria
            expected_improvement = target['expected_improvement']
            rules['stopping_criteria'][model_name] = {
                'min_improvement_threshold': expected_improvement * 0.5,  # At least 50% of expected
                'max_epochs': 50 if target['priority'] == 'HIGH' else 30,
                'early_stopping_patience': 10,
                'validation_metric': 'f1_score',
                'rationale': f"Expect +{expected_improvement:.3f} F1 improvement"
            }
        
        self.retraining_rules = rules
        
        logger.info(f"📊 Created retraining rules for {len(rules['retraining_priorities'])} models")
        logger.info(f"   Scenario emphasis defined: {len(rules['scenario_emphasis'])} models")
        logger.info(f"   Difficulty adaptation defined: {len(rules['difficulty_adaptation'])} models")
        
        return rules
    
    def prepare_retraining_configurations(self,
                                         targets: Optional[List[Dict]] = None,
                                         rules: Optional[Dict] = None) -> Dict:
        """
        Prepare detailed retraining configurations for each target model
        
        Args:
            targets: Retraining targets
            rules: Retraining rules
            
        Returns:
            Dictionary with retraining configurations
        """
        logger.info("⚙️  Preparing retraining configurations...")
        
        if targets is None:
            targets = self.retraining_targets
        if rules is None:
            rules = self.retraining_rules
        
        configurations = {
            'models': {},
            'metadata': {
                'total_models': len(targets),
                'high_priority_models': sum(1 for t in targets if t['priority'] == 'HIGH'),
                'created_at': datetime.now().isoformat(),
                'stage': '2_continuation',
                'framework_version': '2.0.0'
            }
        }
        
        for target in targets:
            model_name = target['model_name']
            
            # Get rules for this model
            model_rules = {
                'priority': rules['retraining_priorities'].get(model_name, {}),
                'scenario_emphasis': rules['scenario_emphasis'].get(model_name, {}),
                'difficulty_adaptation': rules['difficulty_adaptation'].get(model_name, {}),
                'feature_focus': rules['feature_focus'].get(model_name, {}),
                'stopping_criteria': rules['stopping_criteria'].get(model_name, {})
            }
            
            # Prepare configuration
            config = {
                'model_name': model_name,
                'retraining_required': target['retraining_required'],
                'priority': target['priority'],
                'base_configuration': {
                    'model_type': self._infer_model_type(model_name),
                    'params_to_tune': target['params_to_tune'],
                    'initial_learning_rate': 0.001 if 'LSTM' in model_name or 'Autoencoder' in model_name else 0.1
                },
                'data_configuration': {
                    'scenario_emphasis': model_rules['scenario_emphasis'],
                    'difficulty_progression': model_rules['difficulty_adaptation'],
                    'feature_focus': model_rules['feature_focus'],
                    'data_augmentation': 'difficulty_aware_scenarios'
                },
                'training_configuration': {
                    'stopping_criteria': model_rules['stopping_criteria'],
                    'validation_strategy': 'time_series_split',
                    'class_balancing': 'adaptive_based_on_difficulty'
                },
                'expected_outcomes': {
                    'target_f1_improvement': target['expected_improvement'],
                    'minimum_acceptable_improvement': target['expected_improvement'] * 0.5,
                    'evaluation_metrics': ['f1_score', 'precision', 'recall', 'robustness_score']
                },
                'stage2_integration': {
                    'uses_difficulty_aware_scenarios': True,
                    'scenario_count': len(self.stage2_scenarios),
                    'difficulty_levels_available': list(set(s.get('difficulty_level', 'UNKNOWN') for s in self.stage2_scenarios))
                }
            }
            
            configurations['models'][model_name] = config
            
            logger.info(f"  {model_name}: Configuration prepared (Priority: {target['priority']})")
        
        self.retraining_metadata = configurations
        
        logger.info(f"✅ Prepared configurations for {len(configurations['models'])} models")
        
        return configurations
    
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name"""
        if 'LSTM' in model_name:
            return 'sequence_based_deep_learning'
        elif 'Autoencoder' in model_name:
            return 'reconstruction_based_deep_learning'
        elif 'RandomForest' in model_name or 'XGBoost' in model_name:
            return 'tree_based_ensemble'
        elif 'SVM' in model_name:
            return 'boundary_based_classifier'
        elif 'IsolationForest' in model_name or 'OneClassSVM' in model_name:
            return 'unsupervised_anomaly_detector'
        else:
            return 'unknown'
    
    def save_retraining_plans(self,
                             configurations: Optional[Dict] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Save retraining plans and metadata
        
        Args:
            configurations: Retraining configurations from prepare_retraining_configurations()
            
        Returns:
            Tuple of (plans_json_path, summary_csv_path) or (None, None)
        """
        logger.info("💾 Saving retraining plans...")
        
        if configurations is None:
            configurations = self.retraining_metadata
        
        if not configurations or 'models' not in configurations:
            logger.warning("⚠️  No configurations to save")
            return None, None
        
        # Save as JSON
        plans_json_path = self.retrained_models_dir / 'retraining_plans.json'
        with open(plans_json_path, 'w', encoding='utf-8') as f:
            json.dump(configurations, f, indent=2)
        logger.info(f"💾 Saved retraining plans JSON: {plans_json_path}")
        
        # Save summary as CSV
        summary_data = []
        for model_name, config in configurations['models'].items():
            summary_data.append({
                'model_name': model_name,
                'priority': config['priority'],
                'retraining_required': config['retraining_required'],
                'model_type': config['base_configuration']['model_type'],
                'target_f1_improvement': config['expected_outcomes']['target_f1_improvement'],
                'min_acceptable_improvement': config['expected_outcomes']['minimum_acceptable_improvement'],
                'max_epochs': config['training_configuration']['stopping_criteria'].get('max_epochs', 30),
                'scenario_count': config['stage2_integration']['scenario_count']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = self.retrained_models_dir / 'retraining_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"💾 Saved retraining summary CSV: {summary_csv_path}")
        
        return plans_json_path, summary_csv_path
    
    def generate_stage2_continuation_report(self,
                                           loading_status: Dict,
                                           targets: List[Dict],
                                           rules: Dict,
                                           configurations: Dict,
                                           output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Stage 2 continuation report
        
        Args:
            loading_status: Artifact loading status
            targets: Retraining targets
            rules: Retraining rules
            configurations: Retraining configurations
            output_path: Path to save report
            
        Returns:
            Formatted report string
        """
        logger.info("📝 Generating Stage 2 continuation report...")
        
        report = []
        report.append("="*80)
        report.append("STAGE 2 CONTINUATION: RETRAINING PREPARATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 2.0.0")
        report.append("")
        
        # Artifact Loading Summary
        report.append("ARTIFACT LOADING SUMMARY:")
        report.append(f"  - Stage 1 optimization plan: {loading_status.get('optimization_plan_loaded', False)}")
        report.append(f"  - Stage 1 failure analysis: {loading_status.get('failure_analysis_loaded', False)}")
        report.append(f"  - Stage 2 scenarios: {loading_status.get('scenarios_loaded', False)}")
        report.append(f"  - Overall status: {loading_status.get('status', 'unknown')}")
        report.append("")
        
        # Retraining Targets
        report.append("RETRAINING TARGETS:")
        report.append(f"  - Total models identified: {len(targets)}")
        report.append(f"  - High priority: {sum(1 for t in targets if t['priority'] == 'HIGH')}")
        report.append(f"  - Medium priority: {sum(1 for t in targets if t['priority'] == 'MEDIUM')}")
        report.append(f"  - Low priority: {sum(1 for t in targets if t['priority'] == 'LOW')}")
        report.append("")
        
        for target in targets:
            report.append(f"  Model: {target['model_name']}")
            report.append(f"    Priority: {target['priority']}")
            report.append(f"    Reason: {target['reason']}")
            report.append(f"    Target anomalies: {', '.join(target['target_anomalies']) if target['target_anomalies'] else 'N/A'}")
            report.append(f"    Expected F1 improvement: +{target['expected_improvement']:.3f}")
            report.append(f"    Stage 1 weakness score: {target['stage1_weakness_score']:.3f}")
            report.append(f"    Retraining required: {target['retraining_required']}")
            report.append("")
        
        # Retraining Rules Summary
        report.append("RETRAINING RULES SUMMARY:")
        report.append(f"  - Models with priority rules: {len(rules.get('retraining_priorities', {}))}")
        report.append(f"  - Models with scenario emphasis: {len(rules.get('scenario_emphasis', {}))}")
        report.append(f"  - Models with difficulty adaptation: {len(rules.get('difficulty_adaptation', {}))}")
        report.append(f"  - Models with feature focus: {len(rules.get('feature_focus', {}))}")
        report.append("")
        
        # Scenario Integration
        report.append("STAGE 2 SCENARIO INTEGRATION:")
        report.append(f"  - Total scenarios available: {len(self.stage2_scenarios)}")
        
        # Count scenarios by difficulty
        difficulty_counts = {}
        for scenario in self.stage2_scenarios:
            level = scenario.get('difficulty_level', 'UNKNOWN')
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        report.append(f"  - Difficulty distribution: {difficulty_counts}")
        
        # Count scenarios by anomaly type
        anomaly_counts = {}
        for scenario in self.stage2_scenarios:
            anomaly = scenario.get('anomaly_type', 'UNKNOWN')
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        report.append(f"  - Anomaly type distribution: {anomaly_counts}")
        report.append("")
        
        # Next Steps - Stage 2 Continuation
        report.append("NEXT STEPS FOR STAGE 2 CONTINUATION:")
        report.append("  1. Execute retraining configurations for selected models")
        report.append("  2. Validate retrained models on held-out test set")
        report.append("  3. Compare Stage 1 vs. Stage 2 vs. retrained model performance")
        report.append("  4. Proceed to Stage 3: Overall Framework Evaluation")
        report.append("  5. Generate final thesis results and comparative analysis")
        report.append("")
        
        # Scientific Notes - Stage 2 Continuation
        report.append("SCIENTIFIC NOTES:")
        report.append("  - Retraining rules are based on empirical failure patterns from Stage 1")
        report.append("  - Difficulty-aware scenarios from Stage 2 enable adaptive retraining")
        report.append("  - This preparation phase does NOT execute full retraining")
        report.append("  - Actual retraining execution occurs in Stage 2 continuation pipeline")
        report.append("  - Stage 3 will evaluate overall framework AFTER retraining is complete")
        report.append("  - All configurations are saved for reproducibility and later execution")
        report.append("")
        
        report.append("="*80)
        report.append("END OF STAGE 2 CONTINUATION REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path is None:
            output_path = self.reports_dir / 'stage2_retraining_preparation_report.txt'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved Stage 2 continuation report: {output_path}")
        
        return report_text
    
    def execute_stage2_continuation(self) -> Dict:
        """
        Execute complete Stage 2 continuation workflow
        
        Returns:
            Dictionary with execution results
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 EXECUTING STAGE 2 CONTINUATION: RETRAINING PREPARATION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'targets_identified': 0,
            'rules_created': 0,
            'configurations_prepared': 0,
            'outputs_saved': []
        }
        
        # Step 1: Load Stage 1 and Stage 2 artifacts
        loading_status = self.load_stage_artifacts()
        
        # Step 2: Identify retraining targets
        targets = self.identify_retraining_targets()
        results['targets_identified'] = len(targets)
        
        # Step 3: Create retraining rules
        rules = self.create_retraining_rules(targets)
        results['rules_created'] = len(rules.get('retraining_priorities', {}))
        
        # Step 4: Prepare retraining configurations
        configurations = self.prepare_retraining_configurations(targets, rules)
        results['configurations_prepared'] = len(configurations.get('models', {}))
        
        # Step 5: Save retraining plans
        json_path, csv_path = self.save_retraining_plans(configurations)
        if json_path:
            results['outputs_saved'].append(str(json_path))
        if csv_path:
            results['outputs_saved'].append(str(csv_path))
        
        # Step 6: Generate Stage 2 continuation report - Direct call without unused variable
        self.generate_stage2_continuation_report(
            loading_status=loading_status,
            targets=targets,
            rules=rules,
            configurations=configurations
        )
        results['outputs_saved'].append(str(self.reports_dir / 'stage2_retraining_preparation_report.txt'))
        
        results['status'] = 'complete'
        
        logger.info("\n" + "="*80)
        logger.info("✅ STAGE 2 CONTINUATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"📊 Retraining targets identified: {results['targets_identified']}")
        logger.info(f"📊 Retraining rules created: {results['rules_created']}")
        logger.info(f"📊 Configurations prepared: {results['configurations_prepared']}")
        logger.info(f"💾 Outputs saved to: {self.stage2_dir}")
        for output in results['outputs_saved']:
            logger.info(f"   - {output}")
        logger.info("="*80)
        
        return results