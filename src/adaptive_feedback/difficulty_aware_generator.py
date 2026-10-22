"""
Difficulty-Aware Generator - Stage 2 Component
Generates adaptive anomaly scenarios based on Stage 1 failure analysis

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
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DifficultyAwareGenerator:
    """
    Generates difficulty-aware anomaly scenarios based on Stage 1 feedback
    
    Research Purpose:
    -----------------
    This component:
    1. Loads failure analysis and optimization plan from Stage 1
    2. Identifies weak anomaly types and vulnerable models
    3. Defines difficulty levels (easy, medium, hard) based on model weaknesses
    4. Generates adaptive anomaly specifications targeting weak points
    5. Saves generated scenarios for Stage 2 retraining
    
    Key Innovation: Moves from fixed perturbation to adaptive difficulty
    based on empirical model weaknesses identified in Stage 1
    
    Input: Stage 1 failure analysis report + optimization plan
    Output: Difficulty-aware anomaly scenarios + metadata for retraining
    """
    
    def __init__(self, 
                 adaptive_dir: Path,
                 stage1_dir: Optional[Path] = None,
                 stage2_dir: Optional[Path] = None):
        """
        Initialize Difficulty-Aware Generator
        
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
        self.generated_scenarios_dir = self.stage2_dir / 'generated_scenarios'
        self.reports_dir = self.stage2_dir / 'reports'
        self.generated_scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.stage1_failure_analysis = {}
        self.stage1_optimization_plan = {}
        self.difficulty_profiles = {}
        self.generated_scenarios = []
        self.scenario_metadata = {}
        
        logger.info(f"✅ Initialized DifficultyAwareGenerator")
        logger.info(f"📁 Stage 1 directory: {self.stage1_dir}")
        logger.info(f"📁 Stage 2 directory: {self.stage2_dir}")
        
    def load_stage1_feedback(self,
                            failure_report_path: Optional[str] = None,
                            optimization_plan_path: Optional[str] = None) -> Dict:
        """
        Load failure analysis and optimization plan from Stage 1
        
        Args:
            failure_report_path: Path to failure_analysis_report.json
            optimization_plan_path: Path to optimization_plan.json
            
        Returns:
            Dictionary with loaded Stage 1 feedback
        """
        logger.info("📊 Loading Stage 1 feedback...")
        
        feedback = {
            'failure_analysis_loaded': False,
            'optimization_plan_loaded': False,
            'status': 'missing'
        }
        
        # Load failure analysis report
        if failure_report_path is None:
            failure_report_path = self.stage1_dir / 'reports' / 'failure_analysis_report.json'
        
        if Path(failure_report_path).exists():
            try:
                with open(failure_report_path, 'r', encoding='utf-8') as f:
                    self.stage1_failure_analysis = json.load(f)
                feedback['failure_analysis_loaded'] = True
                logger.info(f"✅ Loaded failure analysis: {self.stage1_failure_analysis.get('summary', {}).get('total_models_analyzed', 0)} models")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load failure analysis: {e}")
        else:
            logger.warning(f"⚠️  Failure analysis not found: {failure_report_path}")
            logger.info("ℹ️  Will use default difficulty profiles based on common weak anomaly types")
        
        # Load optimization plan
        if optimization_plan_path is None:
            optimization_plan_path = self.stage1_dir / 'optimization_plan.json'
        
        if Path(optimization_plan_path).exists():
            try:
                with open(optimization_plan_path, 'r', encoding='utf-8') as f:
                    self.stage1_optimization_plan = json.load(f)
                feedback['optimization_plan_loaded'] = True
                logger.info(f"✅ Loaded optimization plan: {self.stage1_optimization_plan.get('optimization_plan', {}).get('selected_models', 0)} models selected")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load optimization plan: {e}")
        else:
            logger.warning(f"⚠️  Optimization plan not found: {optimization_plan_path}")
        
        # Fix: Correct status logic
        if feedback['failure_analysis_loaded'] and feedback['optimization_plan_loaded']:
            feedback['status'] = 'complete'
        elif feedback['failure_analysis_loaded'] or feedback['optimization_plan_loaded']:
            feedback['status'] = 'partial'
        else:
            feedback['status'] = 'missing'
        
        logger.info(f"📊 Stage 1 feedback status: {feedback['status']}")
        
        # Save loading metadata
        metadata_path = self.reports_dir / 'stage1_feedback_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, indent=2)
        logger.info(f"💾 Saved feedback loading metadata: {metadata_path}")
        
        return feedback
    
    def identify_target_anomaly_types(self) -> List[Dict]:
        """
        Identify weak anomaly types from Stage 1 failure analysis
        
        Returns:
            List of target anomaly types with priority scores
        """
        logger.info("🎯 Identifying target anomaly types from Stage 1...")
        
        target_anomalies = []
        
        # Extract from failure analysis if available
        if self.stage1_failure_analysis.get('summary', {}).get('most_common_weak_anomaly'):
            most_common_weak = self.stage1_failure_analysis['summary']['most_common_weak_anomaly']
            target_anomalies.append({
                'anomaly_type': most_common_weak,
                'priority': 'HIGH',
                'reason': 'most_common_weak_across_models',
                'source': 'failure_analysis_summary'
            })
        
        # Extract from model failures
        if 'model_failures' in self.stage1_failure_analysis:
            anomaly_frequency = {}
            for model_name, failures in self.stage1_failure_analysis['model_failures'].items():
                if model_name == '_summary':
                    continue
                if 'severe_anomalies' in failures:
                    for anomaly in failures['severe_anomalies']:
                        anomaly_frequency[anomaly] = anomaly_frequency.get(anomaly, 0) + 1
            
            # Sort by frequency
            sorted_anomalies = sorted(anomaly_frequency.items(), key=lambda x: x[1], reverse=True)
            
            for anomaly, freq in sorted_anomalies[:5]:  # Top 5 weak anomalies
                if not any(t['anomaly_type'] == anomaly for t in target_anomalies):
                    target_anomalies.append({
                        'anomaly_type': anomaly,
                        'priority': 'HIGH' if freq >= 3 else 'MEDIUM',
                        'reason': f'caused_failures_in_{freq}_models',
                        'source': 'model_failure_patterns'
                    })
        
        # If no Stage 1 data, use default weak anomaly types based on literature
        if len(target_anomalies) == 0:
            logger.info("ℹ️  No Stage 1 failure data available. Using default weak anomaly types.")
            default_weak_anomalies = [
                {'anomaly_type': 'sensor_drift', 'priority': 'HIGH', 'reason': 'literature_known_hard', 'source': 'default'},
                {'anomaly_type': 'delay_shift', 'priority': 'HIGH', 'reason': 'literature_known_hard', 'source': 'default'},
                {'anomaly_type': 'gaussian_noise', 'priority': 'MEDIUM', 'reason': 'moderate_difficulty', 'source': 'default'},
                {'anomaly_type': 'stuck_sensor', 'priority': 'MEDIUM', 'reason': 'moderate_difficulty', 'source': 'default'}
            ]
            target_anomalies.extend(default_weak_anomalies)
        
        logger.info(f"📊 Identified {len(target_anomalies)} target anomaly types")
        for target in target_anomalies:
            logger.info(f"  - {target['anomaly_type']} (Priority: {target['priority']}, Reason: {target['reason']})")
        
        return target_anomalies
    
    def define_difficulty_profiles(self, 
                                   target_anomalies: List[Dict]) -> Dict:
        """
        Define difficulty profiles (easy, medium, hard) for anomaly generation
        
        Scientific Consistency:
        -----------------------
        - EASY = stronger / more obvious anomalies (easier to detect)
        - MEDIUM = moderately difficult anomalies
        - HARD = weaker / subtler / harder-to-detect anomalies
        
        Args:
            target_anomalies: List of target anomaly types from identify_target_anomaly_types()
            
        Returns:
            Dictionary with difficulty profiles
        """
        logger.info("📊 Defining difficulty profiles...")
        
        # Base difficulty profiles (scientifically consistent ordering)
        difficulty_profiles = {
            'EASY': {
                'perturbation_scale': 0.15,  # Stronger perturbations (easier to detect)
                'features_affected': 3,  # More features affected (more obvious)
                'expected_detection_difficulty_range': [0.85, 1.0],
                'description': 'Obvious anomalies with strong perturbations, easier for models to detect',
                'purpose': 'Validate baseline detection capability on clear anomaly patterns'
            },
            'MEDIUM': {
                'perturbation_scale': 0.08,  # Moderate perturbations
                'features_affected': 2,  # 2 features per sample
                'expected_detection_difficulty_range': [0.60, 0.85],
                'description': 'Moderately difficult anomalies with balanced perturbation strength',
                'purpose': 'Assess model robustness under realistic stress conditions'
            },
            'HARD': {
                'perturbation_scale': 0.03,  # Weaker perturbations (harder to detect)
                'features_affected': 1,  # Fewer features affected (more subtle)
                'expected_detection_difficulty_range': [0.30, 0.60],
                'description': 'Subtle anomalies with weak perturbations, challenging for models to detect',
                'purpose': 'Stress-test model limits on hard-to-detect anomaly patterns'
            }
        }
        
        # Adaptive perturbation adjustment based on Stage 1 vulnerability
        # Empirical adaptive heuristic: preserves EASY > MEDIUM > HARD ordering
        avg_vulnerability = self.stage1_failure_analysis.get('summary', {}).get('average_max_f1_drop', 0)
        
        if avg_vulnerability > 0.50:
            # Models show high vulnerability - apply differentiated adjustments
            logger.info("⚠️  Stage 1 shows high vulnerability (avg F1 drop > 0.50). Applying adaptive adjustments.")
            
            # Preserve meaningful gaps between levels while reducing overall intensity
            difficulty_profiles['EASY']['perturbation_scale'] = 0.12  # Reduced from 0.15
            difficulty_profiles['MEDIUM']['perturbation_scale'] = 0.06  # Reduced from 0.08
            difficulty_profiles['HARD']['perturbation_scale'] = 0.025  # Reduced from 0.03
            
            logger.info("  ✅ EASY: 0.15 → 0.12 (slight reduction)")
            logger.info("  ✅ MEDIUM: 0.08 → 0.06 (moderate reduction)")
            logger.info("  ✅ HARD: 0.03 → 0.025 (minimal reduction,保持 subtle)")
        else:
            logger.info("  ✅ Stage 1 vulnerability within acceptable range. Using base profiles.")
        
        # Assign difficulty levels to target anomalies
        for target in target_anomalies:
            anomaly_type = target['anomaly_type']
            priority = target['priority']
            
            if priority == 'HIGH':
                # High priority = generate at all difficulty levels
                target['difficulty_levels'] = ['EASY', 'MEDIUM', 'HARD']
            elif priority == 'MEDIUM':
                # Medium priority = generate at medium and hard
                target['difficulty_levels'] = ['MEDIUM', 'HARD']
            else:
                # Low priority = generate only at hard level
                target['difficulty_levels'] = ['HARD']
        
        self.difficulty_profiles = difficulty_profiles
        
        logger.info(f"📊 Defined {len(difficulty_profiles)} difficulty profiles")
        for level, profile in difficulty_profiles.items():
            logger.info(f"  {level}: perturbation={profile['perturbation_scale']}, "
                       f"features={profile['features_affected']}, "
                       f"expected_difficulty={profile['expected_detection_difficulty_range']}")
        
        return difficulty_profiles
    
    def generate_difficulty_aware_scenarios(self,
                                           target_anomalies: List[Dict],
                                           difficulty_profiles: Dict,
                                           num_scenarios_per_level: int = 3) -> List[Dict]:
        """
        Generate difficulty-aware anomaly scenarios
        
        Args:
            target_anomalies: List of target anomaly types
            difficulty_profiles: Difficulty profiles from define_difficulty_profiles()
            num_scenarios_per_level: Number of scenarios per difficulty level (default: 3)
            
        Returns:
            List of generated scenario specifications
        """
        logger.info("🔬 Generating difficulty-aware anomaly scenarios...")
        
        scenarios = []
        scenario_id = 0
        
        # Extract critical features from Stage 1 if available
        critical_features = []
        if 'feature_correlations' in self.stage1_failure_analysis:
            for model_name, features in self.stage1_failure_analysis['feature_correlations'].items():
                if 'top_features' in features:
                    critical_features.extend(features['top_features'])
            critical_features = list(set(critical_features))[:10]  # Top 10 unique
        
        for target in target_anomalies:
            anomaly_type = target['anomaly_type']
            
            for difficulty_level in target.get('difficulty_levels', ['MEDIUM']):
                profile = difficulty_profiles.get(difficulty_level, difficulty_profiles['MEDIUM'])
                
                for i in range(num_scenarios_per_level):
                    scenario_id += 1
                    
                    # Include critical feature guidance in each scenario
                    scenario = {
                        'scenario_id': f'S2_{scenario_id:03d}',
                        'anomaly_type': anomaly_type,
                        'difficulty_level': difficulty_level,
                        'priority': target['priority'],
                        'perturbation_scale': profile['perturbation_scale'],
                        'features_affected': profile['features_affected'],
                        'expected_detection_difficulty_range': profile['expected_detection_difficulty_range'],
                        'generation_purpose': profile['purpose'],
                        'based_on_stage1': target['source'],
                        'rationale': target['reason'],
                        'critical_features_involved': critical_features if critical_features else [],
                        'metadata': {
                            'created_at': datetime.now().isoformat(),
                            'framework_version': '2.0.0',
                            'stage': 2,
                            'component': 'difficulty_aware_generator'
                        }
                    }
                    
                    scenarios.append(scenario)
                    logger.info(f"  Generated scenario {scenario['scenario_id']}: "
                               f"{anomaly_type} @ {difficulty_level} "
                               f"(scale={profile['perturbation_scale']}, features={profile['features_affected']})")
        
        self.generated_scenarios = scenarios
        
        logger.info(f"✅ Generated {len(scenarios)} difficulty-aware scenarios")
        
        return scenarios
    
    def save_generated_scenarios(self, 
                                scenarios: Optional[List[Dict]] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Save generated scenarios to Stage 2 output directories
        
        Args:
            scenarios: List of scenarios (if None, uses self.generated_scenarios)
            
        Returns:
            Tuple of (scenarios_json_path, scenarios_csv_path) or (None, None)
        """
        logger.info("💾 Saving generated scenarios...")
        
        if scenarios is None:
            scenarios = self.generated_scenarios
        
        if len(scenarios) == 0:
            logger.warning("⚠️  No scenarios to save")
            return None, None
        
        # Save as JSON
        scenarios_json_path = self.generated_scenarios_dir / 'difficulty_aware_scenarios.json'
        with open(scenarios_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.scenario_metadata,
                'scenarios': scenarios
            }, f, indent=2)
        logger.info(f"💾 Saved scenarios JSON: {scenarios_json_path}")
        
        # Save as CSV for easy viewing
        scenarios_df = pd.DataFrame(scenarios)
        scenarios_csv_path = self.generated_scenarios_dir / 'difficulty_aware_scenarios.csv'
        scenarios_df.to_csv(scenarios_csv_path, index=False)
        logger.info(f"💾 Saved scenarios CSV: {scenarios_csv_path}")
        
        return scenarios_json_path, scenarios_csv_path
    
    def generate_scenario_metadata(self,
                                  target_anomalies: List[Dict],
                                  difficulty_profiles: Dict,
                                  scenarios: List[Dict]) -> Dict:
        """
        Generate metadata describing the generated scenarios
        
        Args:
            target_anomalies: List of target anomaly types
            difficulty_profiles: Difficulty profiles
            scenarios: Generated scenarios
            
        Returns:
            Dictionary with scenario metadata
        """
        logger.info("📝 Generating scenario metadata...")
        
        # Count scenarios by difficulty
        difficulty_counts = {}
        for scenario in scenarios:
            level = scenario['difficulty_level']
            difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        
        # Count scenarios by anomaly type
        anomaly_counts = {}
        for scenario in scenarios:
            anomaly = scenario['anomaly_type']
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        # Identify critical features from Stage 1
        critical_features = []
        if 'feature_correlations' in self.stage1_failure_analysis:
            for model_name, features in self.stage1_failure_analysis['feature_correlations'].items():
                if 'top_features' in features:
                    critical_features.extend(features['top_features'])
            critical_features = list(set(critical_features))[:10]  # Top 10 unique
        
        metadata = {
            'total_scenarios': len(scenarios),
            'difficulty_distribution': difficulty_counts,
            'anomaly_type_distribution': anomaly_counts,
            'target_models': self.stage1_optimization_plan.get('optimization_plan', {}).get('models_to_optimize', []),
            'target_anomaly_types': list(anomaly_counts.keys()),
            'difficulty_levels': list(difficulty_counts.keys()),
            'critical_features': critical_features,
            'generation_priorities': {
                'focus_on_weak_anomalies': True,
                'focus_on_hard_to_detect': True,
                'focus_on_critical_features': len(critical_features) > 0,
                'adaptive_difficulty': True
            },
            'stage1_integration': {
                'failure_analysis_used': len(self.stage1_failure_analysis) > 0,
                'optimization_plan_used': len(self.stage1_optimization_plan) > 0,
                'weakness_guided': True
            },
            'next_stage': {
                'stage': '2_continuation',
                'component': 'retraining_manager',
                'expected_action': 'retrain_models_with_generated_scenarios',
                'note': 'Retraining is part of Stage 2, not a separate Stage 3'
            },
            'created_at': datetime.now().isoformat()
        }
        
        self.scenario_metadata = metadata
        
        logger.info(f"📊 Metadata summary:")
        logger.info(f"   Total scenarios: {metadata['total_scenarios']}")
        logger.info(f"   Difficulty distribution: {metadata['difficulty_distribution']}")
        logger.info(f"   Target models: {metadata['target_models']}")
        logger.info(f"   Critical features: {len(metadata['critical_features'])} features identified")
        
        return metadata
    
    def generate_stage2_report(self,
                              feedback: Dict,
                              target_anomalies: List[Dict],
                              difficulty_profiles: Dict,
                              scenarios: List[Dict],
                              metadata: Dict,
                              output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Stage 2 scenario generation report
        
        Args:
            feedback: Stage 1 feedback loading status
            target_anomalies: Target anomaly types
            difficulty_profiles: Difficulty profiles
            scenarios: Generated scenarios
            metadata: Scenario metadata
            output_path: Path to save report
            
        Returns:
            Formatted report string
        """
        logger.info("📝 Generating Stage 2 scenario generation report...")
        
        report = []
        report.append("="*80)
        report.append("STAGE 2: DIFFICULTY-AWARE ANOMALY SCENARIO GENERATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework Version: 2.0.0")
        report.append("")
        
        # Stage 1 Integration Summary
        report.append("STAGE 1 INTEGRATION SUMMARY:")
        report.append(f"  - Failure analysis loaded: {feedback.get('failure_analysis_loaded', False)}")
        report.append(f"  - Optimization plan loaded: {feedback.get('optimization_plan_loaded', False)}")
        report.append(f"  - Loading status: {feedback.get('status', 'unknown')}")
        report.append("")
        
        # Target Anomaly Types
        report.append("TARGET ANOMALY TYPES:")
        for target in target_anomalies:
            report.append(f"  - {target['anomaly_type']}")
            report.append(f"    Priority: {target['priority']}")
            report.append(f"    Difficulty levels: {', '.join(target.get('difficulty_levels', []))}")
            report.append(f"    Reason: {target['reason']}")
            report.append(f"    Source: {target['source']}")
        report.append("")
        
        # Difficulty Profiles
        report.append("DIFFICULTY PROFILES:")
        for level, profile in difficulty_profiles.items():
            report.append(f"  {level}:")
            report.append(f"    Perturbation scale: {profile['perturbation_scale']}")
            report.append(f"    Features affected: {profile['features_affected']}")
            report.append(f"    Expected detection difficulty: {profile['expected_detection_difficulty_range']}")
            report.append(f"    Description: {profile['description']}")
            report.append(f"    Purpose: {profile['purpose']}")
        report.append("")
        
        # Scenario Summary
        report.append("GENERATED SCENARIOS SUMMARY:")
        report.append(f"  - Total scenarios: {metadata['total_scenarios']}")
        report.append(f"  - Difficulty distribution: {metadata['difficulty_distribution']}")
        report.append(f"  - Anomaly type distribution: {metadata['anomaly_type_distribution']}")
        report.append(f"  - Target models: {metadata['target_models']}")
        report.append(f"  - Critical features: {len(metadata['critical_features'])} identified")
        report.append("")
        
        # Generation Priorities
        report.append("GENERATION PRIORITIES:")
        priorities = metadata['generation_priorities']
        report.append(f"  - Focus on weak anomalies: {priorities['focus_on_weak_anomalies']}")
        report.append(f"  - Focus on hard-to-detect: {priorities['focus_on_hard_to_detect']}")
        report.append(f"  - Focus on critical features: {priorities['focus_on_critical_features']}")
        report.append(f"  - Adaptive difficulty: {priorities['adaptive_difficulty']}")
        report.append("")
        
        # Next steps for Stage 2 continuation
        report.append("NEXT STEPS FOR STAGE 2 CONTINUATION:")
        report.append("  1. Load generated scenarios from Stage 2 scenario generation")
        report.append("  2. Prepare retraining data based on scenario specifications")
        report.append("  3. Retrain target models with difficulty-aware anomalies (Stage 2 continuation)")
        report.append("  4. Validate improvement on held-out test set")
        report.append("  5. Compare Stage 1 vs. Stage 2 F1 scores after retraining")
        report.append("")
        
        report.append("="*80)
        report.append("END OF STAGE 2 REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path is None:
            output_path = self.reports_dir / 'stage2_scenario_generation_report.txt'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved Stage 2 report: {output_path}")
        
        return report_text
    
    def execute_stage2_generation(self,
                                 num_scenarios_per_level: int = 3) -> Dict:
        """
        Execute complete Stage 2 scenario generation workflow
        
        Args:
            num_scenarios_per_level: Number of scenarios per difficulty level
            
        Returns:
            Dictionary with execution results
        """
        logger.info("\n" + "="*80)
        logger.info("🎯 EXECUTING STAGE 2: DIFFICULTY-AWARE SCENARIO GENERATION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'scenarios_generated': 0,
            'outputs_saved': []
        }
        
        # Step 1: Load Stage 1 feedback
        feedback = self.load_stage1_feedback()
        
        # Step 2: Identify target anomaly types
        target_anomalies = self.identify_target_anomaly_types()
        
        # Step 3: Define difficulty profiles
        difficulty_profiles = self.define_difficulty_profiles(target_anomalies)
        
        # Step 4: Generate scenarios
        scenarios = self.generate_difficulty_aware_scenarios(
            target_anomalies=target_anomalies,
            difficulty_profiles=difficulty_profiles,
            num_scenarios_per_level=num_scenarios_per_level
        )
        
        # Step 5: Generate metadata
        metadata = self.generate_scenario_metadata(
            target_anomalies=target_anomalies,
            difficulty_profiles=difficulty_profiles,
            scenarios=scenarios
        )
        
        # Step 6: Save scenarios
        json_path, csv_path = self.save_generated_scenarios(scenarios)
        if json_path:
            results['outputs_saved'].append(str(json_path))
        if csv_path:
            results['outputs_saved'].append(str(csv_path))
        
        # Step 7: Generate report
        report = self.generate_stage2_report(
            feedback=feedback,
            target_anomalies=target_anomalies,
            difficulty_profiles=difficulty_profiles,
            scenarios=scenarios,
            metadata=metadata
        )
        results['outputs_saved'].append(str(self.reports_dir / 'stage2_scenario_generation_report.txt'))
        
        results['status'] = 'complete'
        results['scenarios_generated'] = len(scenarios)
        
        logger.info("\n" + "="*80)
        logger.info("✅ STAGE 2 COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"📊 Scenarios generated: {results['scenarios_generated']}")
        logger.info(f"💾 Outputs saved to: {self.stage2_dir}")
        for output in results['outputs_saved']:
            logger.info(f"   - {output}")
        logger.info("="*80)
        
        return results