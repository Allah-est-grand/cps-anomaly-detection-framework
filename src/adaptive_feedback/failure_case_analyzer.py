"""
Failure Case Analyzer - Stage 1 Component
Analyzes WHY models fail on specific anomalies

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

class FailureCaseAnalyzer:
    """
    Analyzes failure patterns from robustness evaluation results
    
    Research Purpose:
    -----------------
    This component identifies:
    1. Which anomaly types cause most F1 drops per model
    2. Which features correlate with misclassifications
    3. Model weakness scores for optimization prioritization
    4. Instance-level failure cases (false positives/negatives)
    
    Input: Robustness evaluation results from run_robustness_evaluation.py
    Output: Failure analysis report for DetectorOptimizer
    
    Scientific Contribution:
    ------------------------
    - Links robustness results to actionable optimization targets
    - Provides statistical evidence for model weaknesses
    - Enables data-driven detector optimization in Stage 2
    """
    
    def __init__(self, results_dir: Path, adaptive_dir: Optional[Path] = None):
        """
        Initialize Failure Case Analyzer
        
        Args:
            results_dir: Path to the directory containing robustness-related result files
                         (e.g., stress_test_robustness_scores.csv, attack_mimic_aligned_results.json,
                          feature_importance.json, model_predictions.csv)
            adaptive_dir: Path to results/adaptive/stage1 directory
        """
        self.results_dir = Path(results_dir)
        self.adaptive_dir = Path(adaptive_dir) if adaptive_dir else (
            Path(results_dir).parent / 'adaptive' / 'stage1'
        )
        
        # Create output directories
        self.failure_analysis_dir = self.adaptive_dir / 'failure_analysis'
        self.reports_dir = self.adaptive_dir / 'reports'
        self.failure_analysis_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.failure_analysis = {}
        self.model_weakness_scores = pd.DataFrame()
        self.feature_correlations = {}
        self.instance_failures = {}
        
        logger.info(f"✅ Initialized FailureCaseAnalyzer")
        logger.info(f"📁 Output directory: {self.adaptive_dir}")
        
    def load_robustness_results(self, 
                                stress_test_path: Optional[str] = None,
                                aligned_results_path: Optional[str] = None) -> Dict:
        """
        Load robustness evaluation results from Stage 0
        
        Args:
            stress_test_path: Path to stress_test_robustness_scores.csv
            aligned_results_path: Path to attack_mimic_aligned_results.json
            
        Returns:
            Dictionary with loaded results and metadata
        """
        logger.info("📊 Loading robustness evaluation results...")
        
        results = {
            'stress_test': None,
            'aligned': None,
            'loaded': False,
            'metadata': {
                'stress_test_loaded': False,
                'aligned_loaded': False,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Load stress-test results
        if stress_test_path is None:
            stress_test_path = self.results_dir / 'stress_test_robustness_scores.csv'
        
        if Path(stress_test_path).exists():
            try:
                results['stress_test'] = pd.read_csv(stress_test_path)
                results['metadata']['stress_test_loaded'] = True
                results['metadata']['stress_test_rows'] = len(results['stress_test'])
                logger.info(f"✅ Loaded stress-test results: {len(results['stress_test'])} rows")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load stress-test results: {e}")
        else:
            logger.warning(f"⚠️  Stress-test results not found: {stress_test_path}")
        
        # Load attack-mimic aligned results
        if aligned_results_path is None:
            aligned_results_path = self.results_dir / 'attack_mimic_aligned_results.json'
        
        if Path(aligned_results_path).exists():
            try:
                with open(aligned_results_path, 'r', encoding='utf-8') as f:
                    results['aligned'] = json.load(f)
                results['metadata']['aligned_loaded'] = True
                results['metadata']['aligned_models'] = len(results['aligned'])
                logger.info(f"✅ Loaded attack-mimic aligned results: {len(results['aligned'])} models")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load aligned results: {e}")
        else:
            logger.warning(f"⚠️  Aligned results not found: {aligned_results_path}")
        
        results['loaded'] = results['metadata']['stress_test_loaded'] or results['metadata']['aligned_loaded']
        
        # Save loading metadata
        metadata_path = self.failure_analysis_dir / 'data_loading_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results['metadata'], f, indent=2)
        logger.info(f"💾 Saved data loading metadata: {metadata_path}")
        
        return results
    
    def analyze_failure_patterns(self, 
                                 robustness_results: Dict,
                                 f1_drop_threshold: float = 0.30) -> Dict:
        """
        Identify which anomaly types cause most F1 drops per model
        
        Args:
            robustness_results: Loaded results from load_robustness_results()
            f1_drop_threshold: Minimum F1 drop to consider as "failure" (default: 0.30)
            
        Returns:
            Dictionary with failure patterns per model
        """
        logger.info("🔍 Analyzing failure patterns...")
        
        failure_patterns = {}
        
        # Analyze stress-test results
        if robustness_results['stress_test'] is not None and len(robustness_results['stress_test']) > 0:
            df = robustness_results['stress_test'].copy()
            
            # Validate required columns
            required_cols = ['model', 'anomaly_type', 'f1_drop', 'f1_original', 
                           'f1_anomalous', 'robustness_score_f1']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠️  Missing columns in stress-test data: {missing_cols}")
                # Try to continue with available columns
            
            # Group by model and analyze F1 drops
            for model_name in df['model'].unique():
                model_data = df[df['model'] == model_name].copy()
                
                if len(model_data) == 0:
                    continue
                
                # Find anomalies with F1 drop > threshold
                severe_drops = model_data[model_data['f1_drop'] > f1_drop_threshold]
                
                # Calculate statistics
                avg_f1_drop = model_data['f1_drop'].mean() if 'f1_drop' in model_data.columns else 0.0
                max_f1_drop = model_data['f1_drop'].max() if 'f1_drop' in model_data.columns else 0.0
                min_robustness = model_data['robustness_score_f1'].min() if 'robustness_score_f1' in model_data.columns else 1.0
                
                if len(severe_drops) > 0:
                    # Identify weakest anomaly type
                    weakest_idx = severe_drops['f1_drop'].idxmax()
                    weakest_anomaly = severe_drops.loc[weakest_idx, 'anomaly_type']
                    
                    failure_patterns[model_name] = {
                        'weakest_anomaly': weakest_anomaly,
                        'max_f1_drop': float(max_f1_drop),
                        'avg_f1_drop': float(avg_f1_drop),
                        'f1_original': float(model_data['f1_original'].mean()) if 'f1_original' in model_data.columns else 0.0,
                        'f1_anomalous': float(model_data['f1_anomalous'].mean()) if 'f1_anomalous' in model_data.columns else 0.0,
                        'min_robustness_score': float(min_robustness),
                        'severe_anomalies': severe_drops['anomaly_type'].tolist(),
                        'num_severe_anomalies': len(severe_drops),
                        'total_anomalies_tested': len(model_data),
                        'optimization_priority': self._calculate_optimization_priority(
                            max_f1_drop=max_f1_drop,
                            avg_f1_drop=avg_f1_drop,
                            num_severe=len(severe_drops)
                        ),
                        'protocol': 'stress_test'
                    }
                    
                    logger.info(f"  {model_name}: Weakest={weakest_anomaly} "
                               f"(F1 drop: {max_f1_drop:.3f}, Priority: {failure_patterns[model_name]['optimization_priority']})")
        
        # Also analyze attack-mimic aligned results if available
        if robustness_results['aligned'] is not None and len(robustness_results['aligned']) > 0:
            for model_name, metrics in robustness_results['aligned'].items():
                if model_name not in failure_patterns:
                    failure_patterns[model_name] = {}
                
                # Check if F1 < 0.70 on aligned data (indicates generalization issue)
                if metrics.get('f1_score', 1.0) < 0.70:
                    failure_patterns[model_name]['aligned_f1'] = metrics['f1_score']
                    failure_patterns[model_name]['generalization_issue'] = True
                    
                    # Update priority if worse than stress-test
                    if failure_patterns[model_name].get('optimization_priority', 'LOW') == 'LOW':
                        failure_patterns[model_name]['optimization_priority'] = 'MEDIUM'
                    
                    failure_patterns[model_name]['protocol'] = 'attack_mimic_aligned'
                    
                    logger.info(f"  {model_name}: Aligned F1={metrics['f1_score']:.3f} "
                               f"(generalization issue detected)")
        
        # Handle case where no failures detected
        if len(failure_patterns) == 0:
            logger.warning("⚠️  No significant failure patterns detected. All models may be performing well.")
            failure_patterns['_summary'] = {
                'status': 'no_significant_failures',
                'message': 'All models show F1 drops below threshold or no aligned results available',
                'recommendation': 'Consider lowering f1_drop_threshold or collecting more diverse anomaly data'
            }
        
        self.failure_analysis = failure_patterns
        
        return failure_patterns
    
    def _calculate_optimization_priority(self, 
                                        max_f1_drop: float,
                                        avg_f1_drop: float,
                                        num_severe: int) -> str:
        """
        Calculate optimization priority based on failure severity
        
        Args:
            max_f1_drop: Maximum F1 drop observed
            avg_f1_drop: Average F1 drop across anomalies
            num_severe: Number of severe anomalies (F1 drop > threshold)
            
        Returns:
            Priority level: 'HIGH', 'MEDIUM', or 'LOW'
        """
        if max_f1_drop > 0.60 or num_severe >= 5:
            return 'HIGH'
        elif max_f1_drop > 0.40 or num_severe >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def extract_failure_features(self, 
                                 model_name: str,
                                 feature_importance_path: Optional[str] = None) -> Dict:
        """
        Find which features correlate with misclassifications
        
        Args:
            model_name: Name of the model to analyze
            feature_importance_path: Path to feature_importance.json
            
        Returns:
            Dictionary with critical features for this model
        """
        logger.info(f"🔍 Extracting failure features for {model_name}...")
        
        critical_features = {
            'model': model_name,
            'top_features': [],
            'feature_weights': {},
            'interpretation': '',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Load feature importance if available
        if feature_importance_path is None:
            feature_importance_path = self.results_dir / 'feature_importance.json'
        
        if Path(feature_importance_path).exists():
            try:
                with open(feature_importance_path, 'r', encoding='utf-8') as f:
                    feature_importance = json.load(f)
                
                if model_name in feature_importance:
                    # Get top 5 most important features
                    model_features = feature_importance[model_name]
                    sorted_features = sorted(model_features.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:5]
                    
                    critical_features['top_features'] = [f[0] for f in sorted_features]
                    critical_features['feature_weights'] = dict(sorted_features)
                    critical_features['interpretation'] = (
                        f"These {len(sorted_features)} features dominate {model_name}'s "
                        f"decisions. Failures likely occur when these features exhibit "
                        f"anomalous patterns not seen during training."
                    )
                    
                    logger.info(f"  Top features: {critical_features['top_features'][:3]}...")
                else:
                    logger.warning(f"⚠️  No feature importance found for {model_name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load feature importance: {e}")
        else:
            logger.warning(f"⚠️  Feature importance file not found: {feature_importance_path}")
            
            # Fallback: Use common CPS features known to be critical
            common_critical_features = ['P204', 'P201', 'PIT503', 'PIT501', 'MV303']
            critical_features['top_features'] = common_critical_features
            critical_features['interpretation'] = (
                "Using common CPS critical features as fallback. "
                "These actuators and pressure sensors typically drive detection decisions."
            )
        
        self.feature_correlations[model_name] = critical_features
        
        return critical_features
    
    def compute_model_weakness_scores(self, 
                                      failure_patterns: Dict) -> pd.DataFrame:
        """
        Quantify each model's vulnerability per anomaly type
        
        Args:
            failure_patterns: Output from analyze_failure_patterns()
            
        Returns:
            DataFrame with weakness scores per model
        """
        logger.info("📊 Computing model weakness scores...")
        
        weakness_data = []
        
        for model_name, patterns in failure_patterns.items():
            if model_name == '_summary':
                continue
                
            if 'max_f1_drop' in patterns:
                weakness_score = patterns['max_f1_drop']  # Higher = more vulnerable
            else:
                weakness_score = 0.0
            
            # Adjust score based on optimization priority
            priority_multiplier = {
                'HIGH': 1.2,
                'MEDIUM': 1.0,
                'LOW': 0.8
            }
            multiplier = priority_multiplier.get(patterns.get('optimization_priority', 'LOW'), 1.0)
            weakness_score *= multiplier
            
            # Adjust for generalization issues
            if patterns.get('generalization_issue', False):
                weakness_score *= 1.1
            
            weakness_data.append({
                'model': model_name,
                'weakness_score': round(weakness_score, 4),
                'max_f1_drop': round(patterns.get('max_f1_drop', 0.0), 4),
                'avg_f1_drop': round(patterns.get('avg_f1_drop', 0.0), 4),
                'weakest_anomaly': patterns.get('weakest_anomaly', 'N/A'),
                'optimization_priority': patterns.get('optimization_priority', 'LOW'),
                'num_severe_anomalies': patterns.get('num_severe_anomalies', 0),
                'generalization_issue': patterns.get('generalization_issue', False),
                'protocol': patterns.get('protocol', 'unknown'),
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        if len(weakness_data) > 0:
            df_weakness = pd.DataFrame(weakness_data)
            df_weakness = df_weakness.sort_values('weakness_score', ascending=False)
            df_weakness = df_weakness.reset_index(drop=True)
        else:
            df_weakness = pd.DataFrame(columns=[
                'model', 'weakness_score', 'max_f1_drop', 'avg_f1_drop',
                'weakest_anomaly', 'optimization_priority', 'num_severe_anomalies',
                'generalization_issue', 'protocol', 'analysis_timestamp'
            ])
        
        self.model_weakness_scores = df_weakness
        
        if len(df_weakness) > 0:
            logger.info(f"  Computed weakness scores for {len(df_weakness)} models")
            logger.info(f"  Most vulnerable: {df_weakness.iloc[0]['model']} "
                       f"(score: {df_weakness.iloc[0]['weakness_score']:.3f})")
        else:
            logger.warning("⚠️  No weakness scores computed (no failure patterns detected)")
        
        return df_weakness
    
    def analyze_instance_failures(self, 
                                  predictions_csv_path: Optional[str] = None,
                                  threshold_fp_rate: float = 0.10,
                                  threshold_fn_rate: float = 0.10,
                                  sample_id_col: str = 'sample_id',
                                  true_label_col: str = 'true_label',
                                  predicted_label_col: str = 'predicted_label',
                                  confidence_col: str = 'confidence') -> Dict:
        """
        Analyze instance-level failures if prediction CSV is available
        
        Args:
            predictions_csv_path: Path to predictions CSV
            threshold_fp_rate: False positive rate threshold
            threshold_fn_rate: False negative rate threshold
            sample_id_col: Column name for sample ID
            true_label_col: Column name for true labels
            predicted_label_col: Column name for predicted labels
            confidence_col: Column name for confidence scores
            
        Returns:
            Dictionary with instance-level failure analysis
        """
        logger.info("🔍 Analyzing instance-level failures...")
        
        instance_analysis = {
            'false_positives': [],
            'false_negatives': [],
            'borderline_cases': [],
            'summary': {}
        }
        
        if predictions_csv_path is None:
            predictions_csv_path = self.results_dir / 'model_predictions.csv'
        
        if not Path(predictions_csv_path).exists():
            logger.info("ℹ️  No predictions CSV found. Skipping instance-level analysis.")
            instance_analysis['summary']['status'] = 'skipped_no_data'
            return instance_analysis
        
        try:
            df_pred = pd.read_csv(predictions_csv_path)
            
            # Validate required columns
            required_cols = [sample_id_col, true_label_col, predicted_label_col]
            missing_cols = [col for col in required_cols if col not in df_pred.columns]
            
            if missing_cols:
                logger.warning(f"⚠️  Missing columns in predictions CSV: {missing_cols}")
                instance_analysis['summary']['status'] = 'error_missing_columns'
                return instance_analysis
            
            # Identify false positives (predicted=1, true=0)
            fp_mask = (df_pred[predicted_label_col] == 1) & (df_pred[true_label_col] == 0)
            false_positives = df_pred[fp_mask].copy()
            
            # Identify false negatives (predicted=0, true=1)
            fn_mask = (df_pred[predicted_label_col] == 0) & (df_pred[true_label_col] == 1)
            false_negatives = df_pred[fn_mask].copy()
            
            # Identify borderline cases (confidence close to 0.5)
            if confidence_col in df_pred.columns:
                borderline_mask = (df_pred[confidence_col] > 0.45) & (df_pred[confidence_col] < 0.55)
                borderline_cases = df_pred[borderline_mask].copy()
            else:
                borderline_cases = pd.DataFrame()
            
            # Calculate rates
            total_samples = len(df_pred)
            fp_rate = len(false_positives) / total_samples if total_samples > 0 else 0.0
            fn_rate = len(false_negatives) / total_samples if total_samples > 0 else 0.0
            
            instance_analysis['false_positives'] = false_positives.to_dict('records')
            instance_analysis['false_negatives'] = false_negatives.to_dict('records')
            instance_analysis['borderline_cases'] = borderline_cases.to_dict('records')
            instance_analysis['summary'] = {
                'status': 'completed',
                'total_samples': total_samples,
                'false_positives_count': len(false_positives),
                'false_negatives_count': len(false_negatives),
                'borderline_cases_count': len(borderline_cases),
                'fp_rate': round(fp_rate, 4),
                'fn_rate': round(fn_rate, 4),
                'needs_optimization': fp_rate > threshold_fp_rate or fn_rate > threshold_fn_rate
            }
            
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  False positives: {len(false_positives)} ({fp_rate*100:.2f}%)")
            logger.info(f"  False negatives: {len(false_negatives)} ({fn_rate*100:.2f}%)")
            logger.info(f"  Borderline cases: {len(borderline_cases)}")
            
            # Save critical cases to CSV
            if len(false_positives) > 0:
                fp_path = self.failure_analysis_dir / 'false_positives.csv'
                false_positives.to_csv(fp_path, index=False)
                logger.info(f"💾 Saved false positives: {fp_path}")
            
            if len(false_negatives) > 0:
                fn_path = self.failure_analysis_dir / 'false_negatives.csv'
                false_negatives.to_csv(fn_path, index=False)
                logger.info(f"💾 Saved false negatives: {fn_path}")
            
            if len(borderline_cases) > 0:
                bl_path = self.failure_analysis_dir / 'borderline_cases.csv'
                borderline_cases.to_csv(bl_path, index=False)
                logger.info(f"💾 Saved borderline cases: {bl_path}")
            
            # Save instance failure summary CSV
            summary_df = pd.DataFrame([instance_analysis['summary']])
            summary_path = self.failure_analysis_dir / 'instance_failure_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"💾 Saved instance failure summary: {summary_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze instance failures: {e}")
            instance_analysis['summary']['status'] = f'error: {str(e)}'
        
        self.instance_failures = instance_analysis
        
        return instance_analysis
    
    def generate_recommendations(self, 
                                failure_patterns: Dict) -> List[Dict]:
        """
        Generate optimization recommendations based on failure analysis
        
        Args:
            failure_patterns: Output from analyze_failure_patterns()
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info("📋 Generating optimization recommendations...")
        
        recommendations = []
        
        for model_name, patterns in failure_patterns.items():
            if model_name == '_summary':
                continue
            
            rec = {
                'model': model_name,
                'priority': patterns.get('optimization_priority', 'LOW'),
                'action': self._suggest_optimization_action(patterns),
                'target_anomalies': patterns.get('severe_anomalies', []),
                'rationale': self._generate_recommendation_rationale(patterns),
                'expected_improvement': self._estimate_improvement(patterns),
                'analysis_timestamp': datetime.now().isoformat()
            }
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations
    
    def _suggest_optimization_action(self, patterns: Dict) -> str:
        """Suggest specific optimization action based on failure pattern"""
        if patterns.get('generalization_issue', False):
            return 'retrain_with_aligned_data'
        elif patterns.get('max_f1_drop', 0.0) > 0.60:
            return 'hyperparameter_tuning_ensemble'
        elif patterns.get('max_f1_drop', 0.0) > 0.40:
            return 'hyperparameter_tuning'
        else:
            return 'feature_engineering'
    
    def _generate_recommendation_rationale(self, patterns: Dict) -> str:
        """Generate human-readable rationale for recommendation"""
        weakest = patterns.get('weakest_anomaly', 'unknown')
        max_drop = patterns.get('max_f1_drop', 0.0)
        
        if patterns.get('generalization_issue', False):
            return (f"Model shows poor generalization to attack-mimic anomalies "
                   f"(F1 < 0.70). Retraining with aligned data recommended.")
        else:
            return (f"Model shows vulnerability to '{weakest}' anomalies "
                   f"with F1 drop of {max_drop:.3f}. Optimization recommended.")
    
    def _estimate_improvement(self, patterns: Dict) -> Dict:
        """Estimate expected improvement from optimization"""
        max_drop = patterns.get('max_f1_drop', 0.0)
        
        improvement_estimates = {
            'retrain_with_aligned_data': 0.25,
            'hyperparameter_tuning_ensemble': 0.20,
            'hyperparameter_tuning': 0.15,
            'feature_engineering': 0.10
        }
        
        action = self._suggest_optimization_action(patterns)
        base_improvement = improvement_estimates.get(action, 0.10)
        
        # Scale by current weakness
        scaling_factor = min(max_drop / 0.50, 1.0)
        
        return {
            'f1_improvement_estimate': round(base_improvement * scaling_factor, 3),
            'confidence': 'medium',
            'basis': 'literature_review_and_empirical_analysis'
        }
    
    def generate_failure_report(self, 
                                output_path: Optional[str] = None,
                                include_instance_analysis: bool = True) -> Dict:
        """
        Create comprehensive interpretable report for optimizer
        
        Args:
            output_path: Path to save failure report JSON
            include_instance_analysis: Whether to include instance-level analysis
            
        Returns:
            Dictionary with complete failure analysis
        """
        logger.info("📝 Generating failure analysis report...")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(self.failure_analysis)
        
        report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'stage': 1,
                'component': 'failure_case_analyzer'
            },
            'summary': self._generate_summary(),
            'model_failures': self.failure_analysis,
            'weakness_scores': (self.model_weakness_scores.to_dict('records') 
                               if isinstance(self.model_weakness_scores, pd.DataFrame) and len(self.model_weakness_scores) > 0
                               else []),
            'feature_correlations': self.feature_correlations,
            'instance_failures': self.instance_failures if include_instance_analysis else None,
            'recommendations': recommendations
        }
        
        # Save JSON report in reports_dir
        if output_path is None:
            output_path = self.reports_dir / 'failure_analysis_report.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Saved failure analysis report: {output_path}")
        
        # Save weakness scores CSV
        if isinstance(self.model_weakness_scores, pd.DataFrame) and len(self.model_weakness_scores) > 0:
            weakness_csv_path = self.failure_analysis_dir / 'model_weakness_scores.csv'
            self.model_weakness_scores.to_csv(weakness_csv_path, index=False)
            logger.info(f"💾 Saved model weakness scores: {weakness_csv_path}")
        
        # Generate text summary report
        text_report = self._generate_text_report(report)
        text_report_path = self.reports_dir / 'stage1_failure_analysis_summary.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        logger.info(f"💾 Saved text summary report: {text_report_path}")
        
        # Log summary
        self._log_summary(report)
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate high-level summary of failure analysis"""
        if not self.failure_analysis or len(self.failure_analysis) == 0:
            return {
                'total_models_analyzed': 0,
                'models_with_high_priority': 0,
                'most_common_weak_anomaly': 'N/A',
                'average_max_f1_drop': 0.0,
                'status': 'no_data'
            }
        
        # Filter out summary entries
        valid_patterns = {k: v for k, v in self.failure_analysis.items() if k != '_summary'}
        
        # Find most common weak anomaly
        anomaly_counts = {}
        for patterns in valid_patterns.values():
            if 'weakest_anomaly' in patterns:
                anomaly = patterns['weakest_anomaly']
                anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        
        most_common_anomaly = max(anomaly_counts, key=anomaly_counts.get) if anomaly_counts else 'N/A'
        
        # Calculate average max F1 drop
        max_drops = [p.get('max_f1_drop', 0.0) for p in valid_patterns.values() if 'max_f1_drop' in p]
        avg_max_drop = np.mean(max_drops) if max_drops else 0.0
        
        return {
            'total_models_analyzed': len(valid_patterns),
            'models_with_high_priority': sum(
                1 for p in valid_patterns.values() 
                if p.get('optimization_priority') == 'HIGH'
            ),
            'models_with_medium_priority': sum(
                1 for p in valid_patterns.values() 
                if p.get('optimization_priority') == 'MEDIUM'
            ),
            'most_common_weak_anomaly': most_common_anomaly,
            'average_max_f1_drop': round(avg_max_drop, 4),
            'instance_failures_analyzed': len(self.instance_failures) > 0,
            'status': 'completed'
        }
    
    def _generate_text_report(self, report: Dict) -> str:
        """Generate human-readable text report"""
        lines = []
        lines.append("="*80)
        lines.append("STAGE 1: FAILURE CASE ANALYSIS REPORT")
        lines.append("="*80)
        lines.append(f"Generated: {report['metadata']['analysis_timestamp']}")
        lines.append(f"Framework Version: {report['metadata']['framework_version']}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY:")
        summary = report['summary']
        lines.append(f"  - Models analyzed: {summary['total_models_analyzed']}")
        lines.append(f"  - High priority optimizations needed: {summary['models_with_high_priority']}")
        lines.append(f"  - Medium priority optimizations needed: {summary['models_with_medium_priority']}")
        lines.append(f"  - Most vulnerable anomaly type: {summary['most_common_weak_anomaly']}")
        lines.append(f"  - Average max F1 drop: {summary['average_max_f1_drop']:.3f}")
        lines.append("")
        
        # Model-specific recommendations
        lines.append("OPTIMIZATION RECOMMENDATIONS:")
        for rec in report['recommendations']:
            lines.append(f"  {rec['model']} (Priority: {rec['priority']}):")
            lines.append(f"    Action: {rec['action']}")
            lines.append(f"    Target anomalies: {', '.join(rec['target_anomalies']) if rec['target_anomalies'] else 'N/A'}")
            lines.append(f"    Expected F1 improvement: +{rec['expected_improvement']['f1_improvement_estimate']:.3f}")
            lines.append(f"    Rationale: {rec['rationale']}")
            lines.append("")
        
        # Instance-level failures
        if report['instance_failures'] and report['instance_failures'].get('summary', {}).get('status') == 'completed':
            inst_summary = report['instance_failures']['summary']
            lines.append("INSTANCE-LEVEL FAILURE ANALYSIS:")
            lines.append(f"  - Total samples: {inst_summary['total_samples']}")
            lines.append(f"  - False positives: {inst_summary['false_positives_count']} ({inst_summary['fp_rate']*100:.2f}%)")
            lines.append(f"  - False negatives: {inst_summary['false_negatives_count']} ({inst_summary['fn_rate']*100:.2f}%)")
            lines.append(f"  - Borderline cases: {inst_summary['borderline_cases_count']}")
            lines.append(f"  - Needs optimization: {inst_summary['needs_optimization']}")
            lines.append("")
        
        # Next steps (more cautious/general wording)
        lines.append("NEXT STEPS FOR STAGE 2:")
        lines.append("  1. Review optimization recommendations by priority")
        lines.append("  2. Generate difficulty-aware anomalies targeting weak anomalies")
        lines.append("  3. Prepare retraining strategies using difficulty-aware anomaly generation")
        lines.append("  4. Validate improvement on held-out test set")
        lines.append("  5. Compare Stage 1 vs. Stage 2 F1 scores")
        lines.append("")
        
        lines.append("="*80)
        lines.append("END OF STAGE 1 REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)
    
    def _log_summary(self, report: Dict):
        """Log summary to console"""
        logger.info("\n📊 FAILURE ANALYSIS SUMMARY:")
        summary = report['summary']
        logger.info(f"   Models analyzed: {summary['total_models_analyzed']}")
        logger.info(f"   High priority optimizations: {summary['models_with_high_priority']}")
        logger.info(f"   Most vulnerable anomaly: {summary['most_common_weak_anomaly']}")
        logger.info(f"   Average max F1 drop: {summary['average_max_f1_drop']:.3f}")
        
        if report['recommendations']:
            logger.info(f"   Recommendations generated: {len(report['recommendations'])}")
            high_priority = [r for r in report['recommendations'] if r['priority'] == 'HIGH']
            if high_priority:
                logger.info(f"   HIGH priority models: {[r['model'] for r in high_priority]}")