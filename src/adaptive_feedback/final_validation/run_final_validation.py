"""
Final Validation Pipeline Runner
Main orchestrator for the final validation stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This file orchestrates the complete final validation pipeline by running:
1. Multi-seed Stability Analysis
2. Ablation Analysis
3. Error Analysis
4. Final Integrated Visualization

This file does NOT duplicate logic from previous modules.
It imports and calls the already completed modules.

EXECUTION:
----------
python -m src.adaptive_feedback.final_validation.run_final_validation

OUTPUTS:
--------
- results/adaptive/final_validation/reports/final_validation_pipeline_report.txt
- All outputs from individual validation modules
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
import sys
import time
import logging
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add project root to sys.path for config import
# This file is at: src/adaptive_feedback/final_validation/run_final_validation.py
# We need to go up 3 levels to reach project root where config.py exists
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now config.py can be imported from project root
from config import FrameworkConfig
from src.utils import setup_logging

# Import previous modules safely
try:
    from src.adaptive_feedback.final_validation.multi_seed_validator import MultiSeedValidator
    MULTI_SEED_AVAILABLE = True
except ImportError as e:
    MULTI_SEED_AVAILABLE = False
    print(f"⚠️  Warning: Could not import MultiSeedValidator: {e}")

try:
    from src.adaptive_feedback.final_validation.ablation_analyzer import AblationAnalyzer
    ABLATION_AVAILABLE = True
except ImportError as e:
    ABLATION_AVAILABLE = False
    print(f"⚠️  Warning: Could not import AblationAnalyzer: {e}")

try:
    from src.adaptive_feedback.final_validation.error_analysis import ErrorAnalyzer
    ERROR_ANALYSIS_AVAILABLE = True
except ImportError as e:
    ERROR_ANALYSIS_AVAILABLE = False
    print(f"⚠️  Warning: Could not import ErrorAnalyzer: {e}")

try:
    from src.adaptive_feedback.final_validation.final_validation_visualizer import FinalValidationVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    VISUALIZER_AVAILABLE = False
    print(f"⚠️  Warning: Could not import FinalValidationVisualizer: {e}")

# =============================================================================
# EXECUTION OPTIONS
# =============================================================================
RUN_MULTI_SEED = True
RUN_ABLATION = True
RUN_ERROR_ANALYSIS = True
RUN_FINAL_VISUALIZATION = True

SKIP_EXISTING_OUTPUTS = True
FORCE_RERUN = False

# =============================================================================
# FINAL VALIDATION RUNNER CLASS
# =============================================================================
class FinalValidationRunner:
    """
    Orchestrates the complete final validation pipeline.
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize Final Validation Runner
        
        Args:
            base_dir: Base results directory
        """
        self.base_dir = Path(base_dir)
        self.validation_dir = self.base_dir / 'adaptive' / 'final_validation'
        
        # Create output directories
        (self.validation_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.validation_dir / 'final_visualizations' / 'figures').mkdir(parents=True, exist_ok=True)
        (self.validation_dir / 'final_visualizations' / 'reports').mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.start_time = None
        self.stages_completed = []
        self.stages_skipped = []
        self.stages_failed = []
        self.pipeline_status = 'pending'
        self.scientific_results = {}
        
        # Required output files
        self.multi_seed_required = [
            self.validation_dir / 'metrics' / 'multi_seed_results.csv',
            self.validation_dir / 'metrics' / 'multi_seed_summary.json'
        ]
        
        self.ablation_required = [
            self.validation_dir / 'ablation' / 'metrics' / 'ablation_results.csv',
            self.validation_dir / 'ablation' / 'metrics' / 'ablation_summary.json'
        ]
        
        self.error_required = [
            self.validation_dir / 'error_analysis' / 'metrics' / 'error_analysis_results.csv',
            self.validation_dir / 'error_analysis' / 'metrics' / 'error_transitions.csv',
            self.validation_dir / 'error_analysis' / 'metrics' / 'error_analysis_summary.json'
        ]
        
        self.visualization_required = [
            self.validation_dir / 'final_visualizations' / 'figures' / 'final_performance_overview.png',
            self.validation_dir / 'final_visualizations' / 'figures' / 'final_adaptive_improvement_summary.png',
            self.validation_dir / 'final_visualizations' / 'figures' / 'final_ablation_contribution_summary.png',
            self.validation_dir / 'final_visualizations' / 'figures' / 'final_error_tradeoff_summary.png',
            self.validation_dir / 'final_visualizations' / 'figures' / 'final_model_comparison_dashboard.png',
            self.validation_dir / 'final_visualizations' / 'reports' / 'final_visualization_report.txt'
        ]
        
        logger.info(f"✅ Initialized FinalValidationRunner")
        logger.info(f"📁 Validation directory: {self.validation_dir}")
    
    def _get_config_stats(self, model_stats: Dict, possible_keys: List[str]) -> Dict:
        """Retrieve configuration stats using multiple possible key names"""
        for key in possible_keys:
            if key in model_stats:
                return model_stats[key]
        return {}
    
    def _get_baseline_and_weighted(self, stats: Dict, model: str) -> Tuple[Dict, Dict]:
        """Get baseline and weighted stats with flexible key matching"""
        model_stats = stats.get(model, {})
        
        baseline = self._get_config_stats(
            model_stats, 
            ['Baseline', 'baseline', 'BASELINE']
        )
        
        weighted = self._get_config_stats(
            model_stats,
            ['Weighted_Retraining', 'weighted_retraining', 'Weighted Retraining', 'weighted retraining', 'WeightedRetraining']
        )
        
        return baseline, weighted
    
    def _extract_scientific_results(self) -> Dict:
        """Extract scientific results from multi-seed and error analysis JSON files"""
        results = {}
        
        # Try to load multi-seed summary
        multi_seed_json = self.validation_dir / 'metrics' / 'multi_seed_summary.json'
        if multi_seed_json.exists():
            try:
                with open(multi_seed_json, 'r', encoding='utf-8') as f:
                    multi_seed_data = json.load(f)
                
                stats = multi_seed_data.get('statistics', {})
                
                for model in ['XGBoost', 'RandomForest']:
                    if model not in stats:
                        continue
                    
                    baseline, weighted = self._get_baseline_and_weighted(stats, model)
                    
                    baseline_f1 = baseline.get('mean_f1', 0)
                    weighted_f1 = weighted.get('mean_f1', 0)
                    baseline_recall = baseline.get('mean_recall', 0)
                    weighted_recall = weighted.get('mean_recall', 0)
                    baseline_fn = baseline.get('mean_fn', 0)
                    weighted_fn = weighted.get('mean_fn', 0)
                    
                    results[model] = {
                        'f1_improvement': round(weighted_f1 - baseline_f1, 4),
                        'recall_improvement': round(weighted_recall - baseline_recall, 4),
                        'fn_reduction': round(baseline_fn - weighted_fn, 1)
                    }
            except Exception as e:
                logger.warning(f"⚠️  Could not extract multi-seed results: {e}")
        
        # Try to load error analysis summary for FP change
        error_json = self.validation_dir / 'error_analysis' / 'metrics' / 'error_analysis_summary.json'
        if error_json.exists():
            try:
                with open(error_json, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                
                stats = error_data.get('statistics', {})
                
                for model in ['XGBoost', 'RandomForest']:
                    if model not in stats or model not in results:
                        continue
                    
                    baseline, weighted = self._get_baseline_and_weighted(stats, model)
                    baseline_fp = baseline.get('mean_fp', 0)
                    weighted_fp = weighted.get('mean_fp', 0)
                    
                    results[model]['fp_change'] = round(weighted_fp - baseline_fp, 1)
            except Exception as e:
                logger.warning(f"⚠️  Could not extract error analysis results: {e}")
        
        self.scientific_results = results
        return results
    
    def check_prerequisites(self) -> bool:
        """Check if required modules are available"""
        logger.info("🔍 Checking prerequisites...")
        
        available = []
        missing = []
        
        if MULTI_SEED_AVAILABLE:
            available.append('MultiSeedValidator')
        else:
            missing.append('MultiSeedValidator')
        
        if ABLATION_AVAILABLE:
            available.append('AblationAnalyzer')
        else:
            missing.append('AblationAnalyzer')
        
        if ERROR_ANALYSIS_AVAILABLE:
            available.append('ErrorAnalyzer')
        else:
            missing.append('ErrorAnalyzer')
        
        if VISUALIZER_AVAILABLE:
            available.append('FinalValidationVisualizer')
        else:
            missing.append('FinalValidationVisualizer')
        
        logger.info(f"  ✅ Available modules: {', '.join(available)}")
        
        if missing:
            logger.warning(f"  ⚠️  Missing modules: {', '.join(missing)}")
            logger.warning(f"  ⚠️  Some stages may be skipped")
        
        return len(available) > 0
    
    def _check_outputs_exist(self, required_files: List[Path]) -> bool:
        """Check if all required output files exist"""
        return all(f.exists() for f in required_files)
    
    def run_multi_seed_validation(self) -> bool:
        """Run multi-seed stability analysis"""
        logger.info("\n" + "="*80)
        logger.info("🔢 STAGE 1: MULTI-SEED STABILITY ANALYSIS")
        logger.info("="*80)
        
        # Check if outputs already exist
        if SKIP_EXISTING_OUTPUTS and not FORCE_RERUN:
            if self._check_outputs_exist(self.multi_seed_required):
                logger.info("⏭️  Skipping multi-seed validation (outputs already exist)")
                self.stages_skipped.append('Multi-Seed Validation')
                return True
        
        if not MULTI_SEED_AVAILABLE:
            logger.error("❌ MultiSeedValidator module not available")
            self.stages_failed.append('Multi-Seed Validation')
            return False
        
        try:
            validator = MultiSeedValidator(base_dir=self.base_dir)
            results = validator.run_full_validation()
            
            if results.get('visualization_complete', False):
                logger.info("✅ Multi-seed validation completed successfully")
                self.stages_completed.append('Multi-Seed Validation')
                return True
            else:
                logger.error("❌ Multi-seed validation failed or partial")
                self.stages_failed.append('Multi-Seed Validation')
                return False
                
        except Exception as e:
            logger.error(f"❌ Multi-seed validation failed: {e}")
            logger.error(traceback.format_exc())
            self.stages_failed.append('Multi-Seed Validation')
            return False
    
    def run_ablation_analysis(self) -> bool:
        """Run ablation/component contribution analysis"""
        logger.info("\n" + "="*80)
        logger.info("🔍 STAGE 2: ABLATION ANALYSIS")
        logger.info("="*80)
        
        # Check if outputs already exist
        if SKIP_EXISTING_OUTPUTS and not FORCE_RERUN:
            if self._check_outputs_exist(self.ablation_required):
                logger.info("⏭️  Skipping ablation analysis (outputs already exist)")
                self.stages_skipped.append('Ablation Analysis')
                return True
        
        if not ABLATION_AVAILABLE:
            logger.error("❌ AblationAnalyzer module not available")
            self.stages_failed.append('Ablation Analysis')
            return False
        
        try:
            analyzer = AblationAnalyzer(base_dir=self.base_dir)
            results = analyzer.run_full_analysis()
            
            if results.get('analysis_complete', False):
                logger.info("✅ Ablation analysis completed successfully")
                self.stages_completed.append('Ablation Analysis')
                return True
            else:
                logger.error("❌ Ablation analysis failed or partial")
                self.stages_failed.append('Ablation Analysis')
                return False
                
        except Exception as e:
            logger.error(f"❌ Ablation analysis failed: {e}")
            logger.error(traceback.format_exc())
            self.stages_failed.append('Ablation Analysis')
            return False
    
    def run_error_analysis(self) -> bool:
        """Run error behavior analysis"""
        logger.info("\n" + "="*80)
        logger.info("🔍 STAGE 3: ERROR ANALYSIS")
        logger.info("="*80)
        
        # Check if outputs already exist
        if SKIP_EXISTING_OUTPUTS and not FORCE_RERUN:
            if self._check_outputs_exist(self.error_required):
                logger.info("⏭️  Skipping error analysis (outputs already exist)")
                self.stages_skipped.append('Error Analysis')
                return True
        
        if not ERROR_ANALYSIS_AVAILABLE:
            logger.error("❌ ErrorAnalyzer module not available")
            self.stages_failed.append('Error Analysis')
            return False
        
        try:
            analyzer = ErrorAnalyzer(base_dir=self.base_dir)
            results = analyzer.run_full_analysis()
            
            if results.get('analysis_complete', False):
                logger.info("✅ Error analysis completed successfully")
                self.stages_completed.append('Error Analysis')
                return True
            else:
                logger.error("❌ Error analysis failed or partial")
                self.stages_failed.append('Error Analysis')
                return False
                
        except Exception as e:
            logger.error(f"❌ Error analysis failed: {e}")
            logger.error(traceback.format_exc())
            self.stages_failed.append('Error Analysis')
            return False
    
    def run_final_visualization(self) -> bool:
        """Run final integrated visualization"""
        logger.info("\n" + "="*80)
        logger.info("🎨 STAGE 4: FINAL INTEGRATED VISUALIZATION")
        logger.info("="*80)
        
        # Check if outputs already exist
        if SKIP_EXISTING_OUTPUTS and not FORCE_RERUN:
            if self._check_outputs_exist(self.visualization_required):
                logger.info("⏭️  Skipping final visualization (outputs already exist)")
                self.stages_skipped.append('Final Visualization')
                return True
        
        if not VISUALIZER_AVAILABLE:
            logger.error("❌ FinalValidationVisualizer module not available")
            self.stages_failed.append('Final Visualization')
            return False
        
        try:
            visualizer = FinalValidationVisualizer(base_dir=self.base_dir)
            results = visualizer.run_full_visualization()
            
            if results.get('visualization_complete', False):
                logger.info("✅ Final visualization completed successfully")
                self.stages_completed.append('Final Visualization')
                return True
            else:
                logger.warning("⚠️  Final visualization partial or failed")
                self.stages_failed.append('Final Visualization')
                return False
                
        except Exception as e:
            logger.error(f"❌ Final visualization failed: {e}")
            logger.error(traceback.format_exc())
            self.stages_failed.append('Final Visualization')
            return False
    
    def verify_outputs(self) -> Dict[str, bool]:
        """Verify all expected output files exist"""
        logger.info("\n🔍 Verifying output files...")
        
        verification = {
            'multi_seed': self._check_outputs_exist(self.multi_seed_required),
            'ablation': self._check_outputs_exist(self.ablation_required),
            'error_analysis': self._check_outputs_exist(self.error_required),
            'visualization': self._check_outputs_exist(self.visualization_required)
        }
        
        for stage, exists in verification.items():
            if exists:
                logger.info(f"  ✅ {stage.replace('_', ' ').title()} outputs verified")
            else:
                logger.warning(f"  ⚠️  {stage.replace('_', ' ').title()} outputs missing")
        
        return verification
    
    def generate_pipeline_report(self) -> str:
        """Generate final pipeline execution report"""
        logger.info("📝 Generating pipeline report...")
        
        # Extract scientific results
        self._extract_scientific_results()
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        lines = []
        lines.append("="*80)
        lines.append("FINAL VALIDATION PIPELINE REPORT")
        lines.append("Adaptive Simulation-Based Framework for CPS Anomaly Detection")
        lines.append("="*80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Framework Version: 1.0.0")
        lines.append(f"Pipeline Status: {self.pipeline_status.upper()}")
        lines.append(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        lines.append("")
        
        # Stages summary
        lines.append("PIPELINE STAGES:")
        lines.append(f"  - Stages completed: {len(self.stages_completed)}")
        for stage in self.stages_completed:
            lines.append(f"      ✅ {stage}")
        
        lines.append(f"  - Stages skipped: {len(self.stages_skipped)}")
        for stage in self.stages_skipped:
            lines.append(f"      ⏭️  {stage}")
        
        lines.append(f"  - Stages failed: {len(self.stages_failed)}")
        for stage in self.stages_failed:
            lines.append(f"      ❌ {stage}")
        lines.append("")
        
        # Scientific results summary
        lines.append("MAIN VALIDATED FINDINGS:")
        lines.append("")
        
        if self.scientific_results:
            for model, results in self.scientific_results.items():
                lines.append(f"  {model}:")
                lines.append(f"    - F1 improvement: {results.get('f1_improvement', 0):+.4f}")
                lines.append(f"    - Recall improvement: {results.get('recall_improvement', 0):+.4f}")
                lines.append(f"    - False negative reduction: {results.get('fn_reduction', 0):.1f}")
                if 'fp_change' in results:
                    lines.append(f"    - False positive change: {results.get('fp_change', 0):+.1f}")
                lines.append("")
        else:
            lines.append("  ⚠️  Could not extract scientific results from output files")
            lines.append("")
        
        # Scientific interpretation
        lines.append("SCIENTIFIC INTERPRETATION:")
        lines.append("")
        lines.append("  Under the tested setting, the results indicate that the adaptive")
        lines.append("  feedback strategy reduces false negatives. The improvement is")
        lines.append("  model-dependent:")
        lines.append("")
        lines.append("  - XGBoost shows higher false negative reduction with a larger false")
        lines.append("    positive trade-off")
        lines.append("  - RandomForest shows strong false negative reduction with a smaller")
        lines.append("    false positive trade-off")
        lines.append("")
        lines.append("  The increase in false positives should be interpreted as a deployment")
        lines.append("  trade-off, particularly for safety-critical CPS applications where")
        lines.append("  both false negatives and false positives have operational costs.")
        lines.append("")
        
        lines.append("  ⚠️  CAUTION:")
        lines.append("  ⚠️  - These results are from validation on a single dataset")
        lines.append("  ⚠️  - Do not claim universal improvement or perfect detection")
        lines.append("  ⚠️  - External validation is recommended before production deployment")
        lines.append("  ⚠️  - The adaptive feedback strategy should be evaluated in the")
        lines.append("  ⚠️    specific CPS deployment context before operational use")
        lines.append("")
        
        lines.append("="*80)
        lines.append("END OF PIPELINE REPORT")
        lines.append("="*80)
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.validation_dir / 'reports' / 'final_validation_pipeline_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved pipeline report: {report_path}")
        
        return report_text
    
    def run_full_pipeline(self) -> Dict:
        """Run complete final validation pipeline"""
        self.start_time = time.time()
        
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING FINAL VALIDATION PIPELINE")
        logger.info("="*80)
        logger.info(f"📁 Base directory: {self.base_dir}")
        logger.info(f"📁 Validation directory: {self.validation_dir}")
        logger.info(f"⚙️  Configuration:")
        logger.info(f"   - Run Multi-Seed: {RUN_MULTI_SEED}")
        logger.info(f"   - Run Ablation: {RUN_ABLATION}")
        logger.info(f"   - Run Error Analysis: {RUN_ERROR_ANALYSIS}")
        logger.info(f"   - Run Final Visualization: {RUN_FINAL_VISUALIZATION}")
        logger.info(f"   - Skip Existing Outputs: {SKIP_EXISTING_OUTPUTS}")
        logger.info(f"   - Force Rerun: {FORCE_RERUN}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            self.pipeline_status = 'failed'
            # FIXED Issue 3: Return complete dictionary even on failure
            return {
                'status': self.pipeline_status,
                'stages_completed': self.stages_completed,
                'stages_skipped': self.stages_skipped,
                'stages_failed': self.stages_failed,
                'verification': {},
                'scientific_results': {}
            }
        
        # Run stages
        if RUN_MULTI_SEED:
            self.run_multi_seed_validation()
        
        if RUN_ABLATION:
            self.run_ablation_analysis()
        
        if RUN_ERROR_ANALYSIS:
            self.run_error_analysis()
        
        if RUN_FINAL_VISUALIZATION:
            self.run_final_visualization()
        
        # FIXED Issue 1 & 2: Verify outputs and determine status BEFORE generating report
        verification = self.verify_outputs()
        
        # Determine pipeline status based on verification and stages
        all_outputs_verified = all(verification.values()) if verification else False
        any_stage_done_or_skipped = (
            len(self.stages_completed) > 0 or len(self.stages_skipped) > 0
        )
        
        # FIXED Issue 4: Skipped stages with verified outputs count as success
        if len(self.stages_failed) == 0 and all_outputs_verified and any_stage_done_or_skipped:
            self.pipeline_status = 'complete'
        elif all_outputs_verified and len(self.stages_failed) > 0:
            self.pipeline_status = 'partial'
        elif any(verification.values()):
            self.pipeline_status = 'partial'
        else:
            self.pipeline_status = 'failed'
        
        # FIXED Issue 1: Generate report AFTER status is determined
        self.generate_pipeline_report()
        
        return {
            'status': self.pipeline_status,
            'stages_completed': self.stages_completed,
            'stages_skipped': self.stages_skipped,
            'stages_failed': self.stages_failed,
            'verification': verification,
            'scientific_results': self.scientific_results
        }


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute final validation pipeline"""
    global logger
    
    # Initialize configuration
    config = FrameworkConfig()
    
    # Setup logging
    logger = setup_logging(config.output.logs_dir, 'final_validation_pipeline')
    
    logger.info("\n" + "="*80)
    logger.info("🎯 FINAL VALIDATION PIPELINE")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize paths
    base_dir = Path(config.output.base_dir)
    
    # Run pipeline
    runner = FinalValidationRunner(base_dir=base_dir)
    results = runner.run_full_pipeline()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🎉 FINAL VALIDATION PIPELINE COMPLETED!")
    logger.info("="*80)
    logger.info(f"📊 Status: {results['status'].upper()}")
    logger.info(f"✅ Stages completed: {len(results['stages_completed'])}")
    for stage in results['stages_completed']:
        logger.info(f"   - {stage}")
    logger.info(f"⏭️  Stages skipped: {len(results['stages_skipped'])}")
    for stage in results['stages_skipped']:
        logger.info(f"   - {stage}")
    logger.info(f"❌ Stages failed: {len(results['stages_failed'])}")
    for stage in results['stages_failed']:
        logger.info(f"   - {stage}")
    logger.info(f"💾 Pipeline report: {base_dir / 'adaptive' / 'final_validation' / 'reports' / 'final_validation_pipeline_report.txt'}")
    logger.info("="*80)


if __name__ == '__main__':
    main()