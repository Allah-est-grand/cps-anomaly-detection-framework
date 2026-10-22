"""
Adaptive Visualizer - Retraining Results Visualization Module
Generates publication-quality figures for adaptive retraining evaluation

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

This module visualizes the results from the experimental retraining stage.
It reads saved outputs from the retraining pipeline and generates figures
that honestly reflect which models were actually retrained vs. analyzed.

Key Design Principles:
----------------------
1. Scientific Honesty: Only plot what was actually retrained
2. Graceful Degradation: Skip figures if data is missing, don't crash
3. Publication Quality: High-resolution figures suitable for thesis/report
4. Clear Communication: Labels and titles that accurately describe the data

Input Files:
------------
- results/adaptive/retraining/models/retrained_models_summary.json
- results/adaptive/retraining/metrics/retraining_evaluation_results.json
- results/adaptive/retraining/metrics/baseline_vs_retrained_comparison.json
- results/adaptive/retraining/reports/experimental_retraining_report.txt

Output Directory:
-----------------
- results/figures/retraining/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


class AdaptiveVisualizer:
    """
    Visualizes adaptive retraining results for thesis/mid-term report
    
    This class loads retraining outputs and generates publication-quality
    figures that honestly reflect the adaptive retraining workflow:
    - Which models were analyzed (all 7 baseline models)
    - Which models were selected for retraining (priority subset)
    - Which models were actually retrained (successful retraining)
    - Observed performance improvements (only for successfully retrained models)
    
    Attributes:
        base_dir: Base results directory
        retraining_dir: Path to retraining outputs
        figures_dir: Path to save generated figures
        retraining_summary: Loaded retraining summary data
        evaluation_results: Loaded evaluation metrics
        comparison_results: Loaded baseline vs retrained comparison
    """
    
    def __init__(self, 
                 base_dir: Optional[Path] = None,
                 retraining_dir: Optional[Path] = None,
                 figures_dir: Optional[Path] = None):
        """
        Initialize Adaptive Visualizer
        
        Args:
            base_dir: Base results directory (default: results/)
            retraining_dir: Path to retraining outputs (default: results/adaptive/retraining/)
            figures_dir: Path to save figures (default: results/figures/retraining/)
        """
        self.base_dir = Path(base_dir) if base_dir else Path('results')
        self.retraining_dir = Path(retraining_dir) if retraining_dir else (
            self.base_dir / 'adaptive' / 'retraining'
        )
        self.figures_dir = Path(figures_dir) if figures_dir else (
            self.base_dir / 'figures' / 'retraining'
        )
        
        # Ensure figures directory exists
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.retraining_summary = {}
        self.evaluation_results = {}
        self.comparison_results = {}
        self.report_text = ""
        
        # Track data availability
        self.data_availability = {
            'retraining_summary': False,
            'evaluation_results': False,
            'comparison_results': False,
            'report': False
        }
        
        logger.info(f"✅ Initialized AdaptiveVisualizer")
        logger.info(f"📁 Retraining directory: {self.retraining_dir}")
        logger.info(f"📁 Figures directory: {self.figures_dir}")
    
    def load_retraining_outputs(self) -> Dict[str, bool]:
        """
        Load all retraining output files
        
        Returns:
            Dictionary indicating which files were successfully loaded
        """
        logger.info("📊 Loading retraining outputs...")
        
        # Load retraining summary
        summary_path = self.retraining_dir / 'models' / 'retrained_models_summary.json'
        if summary_path.exists():
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    self.retraining_summary = json.load(f)
                self.data_availability['retraining_summary'] = True
                logger.info(f"✅ Loaded retraining summary: {summary_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load retraining summary: {e}")
        else:
            logger.warning(f"⚠️  Retraining summary not found: {summary_path}")
        
        # Load evaluation results
        eval_path = self.retraining_dir / 'metrics' / 'retraining_evaluation_results.json'
        if eval_path.exists():
            try:
                with open(eval_path, 'r', encoding='utf-8') as f:
                    self.evaluation_results = json.load(f)
                self.data_availability['evaluation_results'] = True
                logger.info(f"✅ Loaded evaluation results: {eval_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load evaluation results: {e}")
        else:
            logger.warning(f"⚠️  Evaluation results not found: {eval_path}")
        
        # Load comparison results
        comparison_path = self.retraining_dir / 'metrics' / 'baseline_vs_retrained_comparison.json'
        if comparison_path.exists():
            try:
                with open(comparison_path, 'r', encoding='utf-8') as f:
                    self.comparison_results = json.load(f)
                self.data_availability['comparison_results'] = True
                logger.info(f"✅ Loaded comparison results: {comparison_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load comparison results: {e}")
        else:
            logger.warning(f"⚠️  Comparison results not found: {comparison_path}")
        
        # Load report text
        report_path = self.retraining_dir / 'reports' / 'experimental_retraining_report.txt'
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    self.report_text = f.read()
                self.data_availability['report'] = True
                logger.info(f"✅ Loaded retraining report: {report_path}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load retraining report: {e}")
        else:
            logger.warning(f"⚠️  Retraining report not found: {report_path}")
        
        # Log summary
        loaded_count = sum(self.data_availability.values())
        logger.info(f"📊 Loaded {loaded_count}/4 retraining output files")
        
        return self.data_availability
    
    def generate_all_figures(self) -> List[str]:
        """
        Generate all available figures based on loaded data
        
        Returns:
            List of successfully generated figure paths
        """
        logger.info("🎨 Generating all retraining figures...")
        
        generated_figures = []
        
        # Figure 1: Retraining Status Summary (requires retraining_summary)
        if self.data_availability['retraining_summary']:
            fig_path = self.generate_retraining_status_summary()
            if fig_path:
                generated_figures.append(fig_path)
        else:
            logger.warning("⚠️  Skipping Figure 1: retraining_summary not available")
        
        # Figure 2: Baseline vs Retrained F1 (requires comparison_results)
        if self.data_availability['comparison_results']:
            fig_path = self.generate_baseline_vs_retrained_f1()
            if fig_path:
                generated_figures.append(fig_path)
        else:
            logger.warning("⚠️  Skipping Figure 2: comparison_results not available")
        
        # Figure 3: Observed Improvement Per Model (requires comparison_results)
        if self.data_availability['comparison_results']:
            fig_path = self.generate_observed_improvement_per_model()
            if fig_path:
                generated_figures.append(fig_path)
        else:
            logger.warning("⚠️  Skipping Figure 3: comparison_results not available")
        
        # Figure 4: Retraining Workflow Overview (requires retraining_summary)
        if self.data_availability['retraining_summary']:
            fig_path = self.generate_retraining_workflow_overview()
            if fig_path:
                generated_figures.append(fig_path)
        else:
            logger.warning("⚠️  Skipping Figure 4: retraining_summary not available")
        
        # Figure 5: Evaluation Metrics Per Model (requires evaluation_results)
        if self.data_availability['evaluation_results']:
            fig_path = self.generate_evaluation_metrics_per_model()
            if fig_path:
                generated_figures.append(fig_path)
        else:
            logger.warning("⚠️  Skipping Figure 5: evaluation_results not available")
        
        logger.info(f"✅ Generated {len(generated_figures)} figures")
        
        return generated_figures
    
    def generate_retraining_status_summary(self) -> Optional[str]:
        """
        Generate Figure 1: Retraining Status Summary
        
        Shows:
        - Total target models
        - Successfully retrained/refit
        - Skipped models
        - Fallback models
        
        Returns:
            Path to saved figure or None if generation failed
        """
        logger.info("🎨 Generating Figure 1: Retraining Status Summary...")
        
        try:
            # Extract counts from retraining summary
            total_target = self.retraining_summary.get('total_target_models', 0)
            total_retrained = self.retraining_summary.get('total_retrained', 0)
            total_refit = self.retraining_summary.get('total_refit', 0)
            total_skipped = self.retraining_summary.get('total_skipped', 0)
            total_fallback = self.retraining_summary.get('total_fallback', 0)
            
            # Calculate successful retraining (retrained + refit)
            total_successful = total_retrained + total_refit
            
            # Prepare data
            categories = ['Target\nModels', 'Successfully\nRetrained', 
                         'Skipped', 'Fallback']
            counts = [total_target, total_successful, total_skipped, total_fallback]
            colors = ['#3498db', '#27ae60', '#e74c3c', '#f39c12']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
            
            # Configure axes
            ax.set_ylabel('Number of Models', fontsize=12, fontweight='bold')
            ax.set_title('Adaptive Retraining Status Summary\n(Stage 2 Selected Priority Models Only)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_ylim(0, max(counts) * 1.3)
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add note about scientific honesty
            fig.text(0.5, 0.02, 
                    'Note: Only priority models from Stage 2 were selected for retraining.\n'
                    'Not all 7 baseline models were retrained - this reflects the adaptive workflow.',
                    ha='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Save figure
            fig_path = self.figures_dir / 'retraining_status_summary.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved: {fig_path}")
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate retraining status summary: {e}")
            return None
    
    def generate_baseline_vs_retrained_f1(self) -> Optional[str]:
        """
        Generate Figure 2: Baseline vs Retrained F1 Comparison
        
        Shows:
        - Grouped bar chart comparing baseline F1 vs retrained F1
        - Only for models where both values are available
        
        Returns:
            Path to saved figure or None if generation failed
        """
        logger.info("🎨 Generating Figure 2: Baseline vs Retrained F1...")
        
        try:
            # Extract comparison data
            comparison_data = self.comparison_results.get('per_model_comparison', [])
            
            if not comparison_data:
                logger.warning("⚠️  No comparison data available for F1 chart")
                return None
            
            # Filter models with both baseline and retrained F1
            valid_models = []
            baseline_f1 = []
            retrained_f1 = []
            
            for item in comparison_data:
                if item.get('baseline_f1') is not None and item.get('retrained_f1') is not None:
                    valid_models.append(item.get('model', 'Unknown'))
                    baseline_f1.append(item['baseline_f1'])
                    retrained_f1.append(item['retrained_f1'])
            
            if len(valid_models) == 0:
                logger.warning("⚠️  No models with both baseline and retrained F1 available")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = range(len(valid_models))
            width = 0.35
            
            # Grouped bars
            bars1 = ax.bar([i - width/2 for i in x], baseline_f1, width, 
                          label='Baseline F1', color='#3498db', edgecolor='black', linewidth=1.2)
            bars2 = ax.bar([i + width/2 for i in x], retrained_f1, width, 
                          label='Retrained F1', color='#27ae60', edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
            
            # Configure axes
            ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
            ax.set_title('Baseline vs Retrained F1-Score Comparison\n(Only Successfully Retrained Models)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(valid_models, rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=11, loc='lower right')
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add note
            fig.text(0.5, 0.02, 
                    'Note: Only models with successful retraining and available comparison data are shown.',
                    ha='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Save figure
            fig_path = self.figures_dir / 'baseline_vs_retrained_f1.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved: {fig_path}")
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate baseline vs retrained F1: {e}")
            return None
    
    def generate_observed_improvement_per_model(self) -> Optional[str]:
        """
        Generate Figure 3: Observed F1 Improvement Per Model (IMPROVED VERSION)
        
        Shows:
        - Bar chart of observed F1 improvement (retrained - baseline)
        - Positive improvements in green, negative in red
        - Compact layout with tight y-axis scaling
        - Clear zero reference line
        - 4 decimal place precision on labels
        
        Returns:
            Path to saved figure or None if generation failed
        """
        logger.info("🎨 Generating Figure 3: Observed Improvement Per Model...")
        
        try:
            # Extract comparison data
            comparison_data = self.comparison_results.get('per_model_comparison', [])
            
            if not comparison_data:
                logger.warning("⚠️  No comparison data available for improvement chart")
                return None
            
            # Calculate improvements
            models = []
            improvements = []
            
            for item in comparison_data:
                if item.get('improvement') is not None:
                    models.append(item.get('model', 'Unknown'))
                    improvements.append(item['improvement'])
            
            if len(models) == 0:
                logger.warning("⚠️  No models with improvement data available")
                return None
            
            # Determine colors based on positive/negative improvement
            colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
            
            # Create figure with compact size
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = range(len(models))
            bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=1.2)
            
            # Compute y-axis limits with padding
            min_imp = min(improvements)
            max_imp = max(improvements)
            
            # Add padding around the range
            y_range = max_imp - min_imp
            if y_range > 0:
                padding = y_range * 0.15  # 15% padding
                y_min = min_imp - padding
                y_max = max_imp + padding
            else:
                # If all values are the same, add fixed padding
                y_min = min_imp - 0.05
                y_max = max_imp + 0.05
            
            # Ensure zero is visible in the plot
            if min_imp < 0 and max_imp > 0:
                y_min = min(y_min, -0.01)
                y_max = max(y_max, 0.01)
            
            ax.set_ylim(y_min, y_max)
            
            # Add horizontal zero reference line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            
            # Add value labels with 4 decimal places
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                # Position label: above for positive, below for negative
                if height >= 0:
                    y_pos = height + (y_max - y_min) * 0.015  # Small offset above
                    va = 'bottom'
                    ha = 'center'
                else:
                    y_pos = height - (y_max - y_min) * 0.015  # Small offset below
                    va = 'top'
                    ha = 'center'
                
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{imp:+.4f}', ha=ha, va=va, 
                       fontsize=9, fontweight='bold')
            
            # Configure axes
            ax.set_ylabel('F1-score Improvement', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_title('Observed F1 Improvement Per Retrained Model\n(Retrained F1 - Baseline F1)', 
                        fontsize=13, fontweight='bold', pad=12)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            
            # Add grid for readability
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            
            # Add legend for colors
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60', 
                      markersize=10, label='Positive Improvement'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c', 
                      markersize=10, label='Negative Improvement')
            ]
            ax.legend(handles=legend_elements, fontsize=10, loc='upper right', framealpha=0.9)
            
            # Add scientific honesty note
            note_text = (
                'Note: Improvement = Retrained F1 - Baseline F1. '
                'Only successfully retrained models with comparison data shown.'
            )
            fig.text(0.5, 0.01, note_text, 
                    ha='center', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            
            # Save figure
            fig_path = self.figures_dir / 'observed_improvement_per_model.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved: {fig_path}")
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate observed improvement per model: {e}")
            return None
    
    def generate_retraining_workflow_overview(self) -> Optional[str]:
        """
        Generate Figure 4: Retraining Workflow Overview
        
        Shows:
        - Simple diagram showing the adaptive retraining workflow
        - 7 models analyzed → Stage 2 selected subset → Actually retrained
        
        Returns:
            Path to saved figure or None if generation failed
        """
        logger.info("🎨 Generating Figure 4: Retraining Workflow Overview...")
        
        try:
            # Extract counts
            total_analyzed = 7  # All baseline models
            total_target = self.retraining_summary.get('total_target_models', 0)
            total_retrained = self.retraining_summary.get('total_retrained', 0)
            total_refit = self.retraining_summary.get('total_refit', 0)
            total_successful = total_retrained + total_refit
            total_skipped = self.retraining_summary.get('total_skipped', 0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 8)
            ax.axis('off')
            
            # Title
            ax.text(7, 7.5, 'Adaptive Retraining Workflow Overview', 
                   fontsize=16, fontweight='bold', ha='center', va='top')
            
            # Stage 1: All 7 models analyzed
            box1_x, box1_y = 1, 5
            box1_w, box1_h = 3, 2
            rect1 = Rectangle((box1_x, box1_y), box1_w, box1_h, 
                             linewidth=2, edgecolor='#3498db', facecolor='#ebf5fb')
            ax.add_patch(rect1)
            ax.text(box1_x + box1_w/2, box1_y + box1_h/2, 
                   f'Baseline Analysis\n\n{total_analyzed} Models\nAnalyzed', 
                   fontsize=11, ha='center', va='center', fontweight='bold')
            
            # Arrow 1
            ax.annotate('', xy=(5.5, 6), xytext=(4.2, 6),
                       arrowprops=dict(arrowstyle='->', linewidth=2, color='#3498db'))
            ax.text(4.85, 6.3, 'Stage 1:\nFailure\nAnalysis', 
                   fontsize=9, ha='center', va='bottom', style='italic')
            
            # Stage 2: Selected subset
            box2_x, box2_y = 5.5, 5
            box2_w, box2_h = 3, 2
            rect2 = Rectangle((box2_x, box2_y), box2_w, box2_h, 
                             linewidth=2, edgecolor='#f39c12', facecolor='#fef9e7')
            ax.add_patch(rect2)
            ax.text(box2_x + box2_w/2, box2_y + box2_h/2, 
                   f'Stage 2 Selection\n\n{total_target} Priority\nModels Selected', 
                   fontsize=11, ha='center', va='center', fontweight='bold')
            
            # Arrow 2
            ax.annotate('', xy=(10, 6), xytext=(8.7, 6),
                       arrowprops=dict(arrowstyle='->', linewidth=2, color='#f39c12'))
            ax.text(9.35, 6.3, 'Stage 2:\nAdaptive\nSelection', 
                   fontsize=9, ha='center', va='bottom', style='italic')
            
            # Stage 3: Actually retrained
            box3_x, box3_y = 10, 5
            box3_w, box3_h = 3, 2
            rect3 = Rectangle((box3_x, box3_y), box3_w, box3_h, 
                             linewidth=2, edgecolor='#27ae60', facecolor='#e9f7ef')
            ax.add_patch(rect3)
            ax.text(box3_x + box3_w/2, box3_y + box3_h/2, 
                   f'Experimental\nRetraining\n\n{total_successful} Models\nSuccessfully Retrained', 
                   fontsize=11, ha='center', va='center', fontweight='bold')
            
            # Breakdown below
            # Skipped models
            box4_x, box4_y = 5.5, 2
            box4_w, box4_h = 2.5, 1.5
            rect4 = Rectangle((box4_x, box4_y), box4_w, box4_h, 
                             linewidth=2, edgecolor='#e74c3c', facecolor='#fdedec')
            ax.add_patch(rect4)
            ax.text(box4_x + box4_w/2, box4_y + box4_h/2, 
                   f'Skipped\n\n{total_skipped} Models', 
                   fontsize=10, ha='center', va='center')
            
            # Fallback models
            box5_x, box5_y = 8.5, 2
            box5_w, box5_h = 2.5, 1.5
            total_fallback = self.retraining_summary.get('total_fallback', 0)
            rect5 = Rectangle((box5_x, box5_y), box5_w, box5_h, 
                             linewidth=2, edgecolor='#e67e22', facecolor='#fef5e7')
            ax.add_patch(rect5)
            ax.text(box5_x + box5_w/2, box5_y + box5_h/2, 
                   f'Fallback\n\n{total_fallback} Models', 
                   fontsize=10, ha='center', va='center')
            
            # Legend/note
            note_text = (
                'Scientific Honesty Note:\n'
                '• Not all 7 baseline models were retrained\n'
                '• Only priority models from Stage 2 were selected\n'
                '• This reflects the adaptive, resource-efficient workflow\n'
                '• Figures show actual retraining outcomes, not fabricated data'
            )
            fig.text(0.5, 0.02, note_text, 
                    ha='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save figure
            fig_path = self.figures_dir / 'retraining_workflow_overview.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved: {fig_path}")
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate retraining workflow overview: {e}")
            return None
    
    def generate_evaluation_metrics_per_model(self) -> Optional[str]:
        """
        Generate Figure 5: Evaluation Metrics Per Model
        
        Shows:
        - Grouped bar chart for accuracy, precision, recall, F1-score
        - For evaluated retrained models only
        
        Returns:
            Path to saved figure or None if generation failed
        """
        logger.info("🎨 Generating Figure 5: Evaluation Metrics Per Model...")
        
        try:
            # Extract evaluation data
            evaluation_metrics = self.evaluation_results.get('evaluation_metrics', {})
            
            if not evaluation_metrics:
                logger.warning("⚠️  No evaluation metrics available")
                return None
            
            # Filter evaluated models
            evaluated_models = []
            metrics_data = {}
            
            for model_name, metrics in evaluation_metrics.items():
                if metrics.get('status') == 'evaluated':
                    evaluated_models.append(model_name)
                    metrics_data[model_name] = {
                        'accuracy': metrics.get('accuracy'),
                        'precision': metrics.get('precision'),
                        'recall': metrics.get('recall'),
                        'f1_score': metrics.get('f1_score')
                    }
            
            if len(evaluated_models) == 0:
                logger.warning("⚠️  No evaluated models available for metrics chart")
                return None
            
            # Prepare data for plotting
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = ['#3498db', '#9b59b6', '#e67e22', '#27ae60']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 7))
            
            x = range(len(evaluated_models))
            width = 0.2
            multiplier = 0
            
            # Grouped bars for each metric
            for metric_name, metric_key, color in zip(metric_names, metric_keys, colors):
                values = [metrics_data[model][metric_key] for model in evaluated_models]
                offset = [i + multiplier * width for i in x]
                
                bars = ax.bar(offset, values, width, label=metric_name, 
                             color=color, edgecolor='black', linewidth=1.2)
                
                # Add value labels (only if value is not None)
                for bar, val in zip(bars, values):
                    if val is not None:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.2f}', ha='center', va='bottom', 
                               fontsize=8, fontweight='bold')
                
                multiplier += 1
            
            # Configure axes
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Evaluation Metrics Per Retrained Model\n(Only Successfully Evaluated Models)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks([i + 1.5*width for i in x])
            ax.set_xticklabels(evaluated_models, rotation=45, ha='right')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=11, loc='lower right')
            
            # Add grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add note
            fig.text(0.5, 0.02, 
                    'Note: Only models with successful evaluation are shown. '
                    'Metrics may not be available for all models due to retraining outcomes.',
                    ha='center', fontsize=9, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Save figure
            fig_path = self.figures_dir / 'evaluation_metrics_per_model.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Saved: {fig_path}")
            
            return str(fig_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate evaluation metrics per model: {e}")
            return None
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity across loaded files
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating data integrity...")
        
        validation = {
            'all_files_loaded': all(self.data_availability.values()),
            'retraining_summary_valid': False,
            'evaluation_results_valid': False,
            'comparison_results_valid': False,
            'warnings': []
        }
        
        # Validate retraining summary
        if self.data_availability['retraining_summary']:
            required_keys = ['total_target_models', 'total_retrained', 'total_skipped']
            if all(key in self.retraining_summary for key in required_keys):
                validation['retraining_summary_valid'] = True
            else:
                validation['warnings'].append('Retraining summary missing required keys')
        
        # Validate evaluation results
        if self.data_availability['evaluation_results']:
            if 'evaluation_metrics' in self.evaluation_results:
                validation['evaluation_results_valid'] = True
            else:
                validation['warnings'].append('Evaluation results missing evaluation_metrics')
        
        # Validate comparison results
        if self.data_availability['comparison_results']:
            if 'per_model_comparison' in self.comparison_results:
                validation['comparison_results_valid'] = True
            else:
                validation['warnings'].append('Comparison results missing per_model_comparison')
        
        # Log validation results
        if validation['all_files_loaded']:
            logger.info("✅ All data files loaded and validated")
        else:
            logger.warning(f"⚠️  Some data files missing: {validation['warnings']}")
        
        return validation


def main():
    """
    Main execution function for generating retraining visualization figures
    """
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.utils import setup_logging
    from config import FrameworkConfig
    
    # Initialize configuration and logging
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'adaptive_visualization')
    
    logger.info("\n" + "="*80)
    logger.info("🎨 ADAPTIVE RETRAINING VISUALIZATION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize visualizer
    visualizer = AdaptiveVisualizer(
        base_dir=config.output.base_dir,
        retraining_dir=config.output.base_dir / 'adaptive' / 'retraining',
        figures_dir=config.output.figures_dir / 'retraining'
    )
    
    # Load retraining outputs
    data_availability = visualizer.load_retraining_outputs()
    
    # Validate data integrity
    validation = visualizer.validate_data_integrity()
    
    if validation['warnings']:
        logger.warning("⚠️  Data validation warnings:")
        for warning in validation['warnings']:
            logger.warning(f"   - {warning}")
    
    # Generate all available figures
    generated_figures = visualizer.generate_all_figures()
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🎉 VISUALIZATION COMPLETED!")
    logger.info("="*80)
    logger.info(f"\n📊 Data Availability:")
    for key, available in data_availability.items():
        status = "✅" if available else "❌"
        logger.info(f"   {status} {key}")
    
    logger.info(f"\n📊 Generated Figures: {len(generated_figures)}")
    for fig_path in generated_figures:
        logger.info(f"   - {fig_path}")
    
    logger.info(f"\n💾 Figures saved to: {visualizer.figures_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()