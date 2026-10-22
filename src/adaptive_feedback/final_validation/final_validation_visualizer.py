"""
Final Validation Visualizer - Integrated Summary Visualization Module
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
This module creates final integrated visualizations that combine and summarize
the validated outputs from:
1. Multi-seed Stability Analysis
2. Ablation Analysis
3. Error Analysis

This file does NOT retrain models.
This file does NOT recompute experiments.
It only loads existing CSV/JSON result files and generates clean final summary figures.

INPUT FILES:
------------
- results/adaptive/final_validation/metrics/multi_seed_results.csv
- results/adaptive/final_validation/metrics/multi_seed_summary.json
- results/adaptive/final_validation/ablation/metrics/ablation_results.csv
- results/adaptive/final_validation/ablation/metrics/ablation_summary.json
- results/adaptive/final_validation/error_analysis/metrics/error_analysis_results.csv
- results/adaptive/final_validation/error_analysis/metrics/error_transitions.csv
- results/adaptive/final_validation/error_analysis/metrics/error_analysis_summary.json

OUTPUT FILES:
-------------
- results/adaptive/final_validation/final_visualizations/figures/final_performance_overview.png
- results/adaptive/final_validation/final_visualizations/figures/final_adaptive_improvement_summary.png
- results/adaptive/final_validation/final_visualizations/figures/final_ablation_contribution_summary.png
- results/adaptive/final_validation/final_visualizations/figures/final_error_tradeoff_summary.png
- results/adaptive/final_validation/final_visualizations/figures/final_model_comparison_dashboard.png
- results/adaptive/final_validation/final_visualizations/reports/final_visualization_report.txt
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

# Add project root to sys.path for config import
# This file is at: src/adaptive_feedback/final_validation/final_validation_visualizer.py
# We need to go up 3 levels to reach project root where config.py exists
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now config.py can be imported from project root
from config import FrameworkConfig
from src.utils import setup_logging

# Data processing and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS_OF_INTEREST = ['XGBoost', 'RandomForest']

# Color scheme
COLORS = {
    'baseline': '#3498db',
    'weighted': '#27ae60',
    'retraining': '#3498db',
    'weighting': '#9b59b6',
    'total': '#27ae60',
    'fp_tradeoff': '#e74c3c',
    'fn_reduction': '#27ae60'
}

# =============================================================================
# FINAL VALIDATION VISUALIZER CLASS
# =============================================================================
class FinalValidationVisualizer:
    """
    Creates final integrated visualizations from validated multi-seed, ablation,
    and error analysis results.
    """
    
    def __init__(self, base_dir: Path, viz_dir: Optional[Path] = None):
        """
        Initialize Final Validation Visualizer
        
        Args:
            base_dir: Base results directory
            viz_dir: Path to save visualization outputs
        """
        self.base_dir = Path(base_dir)
        self.viz_dir = Path(viz_dir) if viz_dir else (
            self.base_dir / 'adaptive' / 'final_validation' / 'final_visualizations'
        )
        
        # Create output directories
        (self.viz_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.viz_dir / 'reports').mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.multi_seed_data = {}
        self.ablation_data = {}
        self.error_data = {}
        self.viz_status = 'pending'
        self.loaded_files = 0
        self.generated_figures = []
        
        # Input file paths
        self.multi_seed_csv = self.base_dir / 'adaptive' / 'final_validation' / 'metrics' / 'multi_seed_results.csv'
        self.multi_seed_json = self.base_dir / 'adaptive' / 'final_validation' / 'metrics' / 'multi_seed_summary.json'
        self.ablation_csv = self.base_dir / 'adaptive' / 'final_validation' / 'ablation' / 'metrics' / 'ablation_results.csv'
        self.ablation_json = self.base_dir / 'adaptive' / 'final_validation' / 'ablation' / 'metrics' / 'ablation_summary.json'
        self.error_csv = self.base_dir / 'adaptive' / 'final_validation' / 'error_analysis' / 'metrics' / 'error_analysis_results.csv'
        self.error_transitions_csv = self.base_dir / 'adaptive' / 'final_validation' / 'error_analysis' / 'metrics' / 'error_transitions.csv'
        self.error_json = self.base_dir / 'adaptive' / 'final_validation' / 'error_analysis' / 'metrics' / 'error_analysis_summary.json'
        
        logger.info(f"✅ Initialized FinalValidationVisualizer")
        logger.info(f"📁 Visualization directory: {self.viz_dir}")
    
    def load_results(self) -> bool:
        """Load all existing CSV/JSON results from previous modules"""
        logger.info("📊 Loading results from previous modules...")
        
        files_loaded = 0
        files_expected = 7
        
        # Load multi-seed results
        try:
            if self.multi_seed_csv.exists():
                self.multi_seed_data['csv'] = pd.read_csv(self.multi_seed_csv)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.multi_seed_csv.name}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.multi_seed_csv.name}")
            
            if self.multi_seed_json.exists():
                with open(self.multi_seed_json, 'r', encoding='utf-8') as f:
                    self.multi_seed_data['json'] = json.load(f)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.multi_seed_json.name}")
                # FIXED: Add debug logging for JSON structure
                if self.multi_seed_data['json']:
                    logger.info(f"  🔍 Multi-seed JSON keys: {list(self.multi_seed_data['json'].keys())}")
                    stats = self.multi_seed_data['json'].get('statistics', {})
                    for model in MODELS_OF_INTEREST:
                        if model in stats:
                            logger.info(f"  🔍 {model} keys: {list(stats[model].keys())}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.multi_seed_json.name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load multi-seed data: {e}")
        
        # Load ablation results
        try:
            if self.ablation_csv.exists():
                self.ablation_data['csv'] = pd.read_csv(self.ablation_csv)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.ablation_csv.name}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.ablation_csv.name}")
            
            if self.ablation_json.exists():
                with open(self.ablation_json, 'r', encoding='utf-8') as f:
                    self.ablation_data['json'] = json.load(f)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.ablation_json.name}")
                # FIXED: Add debug logging for JSON structure
                if self.ablation_data['json']:
                    logger.info(f"  🔍 Ablation JSON keys: {list(self.ablation_data['json'].keys())}")
                    contribs = self.ablation_data['json'].get('ablation_contributions', {})
                    for model in MODELS_OF_INTEREST:
                        if model in contribs:
                            logger.info(f"  🔍 {model} contribution keys: {list(contribs[model].keys())}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.ablation_json.name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load ablation data: {e}")
        
        # Load error analysis results
        try:
            if self.error_csv.exists():
                self.error_data['csv'] = pd.read_csv(self.error_csv)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.error_csv.name}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.error_csv.name}")
            
            if self.error_transitions_csv.exists():
                self.error_data['transitions'] = pd.read_csv(self.error_transitions_csv)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.error_transitions_csv.name}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.error_transitions_csv.name}")
            
            if self.error_json.exists():
                with open(self.error_json, 'r', encoding='utf-8') as f:
                    self.error_data['json'] = json.load(f)
                files_loaded += 1
                logger.info(f"  ✅ Loaded: {self.error_json.name}")
                # FIXED: Add debug logging for JSON structure
                if self.error_data['json']:
                    logger.info(f"  🔍 Error analysis JSON keys: {list(self.error_data['json'].keys())}")
                    stats = self.error_data['json'].get('statistics', {})
                    for model in MODELS_OF_INTEREST:
                        if model in stats:
                            logger.info(f"  🔍 {model} error keys: {list(stats[model].keys())}")
            else:
                logger.warning(f"  ⚠️  Missing: {self.error_json.name}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load error analysis data: {e}")
        
        self.loaded_files = files_loaded
        logger.info(f"📊 Loaded {files_loaded}/{files_expected} expected files")
        
        return files_loaded >= 4  # Need at least 4 files to generate meaningful visualizations
    
    def _safe_get_nested_value(self, data: Dict, *keys, default=None) -> Any:
        """Safely get nested dictionary value"""
        result = data
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result
    
    def _get_config_stats(self, model_stats: Dict, possible_keys: List[str]) -> Dict:
        """
        Retrieve configuration stats using multiple possible key names.
        
        Args:
            model_stats: Dictionary containing model statistics
            possible_keys: List of possible key names to try
            
        Returns:
            Dictionary with stats or empty dict if not found
        """
        for key in possible_keys:
            if key in model_stats:
                return model_stats[key]
        return {}
    
    def _get_baseline_and_weighted(self, stats: Dict, model: str) -> Tuple[Dict, Dict]:
        """
        Get baseline and weighted stats for a model using flexible key matching.
        
        Args:
            stats: Statistics dictionary from JSON
            model: Model name ('XGBoost' or 'RandomForest')
            
        Returns:
            Tuple of (baseline_dict, weighted_dict)
        """
        model_stats = stats.get(model, {})
        
        # Get baseline with flexible key matching
        baseline = self._get_config_stats(
            model_stats, 
            ['Baseline', 'baseline', 'BASELINE']
        )
        
        # Get weighted with flexible key matching
        weighted = self._get_config_stats(
            model_stats,
            ['Weighted_Retraining', 'weighted_retraining', 'Weighted Retraining', 'weighted retraining', 'WeightedRetraining']
        )
        
        # Log warnings if values are missing
        if not baseline:
            logger.warning(f"  ⚠️  Could not find baseline stats for {model}. Available keys: {list(model_stats.keys())}")
        if not weighted:
            logger.warning(f"  ⚠️  Could not find weighted stats for {model}. Available keys: {list(model_stats.keys())}")
        
        return baseline, weighted
    
    def _compute_model_improvements(self, stats: Dict, model: str) -> Dict[str, float]:
        """
        Compute model improvements from Baseline and Weighted_Retraining stats.
        
        This is a ROBUST helper that:
        1. Uses _get_baseline_and_weighted() for flexible key matching
        2. Computes improvements directly from mean values if 'improvements' key is missing
        3. Logs warnings if baseline/weighted stats are empty
        
        Args:
            stats: Statistics dictionary from JSON
            model: Model name ('XGBoost' or 'RandomForest')
            
        Returns:
            Dictionary with f1_improvement, recall_improvement, fn_reduction
        """
        # Try to use existing improvements if they exist and are non-zero
        model_stats = stats.get(model, {})
        existing_improvements = model_stats.get('improvements', {})
        if existing_improvements and existing_improvements.get('f1_improvement', 0) != 0:
            return existing_improvements
        
        # Use _get_baseline_and_weighted for flexible key matching
        baseline, weighted = self._get_baseline_and_weighted(stats, model)
        
        # Extract values with defaults
        baseline_f1 = baseline.get('mean_f1', 0)
        weighted_f1 = weighted.get('mean_f1', 0)
        
        baseline_recall = baseline.get('mean_recall', 0)
        weighted_recall = weighted.get('mean_recall', 0)
        
        baseline_fn = baseline.get('mean_fn', 0)
        weighted_fn = weighted.get('mean_fn', 0)
        
        # Compute improvements
        return {
            'f1_improvement': round(weighted_f1 - baseline_f1, 4),
            'recall_improvement': round(weighted_recall - baseline_recall, 4),
            'fn_reduction': round(baseline_fn - weighted_fn, 2)
        }
    
    def _set_dynamic_ylim(self, ax: plt.Axes, values: List[float], stds: Optional[List[float]] = None, 
                         include_zero: bool = True) -> None:
        """Set dynamic y-axis limits with margin for labels"""
        if stds:
            y_max_with_err = max(v + s for v, s in zip(values, stds))
            y_min_with_err = min(v - s for v, s in zip(values, stds))
        else:
            y_max_with_err = max(values)
            y_min_with_err = min(values)
        
        y_range = y_max_with_err - y_min_with_err if y_max_with_err > y_min_with_err else max(abs(y_max_with_err), 1)
        label_margin = y_range * 0.25
        
        if include_zero and y_min_with_err > 0:
            ax.set_ylim(max(0, y_min_with_err - label_margin), y_max_with_err + label_margin * 1.5)
        else:
            ax.set_ylim(y_min_with_err - label_margin, y_max_with_err + label_margin * 1.5)
    
    def _add_value_labels(self, ax: plt.Axes, bars: List, values: List[float], 
                         stds: Optional[List[float]] = None, fontsize: int = 10) -> None:
        """Add value labels above bars (above error bars if present)"""
        # Use enumerate instead of bars.index(bar)
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
        
        for idx, (bar, val) in enumerate(zip(bars, values)):
            std = stds[idx] if stds else 0
            if val >= 0:
                label_y = val + std + offset
                va = 'bottom'
            else:
                label_y = val - std - offset
                va = 'top'
            
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{val:+.4f}' if abs(val) < 1 else f'{val:.1f}',
                   ha='center', va=va, fontsize=fontsize, fontweight='bold')
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> Optional[str]:
        """Save figure with consistent settings"""
        try:
            fig_path = self.viz_dir / 'figures' / filename
            plt.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            self.generated_figures.append(str(fig_path))
            logger.info(f"  ✅ Saved: {filename}")
            return str(fig_path)
        except Exception as e:
            logger.error(f"  ❌ Failed to save {filename}: {e}")
            plt.close()
            return None
    
    def generate_performance_overview(self) -> Optional[str]:
        """Generate Figure 1: Final Performance Overview"""
        logger.info("🎨 Generating performance overview...")
        
        if not self.multi_seed_data.get('json'):
            logger.warning("⚠️  No multi-seed data for performance overview")
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            stats = self._safe_get_nested_value(self.multi_seed_data, 'json', 'statistics', default={})
            models = list(stats.keys())
            
            # FIXED: Use _get_baseline_and_weighted for flexible key matching
            baseline_f1, weighted_f1 = [], []
            baseline_recall, weighted_recall = [], []
            baseline_fn, weighted_fn = [], []
            
            for m in models:
                baseline, weighted = self._get_baseline_and_weighted(stats, m)
                baseline_f1.append(baseline.get('mean_f1', 0))
                weighted_f1.append(weighted.get('mean_f1', 0))
                baseline_recall.append(baseline.get('mean_recall', 0))
                weighted_recall.append(weighted.get('mean_recall', 0))
                baseline_fn.append(baseline.get('mean_fn', 0))
                weighted_fn.append(weighted.get('mean_fn', 0))
            
            # Subplot 1: F1-score comparison
            ax1 = axes[0, 0]
            x = range(len(models))
            width = 0.35
            bars1 = ax1.bar([i - width/2 for i in x], baseline_f1, width, label='Baseline', color=COLORS['baseline'], edgecolor='black')
            bars2 = ax1.bar([i + width/2 for i in x], weighted_f1, width, label='Weighted Retraining', color=COLORS['weighted'], edgecolor='black')
            
            self._set_dynamic_ylim(ax1, baseline_f1 + weighted_f1, include_zero=True)
            self._add_value_labels(ax1, bars1 + bars2, baseline_f1 + weighted_f1, fontsize=10)
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, fontsize=11)
            ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
            ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold', pad=15)
            ax1.legend(fontsize=10)
            ax1.grid(axis='y', alpha=0.3)
            
            # Subplot 2: Recall comparison
            ax2 = axes[0, 1]
            bars3 = ax2.bar([i - width/2 for i in x], baseline_recall, width, label='Baseline', color=COLORS['baseline'], edgecolor='black')
            bars4 = ax2.bar([i + width/2 for i in x], weighted_recall, width, label='Weighted Retraining', color=COLORS['weighted'], edgecolor='black')
            
            self._set_dynamic_ylim(ax2, baseline_recall + weighted_recall, include_zero=True)
            self._add_value_labels(ax2, bars3 + bars4, baseline_recall + weighted_recall, fontsize=10)
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, fontsize=11)
            ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
            ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold', pad=15)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            
            # Subplot 3: False Negatives comparison
            ax3 = axes[1, 0]
            bars5 = ax3.bar([i - width/2 for i in x], baseline_fn, width, label='Baseline', color=COLORS['baseline'], edgecolor='black')
            bars6 = ax3.bar([i + width/2 for i in x], weighted_fn, width, label='Weighted Retraining', color=COLORS['weighted'], edgecolor='black')
            
            self._set_dynamic_ylim(ax3, baseline_fn + weighted_fn, include_zero=True)
            self._add_value_labels(ax3, bars5 + bars6, baseline_fn + weighted_fn, fontsize=10)
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(models, fontsize=11)
            ax3.set_ylabel('False Negatives', fontsize=12, fontweight='bold')
            ax3.set_title('False Negatives Comparison', fontsize=14, fontweight='bold', pad=15)
            ax3.legend(fontsize=10)
            ax3.grid(axis='y', alpha=0.3)
            
            # Subplot 4: Summary metrics table - FIXED: Use _compute_model_improvements
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            table_data = []
            for model in models:
                improvements = self._compute_model_improvements(stats, model)
                table_data.append([model, 
                                f"{improvements['f1_improvement']:+.4f}", 
                                f"{improvements['recall_improvement']:+.4f}", 
                                f"{improvements['fn_reduction']:.1f}"])
            
            table = ax4.table(cellText=table_data,
                             colLabels=['Model', 'ΔF1', 'ΔRecall', 'FN Reduction'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            ax4.set_title('Improvement Summary', fontsize=14, fontweight='bold', pad=15)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return self._save_figure(fig, 'final_performance_overview.png')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance overview: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_adaptive_improvement_summary(self) -> Optional[str]:
        """Generate Figure 2: Final Adaptive Improvement Summary"""
        logger.info("🎨 Generating adaptive improvement summary...")
        
        if not self.multi_seed_data.get('json'):
            logger.warning("⚠️  No multi-seed data for improvement summary")
            return None
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            stats = self._safe_get_nested_value(self.multi_seed_data, 'json', 'statistics', default={})
            models = list(stats.keys())
            
            # Subplot 1: F1 improvement - FIXED: Use _compute_model_improvements
            ax1 = axes[0]
            f1_improvements = [self._compute_model_improvements(stats, m)['f1_improvement'] for m in models]
            
            colors = [COLORS['weighted'] if v > 0 else COLORS['fp_tradeoff'] for v in f1_improvements]
            bars1 = ax1.bar(models, f1_improvements, color=colors, edgecolor='black', alpha=0.8)
            
            self._set_dynamic_ylim(ax1, f1_improvements, include_zero=False)
            self._add_value_labels(ax1, bars1, f1_improvements, fontsize=11)
            
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_ylabel('F1-Score Improvement', fontsize=12, fontweight='bold')
            ax1.set_title('F1-Score Improvement', fontsize=14, fontweight='bold', pad=15)
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontsize=11, rotation=45, ha='right')
            
            # Subplot 2: Recall improvement - FIXED: Use _compute_model_improvements
            ax2 = axes[1]
            recall_improvements = [self._compute_model_improvements(stats, m)['recall_improvement'] for m in models]
            
            colors = [COLORS['weighted'] if v > 0 else COLORS['fp_tradeoff'] for v in recall_improvements]
            bars2 = ax2.bar(models, recall_improvements, color=colors, edgecolor='black', alpha=0.8)
            
            self._set_dynamic_ylim(ax2, recall_improvements, include_zero=False)
            self._add_value_labels(ax2, bars2, recall_improvements, fontsize=11)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Recall Improvement', fontsize=12, fontweight='bold')
            ax2.set_title('Recall Improvement', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, fontsize=11, rotation=45, ha='right')
            
            # Subplot 3: FN reduction - FIXED: Use _compute_model_improvements
            ax3 = axes[2]
            fn_reductions = [self._compute_model_improvements(stats, m)['fn_reduction'] for m in models]
            
            bars3 = ax3.bar(models, fn_reductions, color=COLORS['fn_reduction'], edgecolor='black', alpha=0.8)
            
            self._set_dynamic_ylim(ax3, fn_reductions, include_zero=True)
            self._add_value_labels(ax3, bars3, fn_reductions, fontsize=11)
            
            ax3.set_ylabel('False Negative Reduction', fontsize=12, fontweight='bold')
            ax3.set_title('False Negative Reduction', fontsize=14, fontweight='bold', pad=15)
            ax3.grid(axis='y', alpha=0.3)
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, fontsize=11, rotation=45, ha='right')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return self._save_figure(fig, 'final_adaptive_improvement_summary.png')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate adaptive improvement summary: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_ablation_contribution_summary(self) -> Optional[str]:
        """Generate Figure 3: Final Ablation Contribution Summary"""
        logger.info("🎨 Generating ablation contribution summary...")
        
        if not self.ablation_data.get('json'):
            logger.warning("⚠️  No ablation data for contribution summary")
            return None
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            contributions = self._safe_get_nested_value(self.ablation_data, 'json', 'ablation_contributions', default={})
            models = list(contributions.keys())
            
            # Subplot 1: F1 contribution breakdown
            ax1 = axes[0]
            
            # Collect all F1 values first, then set ylim, then add labels
            all_f1_values = []
            all_f1_bars = []
            
            for model_idx, model in enumerate(models):
                model_contrib = contributions.get(model, {})
                retraining_f1 = self._safe_get_nested_value(model_contrib, 'retraining_contribution', 'f1_contribution', default=0)
                weighting_f1 = self._safe_get_nested_value(model_contrib, 'weighting_contribution', 'f1_contribution', default=0)
                total_f1 = self._safe_get_nested_value(model_contrib, 'total_adaptive_contribution', 'f1_contribution', default=0)
                
                all_f1_values.extend([retraining_f1, weighting_f1, total_f1])
                
                x = [model_idx]
                width = 0.25
                
                bars1 = ax1.bar([i - width for i in x], [retraining_f1], width, label='Retraining' if model_idx == 0 else "", color=COLORS['retraining'], edgecolor='black')
                bars2 = ax1.bar(x, [weighting_f1], width, label='Weighting' if model_idx == 0 else "", color=COLORS['weighting'], edgecolor='black')
                bars3 = ax1.bar([i + width for i in x], [total_f1], width, label='Total' if model_idx == 0 else "", color=COLORS['total'], edgecolor='black')
                
                all_f1_bars.extend([bars1[0], bars2[0], bars3[0]])
            
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax1, all_f1_values, include_zero=False)
            self._add_value_labels(ax1, all_f1_bars, all_f1_values, fontsize=9)
            
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontsize=11)
            ax1.set_ylabel('F1-Score Contribution', fontsize=12, fontweight='bold')
            ax1.set_title('F1-Score Contribution Breakdown', fontsize=14, fontweight='bold', pad=15)
            ax1.legend(fontsize=10)
            ax1.grid(axis='y', alpha=0.3)
            
            # Subplot 2: FN reduction breakdown
            ax2 = axes[1]
            
            # Collect all FN values first, then set ylim, then add labels
            all_fn_values = []
            all_fn_bars = []
            
            for model_idx, model in enumerate(models):
                model_contrib = contributions.get(model, {})
                retraining_fn = self._safe_get_nested_value(model_contrib, 'retraining_contribution', 'fn_reduction', default=0)
                weighting_fn = self._safe_get_nested_value(model_contrib, 'weighting_contribution', 'fn_reduction', default=0)
                total_fn = self._safe_get_nested_value(model_contrib, 'total_adaptive_contribution', 'fn_reduction', default=0)
                
                all_fn_values.extend([retraining_fn, weighting_fn, total_fn])
                
                x = [model_idx]
                width = 0.25
                
                bars4 = ax2.bar([i - width for i in x], [retraining_fn], width, label='Retraining' if model_idx == 0 else "", color=COLORS['retraining'], edgecolor='black')
                bars5 = ax2.bar(x, [weighting_fn], width, label='Weighting' if model_idx == 0 else "", color=COLORS['weighting'], edgecolor='black')
                bars6 = ax2.bar([i + width for i in x], [total_fn], width, label='Total' if model_idx == 0 else "", color=COLORS['total'], edgecolor='black')
                
                all_fn_bars.extend([bars4[0], bars5[0], bars6[0]])
            
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax2, all_fn_values, include_zero=False)
            self._add_value_labels(ax2, all_fn_bars, all_fn_values, fontsize=9)
            
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, fontsize=11)
            ax2.set_ylabel('False Negative Reduction', fontsize=12, fontweight='bold')
            ax2.set_title('False Negative Reduction Breakdown', fontsize=14, fontweight='bold', pad=15)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return self._save_figure(fig, 'final_ablation_contribution_summary.png')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate ablation contribution summary: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_error_tradeoff_summary(self) -> Optional[str]:
        """Generate Figure 4: Final Error Trade-off Summary"""
        logger.info("🎨 Generating error trade-off summary...")
        
        if not self.error_data.get('json'):
            logger.warning("⚠️  No error analysis data for trade-off summary")
            return None
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            stats = self._safe_get_nested_value(self.error_data, 'json', 'statistics', default={})
            models = list(stats.keys())
            
            # Subplot 1: FN reduction - FIXED: Use _get_baseline_and_weighted
            ax1 = axes[0]
            fn_reductions = []
            for model in models:
                baseline, weighted = self._get_baseline_and_weighted(stats, model)
                baseline_fn = baseline.get('mean_fn', 0)
                weighted_fn = weighted.get('mean_fn', 0)
                fn_reductions.append(baseline_fn - weighted_fn)
            
            bars1 = ax1.bar(models, fn_reductions, color=COLORS['fn_reduction'], edgecolor='black', alpha=0.8)
            
            self._set_dynamic_ylim(ax1, fn_reductions, include_zero=True)
            self._add_value_labels(ax1, bars1, fn_reductions, fontsize=11)
            
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontsize=11)
            ax1.set_ylabel('False Negative Reduction', fontsize=12, fontweight='bold')
            ax1.set_title('False Negative Reduction', fontsize=14, fontweight='bold', pad=15)
            ax1.grid(axis='y', alpha=0.3)
            
            # Subplot 2: FP change - FIXED: Use _get_baseline_and_weighted
            ax2 = axes[1]
            fp_changes = []
            for model in models:
                baseline, weighted = self._get_baseline_and_weighted(stats, model)
                baseline_fp = baseline.get('mean_fp', 0)
                weighted_fp = weighted.get('mean_fp', 0)
                fp_changes.append(weighted_fp - baseline_fp)
            
            colors = [COLORS['fp_tradeoff'] if v > 0 else COLORS['fn_reduction'] for v in fp_changes]
            bars2 = ax2.bar(models, fp_changes, color=colors, edgecolor='black', alpha=0.8)
            
            self._set_dynamic_ylim(ax2, fp_changes, include_zero=False)
            self._add_value_labels(ax2, bars2, fp_changes, fontsize=11)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, fontsize=11)
            ax2.set_ylabel('False Positive Change', fontsize=12, fontweight='bold')
            ax2.set_title('False Positive Trade-off', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return self._save_figure(fig, 'final_error_tradeoff_summary.png')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate error trade-off summary: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_model_comparison_dashboard(self) -> Optional[str]:
        """Generate Figure 5: Final Model Comparison Dashboard"""
        logger.info("🎨 Generating model comparison dashboard...")
        
        if not self.multi_seed_data.get('json') or not self.error_data.get('json'):
            logger.warning("⚠️  Insufficient data for dashboard")
            return None
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            multi_stats = self._safe_get_nested_value(self.multi_seed_data, 'json', 'statistics', default={})
            error_stats = self._safe_get_nested_value(self.error_data, 'json', 'statistics', default={})
            models = list(multi_stats.keys())
            
            # Subplot 1: F1 improvement - FIXED: Use _compute_model_improvements
            ax1 = axes[0, 0]
            f1_imps = [self._compute_model_improvements(multi_stats, m)['f1_improvement'] for m in models]
            colors = [COLORS['weighted'] if v > 0 else COLORS['fp_tradeoff'] for v in f1_imps]
            bars1 = ax1.bar(models, f1_imps, color=colors, edgecolor='black', alpha=0.8)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_ylabel('ΔF1', fontsize=11, fontweight='bold')
            ax1.set_title('F1 Improvement', fontsize=13, fontweight='bold', pad=15)
            ax1.grid(axis='y', alpha=0.3)
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax1, f1_imps, include_zero=False)
            self._add_value_labels(ax1, bars1, f1_imps, fontsize=9)
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontsize=11)
            
            # Subplot 2: Recall improvement - FIXED: Use _compute_model_improvements
            ax2 = axes[0, 1]
            recall_imps = [self._compute_model_improvements(multi_stats, m)['recall_improvement'] for m in models]
            colors = [COLORS['weighted'] if v > 0 else COLORS['fp_tradeoff'] for v in recall_imps]
            bars2 = ax2.bar(models, recall_imps, color=colors, edgecolor='black', alpha=0.8)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('ΔRecall', fontsize=11, fontweight='bold')
            ax2.set_title('Recall Improvement', fontsize=13, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3)
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax2, recall_imps, include_zero=False)
            self._add_value_labels(ax2, bars2, recall_imps, fontsize=9)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, fontsize=11)
            
            # Subplot 3: FN reduction - FIXED: Use _compute_model_improvements
            ax3 = axes[0, 2]
            fn_reds = [self._compute_model_improvements(multi_stats, m)['fn_reduction'] for m in models]
            bars3 = ax3.bar(models, fn_reds, color=COLORS['fn_reduction'], edgecolor='black', alpha=0.8)
            ax3.set_ylabel('FN Reduction', fontsize=11, fontweight='bold')
            ax3.set_title('False Negative Reduction', fontsize=13, fontweight='bold', pad=15)
            ax3.grid(axis='y', alpha=0.3)
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax3, fn_reds, include_zero=True)
            self._add_value_labels(ax3, bars3, fn_reds, fontsize=9)
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels(models, fontsize=11)
            
            # Subplot 4: FP change - FIXED: Use _get_baseline_and_weighted for error stats
            ax4 = axes[1, 0]
            fp_changes = []
            for model in models:
                baseline, weighted = self._get_baseline_and_weighted(error_stats, model)
                baseline_fp = baseline.get('mean_fp', 0)
                weighted_fp = weighted.get('mean_fp', 0)
                fp_changes.append(weighted_fp - baseline_fp)
            
            colors = [COLORS['fp_tradeoff'] if v > 0 else COLORS['fn_reduction'] for v in fp_changes]
            bars4 = ax4.bar(models, fp_changes, color=colors, edgecolor='black', alpha=0.8)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax4.set_ylabel('ΔFP', fontsize=11, fontweight='bold')
            ax4.set_title('False Positive Change', fontsize=13, fontweight='bold', pad=15)
            ax4.grid(axis='y', alpha=0.3)
            # Set dynamic ylim BEFORE adding labels
            self._set_dynamic_ylim(ax4, fp_changes, include_zero=False)
            self._add_value_labels(ax4, bars4, fp_changes, fontsize=9)
            ax4.set_xticks(range(len(models)))
            ax4.set_xticklabels(models, fontsize=11)
            
            # Subplot 5: Summary table - FIXED: Use _compute_model_improvements
            ax5 = axes[1, 1]
            ax5.axis('off')
            
            table_data = []
            for model in models:
                improvements = self._compute_model_improvements(multi_stats, model)
                # Get FP change for table
                baseline, weighted = self._get_baseline_and_weighted(error_stats, model)
                baseline_fp = baseline.get('mean_fp', 0)
                weighted_fp = weighted.get('mean_fp', 0)
                fp_chg = weighted_fp - baseline_fp
                table_data.append([model, 
                                f"{improvements['f1_improvement']:+.4f}", 
                                f"{improvements['fn_reduction']:.1f}", 
                                f"{fp_chg:+.1f}"])
            
            table = ax5.table(cellText=table_data,
                             colLabels=['Model', 'ΔF1', 'FN Reduction', 'ΔFP'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            ax5.set_title('Summary Metrics', fontsize=13, fontweight='bold', pad=15)
            
            # Subplot 6: Interpretation
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            interpretations = []
            for model in models:
                fp_chg = fp_changes[models.index(model)]
                fn_red = self._compute_model_improvements(multi_stats, model)['fn_reduction']
                
                if fp_chg > 100:
                    interp = f"{model}: Higher FN reduction\nwith larger FP trade-off"
                else:
                    interp = f"{model}: Strong FN reduction\nwith smaller FP trade-off"
                interpretations.append(interp)
            
            ax6.text(0.5, 0.7, '\n\n'.join(interpretations),
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax6.set_title('Interpretation', fontsize=13, fontweight='bold', pad=15)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return self._save_figure(fig, 'final_model_comparison_dashboard.png')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate model comparison dashboard: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def generate_report(self) -> str:
        """Generate final visualization report"""
        logger.info("📝 Generating final visualization report...")
        
        lines = []
        lines.append("="*80)
        lines.append("FINAL VALIDATION VISUALIZATION REPORT")
        lines.append("Integrated Summary of Multi-Seed, Ablation, and Error Analysis")
        lines.append("="*80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Framework Version: 1.0.0")
        lines.append(f"Visualization Status: {self.viz_status.upper()}")
        lines.append("")
        
        # Data sources
        lines.append("DATA SOURCES:")
        lines.append(f"  - Files loaded: {self.loaded_files}")
        lines.append(f"  - Multi-seed results: {self.multi_seed_csv.name}")
        lines.append(f"  - Ablation results: {self.ablation_csv.name}")
        lines.append(f"  - Error analysis results: {self.error_csv.name}")
        lines.append("")
        
        # Figures generated
        lines.append("FIGURES GENERATED:")
        for fig in self.generated_figures:
            lines.append(f"  - {Path(fig).name}")
        lines.append("")
        
        # Main findings - FIXED: Use _compute_model_improvements
        lines.append("MAIN VALIDATED FINDINGS:")
        lines.append("")
        
        if self.multi_seed_data.get('json'):
            stats = self._safe_get_nested_value(self.multi_seed_data, 'json', 'statistics', default={})
            
            for model in MODELS_OF_INTEREST:
                if model not in stats:
                    continue
                
                improvements = self._compute_model_improvements(stats, model)
                
                lines.append(f"  {model}:")
                lines.append(f"    - F1 improvement: {improvements['f1_improvement']:+.4f}")
                lines.append(f"    - Recall improvement: {improvements['recall_improvement']:+.4f}")
                lines.append(f"    - False negative reduction: {improvements['fn_reduction']:.1f}")
                lines.append("")
        
        # Scientific interpretation
        lines.append("SCIENTIFIC INTERPRETATION:")
        lines.append("")
        lines.append("  The results indicate that the adaptive feedback strategy reduces false")
        lines.append("  negatives under the tested setting. The improvement is model-dependent:")
        lines.append("")
        lines.append("  - XGBoost shows higher false negative reduction with a larger false")
        lines.append("    positive trade-off")
        lines.append("  - RandomForest shows strong false negative reduction with a smaller")
        lines.append("    false positive trade-off")
        lines.append("")
        lines.append("  The false positive increase should be considered as a deployment")
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
        lines.append("END OF FINAL VISUALIZATION REPORT")
        lines.append("="*80)
        
        report_text = "\n".join(lines)
        
        # Save report
        report_path = self.viz_dir / 'reports' / 'final_visualization_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"💾 Saved report: {report_path}")
        
        return report_text
    
    def run_full_visualization(self) -> Dict:
        """Run complete final visualization pipeline"""
        logger.info("\n" + "="*80)
        logger.info("🎨 RUNNING FINAL VALIDATION VISUALIZATION")
        logger.info("="*80)
        
        results = {
            'status': 'pending',
            'outputs_saved': [],
            'visualization_complete': False
        }
        
        # Step 1: Load results
        if not self.load_results():
            logger.error("❌ Failed to load sufficient results")
            results['status'] = 'failed'
            return results
        
        # Step 2: Generate figures
        fig1 = self.generate_performance_overview()
        if fig1:
            results['outputs_saved'].append(fig1)
        
        fig2 = self.generate_adaptive_improvement_summary()
        if fig2:
            results['outputs_saved'].append(fig2)
        
        fig3 = self.generate_ablation_contribution_summary()
        if fig3:
            results['outputs_saved'].append(fig3)
        
        fig4 = self.generate_error_tradeoff_summary()
        if fig4:
            results['outputs_saved'].append(fig4)
        
        fig5 = self.generate_model_comparison_dashboard()
        if fig5:
            results['outputs_saved'].append(fig5)
        
        # Step 3: Set status BEFORE generating report
        if len(self.generated_figures) >= 3:
            self.viz_status = 'complete'
            results['status'] = 'complete'
            results['visualization_complete'] = True
        else:
            self.viz_status = 'partial'
            results['status'] = 'partial'
            results['visualization_complete'] = False
        
        # Step 4: Generate report (with correct status)
        self.generate_report()
        results['outputs_saved'].append(str(self.viz_dir / 'reports' / 'final_visualization_report.txt'))
        
        return results


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute final validation visualization"""
    global logger
    
    start_time = time.time()
    
    # Initialize configuration
    config = FrameworkConfig()
    
    # Setup logging
    logger = setup_logging(config.output.logs_dir, 'final_validation_visualization')
    
    logger.info("\n" + "="*80)
    logger.info("🎨 FINAL VALIDATION VISUALIZATION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    # Initialize paths
    base_dir = Path(config.output.base_dir)
    viz_dir = base_dir / 'adaptive' / 'final_validation' / 'final_visualizations'
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    visualizer = FinalValidationVisualizer(base_dir=base_dir, viz_dir=viz_dir)
    results = visualizer.run_full_visualization()
    
    # Summary
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("🎉 FINAL VISUALIZATION COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
    logger.info(f"📊 Status: {results['status']}")
    logger.info(f"🎨 Figures generated: {len(visualizer.generated_figures)}")
    logger.info(f"💾 Outputs: {len(results['outputs_saved'])} files")
    
    for output in results['outputs_saved']:
        logger.info(f"   - {output}")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()