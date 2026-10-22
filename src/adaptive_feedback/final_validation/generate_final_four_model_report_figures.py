"""
Final Four-Model Report Figure Generation
Part of Additional Controlled Validation Stage

Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY

PURPOSE:
--------
Generate FINAL report figures for the retained four models:
- XGBoost
- RandomForest
- LinearSVM
- LSTM

This script uses FINAL validated data sources ONLY.
It does NOT train any models.
It does NOT modify any validated historical result file.
It only reads validated CSV/JSON outputs and generates figures/reports.

AUTHORITATIVE DATA SOURCES:
---------------------------
1. Final four-model ablation results (FIVE-SEED):
   results/adaptive/final_validation_four_model/ablation/metrics/four_model_ablation_results.csv
   
2. Weight sensitivity results (SEED-42 RETROSPECTIVE DIAGNOSTIC):
   results/adaptive/final_validation_four_model/report_extensions/weight_sensitivity/four_model_weight_sensitivity_results.csv
   
3. Transition summary (FIVE-SEED, LSTM unavailable):
   results/adaptive/final_validation_four_model/report_extensions/transitions/four_model_transitions_summary.csv

OUTPUT:
-------
results/adaptive/final_validation_four_model/figures/
- figure_01_final_f1_comparison.png/svg
- figure_02_final_recall_comparison.png/svg
- figure_03_final_precision_comparison.png/svg
- figure_04_final_false_negatives.png/svg
- figure_05_final_false_positives.png/svg
- figure_06_weight_sensitivity_f1.png/svg
- figure_07_weight_sensitivity_fn.png/svg
- figure_08_fn_transition_summary.png/svg
- figure_09_fp_transition_summary.png/svg
- figure_10_ablation_component_contribution.png/svg (if data available)
- final_four_model_figure_manifest.csv
- final_four_model_figure_report.txt
"""

# =============================================================================
# IMPORTS AND PATH CONFIGURATION
# =============================================================================
from __future__ import annotations

import sys
import os
import time
import logging
import json
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FrameworkConfig
from src.utils import setup_logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =============================================================================
# CONFIGURATION
# =============================================================================
MODELS = ['XGBoost', 'RandomForest', 'LinearSVM', 'LSTM']
EXPECTED_MODEL_SET = {'XGBoost', 'RandomForest', 'LinearSVM', 'LSTM'}
TRANSITION_SEEDS = [42, 123, 2024, 2025, 3407]
EXPECTED_SEED_SET = {42, 123, 2024, 2025, 3407}
EXPECTED_WEIGHT_SENSITIVITY_ROWS = 35
HISTORICAL_WEIGHT_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]

MODEL_COLORS = {
    'XGBoost': '#1f77b4',
    'RandomForest': '#ff7f0e',
    'LinearSVM': '#2ca02c',
    'LSTM': '#d62728'
}

# FIX #1: Required transition summary columns
REQUIRED_TRANSITION_COLUMNS = [
    'mean_corrected_fn', 'std_corrected_fn',
    'mean_persistent_fn', 'std_persistent_fn',
    'mean_newly_introduced_fn', 'std_newly_introduced_fn',
    'mean_fn_reduction', 'std_fn_reduction',
    'mean_corrected_fp', 'std_corrected_fp',
    'mean_persistent_fp', 'std_persistent_fp',
    'mean_newly_introduced_fp', 'std_newly_introduced_fp',
    'mean_fp_reduction', 'std_fp_reduction',
    'n_seeds', 'reproduction_status'
]

# FIX #2: Expected weight-sensitivity combinations
EXPECTED_WS_COMBINATIONS = set()
for model in ['XGBoost', 'RandomForest', 'LSTM']:
    for weight in HISTORICAL_WEIGHT_GRID:
        EXPECTED_WS_COMBINATIONS.add((model, weight, 'default'))
        EXPECTED_WS_COMBINATIONS.add((model, weight, 'test_set_diagnostic'))
for weight in HISTORICAL_WEIGHT_GRID:
    EXPECTED_WS_COMBINATIONS.add(('LinearSVM', weight, 'default'))

# =============================================================================
# FIGURE GENERATION CLASS
# =============================================================================
class FinalFigureGenerator:
    """Generate final four-model report figures from validated data sources."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        
        # Authoritative data source paths
        self.ablation_results_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'ablation' / 'metrics' / 'four_model_ablation_results.csv'
        self.weight_sensitivity_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'report_extensions' / 'weight_sensitivity' / 'four_model_weight_sensitivity_results.csv'
        self.transitions_summary_path = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'report_extensions' / 'transitions' / 'four_model_transitions_summary.csv'
        
        # Output directory
        self.figures_dir = self.base_dir / 'adaptive' / 'final_validation_four_model' / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.ablation_df = None
        self.weight_sensitivity_df = None
        self.transitions_df = None
        
        # Generated figures manifest
        self.generated_figures = []
        
        # Track mandatory figure success
        self.mandatory_figures_success = {i: False for i in range(1, 10)}
        
        logger.info(f"✅ Initialized FinalFigureGenerator")
        logger.info(f"📁 Figures directory: {self.figures_dir}")
    
    def _load_and_validate_ablation(self) -> bool:
        """Load and validate final ablation results (STRICT/FAIL-CLOSED)."""
        logger.info("📊 Loading ablation results...")
        
        if not self.ablation_results_path.exists():
            logger.error(f"❌ Ablation results not found: {self.ablation_results_path}")
            return False
        
        try:
            self.ablation_df = pd.read_csv(self.ablation_results_path)
            logger.info(f"✅ Loaded ablation results: {len(self.ablation_df)} rows")
            
            # Require EXACT model set
            models_found = set(self.ablation_df['model_name'].unique())
            if models_found != EXPECTED_MODEL_SET:
                logger.error(f"❌ Model mismatch. Expected {EXPECTED_MODEL_SET}, got {models_found}")
                return False
            
            # Require EXACT seed set
            seeds_found = set(self.ablation_df['seed'].unique())
            if seeds_found != EXPECTED_SEED_SET:
                logger.error(f"❌ Seed mismatch. Expected {EXPECTED_SEED_SET}, got {seeds_found}")
                return False
            
            # Require essential columns
            required_columns = ['model_name', 'seed', 'variant', 'accuracy', 'precision', 
                               'recall', 'f1_score', 'false_positive', 'false_negative']
            missing_cols = [col for col in required_columns if col not in self.ablation_df.columns]
            if missing_cols:
                logger.error(f"❌ Missing essential columns: {missing_cols}")
                return False
            
            # Validate each model/variant has exactly 5 rows (one per seed)
            required_variants = ['baseline', 'weighted_retraining']
            for model_name in MODELS:
                for variant in required_variants:
                    subset = self.ablation_df[
                        (self.ablation_df['model_name'] == model_name) &
                        (self.ablation_df['variant'] == variant)
                    ]
                    
                    if len(subset) != 5:
                        logger.error(f"❌ {model_name} {variant} should have 5 rows, got {len(subset)}")
                        return False
                    
                    # Check for duplicates
                    if subset.duplicated(subset=['seed']).any():
                        logger.error(f"❌ Duplicate (model_name, variant, seed) for {model_name} {variant}")
                        return False
                    
                    seed_set = set(subset['seed'].unique())
                    if seed_set != EXPECTED_SEED_SET:
                        logger.error(f"❌ {model_name} {variant} seed mismatch. Expected {EXPECTED_SEED_SET}, got {seed_set}")
                        return False
            
            logger.info(f"✅ Ablation validation passed: {len(models_found)} models, {len(seeds_found)} seeds")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ablation results: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_and_validate_weight_sensitivity(self) -> bool:
        """Load and validate weight sensitivity results (FIX #2: EXACT COMBINATION COVERAGE)."""
        logger.info("📊 Loading weight sensitivity results...")
        
        if not self.weight_sensitivity_path.exists():
            logger.error(f"❌ Weight sensitivity results not found: {self.weight_sensitivity_path}")
            return False
        
        try:
            self.weight_sensitivity_df = pd.read_csv(self.weight_sensitivity_path)
            logger.info(f"✅ Loaded weight sensitivity: {len(self.weight_sensitivity_df)} rows")
            
            # Require EXACTLY 35 rows
            if len(self.weight_sensitivity_df) != EXPECTED_WEIGHT_SENSITIVITY_ROWS:
                logger.error(f"❌ Expected {EXPECTED_WEIGHT_SENSITIVITY_ROWS} rows, got {len(self.weight_sensitivity_df)}")
                return False
            
            # Require seed = 42 ONLY
            seed_set = set(self.weight_sensitivity_df['seed'].unique())
            if seed_set != {42}:
                logger.error(f"❌ Weight sensitivity must be seed 42 only, got {seed_set}")
                return False
            
            # Require exact weight set
            weights_found = set(self.weight_sensitivity_df['anomaly_weight'].unique())
            if weights_found != set(HISTORICAL_WEIGHT_GRID):
                logger.error(f"❌ Weight mismatch. Expected {set(HISTORICAL_WEIGHT_GRID)}, got {weights_found}")
                return False
            
            # Require exact model set
            models_found = set(self.weight_sensitivity_df['model_name'].unique())
            if models_found != EXPECTED_MODEL_SET:
                logger.error(f"❌ Model mismatch. Expected {EXPECTED_MODEL_SET}, got {models_found}")
                return False
            
            # FIX #2: Validate EXACT expected combinations
            actual_combinations = set()
            for _, row in self.weight_sensitivity_df.iterrows():
                combo = (row['model_name'], row['anomaly_weight'], row['threshold_mode'])
                actual_combinations.add(combo)
            
            if actual_combinations != EXPECTED_WS_COMBINATIONS:
                missing = EXPECTED_WS_COMBINATIONS - actual_combinations
                extra = actual_combinations - EXPECTED_WS_COMBINATIONS
                if missing:
                    logger.error(f"❌ Missing combinations: {missing}")
                if extra:
                    logger.error(f"❌ Extra combinations: {extra}")
                return False
            
            # Check for duplicates
            if self.weight_sensitivity_df.duplicated(subset=['model_name', 'anomaly_weight', 'threshold_mode']).any():
                logger.error(f"❌ Duplicate (model_name, anomaly_weight, threshold_mode) found")
                return False
            
            # Validate evaluation_n by model
            for _, row in self.weight_sensitivity_df.iterrows():
                model = row.get('model_name')
                eval_n = row.get('evaluation_n')
                
                if model == 'LSTM':
                    if eval_n != 288335:
                        logger.error(f"❌ LSTM evaluation_n should be 288335, got {eval_n}")
                        return False
                else:
                    if eval_n != 288344:
                        logger.error(f"❌ {model} evaluation_n should be 288344, got {eval_n}")
                        return False
            
            logger.info(f"✅ Weight sensitivity validation passed: {len(weights_found)} weights")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load weight sensitivity: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_and_validate_transitions(self) -> bool:
        """Load and validate transition summary (FIX #1: STRICT COLUMN VALIDATION)."""
        logger.info("📊 Loading transition summary...")
        
        if not self.transitions_summary_path.exists():
            logger.error(f"❌ Transitions summary not found: {self.transitions_summary_path}")
            return False
        
        try:
            self.transitions_df = pd.read_csv(self.transitions_summary_path)
            logger.info(f"✅ Loaded transitions: {len(self.transitions_df)} rows")
            
            # Require exactly 4 rows (one per model)
            if len(self.transitions_df) != 4:
                logger.error(f"❌ Expected 4 transition summary rows, got {len(self.transitions_df)}")
                return False
            
            # FIX #1: Require ALL required columns to exist (not skipping missing)
            missing_cols = [col for col in REQUIRED_TRANSITION_COLUMNS if col not in self.transitions_df.columns]
            if missing_cols:
                logger.error(f"❌ Missing required transition columns: {missing_cols}")
                return False
            
            # Validate XGBoost, RandomForest, LinearSVM
            for model_name in ['XGBoost', 'RandomForest', 'LinearSVM']:
                model_data = self.transitions_df[self.transitions_df['model_name'] == model_name]
                if len(model_data) != 1:
                    logger.error(f"❌ {model_name} should have 1 summary row, got {len(model_data)}")
                    return False
                
                # Require n_seeds == 5
                n_seeds = model_data['n_seeds'].values[0]
                if n_seeds != 5:
                    logger.error(f"❌ {model_name} should have 5 seeds, got {n_seeds}")
                    return False
                
                # FIX #1: Require ALL mean/std fields to be finite (no skipping)
                for field in REQUIRED_TRANSITION_COLUMNS:
                    if field.startswith('mean_') or field.startswith('std_'):
                        val = model_data[field].values[0]
                        if pd.isna(val) or not np.isfinite(val):
                            logger.error(f"❌ {model_name} {field} is not finite: {val}")
                            return False
            
            # Validate LSTM
            lstm_data = self.transitions_df[self.transitions_df['model_name'] == 'LSTM']
            if len(lstm_data) != 1:
                logger.error(f"❌ LSTM should have 1 summary row, got {len(lstm_data)}")
                return False
            
            if lstm_data['reproduction_status'].values[0] != 'sample_level_transitions_unavailable':
                logger.error(f"❌ LSTM should be marked as sample_level_transitions_unavailable")
                return False
            
            if lstm_data['n_seeds'].values[0] != 0:
                logger.error(f"❌ LSTM n_seeds should be 0, got {lstm_data['n_seeds'].values[0]}")
                return False
            
            logger.info(f"✅ Transitions validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load transitions: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _compute_ablation_aggregates(self) -> Tuple[pd.DataFrame, bool]:
        """
        Compute mean ± sample std for ablation data (five-seed).
        Aggregate ALL variants present in CSV.
        Returns (aggregates_df, has_retraining_variant).
        """
        logger.info("📊 Computing ablation aggregates...")
        
        aggregates = []
        has_retraining_variant = False
        
        # Get all unique variants in the data
        all_variants = set(self.ablation_df['variant'].unique())
        
        # Include retraining_only or retraining if present
        retraining_variants = [v for v in all_variants if 'retraining' in v.lower() and v != 'weighted_retraining']
        
        if retraining_variants:
            has_retraining_variant = True
            logger.info(f"✅ Found retraining variant(s): {retraining_variants}")
        
        for model_name in MODELS:
            model_data = self.ablation_df[self.ablation_df['model_name'] == model_name]
            
            # Aggregate baseline
            baseline_data = model_data[model_data['variant'] == 'baseline']
            if len(baseline_data) == 5:
                agg_row = self._create_aggregate_row(model_name, 'baseline', baseline_data)
                aggregates.append(agg_row)
            
            # Aggregate retraining variant if available
            for retrain_variant in retraining_variants:
                retrain_data = model_data[model_data['variant'] == retrain_variant]
                if len(retrain_data) == 5:
                    agg_row = self._create_aggregate_row(model_name, retrain_variant, retrain_data)
                    aggregates.append(agg_row)
            
            # Aggregate weighted_retraining
            final_data = model_data[model_data['variant'] == 'weighted_retraining']
            if len(final_data) == 5:
                agg_row = self._create_aggregate_row(model_name, 'weighted_retraining', final_data)
                aggregates.append(agg_row)
        
        return pd.DataFrame(aggregates), has_retraining_variant
    
    def _create_aggregate_row(self, model_name: str, variant: str, data: pd.DataFrame) -> Dict:
        """Helper to create aggregate row for a model/variant."""
        agg_row = {
            'model_name': model_name,
            'variant': variant,
            'n_seeds': len(data)
        }
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'false_positive', 'false_negative']:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) > 0:
                    agg_row[f'{metric}_mean'] = values.mean()
                    agg_row[f'{metric}_std'] = values.std(ddof=1) if len(values) > 1 else 0.0
        
        return agg_row
    
    def generate_figure_01_f1_comparison(self, ablation_agg: pd.DataFrame) -> bool:
        """FIGURE 1: Final F1 comparison - baseline vs weighted_retraining."""
        logger.info("📊 Generating Figure 1: F1 Comparison...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            baseline_means = []
            baseline_stds = []
            final_means = []
            final_stds = []
            
            for i, model_name in enumerate(MODELS):
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 1")
                    return False
                
                baseline_means.append(baseline_data['f1_score_mean'].values[0])
                baseline_stds.append(baseline_data['f1_score_std'].values[0])
                final_means.append(final_data['f1_score_mean'].values[0])
                final_stds.append(final_data['f1_score_std'].values[0])
            
            bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                          label='Baseline', color='#1f77b4', capsize=5)
            bars2 = ax.bar(x + width/2, final_means, width, yerr=final_stds, 
                          label='Final (Weighted Retraining)', color='#2ca02c', capsize=5)
            
            ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            ax.set_title('Final Four-Model F1 Comparison\n(Baseline vs Final Adaptive)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_01_final_f1_comparison.png'
            svg_path = self.figures_dir / 'figure_01_final_f1_comparison.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(1, 'figure_01_final_f1_comparison', 'Ablation results (five-seed)', 
                                 MODELS, 'Five-seed final validation', '')
            
            self.mandatory_figures_success[1] = True
            logger.info(f"✅ Saved Figure 1: {png_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 1: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_figure_02_recall_comparison(self, ablation_agg: pd.DataFrame) -> bool:
        """FIGURE 2: Final Recall comparison."""
        logger.info("📊 Generating Figure 2: Recall Comparison...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            baseline_means = []
            baseline_stds = []
            final_means = []
            final_stds = []
            
            for i, model_name in enumerate(MODELS):
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 2")
                    return False
                
                baseline_means.append(baseline_data['recall_mean'].values[0])
                baseline_stds.append(baseline_data['recall_std'].values[0])
                final_means.append(final_data['recall_mean'].values[0])
                final_stds.append(final_data['recall_std'].values[0])
            
            bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                          label='Baseline', color='#1f77b4', capsize=5)
            bars2 = ax.bar(x + width/2, final_means, width, yerr=final_stds, 
                          label='Final (Weighted Retraining)', color='#2ca02c', capsize=5)
            
            ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
            ax.set_title('Final Four-Model Recall Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_02_final_recall_comparison.png'
            svg_path = self.figures_dir / 'figure_02_final_recall_comparison.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(2, 'figure_02_final_recall_comparison', 'Ablation results (five-seed)', 
                                 MODELS, 'Five-seed final validation', '')
            
            self.mandatory_figures_success[2] = True
            logger.info(f"✅ Saved Figure 2")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 2: {e}")
            return False
    
    def generate_figure_03_precision_comparison(self, ablation_agg: pd.DataFrame) -> bool:
        """FIGURE 3: Final Precision comparison."""
        logger.info("📊 Generating Figure 3: Precision Comparison...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            baseline_means = []
            baseline_stds = []
            final_means = []
            final_stds = []
            
            for i, model_name in enumerate(MODELS):
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 3")
                    return False
                
                baseline_means.append(baseline_data['precision_mean'].values[0])
                baseline_stds.append(baseline_data['precision_std'].values[0])
                final_means.append(final_data['precision_mean'].values[0])
                final_stds.append(final_data['precision_std'].values[0])
            
            bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                          label='Baseline', color='#1f77b4', capsize=5)
            bars2 = ax.bar(x + width/2, final_means, width, yerr=final_stds, 
                          label='Final (Weighted Retraining)', color='#2ca02c', capsize=5)
            
            ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
            ax.set_title('Final Four-Model Precision Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_03_final_precision_comparison.png'
            svg_path = self.figures_dir / 'figure_03_final_precision_comparison.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(3, 'figure_03_final_precision_comparison', 'Ablation results (five-seed)', 
                                 MODELS, 'Five-seed final validation', '')
            
            self.mandatory_figures_success[3] = True
            logger.info(f"✅ Saved Figure 3")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 3: {e}")
            return False
    
    def generate_figure_04_false_negatives(self, ablation_agg: pd.DataFrame) -> bool:
        """FIGURE 4: False Negatives comparison."""
        logger.info("📊 Generating Figure 4: False Negatives...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            baseline_means = []
            baseline_stds = []
            final_means = []
            final_stds = []
            
            for i, model_name in enumerate(MODELS):
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 4")
                    return False
                
                baseline_means.append(baseline_data['false_negative_mean'].values[0])
                baseline_stds.append(baseline_data['false_negative_std'].values[0])
                final_means.append(final_data['false_negative_mean'].values[0])
                final_stds.append(final_data['false_negative_std'].values[0])
            
            bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                          label='Baseline', color='#d62728', capsize=5)
            bars2 = ax.bar(x + width/2, final_means, width, yerr=final_stds, 
                          label='Final (Weighted Retraining)', color='#2ca02c', capsize=5)
            
            ax.set_ylabel('False Negatives (Count)', fontsize=12, fontweight='bold')
            ax.set_title('Final Four-Model False Negatives Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_04_final_false_negatives.png'
            svg_path = self.figures_dir / 'figure_04_final_false_negatives.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(4, 'figure_04_final_false_negatives', 'Ablation results (five-seed)', 
                                 MODELS, 'Five-seed final validation', '')
            
            self.mandatory_figures_success[4] = True
            logger.info(f"✅ Saved Figure 4")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 4: {e}")
            return False
    
    def generate_figure_05_false_positives(self, ablation_agg: pd.DataFrame) -> bool:
        """FIGURE 5: False Positives comparison."""
        logger.info("📊 Generating Figure 5: False Positives...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(MODELS))
            width = 0.35
            
            baseline_means = []
            baseline_stds = []
            final_means = []
            final_stds = []
            
            for i, model_name in enumerate(MODELS):
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 5")
                    return False
                
                baseline_means.append(baseline_data['false_positive_mean'].values[0])
                baseline_stds.append(baseline_data['false_positive_std'].values[0])
                final_means.append(final_data['false_positive_mean'].values[0])
                final_stds.append(final_data['false_positive_std'].values[0])
            
            bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, 
                          label='Baseline', color='#d62728', capsize=5)
            bars2 = ax.bar(x + width/2, final_means, width, yerr=final_stds, 
                          label='Final (Weighted Retraining)', color='#2ca02c', capsize=5)
            
            ax.set_ylabel('False Positives (Count)', fontsize=12, fontweight='bold')
            ax.set_title('Final Four-Model False Positives Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_05_final_false_positives.png'
            svg_path = self.figures_dir / 'figure_05_final_false_positives.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(5, 'figure_05_final_false_positives', 'Ablation results (five-seed)', 
                                 MODELS, 'Five-seed final validation', '')
            
            self.mandatory_figures_success[5] = True
            logger.info(f"✅ Saved Figure 5")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 5: {e}")
            return False
    
    def generate_figure_06_weight_sensitivity_f1(self) -> bool:
        """FIGURE 6: Weight sensitivity F1 analysis (seed-42 retrospective diagnostic)."""
        logger.info("📊 Generating Figure 6: Weight Sensitivity F1...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, model_name in enumerate(MODELS):
                ax = axes[idx]
                
                model_data = self.weight_sensitivity_df[self.weight_sensitivity_df['model_name'] == model_name]
                
                if len(model_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    continue
                
                weights = sorted(HISTORICAL_WEIGHT_GRID)
                
                # Plot default threshold mode
                default_data = model_data[model_data['threshold_mode'] == 'default'].sort_values('anomaly_weight')
                if len(default_data) == 5:
                    default_f1 = default_data['f1_score'].values
                    ax.plot(weights, default_f1, 'o-', label='Default Threshold', 
                           color=MODEL_COLORS[model_name], markersize=8, linewidth=2)
                else:
                    logger.error(f"❌ {model_name} default mode missing weights")
                    return False
                
                # Plot diagnostic threshold mode for probability models
                if model_name != 'LinearSVM':
                    diagnostic_data = model_data[model_data['threshold_mode'] == 'test_set_diagnostic'].sort_values('anomaly_weight')
                    if len(diagnostic_data) == 5:
                        diagnostic_f1 = diagnostic_data['f1_score'].values
                        ax.plot(weights, diagnostic_f1, 's--', label='Best Diagnostic Threshold\n(within predefined grid)', 
                               color='#ff7f0e', markersize=8, linewidth=2)
                    else:
                        logger.error(f"❌ {model_name} diagnostic mode missing weights")
                        return False
                else:
                    ax.text(0.02, 0.98, 'LinearSVM:\nDefault threshold only\n(decision_function ≥ 0)', 
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('Anomaly Weight', fontsize=11, fontweight='bold')
                ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name} - Seed 42 Retrospective Diagnostic', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(weights)
            
            fig.suptitle('Weight Sensitivity Analysis - Seed 42 Retrospective Diagnostic\n(Not five-seed robustness validation)', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_06_weight_sensitivity_f1.png'
            svg_path = self.figures_dir / 'figure_06_weight_sensitivity_f1.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(6, 'figure_06_weight_sensitivity_f1', 'Weight sensitivity results', 
                                 MODELS, 'Seed-42 retrospective diagnostic', 
                                 'LinearSVM: default threshold only; LSTM eval_n=288335')
            
            self.mandatory_figures_success[6] = True
            logger.info(f"✅ Saved Figure 6")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 6: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_figure_07_weight_sensitivity_fn(self) -> bool:
        """FIGURE 7: Weight sensitivity False Negatives (FIX #3: FAIL-CLOSED)."""
        logger.info("📊 Generating Figure 7: Weight Sensitivity FN...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, model_name in enumerate(MODELS):
                ax = axes[idx]
                
                model_data = self.weight_sensitivity_df[self.weight_sensitivity_df['model_name'] == model_name]
                
                if len(model_data) == 0:
                    continue
                
                weights = sorted(HISTORICAL_WEIGHT_GRID)
                
                # FIX #3: Require exactly 5 ordered weights for default mode
                default_data = model_data[model_data['threshold_mode'] == 'default'].sort_values('anomaly_weight')
                if len(default_data) != 5:
                    logger.error(f"❌ {model_name} default mode must have exactly 5 weights, got {len(default_data)}")
                    return False
                
                default_fn = default_data['false_negative'].values
                ax.plot(weights, default_fn, 'o-', label='Default Threshold', 
                       color=MODEL_COLORS[model_name], markersize=8, linewidth=2)
                
                # FIX #3: Require exactly 5 ordered weights for diagnostic mode (XGBoost/RF/LSTM)
                if model_name != 'LinearSVM':
                    diagnostic_data = model_data[model_data['threshold_mode'] == 'test_set_diagnostic'].sort_values('anomaly_weight')
                    if len(diagnostic_data) != 5:
                        logger.error(f"❌ {model_name} diagnostic mode must have exactly 5 weights, got {len(diagnostic_data)}")
                        return False
                    
                    diagnostic_fn = diagnostic_data['false_negative'].values
                    ax.plot(weights, diagnostic_fn, 's--', label='Best Diagnostic Threshold', 
                           color='#ff7f0e', markersize=8, linewidth=2)
                
                ax.set_xlabel('Anomaly Weight', fontsize=11, fontweight='bold')
                ax.set_ylabel('False Negatives (Count)', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_name} - Seed 42', fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(weights)
            
            fig.suptitle('Weight Sensitivity False Negatives - Seed 42 Retrospective Diagnostic', 
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_07_weight_sensitivity_fn.png'
            svg_path = self.figures_dir / 'figure_07_weight_sensitivity_fn.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(7, 'figure_07_weight_sensitivity_fn', 'Weight sensitivity results', 
                                 MODELS, 'Seed-42 retrospective diagnostic', '')
            
            self.mandatory_figures_success[7] = True
            logger.info(f"✅ Saved Figure 7")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 7: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_figure_08_fn_transition_summary(self) -> bool:
        """FIGURE 8: FN transition summary (ACTUAL ERROR BARS)."""
        logger.info("📊 Generating Figure 8: FN Transition Summary...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            transition_models = ['XGBoost', 'RandomForest', 'LinearSVM']
            x = np.arange(len(transition_models))
            width = 0.25
            
            corrected_fn = []
            corrected_fn_std = []
            persistent_fn = []
            persistent_fn_std = []
            newly_fn = []
            newly_fn_std = []
            
            for model_name in transition_models:
                model_data = self.transitions_df[self.transitions_df['model_name'] == model_name]
                if len(model_data) != 1:
                    logger.error(f"❌ Missing transition data for {model_name}")
                    return False
                
                corrected_fn.append(model_data['mean_corrected_fn'].values[0])
                corrected_fn_std.append(model_data['std_corrected_fn'].values[0])
                persistent_fn.append(model_data['mean_persistent_fn'].values[0])
                persistent_fn_std.append(model_data['std_persistent_fn'].values[0])
                newly_fn.append(model_data['mean_newly_introduced_fn'].values[0])
                newly_fn_std.append(model_data['std_newly_introduced_fn'].values[0])
            
            bars1 = ax.bar(x - width, corrected_fn, width, yerr=corrected_fn_std, 
                          label='Corrected FN', color='#2ca02c', capsize=5)
            bars2 = ax.bar(x, persistent_fn, width, yerr=persistent_fn_std, 
                          label='Persistent FN', color='#1f77b4', capsize=5)
            bars3 = ax.bar(x + width, newly_fn, width, yerr=newly_fn_std, 
                          label='Newly Introduced FN', color='#d62728', capsize=5)
            
            ax.set_ylabel('Mean Count (±1 Sample Std)', fontsize=12, fontweight='bold')
            ax.set_title('False Negative Transition Summary\n(Mean ± Sample Std across 5 Seeds)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(transition_models, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            ax.text(0.98, 0.02, 'Note: LSTM sample-level\ntransitions unavailable', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_08_fn_transition_summary.png'
            svg_path = self.figures_dir / 'figure_08_fn_transition_summary.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(8, 'figure_08_fn_transition_summary', 'Transition summary', 
                                 transition_models, 'Five-seed validation', 'LSTM transitions unavailable')
            
            self.mandatory_figures_success[8] = True
            logger.info(f"✅ Saved Figure 8")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 8: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def generate_figure_09_fp_transition_summary(self) -> bool:
        """FIGURE 9: FP transition summary (ACTUAL ERROR BARS)."""
        logger.info("📊 Generating Figure 9: FP Transition Summary...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            transition_models = ['XGBoost', 'RandomForest', 'LinearSVM']
            x = np.arange(len(transition_models))
            width = 0.25
            
            corrected_fp = []
            corrected_fp_std = []
            persistent_fp = []
            persistent_fp_std = []
            newly_fp = []
            newly_fp_std = []
            
            for model_name in transition_models:
                model_data = self.transitions_df[self.transitions_df['model_name'] == model_name]
                if len(model_data) != 1:
                    logger.error(f"❌ Missing transition data for {model_name}")
                    return False
                
                corrected_fp.append(model_data['mean_corrected_fp'].values[0])
                corrected_fp_std.append(model_data['std_corrected_fp'].values[0])
                persistent_fp.append(model_data['mean_persistent_fp'].values[0])
                persistent_fp_std.append(model_data['std_persistent_fp'].values[0])
                newly_fp.append(model_data['mean_newly_introduced_fp'].values[0])
                newly_fp_std.append(model_data['std_newly_introduced_fp'].values[0])
            
            bars1 = ax.bar(x - width, corrected_fp, width, yerr=corrected_fp_std, 
                          label='Corrected FP', color='#2ca02c', capsize=5)
            bars2 = ax.bar(x, persistent_fp, width, yerr=persistent_fp_std, 
                          label='Persistent FP', color='#1f77b4', capsize=5)
            bars3 = ax.bar(x + width, newly_fp, width, yerr=newly_fp_std, 
                          label='Newly Introduced FP', color='#d62728', capsize=5)
            
            ax.set_ylabel('Mean Count (±1 Sample Std)', fontsize=12, fontweight='bold')
            ax.set_title('False Positive Transition Summary\n(Mean ± Sample Std across 5 Seeds)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(transition_models, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            ax.text(0.98, 0.02, 'Note: LSTM sample-level\ntransitions unavailable', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_09_fp_transition_summary.png'
            svg_path = self.figures_dir / 'figure_09_fp_transition_summary.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(9, 'figure_09_fp_transition_summary', 'Transition summary', 
                                 transition_models, 'Five-seed validation', 'LSTM transitions unavailable')
            
            self.mandatory_figures_success[9] = True
            logger.info(f"✅ Saved Figure 9")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 9: {e}")
            return False
    
    def generate_figure_10_ablation_contribution(self, ablation_agg: pd.DataFrame, has_retraining: bool) -> Optional[bool]:
        """FIGURE 10: Ablation component contribution (USE REAL RETRAINING DATA)."""
        logger.info("📊 Generating Figure 10: Ablation Component Contribution...")
        
        try:
            if not has_retraining:
                logger.info("⚠️  Skipping Figure 10: retraining variant not available in ablation data")
                return None
            
            all_variants = set(ablation_agg['variant'].unique())
            retraining_variants = [v for v in all_variants if 'retraining' in v.lower() and v != 'weighted_retraining']
            
            if not retraining_variants:
                logger.info("⚠️  Skipping Figure 10: no retraining variant found")
                return None
            
            retrain_variant = retraining_variants[0]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(MODELS))
            width = 0.25
            
            baseline_f1 = []
            retrain_f1 = []
            final_f1 = []
            
            for model_name in MODELS:
                baseline_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'baseline')]
                retrain_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == retrain_variant)]
                final_data = ablation_agg[(ablation_agg['model_name'] == model_name) & (ablation_agg['variant'] == 'weighted_retraining')]
                
                if len(baseline_data) != 1 or len(retrain_data) != 1 or len(final_data) != 1:
                    logger.error(f"❌ Missing data for {model_name} in Figure 10")
                    return False
                
                baseline_f1.append(baseline_data['f1_score_mean'].values[0])
                retrain_f1.append(retrain_data['f1_score_mean'].values[0])
                final_f1.append(final_data['f1_score_mean'].values[0])
            
            bars1 = ax.bar(x - width, baseline_f1, width, label='Baseline', color='#1f77b4')
            bars2 = ax.bar(x, retrain_f1, width, label=f'Retraining Only', color='#ff7f0e')
            bars3 = ax.bar(x + width, final_f1, width, label='Final (Weighted)', color='#2ca02c')
            
            ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            ax.set_title('Ablation Component Contribution to F1\n(Baseline → Retraining → Final Weighted)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=11)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            ax.text(0.98, 0.02, 'Note: LSTM final selected\nanomaly weight = 1.0\n(no up-weighting applied)', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            png_path = self.figures_dir / 'figure_10_ablation_component_contribution.png'
            svg_path = self.figures_dir / 'figure_10_ablation_component_contribution.svg'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(svg_path, bbox_inches='tight')
            plt.close()
            
            self._add_to_manifest(10, 'figure_10_ablation_component_contribution', 'Ablation results', 
                                 MODELS, 'Five-seed validation', 'LSTM weight=1.0 (no up-weighting)')
            
            logger.info(f"✅ Saved Figure 10")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate Figure 10: {e}")
            return False
    
    def _add_to_manifest(self, figure_num: int, filename_base: str, data_source: str, 
                        models_included: List[str], analysis_scope: str, notes: str):
        """Add figure to manifest list."""
        self.generated_figures.append({
            'figure_number': figure_num,
            'filename_png': f'{filename_base}.png',
            'filename_svg': f'{filename_base}.svg',
            'data_source': data_source,
            'models_included': '; '.join(models_included),
            'analysis_scope': analysis_scope,
            'notes': notes
        })
    
    def save_manifest(self):
        """Save figure manifest CSV."""
        logger.info("💾 Saving figure manifest...")
        
        manifest_df = pd.DataFrame(self.generated_figures)
        manifest_path = self.figures_dir / 'final_four_model_figure_manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        
        logger.info(f"✅ Saved manifest: {manifest_path}")
    
    def generate_report(self):
        """Generate final figure report."""
        logger.info("📝 Generating figure report...")
        
        report = []
        report.append("="*80)
        report.append("FINAL FOUR-MODEL REPORT FIGURES")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Framework: CPS Anomaly Detection Framework")
        report.append(f"Thesis: Master's Thesis Research - Dalian University of Technology")
        report.append("")
        
        report.append("DATA SOURCES:")
        report.append(f"1. Ablation results (FIVE-SEED): {self.ablation_results_path}")
        report.append(f"2. Weight sensitivity (SEED-42 DIAGNOSTIC): {self.weight_sensitivity_path}")
        report.append(f"3. Transition summary (FIVE-SEED): {self.transitions_summary_path}")
        report.append("")
        
        report.append("FIGURES GENERATED:")
        for fig in self.generated_figures:
            report.append(f"Figure {fig['figure_number']}: {fig['filename_png']}")
            report.append(f"  - Data source: {fig['data_source']}")
            report.append(f"  - Models: {fig['models_included']}")
            report.append(f"  - Analysis scope: {fig['analysis_scope']}")
            if fig['notes']:
                report.append(f"  - Notes: {fig['notes']}")
            report.append("")
        
        report.append("IMPORTANT SCIENTIFIC NOTES:")
        report.append("- Final conclusions are model-dependent.")
        report.append("- Supervised classifiers (XGBoost, RF, LinearSVM) benefit from controlled retraining and anomaly-aware weighting.")
        report.append("- LSTM benefits from additional retraining data, but selected final anomaly weight is 1.0 (no up-weighting).")
        report.append("- Weight sensitivity is SEED-42 RETROSPECTIVE DIAGNOSTIC only (not five-seed robustness).")
        report.append("- Main final conclusions come from five-seed final validation/ablation.")
        report.append("- LSTM evaluation population: 288335 sequence endpoints.")
        report.append("- Classical models: 288344 test samples.")
        report.append("- LSTM sample-level transitions: UNAVAILABLE (not fabricated).")
        report.append("- Diagnostic threshold selection is NOT claimed as 'optimal'.")
        report.append("")
        
        report.append("="*80)
        report.append("END OF FINAL FOUR-MODEL FIGURE REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        report_path = self.figures_dir / 'final_four_model_figure_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✅ Saved report: {report_path}")
    
    def run_full_generation(self) -> bool:
        """Run complete figure generation (STRICT SUCCESS CONDITION)."""
        logger.info("\n" + "="*80)
        logger.info("🎯 RUNNING FINAL FOUR-MODEL FIGURE GENERATION")
        logger.info("="*80)
        
        if not self._load_and_validate_ablation():
            logger.error("❌ Ablation validation failed")
            return False
        
        if not self._load_and_validate_weight_sensitivity():
            logger.error("❌ Weight sensitivity validation failed")
            return False
        
        if not self._load_and_validate_transitions():
            logger.error("❌ Transitions validation failed")
            return False
        
        ablation_agg, has_retraining = self._compute_ablation_aggregates()
        
        figures_generated = 0
        
        if self.generate_figure_01_f1_comparison(ablation_agg):
            figures_generated += 1
        
        if self.generate_figure_02_recall_comparison(ablation_agg):
            figures_generated += 1
        
        if self.generate_figure_03_precision_comparison(ablation_agg):
            figures_generated += 1
        
        if self.generate_figure_04_false_negatives(ablation_agg):
            figures_generated += 1
        
        if self.generate_figure_05_false_positives(ablation_agg):
            figures_generated += 1
        
        if self.generate_figure_06_weight_sensitivity_f1():
            figures_generated += 1
        
        if self.generate_figure_07_weight_sensitivity_fn():
            figures_generated += 1
        
        if self.generate_figure_08_fn_transition_summary():
            figures_generated += 1
        
        if self.generate_figure_09_fp_transition_summary():
            figures_generated += 1
        
        fig10_result = self.generate_figure_10_ablation_contribution(ablation_agg, has_retraining)
        if fig10_result is True:
            figures_generated += 1
        
        all_mandatory_success = all(self.mandatory_figures_success[i] for i in range(1, 10))
        
        self.save_manifest()
        self.generate_report()
        
        logger.info("\n" + "="*80)
        
        if all_mandatory_success:
            logger.info("FINAL FOUR-MODEL FIGURE GENERATION: COMPLETE")
            logger.info("="*80)
            logger.info(f"📊 Mandatory figures generated: {sum(1 for v in self.mandatory_figures_success.values() if v)}/9")
            if fig10_result is True:
                logger.info(f"📊 Optional Figure 10 generated: YES")
            logger.info(f"📁 Output directory: {self.figures_dir}")
            
            for fig in self.generated_figures:
                logger.info(f"   - {fig['filename_png']}")
            
            logger.info("="*80)
            return True
        else:
            logger.error("❌ One or more mandatory figures failed")
            logger.error(f"   Success status: {self.mandatory_figures_success}")
            logger.info("="*80)
            return False


# =============================================================================
# GLOBAL LOGGER
# =============================================================================
logger = None


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Execute final figure generation."""
    global logger
    
    start_time = time.time()
    
    config = FrameworkConfig()
    logger = setup_logging(config.output.logs_dir, 'generate_final_four_model_figures')
    
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL FOUR-MODEL REPORT FIGURE GENERATION")
    logger.info("📚 Master's Thesis Research - Dalian University of Technology")
    logger.info("👤 Author: BERETE MAMADY")
    logger.info("="*80 + "\n")
    
    base_dir = Path(config.output.base_dir)
    
    logger.info(f"📁 Base directory: {base_dir}")
    logger.info(f"⚠️  READ-ONLY operation (no model training)")
    logger.info(f"⚠️  Uses authoritative final data sources only")
    
    generator = FinalFigureGenerator(base_dir=base_dir)
    success = generator.run_full_generation()
    
    elapsed_time = time.time() - start_time
    
    if success:
        logger.info(f"⏱️  Execution time: {elapsed_time:.2f}s")
        logger.info("✅ FINAL FOUR-MODEL FIGURE GENERATION: COMPLETE")
    else:
        logger.error("❌ Figure generation FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()