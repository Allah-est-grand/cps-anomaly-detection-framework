"""
Visualization Module - Academic-Quality Figures for Thesis
Master's Thesis Research - Dalian University of Technology
Author: BERETE MAMADY
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CPSVisualizer:
    """
    Visualization framework for CPS anomaly detection research
    Generates publication-quality figures
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for academic publications
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 11
    
    def plot_model_comparison(self, results: Dict, metric: str = 'f1_score',
                             save_name: str = 'model_comparison.png'):
        """
        Plot comparison of models for a specific metric
        
        Args:
            results: Dictionary with model results
            metric: Metric to plot ('accuracy', 'f1_score', etc.)
            save_name: Output filename
        """
        logger.info(f"📊 Generating {metric} comparison plot...")
        
        models = list(results.keys())
        values = [results[model].get(metric, 0) * 100 for model in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#34495e', '#16a085']
        
        bars = ax.bar(models, values, color=colors[:len(models)])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, 100])
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str,
                             class_names: List[str] = None,
                             save_name: str = 'confusion_matrix.png'):
        """
        Plot confusion matrix heatmap
        
        Args:
            cm: Confusion matrix (2x2)
            model_name: Name of the model
            class_names: Names for classes
            save_name: Output filename
        """
        logger.info(f"📊 Generating confusion matrix for {model_name}...")
        
        if class_names is None:
            class_names = ['Normal', 'Attack']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # FIX: Convert confusion matrix to integers before plotting
        cm = np.array(cm).astype(int)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix\n{model_name}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def plot_roc_curves(self, roc_data: Dict, save_name: str = 'roc_curves.png'):
        """
        Plot ROC curves for multiple models
        
        Args:
            roc_data: Dictionary with {model_name: (fpr, tpr, auc)}
            save_name: Output filename
        """
        logger.info("📊 Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#34495e']
        
        for i, (model_name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                   label=f'{model_name} (AUC = {auc:.3f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def plot_precision_recall_curves(self, pr_data: Dict, 
                                    save_name: str = 'precision_recall_curves.png'):
        """
        Plot Precision-Recall curves for multiple models
        
        Args:
            pr_data: Dictionary with {model_name: (precision, recall)}
            save_name: Output filename
        """
        logger.info("📊 Generating Precision-Recall curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#34495e']
        
        for i, (model_name, (precision, recall)) in enumerate(pr_data.items()):
            ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                   label=model_name)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve Comparison', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def plot_feature_importance(self, importance_dict: Dict, top_n: int = 10,
                               save_name: str = 'feature_importance.png'):
        """
        Plot feature importance for tree-based models
        
        Args:
            importance_dict: Dictionary {feature_name: importance_score}
            top_n: Number of top features to show
            save_name: Output filename
        """
        logger.info(f"📊 Generating top {top_n} feature importance plot...")
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Reverse for better visualization (top feature at top)
        features = features[::-1]
        importances = importances[::-1]
        
        bars = ax.barh(range(len(features)), importances, color='#3498db')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important CPS Features', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add value labels
        for bar, imp in zip(bars, importances):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def plot_training_time(self, results: Dict, save_name: str = 'training_time.png'):
        """
        Plot training time comparison with smart unit formatting
        
        Args:
            results: Dictionary with model results (including train_time in seconds)
            save_name: Output filename
        """
        logger.info("📊 Generating training time comparison...")
        
        models = list(results.keys())
        
        # Convert to appropriate units (seconds for < 1 min, minutes otherwise)
        times = []
        time_labels = []
        
        for model in models:
            train_time = results[model].get('train_time', 0)  # Already in seconds
            if train_time < 60:
                times.append(train_time)  # Keep in seconds
                time_labels.append(f"{train_time:.2f}s")
            else:
                times.append(train_time / 60)  # Convert to minutes
                time_labels.append(f"{train_time/60:.2f}m")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#34495e']
        
        bars = ax.bar(models, times, color=colors[:len(models)])
        
        # Add value labels with appropriate units
        for bar, label in zip(bars, time_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   label, ha='center', va='bottom', fontsize=9, rotation=90)
        
        ax.set_ylabel('Training Time', fontsize=12, fontweight='bold')
        ax.set_title('Model Training Time Comparison', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.tick_params(axis='x', rotation=45)
        
        # Add unit annotation for clarity
        ax.text(0.02, 0.98, 'Units: seconds (s) for < 1 min, minutes (m) for ≥ 1 min', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")
    
    def create_comprehensive_dashboard(self, results: Dict, feature_importance: Dict,
                                      save_name: str = 'comprehensive_dashboard.png'):
        """
        Create comprehensive 8-panel dashboard (like your baseline results)
        
        Args:
            results: Dictionary with all model results
            feature_importance: Feature importance dictionary
            save_name: Output filename
        """
        logger.info("📊 Generating comprehensive 8-panel dashboard...")
        
        fig = plt.figure(figsize=(22, 14))
        
        # Panel 1: Class Distribution (placeholder - would need y data)
        ax1 = plt.subplot(2, 4, 1)
        ax1.text(0.5, 0.5, 'Class Distribution\n(See preprocessing stats)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.axis('off')
        
        # Panel 2: Accuracy Comparison
        ax2 = plt.subplot(2, 4, 2)
        models = list(results.keys())
        accuracies = [results[m].get('accuracy', 0) * 100 for m in models]
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c', '#34495e', '#16a085']
        bars = ax2.bar(models, accuracies, color=colors[:len(models)])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Comparison', fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.2f}%', ha='center', fontsize=7)
        
        # Panel 3: F1-Score Comparison
        ax3 = plt.subplot(2, 4, 3)
        f1_scores = [results[m].get('f1_score', 0) * 100 for m in models]
        bars = ax3.bar(models, f1_scores, color=colors[:len(models)])
        ax3.set_ylabel('F1-Score (%)')
        ax3.set_title('F1-Score Comparison', fontweight='bold')
        ax3.set_ylim([0, 100])
        ax3.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.2f}%', ha='center', fontsize=7)
        
        # Panel 4: Training Time (with smart units)
        ax4 = plt.subplot(2, 4, 4)
        times = []
        time_labels = []
        for m in models:
            train_time = results[m].get('train_time', 0)
            if train_time < 60:
                times.append(train_time)
                time_labels.append(f"{train_time:.2f}s")
            else:
                times.append(train_time / 60)
                time_labels.append(f"{train_time/60:.2f}m")
        
        bars = ax4.bar(models, times, color=colors[:len(models)])
        ax4.set_ylabel('Time')
        ax4.set_title('Training Time', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        for bar, label in zip(bars, time_labels):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    label, ha='center', fontsize=7, rotation=90)
        
        # Panel 5: Confusion Matrix (best model)
        ax5 = plt.subplot(2, 4, 5)
        best_model = max(results, key=lambda x: results[x].get('f1_score', 0))
        cm = results[best_model].get('confusion_matrix', np.zeros((2, 2)))
        
        # FIX: Convert confusion matrix to integers before plotting
        cm = np.array(cm).astype(int)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax5.set_title(f'Confusion Matrix\n({best_model})', fontweight='bold')
        ax5.set_ylabel('Actual')
        ax5.set_xlabel('Predicted')
        
        # Panel 6: ROC Curves (placeholder)
        ax6 = plt.subplot(2, 4, 6)
        ax6.text(0.5, 0.5, 'ROC Curves\n(See roc_curves.png)', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.axis('off')
        
        # Panel 7: Precision-Recall (placeholder)
        ax7 = plt.subplot(2, 4, 7)
        ax7.text(0.5, 0.5, 'Precision-Recall\n(See precision_recall_curves.png)', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.axis('off')
        
        # Panel 8: Feature Importance
        ax8 = plt.subplot(2, 4, 8)
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            features = features[::-1]
            importances = importances[::-1]
            ax8.barh(range(len(features)), importances, color='#3498db')
            ax8.set_yticks(range(len(features)))
            ax8.set_yticklabels([f[:12]+'...' if len(f)>15 else f for f in features], fontsize=7)
            ax8.set_title('Top 10 Features', fontweight='bold')
            ax8.set_xlabel('Importance')
        else:
            ax8.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.axis('off')
        
        plt.suptitle('CPS Anomaly Detection - Comprehensive Results Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {filepath}")