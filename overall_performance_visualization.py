#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall Performance Visualization for TranSenti Framework
This script generates comprehensive performance comparison plots for the integrated TranSenti method
against state-of-the-art baselines across both filtering and sentiment analysis tasks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Times New Roman',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def create_overall_performance_comparison():
    """
    Create a comprehensive performance comparison showing the integrated TranSenti framework
    against baselines across both filtering and sentiment analysis tasks.
    """
    
    # Define performance data for integrated pipeline
    methods = [
        'Keyword + BERTweet',
        'Keyword + TweetNLP', 
        'Keyword + RoBERTa Base',
        'Qwen2.5 + BERTweet',
        'Qwen2.5 + TweetNLP',
        'Qwen2.5 + RoBERTa Base',
        'Mixture w/o Inst. + RoBERTa Base',
        'TranSenti (Ours)'
    ]
    
    # End-to-end F1 scores (filtering F1 * sentiment F1)
    filtering_f1 = [49.1, 49.1, 49.1, 70.2, 70.2, 70.2, 72.6, 77.3]
    sentiment_f1 = [71.3, 73.1, 74.6, 71.3, 73.1, 74.6, 74.6, 80.8]
    
    # Calculate end-to-end performance (geometric mean to account for pipeline dependency)
    end_to_end_f1 = [np.sqrt(f * s) for f, s in zip(filtering_f1, sentiment_f1)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Main comprehensive comparison plot
    ax1 = plt.subplot(2, 2, (1, 2))
    
    x_pos = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, filtering_f1, width, label='Filtering F1', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_pos, sentiment_f1, width, label='Sentiment F1', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax1.bar(x_pos + width, end_to_end_f1, width, label='End-to-End F1', 
                   color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight our method
    for i, method in enumerate(methods):
        if 'TranSenti' in method:
            # Add highlight box
            rect = Rectangle((i-0.5, 0), 1, 85, linewidth=2, edgecolor='red', 
                           facecolor='none', linestyle='--', alpha=0.7)
            ax1.add_patch(rect)
    
    ax1.set_xlabel('Methods', fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontweight='bold')
    ax1.set_title('Comprehensive Performance Comparison: Integrated TranSenti Framework', 
                  fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 85)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Component analysis - showing contribution of each component
    ax2 = plt.subplot(2, 2, 3)
    
    components = ['Keyword\nFiltering', 'LLM\nFiltering', 'Mixture w/o\nInstruction', 
                  'Mixture w/\nInstruction', 'RoBERTa\nBase', 'RoBERTa w/\nST Tuning']
    component_scores = [49.1, 70.2, 72.6, 77.3, 74.6, 80.8]
    colors = ['#E74C3C', '#F39C12', '#F39C12', '#27AE60', '#3498DB', '#2E86AB']
    
    bars = ax2.bar(components, component_scores, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('F1 Score (%)', fontweight='bold')
    ax2.set_title('Component-wise Performance Analysis', fontweight='bold')
    ax2.set_ylim(0, 85)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Performance improvement analysis
    ax3 = plt.subplot(2, 2, 4)
    
    improvement_categories = ['Filtering\nAccuracy', 'Sentiment\nAccuracy', 'End-to-End\nPipeline']
    baseline_best = [70.2, 74.6, 72.2]  # Best baseline performance in each category
    tranSenti_performance = [77.3, 80.8, 78.9]
    improvements = [t - b for t, b in zip(tranSenti_performance, baseline_best)]
    
    x_pos = np.arange(len(improvement_categories))
    bars_baseline = ax3.bar(x_pos - 0.2, baseline_best, 0.4, label='Best Baseline', 
                           color='#95A5A6', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars_ours = ax3.bar(x_pos + 0.2, tranSenti_performance, 0.4, label='TranSenti (Ours)', 
                       color='#E67E22', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_ylabel('F1 Score (%)', fontweight='bold')
    ax3.set_title('Performance Improvements over Best Baselines', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(improvement_categories)
    ax3.legend()
    ax3.set_ylim(0, 85)
    
    # Add improvement annotations
    for i, improvement in enumerate(improvements):
        ax3.annotate(f'+{improvement:.1f}%',
                    xy=(i, tranSenti_performance[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/Users/leida/TransBert/TranSent Paper/figs/overall_performance_comparison.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_detailed_metrics_heatmap():
    """
    Create a detailed heatmap showing performance across different metrics and methods.
    """
    
    # Detailed performance data
    methods = ['Keyword+BERTweet', 'Qwen2.5+TweetNLP', 'Mixture w/o Inst.+RoBERTa', 'TranSenti (Ours)']
    metrics = ['Filtering\nAccuracy', 'Filtering\nPrecision', 'Filtering\nRecall', 'Filtering\nF1',
               'Sentiment\nAccuracy', 'Sentiment\nPrecision', 'Sentiment\nRecall', 'Sentiment\nF1']
    
    # Performance matrix
    performance_data = np.array([
        [54.5, 42.3, 58.7, 49.1, 68.2, 70.5, 72.1, 71.3],  # Keyword+BERTweet
        [68.9, 65.7, 72.4, 70.2, 69.8, 72.3, 73.8, 73.1],  # Qwen2.5+TweetNLP
        [71.5, 68.9, 74.2, 72.6, 71.6, 73.8, 75.4, 74.6],  # Mixture w/o Inst.+RoBERTa
        [78.7, 76.4, 79.8, 77.3, 78.7, 80.2, 81.5, 80.8]   # TranSenti (Ours)
    ])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(performance_data, cmap='RdYlGn', aspect='auto', vmin=40, vmax=85)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_yticklabels(methods)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{performance_data[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Detailed Performance Metrics Comparison', fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance (%)', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/leida/TransBert/TranSent Paper/figs/detailed_metrics_heatmap.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def create_ablation_study_plot():
    """
    Create an ablation study plot showing the contribution of different components.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filtering ablation
    filtering_components = ['Base LLM', '+ Mixture', '+ KL Constraint', '+ Instruction\nRefinement']
    filtering_scores = [70.2, 72.6, 75.1, 77.3]
    
    bars1 = ax1.bar(filtering_components, filtering_scores, 
                   color=['#E74C3C', '#F39C12', '#F1C40F', '#27AE60'], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('F1 Score (%)', fontweight='bold')
    ax1.set_title('Filtering Component Ablation Study', fontweight='bold')
    ax1.set_ylim(65, 80)
    
    # Add improvement arrows and values
    for i in range(1, len(filtering_scores)):
        improvement = filtering_scores[i] - filtering_scores[i-1]
        ax1.annotate(f'+{improvement:.1f}%', 
                    xy=(i-0.5, filtering_scores[i-1] + improvement/2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='blue',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    # Sentiment ablation
    sentiment_components = ['RoBERTa Base', '+ Fine-tuning', '+ Temporal\nEncoder', '+ Spatial\nEncoder']
    sentiment_scores = [74.6, 76.4, 78.4, 80.8]
    
    bars2 = ax2.bar(sentiment_components, sentiment_scores,
                   color=['#3498DB', '#9B59B6', '#E67E22', '#2E86AB'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('F1 Score (%)', fontweight='bold')
    ax2.set_title('Sentiment Analysis Component Ablation Study', fontweight='bold')
    ax2.set_ylim(70, 85)
    
    # Add improvement arrows and values
    for i in range(1, len(sentiment_scores)):
        improvement = sentiment_scores[i] - sentiment_scores[i-1]
        ax2.annotate(f'+{improvement:.1f}%',
                    xy=(i-0.5, sentiment_scores[i-1] + improvement/2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='blue',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax = bar.axes
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/leida/TransBert/TranSent Paper/figs/ablation_study_analysis.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


if __name__ == "__main__":
    print("Generating overall performance comparison plots...")
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('/Users/leida/TransBert/TranSent Paper/figs', exist_ok=True)
    
    # Generate all plots
    create_overall_performance_comparison()
    create_detailed_metrics_heatmap()
    create_ablation_study_plot()
    
    print("All plots have been generated and saved to /Users/leida/TransBert/TranSent Paper/figs/")
