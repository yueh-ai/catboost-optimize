#!/usr/bin/env python3

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_report(file_path):
    """Load experiment report"""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_batch_size_performance(report, output_dir):
    """Plot performance vs batch size"""
    batch_results = report['performance']['all_batch_results']
    
    batch_sizes = [r['batchSize'] for r in batch_results]
    pred_per_sec = [r['predictionsPerSecond'] for r in batch_results]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(batch_sizes, pred_per_sec, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Batch Size')
    plt.ylabel('Predictions per Second')
    plt.title(f'Performance vs Batch Size - {report["model"]["name"]}')
    plt.grid(True, alpha=0.3)
    
    # Mark best performance
    best_idx = np.argmax(pred_per_sec)
    plt.plot(batch_sizes[best_idx], pred_per_sec[best_idx], 'r*', markersize=15)
    plt.annotate(f'Best: {pred_per_sec[best_idx]:,.0f} pred/s', 
                xy=(batch_sizes[best_idx], pred_per_sec[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_size_performance.png', dpi=150)
    plt.close()

def plot_error_distribution(report, output_dir):
    """Plot error distribution"""
    error_dist = report['accuracy']['error_distribution']
    
    labels = list(error_dist.keys())
    values = list(error_dist.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), values, color='skyblue', edgecolor='navy')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel('Number of Predictions')
    plt.title(f'Error Distribution - {report["model"]["name"]}')
    plt.yscale('log')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        if v > 0:
            plt.text(i, v, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150)
    plt.close()

def plot_percentile_errors(report, output_dir):
    """Plot error percentiles"""
    percentiles = report['accuracy']['error_percentiles']
    
    p_values = [50, 90, 95, 99, 99.9]
    p_labels = ['p50', 'p90', 'p95', 'p99', 'p99.9']
    errors = [percentiles.get(label, 0) for label in p_labels]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(p_values, errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Percentile')
    plt.ylabel('Absolute Error')
    plt.title(f'Error Percentiles - {report["model"]["name"]}')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (p, e) in enumerate(zip(p_values, errors)):
        plt.annotate(f'{e:.6f}', 
                    xy=(p, e),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_percentiles.png', dpi=150)
    plt.close()

def generate_summary_plot(report, output_dir):
    """Generate a summary plot with key metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Experiment Summary - {report["model"]["name"]}', fontsize=16)
    
    # Performance metrics
    perf = report['performance']
    ax1.text(0.1, 0.8, f"Predictions/sec: {perf['predictions_per_second']:,}", fontsize=14)
    ax1.text(0.1, 0.6, f"Mean time: {perf['mean_prediction_time_us']:.2f} μs", fontsize=14)
    ax1.text(0.1, 0.4, f"Speedup vs baseline: {perf['speedup_vs_baseline']:.1f}x", fontsize=14)
    ax1.text(0.1, 0.2, f"Best batch size: {perf['best_batch_size']}", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Performance Metrics')
    
    # Accuracy metrics
    acc = report['accuracy']
    ax2.text(0.1, 0.8, f"Max error: {acc['max_absolute_error']:.6f}", fontsize=14)
    ax2.text(0.1, 0.6, f"Mean error: {acc['mean_absolute_error']:.6f}", fontsize=14)
    ax2.text(0.1, 0.4, f"RMSE: {acc['rmse']:.6f}", fontsize=14)
    ax2.text(0.1, 0.2, f"Exact matches: {acc['exact_matches_ratio']*100:.1f}%", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Accuracy Metrics')
    
    # Model info
    model = report['model']
    ax3.text(0.1, 0.8, f"WASM size: {model['wasm_size_kb']} KB", fontsize=14)
    ax3.text(0.1, 0.6, f"Compilation: {model['compilation_flags']}", fontsize=12)
    ax3.text(0.1, 0.4, f"Platform: {report['environment']['platform']}", fontsize=12)
    ax3.text(0.1, 0.2, f"Node: {report['environment']['node_version']}", fontsize=12)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Model & Environment')
    
    # Status
    if acc['regression_detected']:
        status_color = 'red'
        status_text = '⚠️ REGRESSION DETECTED'
    else:
        status_color = 'green'
        status_text = '✅ PASSED'
    
    ax4.text(0.5, 0.5, status_text, fontsize=24, ha='center', va='center',
             color=status_color, weight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Status')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('report', help='Report JSON file')
    parser.add_argument('--output-dir', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load report
    report = load_report(args.report)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.report).parent / f"{report['experiment_id']}_plots"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Generating visualizations for {report['experiment_id']}...")
    
    # Generate plots
    plot_batch_size_performance(report, output_dir)
    print("   ✓ Batch size performance plot")
    
    plot_error_distribution(report, output_dir)
    print("   ✓ Error distribution plot")
    
    plot_percentile_errors(report, output_dir)
    print("   ✓ Error percentiles plot")
    
    generate_summary_plot(report, output_dir)
    print("   ✓ Summary plot")
    
    print(f"\n✅ Plots saved to: {output_dir}")

if __name__ == '__main__':
    main()