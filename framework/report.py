#!/usr/bin/env python3

import json
import argparse
import numpy as np
from pathlib import Path

def calculate_accuracy(predictions, ground_truth):
    """Calculate accuracy metrics"""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    errors = np.abs(predictions - ground_truth)
    
    return {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "exact_matches": int(np.sum(errors < 1e-6))
    }

def main():
    parser = argparse.ArgumentParser(description='Generate simple performance report')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--cpp-file', required=True)
    parser.add_argument('--wasm-size', type=int, required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--em-flags', required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Calculate accuracy on the first 1000 samples
    accuracy = calculate_accuracy(
        results['predictions'], 
        results['groundTruth']
    )
    
    # Check for baseline results
    results_dir = Path(args.output).parent
    baseline_path = results_dir / 'baseline_simple_report.json'
    speedup = 1.0
    
    if baseline_path.exists() and 'baseline' not in args.experiment_name:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            baseline_pps = baseline['predictions_per_second']
            current_pps = results['predictionsPerSecond']
            speedup = current_pps / baseline_pps
    
    # Create report
    report = {
        "experiment_name": args.experiment_name,
        "cpp_file": args.cpp_file,
        "timestamp": results['timestamp'],
        "total_samples": results['totalSamples'],
        "total_time_ms": results['totalTimeMs'],
        "worker_time_ms": results['workerTimeMs'],
        "predictions_per_second": results['predictionsPerSecond'],
        "speedup_vs_baseline": speedup,
        "wasm_size_bytes": args.wasm_size,
        "wasm_size_kb": args.wasm_size / 1024,
        "compilation_flags": args.em_flags,
        "accuracy": accuracy
    }
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save as baseline if this is the baseline
    if 'baseline' in args.experiment_name.lower():
        with open(results_dir / 'baseline_simple_report.json', 'w') as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()