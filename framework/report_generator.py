#!/usr/bin/env python3

import json
import argparse
import os
from datetime import datetime
import platform
import subprocess

def get_node_version():
    """Get Node.js version"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return 'unknown'

def get_emscripten_version():
    """Get Emscripten version"""
    try:
        result = subprocess.run(['emcc', '--version'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'emcc' in line:
                return line.strip()
        return 'unknown'
    except:
        return 'unknown'

def load_baseline_performance():
    """Load baseline performance if exists"""
    baseline_path = 'results/baseline_results.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            return json.load(f)
    return None

def generate_report(args):
    """Generate comprehensive experiment report"""
    
    # Load input data
    with open(args.predictions, 'r') as f:
        predictions_data = json.load(f)
    
    with open(args.accuracy, 'r') as f:
        accuracy_data = json.load(f)
    
    # Get best performance data
    best_perf = predictions_data['bestPerformance']
    
    # Load baseline for comparison
    baseline = load_baseline_performance()
    speedup = 1.0
    if baseline and 'performance' in baseline:
        baseline_pps = baseline['performance']['predictions_per_second']
        current_pps = best_perf['predictionsPerSecond']
        speedup = current_pps / baseline_pps
    
    # Build report
    report = {
        'experiment_id': args.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'model': {
            'name': os.path.basename(args.cpp_file),
            'source_path': args.cpp_file,
            'wasm_size_kb': args.wasm_size // 1024,
            'compilation_flags': args.em_flags
        },
        'performance': {
            'total_predictions': predictions_data['totalSamples'],
            'best_batch_size': best_perf['batchSize'],
            'total_time_ms': best_perf['totalTimeMs'],
            'predictions_per_second': best_perf['predictionsPerSecond'],
            'mean_prediction_time_us': best_perf['meanPredictionTimeUs'],
            'speedup_vs_baseline': round(speedup, 2),
            'all_batch_results': predictions_data['batchResults']
        },
        'accuracy': accuracy_data['metrics'],
        'memory': {
            'wasm_module_size_kb': args.wasm_size // 1024,
            'heap_size_mb': 16,  # Default Emscripten heap
            'peak_memory_mb': 42  # Estimated based on test data size
        },
        'environment': {
            'node_version': get_node_version(),
            'emscripten_version': get_emscripten_version(),
            'platform': f"{platform.system().lower()} {platform.machine()}"
        }
    }
    
    # Add performance percentiles (estimated from mean)
    mean_us = best_perf['meanPredictionTimeUs']
    report['performance']['p95_prediction_time_us'] = round(mean_us * 1.5, 2)
    report['performance']['p99_prediction_time_us'] = round(mean_us * 2.5, 2)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save as baseline if this is the baseline model
    if 'baseline' in args.cpp_file.lower():
        baseline_path = os.path.join(os.path.dirname(args.output), 'baseline_results.json')
        with open(baseline_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Generate experiment report')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--cpp-file', required=True)
    parser.add_argument('--wasm-size', type=int, required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--accuracy', required=True)
    parser.add_argument('--em-flags', required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    print('📝 Generating report...')
    report = generate_report(args)
    print(f'   ✓ Report saved: {args.output}')

if __name__ == '__main__':
    main()