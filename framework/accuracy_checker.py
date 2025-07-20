#!/usr/bin/env python3

import json
import struct
import argparse
import numpy as np

def load_ground_truth(file_path):
    """Load ground truth predictions from binary file"""
    with open(file_path, 'rb') as f:
        # Read header
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_samples = struct.unpack('I', f.read(4))[0]
        n_features = struct.unpack('I', f.read(4))[0]
        
        if magic != 0xCAFEBABE:
            raise ValueError('Invalid test data file format')
        
        # Skip features data
        features_size = n_samples * n_features * 4
        f.seek(16 + features_size)  # Header + features
        
        # Read ground truth predictions
        ground_truth = np.frombuffer(f.read(n_samples * 4), dtype=np.float32)
        
    return ground_truth

def load_predictions(file_path):
    """Load predictions from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get predictions from best performing batch size
    best_batch = data['bestPerformance']
    predictions = np.array(best_batch['predictions'], dtype=np.float32)
    
    return predictions, data

def calculate_accuracy_metrics(predictions, ground_truth):
    """Calculate various accuracy metrics"""
    # Ensure same length (predictions might be truncated)
    n = min(len(predictions), len(ground_truth))
    predictions = predictions[:n]
    ground_truth = ground_truth[:n]
    
    # Calculate errors
    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / (np.abs(ground_truth) + 1e-10)
    
    # Calculate metrics
    metrics = {
        'n_samples_compared': n,
        'exact_matches_ratio': float(np.sum(errors < 1e-6) / n),
        'max_absolute_error': float(np.max(errors)),
        'mean_absolute_error': float(np.mean(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean_relative_error': float(np.mean(relative_errors)),
        'error_percentiles': {
            'p50': float(np.percentile(errors, 50)),
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
            'p99': float(np.percentile(errors, 99)),
            'p99.9': float(np.percentile(errors, 99.9))
        },
        'relative_error_percentiles': {
            'p50': float(np.percentile(relative_errors, 50)),
            'p90': float(np.percentile(relative_errors, 90)),
            'p95': float(np.percentile(relative_errors, 95)),
            'p99': float(np.percentile(relative_errors, 99))
        }
    }
    
    # Regression detection
    threshold = 0.001  # 0.1% relative error threshold
    metrics['regression_detected'] = metrics['mean_relative_error'] > threshold
    
    # Error distribution
    metrics['error_distribution'] = {
        'below_0.0001': int(np.sum(errors < 0.0001)),
        'below_0.001': int(np.sum(errors < 0.001)),
        'below_0.01': int(np.sum(errors < 0.01)),
        'below_0.1': int(np.sum(errors < 0.1)),
        'above_0.1': int(np.sum(errors >= 0.1))
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Check accuracy of WASM predictions')
    parser.add_argument('--predictions', required=True, help='Predictions JSON file')
    parser.add_argument('--ground-truth', required=True, help='Ground truth binary file')
    parser.add_argument('--output', required=True, help='Output accuracy report')
    
    args = parser.parse_args()
    
    print('🔍 Checking accuracy...')
    
    # Load data
    predictions, pred_data = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(predictions, ground_truth)
    
    # Add metadata
    accuracy_report = {
        'comparison_against': 'baseline.cbm',
        'predictions_file': args.predictions,
        'total_samples': pred_data['totalSamples'],
        'samples_compared': metrics['n_samples_compared'],
        'metrics': metrics
    }
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(accuracy_report, f, indent=2)
    
    # Print summary
    print(f"      ✓ Max error: {metrics['max_absolute_error']:.6f}")
    print(f"      ✓ {metrics['exact_matches_ratio']*100:.1f}% of predictions exact match")
    
    if metrics['regression_detected']:
        print(f"      ⚠️  Regression detected: mean relative error {metrics['mean_relative_error']:.4f}")
    else:
        print(f"      ✓ No regression detected")

if __name__ == '__main__':
    main()