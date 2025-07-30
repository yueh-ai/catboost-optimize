#!/usr/bin/env python3

import json
import struct
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def generate_test_data():
    print("🎲 Generating test data (with string categoricals)...")

    # Load metadata
    with open("../models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of test samples
    n_samples = 1_000_000
    print(f"   Generating {n_samples:,} test vectors")

    # Generate test data
    test_data = {}

    # Generate numeric features
    for feature, ranges in metadata["feature_ranges"].items():
        mean = ranges["mean"]
        std = ranges["std"]
        min_val = ranges["min"]
        max_val = ranges["max"]

        values = np.random.normal(mean, std, n_samples)
        values = np.clip(values, min_val, max_val)
        test_data[feature] = values
        print(f"   Generated {feature}: [{min_val:.2f}, {max_val:.2f}]")

    # Generate categorical features (keep as strings)
    for feature, categories in metadata["categorical_values"].items():
        if feature == "cut":
            weights = [0.05, 0.15, 0.25, 0.30, 0.25]
        elif feature == "color":
            n_colors = len(categories)
            weights = np.exp(-0.5 * ((np.arange(n_colors) - n_colors / 2) / 2) ** 2)
            weights = weights / weights.sum()
        elif feature == "clarity":
            weights = [0.02, 0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.02]
        else:
            weights = None

        values = np.random.choice(categories, n_samples, p=weights)
        test_data[feature] = values
        print(f"   Generated {feature}: {len(categories)} categories")

    # Create DataFrame
    test_df = pd.DataFrame(test_data)
    test_df = test_df[metadata["features"]]

    # Generate ground truth predictions
    print("\n🔮 Generating ground truth predictions...")
    model = CatBoostRegressor()
    model.load_model("../models/baseline.cbm")

    batch_size = 10000
    predictions = []

    for i in range(0, n_samples, batch_size):
        batch = test_df.iloc[i : i + batch_size]
        batch_pred = model.predict(batch)
        predictions.extend(batch_pred)
        if i % 100000 == 0:
            print(f"   Progress: {i:,}/{n_samples:,}")

    predictions = np.array(predictions, dtype=np.float32)

    # Save test data in new format
    print("\n💾 Saving test data...")

    # Create a JSON format that preserves string categoricals
    test_data_dict = {
        "n_samples": n_samples,
        "numeric_features": metadata["numeric_features"],
        "categorical_features": metadata["categorical_features"],
        "data": [],
    }

    # For efficiency, we'll still use binary format but with a different structure
    # Format: float features (6 floats) + categorical indices (3 uint8) + padding

    # First, create encoding for categoricals
    cat_encodings = {}
    for feature in metadata["categorical_features"]:
        categories = metadata["categorical_values"][feature]
        cat_encodings[feature] = {cat: i for i, cat in enumerate(categories)}

    # Binary format V2:
    # Header: magic (4), version (4), n_samples (4), n_float_features (4), n_cat_features (4)
    # For each sample: 6 floats + 3 uint8 categorical indices + 1 byte padding

    binary_path = "../models/test_data.bin"
    with open(binary_path, "wb") as f:
        # Write header
        header = struct.pack("IIIII", 0xCAFEBABE, 2, n_samples, 6, 3)
        f.write(header)

        # Pre-extract numeric data as numpy arrays for efficient access
        numeric_data = {
            feat: test_df[feat].values.astype(np.float32)
            for feat in metadata["numeric_features"]
        }

        # Pre-encode categorical data
        categorical_indices = {}
        for feat in metadata["categorical_features"]:
            cat_values = test_df[feat].values
            encoding = cat_encodings[feat]
            indices = np.array([encoding[val] for val in cat_values], dtype=np.uint8)
            categorical_indices[feat] = indices

        # Write data in batches for better I/O performance
        batch_size = 50000
        buffer = bytearray(
            batch_size * 28
        )  # 6 floats (24 bytes) + 4 bytes (3 uint8 + padding)

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_samples = batch_end - batch_start

            # Fill buffer with batch data
            offset = 0
            for i in range(batch_start, batch_end):
                # Pack 6 floats
                for feat in metadata["numeric_features"]:
                    struct.pack_into("f", buffer, offset, numeric_data[feat][i])
                    offset += 4

                # Pack 3 categorical indices + padding
                for feat in metadata["categorical_features"]:
                    buffer[offset] = categorical_indices[feat][i]
                    offset += 1
                buffer[offset] = 0  # padding
                offset += 1

            # Write only the filled portion of the buffer
            f.write(buffer[: batch_samples * 28])

            if batch_start % 200000 == 0:
                print(f"   Saving progress: {batch_start:,}/{n_samples:,}")

        # Write predictions at the end
        predictions.astype(np.float32).tofile(f)

    file_size_mb = os.path.getsize(binary_path) / (1024 * 1024)
    print(f"   Saved binary data: {binary_path} ({file_size_mb:.1f} MB)")

    # Save categorical mappings
    cat_mappings = {"encodings": cat_encodings, "decodings": {}}

    for feature in metadata["categorical_features"]:
        categories = metadata["categorical_values"][feature]
        cat_mappings["decodings"][feature] = categories

    mappings_path = "../models/categorical_mappings.json"
    with open(mappings_path, "w") as f:
        json.dump(cat_mappings, f, indent=2)
    print(f"   Saved categorical mappings: {mappings_path}")

    print("\n✅ Test data generation completed!")


if __name__ == "__main__":
    generate_test_data()
