"""
Evaluation script for uLEAD-TabPFN on ADBench-format anomaly detection datasets.

Runs the full evaluation pipeline over all ADBench datasets and saves per-dataset metrics
(ROC-AUC, AP, PR-AUC, F1) to a CSV file.  Expensive TabPFN predictions are cached so that
only the scoring step needs to be re-run when tuning post-hoc parameters.

Usage
-----
    # Run all datasets with default settings (configure data path in config.py first):
    python run.py

    # Override key settings via command-line arguments:
    python run.py --data-path /path/to/adbench/ --output-csv results/my_results.csv
    python run.py --dataset-id 26          # run a single dataset
    python run.py --from-scratch           # ignore cached artifacts and retrain
    python run.py --epochs 50              # override number of training epochs
    python run.py --seeds 0 1 2            # run seeds 0, 1, 2

See config.py for the full list of tunable hyperparameters.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    f1_score
)
import sys
from lead import LeadTabPFN
from config import CONFIG, group1, group2, group3, group4, group5, group6, group7, group8, group9, group10, group11, group12, group13, group14, group15
from pathlib import WindowsPath, PosixPath
import json
import shutil
import torch
import platform
from sklearn.cluster import KMeans


class Tee:
    """Write output to both console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def reconstruct_test_indices(n_samples, context_indices):
    """
    Reconstruct test_indices from full dataset size and context_indices.

    Args:
        n_samples (int): Total number of samples in dataset
        context_indices (ndarray): Indices of context (training) samples

    Returns:
        ndarray: Indices of test samples (non-context)
    """
    context_set = set(context_indices)
    test_indices = np.array([i for i in range(n_samples) if i not in context_set])
    return test_indices


def reconstruct_test_indices_from_saved(n_samples, context_indices, initial_indices):
    """
    Reconstruct test_indices based on CONFIG['test_set_exclude'].
    """
    if CONFIG.get('test_set_exclude', 'context') == 'initial_normals':
        if initial_indices is not None and len(initial_indices) > 0:
            exclude_set = set(initial_indices)
        else:
            print("  WARNING: initial_normal_indices missing; falling back to context_indices for test set")
            exclude_set = set(context_indices) if context_indices is not None else set()
    else:
        exclude_set = set(context_indices) if context_indices is not None else set()

    return np.array([i for i in range(n_samples) if i not in exclude_set])

def extract_data_id(filename, category):
    """
    Extract numeric ID from dataset filename.

    Naming patterns:
    - Classical: {id}_{name}.npz (e.g., 35_SpamBase.npz -> 35)
    - CV/NLP: {name}_{id}.npz or {name}.npz (e.g., CIFAR10_0.npz -> 0)
    - MVTec-AD: MVTec-AD_{name}.npz (no numeric id)
    - MNIST-C: MNIST-C_{corruption}.npz (no numeric id)
    """
    name = filename.replace('.npz', '')

    if category == 'Classical':
        # Pattern: {id}_{name}
        parts = name.split('_', 1)
        if parts[0].isdigit():
            return int(parts[0])
    elif category.startswith('CV_') or category.startswith('NLP_'):
        # Pattern: {name}_{id} or just {name}
        parts = name.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])

    return None


def compute_metrics(y_true, y_scores):
    """
    Compute all evaluation metrics.

    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores (higher = more anomalous)

    Returns:
        dict with roc_auc, ap, pr_auc, f1_at_anomaly_pct
    """
    anomaly_pct = 100 * np.mean(y_true == 1)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_scores)

    # Average Precision (AP)
    ap = average_precision_score(y_true, y_scores)

    # PR-AUC (Area under Precision-Recall curve)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # F1 at anomaly percentage threshold
    # Set threshold to predict top anomaly_pct% as anomalies
    threshold = np.percentile(y_scores, 100 - anomaly_pct)
    y_pred = (y_scores >= threshold).astype(int)
    f1_at_pct = f1_score(y_true, y_pred)

    return {
        'roc_auc': roc_auc,
        'ap': ap,
        'pr_auc': pr_auc,
        'f1_at_anomaly_pct': f1_at_pct
    }


def flatten_config_for_csv(config_dict):
    """
    Flatten CONFIG dictionary into a dict suitable for CSV export.

    Converts nested structures and non-serializable types to strings.
    Skips fields that shouldn't be in results CSV (like data_path).

    Args:
        config_dict: CONFIG dictionary

    Returns:
        dict: Flattened config with string/numeric values
    """
    from pathlib import Path

    # Fields to exclude from CSV (not relevant for results)
    EXCLUDE_FIELDS = {
        'data_path',      # Machine-specific, not a hyperparameter
        'datasets_json',  # Infrastructure, not a hyperparameter
        'output_csv',     # Output control, not a hyperparameter
        'save_path',      # Output control, not a hyperparameter
        'test_data_id',   # Execution control, not a hyperparameter
        'n_iter',         # Execution control, not a hyperparameter
        'seeds',          # Execution control (individual seed saved separately)
        'save_results',   # Execution control, not a hyperparameter
    }

    flat_config = {}

    for key, value in config_dict.items():
        # Skip excluded fields
        if key in EXCLUDE_FIELDS:
            continue

        # Convert to CSV-compatible type
        if isinstance(value, (int, float, str, bool)):
            flat_config[key] = value
        elif isinstance(value, Path):
            continue  # Skip Path objects (like data_path)
        elif value is None:
            flat_config[key] = ''  # Empty string for None
        else:
            # Convert complex types to string
            flat_config[key] = str(value)

    return flat_config


def append_to_csv_with_schema_update(data_dict, output_csv_path):
    """
    Append a row to CSV with automatic schema reconciliation.

    Handles dynamic CONFIG changes:
    - New columns: Add to header, fill '' in existing rows
    - Deleted columns: Keep in header, fill '' in new row
    - Column ordering: dataset, file_name, roc_auc first; timestamp last

    Args:
        data_dict (dict): Row data to append
        output_csv_path (str): Path to CSV file

    Returns:
        None
    """
    output_csv_path = Path(output_csv_path)

    # Case 1: CSV doesn't exist - create new file
    if not output_csv_path.exists():
        df = pd.DataFrame([data_dict])
        df = _apply_column_order(df)
        df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
        return

    # Case 2: CSV exists - handle schema reconciliation
    # Read existing CSV
    df_existing = pd.read_csv(output_csv_path, quoting=csv.QUOTE_ALL)
    existing_columns = set(df_existing.columns)
    new_data_columns = set(data_dict.keys())

    # Identify schema changes
    new_columns = new_data_columns - existing_columns  # In data but not in CSV
    deleted_columns = existing_columns - new_data_columns  # In CSV but not in data

    # Handle new columns: Add to existing data with blank values
    for col in new_columns:
        df_existing[col] = ''

    # Handle deleted columns: Add to new data with blank values
    for col in deleted_columns:
        data_dict[col] = ''

    # Append new row
    df_new_row = pd.DataFrame([data_dict])
    df_combined = pd.concat([df_existing, df_new_row], ignore_index=True)

    # Apply column ordering
    df_combined = _apply_column_order(df_combined)

    # Write entire CSV with updated schema
    df_combined.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)


def _apply_column_order(df):
    """
    Enforce column order for results CSV:
    - First: dataset, file_name, roc_auc
    - Middle: All other columns (alphabetically sorted)
    - Last: timestamp

    Args:
        df (pd.DataFrame): DataFrame to reorder

    Returns:
        pd.DataFrame: Reordered DataFrame
    """
    first_cols = ['dataset', 'file_name', 'roc_auc']
    last_cols = ['timestamp']

    # Filter to only existing columns
    first_cols_present = [c for c in first_cols if c in df.columns]
    last_cols_present = [c for c in last_cols if c in df.columns]

    # Get middle columns and sort alphabetically
    middle_cols = [col for col in df.columns
                   if col not in first_cols + last_cols]
    middle_cols_sorted = sorted(middle_cols)

    # Construct final column order
    column_order = first_cols_present + middle_cols_sorted + last_cols_present

    return df[column_order]


def sanitize_filename(filename):
    """Remove .npz extension from dataset filename."""
    return filename.replace('.npz', '')


def extract_dataset_name(file_name, category):
    """
    Extract simplified dataset name from filename based on category.

    Args:
        file_name (str): Full filename (e.g., "14_glass.npz", "CIFAR10_1.npz")
        category (str): Dataset category (e.g., "Classical", "NLP_by_BERT", "CV_by_ResNet18")

    Returns:
        str: Simplified dataset name

    Examples:
        Classical: "14_glass.npz" -> "glass"
        NLP_by_BERT: "agnews_3.npz" -> "agnews"
        CV_by_ResNet18: "CIFAR10_1.npz" -> "CIFAR10"
        CV_by_ResNet18: "MVTec-AD_hazelnut.npz" -> "MVTec-AD"
        NLP_by_BERT: "yelp.npz" -> "yelp"
    """
    # Remove .npz extension
    stem = file_name.replace('.npz', '')

    if category == 'Classical':
        # Remove numeric prefix: "14_glass" -> "glass"
        # Find first underscore and take everything after it
        if '_' in stem:
            parts = stem.split('_', 1)  # Split on first underscore only
            return parts[1]
        else:
            return stem  # Edge case: no underscore

    elif category in ['NLP_by_BERT', 'CV_by_ResNet18', 'CV_by_ViT', 'NLP_by_RoBERTa']:
        # Remove suffix after last underscore: "CIFAR10_1" -> "CIFAR10"
        # Special case: if no underscore, return as-is: "yelp" -> "yelp"
        if '_' in stem:
            parts = stem.rsplit('_', 1)  # Split on last underscore only
            return parts[0]
        else:
            return stem  # No underscore, return as-is

    else:
        # Unknown category - return stem as-is
        return stem


def generate_context_indices(X, y, seed):
    """Generate context indices using the same selection strategy as LeadTabPFN."""
    normal_indices = np.where(y == 0)[0]
    n_normal = len(normal_indices)
    n_initial = int(n_normal * CONFIG['context_fraction'])
    if n_initial < 1:
        raise ValueError(
            f"No normal samples selected with context_fraction={CONFIG['context_fraction']}. "
            f"Need at least 1 normal sample."
        )

    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(normal_indices, size=n_initial, replace=False)

    if n_initial <= CONFIG['context_cap']:
        return initial_indices, initial_indices

    n_clusters = CONFIG['context_clustering_n_clusters']
    if n_clusters is None:
        selected = rng.choice(initial_indices, size=CONFIG['context_cap'], replace=False)
        return selected, initial_indices

    X_initial = X[initial_indices]
    n_clusters = min(n_clusters, len(X_initial))
    if n_clusters < 2:
        selected = rng.choice(initial_indices, size=CONFIG['context_cap'], replace=False)
        return selected, initial_indices

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300
    )
    cluster_labels = kmeans.fit_predict(X_initial)

    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    samples_per_cluster = np.round(
        (cluster_counts / len(initial_indices)) * CONFIG['context_cap']
    ).astype(int)

    diff = CONFIG['context_cap'] - np.sum(samples_per_cluster)
    if diff > 0:
        largest_clusters = np.argsort(samples_per_cluster)[-diff:]
        samples_per_cluster[largest_clusters] += 1
    elif diff < 0:
        largest_clusters = np.argsort(samples_per_cluster)[diff:]
        samples_per_cluster[largest_clusters] -= 1

    samples_per_cluster = np.maximum(samples_per_cluster, 1)

    selected_indices = []
    for cluster_id, n_take in zip(unique_clusters, samples_per_cluster):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = initial_indices[cluster_mask]
        if len(cluster_indices) <= n_take:
            selected_indices.extend(cluster_indices.tolist())
            continue

        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(X_initial[cluster_mask] - center, axis=1)
        sorted_idx = np.argsort(distances)
        if n_take == 1:
            chosen_idx = sorted_idx[:1]
        else:
            n_close = n_take // 2
            n_far = n_take - n_close
            close_idx = sorted_idx[:n_close]
            far_idx = sorted_idx[-n_far:]
            chosen_idx = np.unique(np.concatenate([close_idx, far_idx]))

            if len(chosen_idx) < n_take:
                needed = n_take - len(chosen_idx)
                middle_pool = sorted_idx[n_close:len(sorted_idx) - n_far]
                if len(middle_pool) > 0:
                    extra = middle_pool[:needed]
                    chosen_idx = np.unique(np.concatenate([chosen_idx, extra]))

        selected_indices.extend(cluster_indices[chosen_idx].tolist())

    selected = np.array(selected_indices[:CONFIG['context_cap']], dtype=int)
    return selected, initial_indices


def normalize_with_context(X_full, context_indices):
    """Apply LeadTabPFN-style min-max + median/MAD normalization using context stats."""
    X_context = X_full[context_indices]
    normalization_min = np.min(X_context, axis=0, keepdims=True)
    normalization_max = np.max(X_context, axis=0, keepdims=True)
    range_vals = normalization_max - normalization_min
    range_vals[range_vals == 0] = 1.0

    X_minmax = (X_full - normalization_min) / range_vals
    X_minmax = np.clip(X_minmax, 0, 1)

    X_context_minmax = X_minmax[context_indices]
    normalization_median = np.median(X_context_minmax, axis=0, keepdims=True)
    normalization_mad = np.median(
        np.abs(X_context_minmax - normalization_median),
        axis=0,
        keepdims=True
    )

    eps = 1e-8
    X_normalized = (X_minmax - normalization_median) / (normalization_mad + eps)
    X_normalized = np.clip(X_normalized, -5, 5)
    return X_normalized


def check_results_exist_in_csv(output_csv, dataset_file, seed, mode):
    """
    Check if results already exist in CSV for given dataset, seed, and mode.

    Requires EXACT match on all three identifiers:
    - file_name: Full dataset filename (e.g., '10_cover.npz')
    - seed: Random seed integer
    - mode: Experimental mode string (save_path + scoring_method)

    Different modes trigger re-run even for same dataset/seed, as they
    represent different experimental setups (e.g., different encoder strategies).

    Handles CSVs with varying column counts by reading header and matching
    based on available columns. Malformed rows are skipped silently.

    Supports backward compatibility: checks 'file_name' column (new format)
    or 'dataset' column (old format) for the full filename.

    Args:
        output_csv (str): Path to CSV file
        dataset_file (str): Full dataset filename (e.g., '10_cover.npz')
        seed (int): Random seed
        mode (str): Experimental mode (SAVE_PATH stem + scoring method)

    Returns:
        bool: True if exact match exists, False otherwise
    """
    import os
    import pandas as pd
    import csv

    # If CSV doesn't exist, no results exist
    if not os.path.exists(output_csv):
        return False

    try:
        # Read CSV manually to handle field count mismatches
        # Standard pd.read_csv skips rows with mismatched field counts
        import csv as csv_module

        rows = []
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv_module.reader(f, quoting=csv_module.QUOTE_ALL)
            header = next(reader)

            for line_num, row in enumerate(reader, start=2):
                if len(row) == len(header):
                    rows.append(row)
                elif len(row) > len(header):
                    # Extra fields - truncate to header length
                    rows.append(row[:len(header)])
                else:
                    # Missing fields - pad with empty strings
                    rows.append(row + [''] * (len(header) - len(row)))

        df = pd.DataFrame(rows, columns=header)

  

        # Strip any remaining quotes from string columns (for mixed-format CSVs)
        # Some older rows may have been written without quotes, newer rows with quotes
        for col in ['dataset', 'file_name', 'mode', 'hostname', 'category']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip('"')

        # Use file_name column exclusively (standard format)
        if 'file_name' not in df.columns:
            print(f"  WARNING: CSV missing 'file_name' column")
            return False

        filename_col = 'file_name'

        # Check if required columns exist (file_name is already checked above)
        required_cols = ['seed', 'mode']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  WARNING: CSV missing required columns: {missing_cols}")
            return False

        # Check for matching row (file_name/dataset, seed, mode)
        # Requires EXACT match - different modes = different experiments = re-run
        # Use robust comparison to handle formatting variations (whitespace, type mismatches)
        try:
            # More defensive: handle potential type errors
            # Convert filename column to string, handling any non-string values
            file_match = df[filename_col].apply(lambda x: str(x).strip() == str(dataset_file).strip())

            # Convert seed to int, handling non-integer values gracefully
            def safe_seed_match(x):
                try:
                    return int(x) == int(seed)
                except (ValueError, TypeError):
                    return False

            seed_match = df['seed'].apply(safe_seed_match)

            # Convert mode column to string
            mode_match = df['mode'].apply(lambda x: str(x).strip() == str(mode).strip())

            # Combine conditions
            mask = file_match & seed_match & mode_match
            exists = mask.any()

            return exists

        except Exception as e:
            # Make exceptions more visible
            print(f"  ⚠ ERROR in check_results_exist_in_csv: {type(e).__name__}: {e}")
            print(f"    Dataset: {dataset_file}, Seed: {seed}, Mode: {mode}")
            import traceback
            traceback.print_exc()
            return False  # On error, assume no results exist (safer to recompute)

    except Exception as e:
        # Outer exception handler for pd.read_csv errors
        print(f"  WARNING: Could not read CSV to check existing results: {e}")
        return False



def parse_args():
    """Parse command-line arguments. All flags override the corresponding config.py value."""
    parser = argparse.ArgumentParser(
        description='Evaluate uLEAD-TabPFN on ADBench-format anomaly detection datasets.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to ADBench dataset directory (overrides ADBENCH_DATA_PATH and config.py)')
    parser.add_argument('--datasets-json', type=str, default=None,
                        help='Dataset catalog JSON file')
    parser.add_argument('--dataset-id', type=int, default=None,
                        help='Run a single dataset by its numeric ID (None = all datasets)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV file path for results')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory for saving models and cached artifacts')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of encoder training epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds to evaluate (e.g. --seeds 0 1 2)')
    parser.add_argument('--from-scratch', action='store_true', default=False,
                        help='Ignore cached artifacts and retrain from scratch')
    parser.add_argument('--no-ddm', action='store_true', default=False,
                        help='Disable distributional dependency modeling for a faster smoke test')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device for PyTorch/TabPFN (default: cuda if available)')
    return parser.parse_args()


def main():
    """Run comprehensive evaluation on all ADBench datasets."""

    # =================== CENTRALIZED CONFIGURATION ============================
    global CONFIG

    # Apply command-line overrides to CONFIG
    args = parse_args()
    if args.data_path is not None:
        from pathlib import Path as _Path
        CONFIG['data_path'] = _Path(args.data_path)
    if args.datasets_json is not None:
        CONFIG['datasets_json'] = args.datasets_json
    if args.dataset_id is not None:
        CONFIG['test_data_id'] = args.dataset_id
    if args.output_csv is not None:
        CONFIG['output_csv'] = args.output_csv
    if args.save_dir is not None:
        CONFIG['save_path'] = args.save_dir
    if args.epochs is not None:
        CONFIG['ae_epochs'] = args.epochs
    if args.seeds is not None:
        CONFIG['seeds'] = args.seeds
    if args.from_scratch:
        CONFIG['from_scratch'] = True
    if args.no_ddm:
        CONFIG['use_ddm'] = False
    if args.device is not None:
        CONFIG['device'] = args.device

    # Load configuration from config.py
    DATA_PATH = Path(CONFIG['data_path'])
    DATASETS_JSON = CONFIG['datasets_json']
    OUTPUT_CSV = CONFIG['output_csv']
    SAVE_PATH = CONFIG['save_path']    
    SEEDS = CONFIG['seeds']
    SAVE_RESULTS = CONFIG['save_results']
    TEST_DATA_ID = CONFIG['test_data_id']
    TEST_N_FEATURES = CONFIG['test_n_features']
    TEST_N_SAMPLE_CAP = CONFIG['test_n_sample_cap']
    DATASET_GROUP_ID = CONFIG['dataset_group_id']

    # Create mapping from group ID to group list
    GROUP_MAPPING = {
        1: group1, 2: group2, 3: group3, 4: group4, 5: group5,
        6: group6, 7: group7, 8: group8, 9: group9, 10: group10,
        11: group11, 12: group12, 13: group13, 14: group14, 15: group15
    }

    print(f"Data path: {DATA_PATH}")

    # When not saving results, use a temporary directory that is cleaned up later
    if not SAVE_RESULTS:
        import tempfile
        SAVE_PATH = tempfile.mkdtemp(prefix='lead_tmp_')

    print("="*80)
    print("LeadTabPFN - Comprehensive Evaluation on ADBench")
    print("="*80)
    print(f"Data path: {DATA_PATH}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"\nCONFIG:")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print("="*80)

    # Ensure directories exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Optional logging to file
    log_path = CONFIG.get('log_path')
    if log_path is None:
        log_path = os.path.join(SAVE_PATH, "run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    # save CONFIG as json file
    # convert path objects to strings
    config_for_export = {
        k: (str(v) if isinstance(v, (WindowsPath, PosixPath)) else v)
        for k, v in CONFIG.items()
    }
    with open(os.path.join(SAVE_PATH, "config.json"), "w") as f:
        json.dump(config_for_export, f, indent=4)

    # copy scripts
    os.makedirs(os.path.join(SAVE_PATH, "scripts"), exist_ok=True)
    shutil.copyfile('lead.py', os.path.join(SAVE_PATH, 'scripts', 'lead.py'))
    shutil.copyfile('config.py', os.path.join(SAVE_PATH, 'scripts', 'config.py'))
    shutil.copyfile('run.py', os.path.join(SAVE_PATH, 'scripts', 'run.py'))

    # Load dataset catalog
    print(f"\nLoading dataset catalog from {DATASETS_JSON}...")
    with open(DATASETS_JSON, 'r') as f:
        datasets_catalog = json.load(f)

    # Count total datasets
    total_datasets = sum(len(files) for files in datasets_catalog.values())
    print(f"Found {total_datasets} datasets across {len(datasets_catalog)} categories")

    # Initialize results list
    results = []
    dataset_idx = 0

    # Iterate through categories and datasets
    # categories: Classical, CV_by_ResNet18, NLP_by_BERT      
    for category, dataset_files in datasets_catalog.items(): 
        if category != 'Classical':
            SEEDS = [0]
        else:
            SEEDS = CONFIG['seeds']



        print(f"\n{'='*80}")
        print(f"Category: {category} ({len(dataset_files)} datasets)")
        print(f"{'='*80}")

        category_path = DATA_PATH / category
        if not category_path.exists():
            print(f"  WARNING: Category path does not exist: {category_path}")
            print(f"  Skipping {category}...")
            continue

        for dataset_file in dataset_files:

            # Filter by dataset group if specified
            if TEST_DATA_ID is None and DATASET_GROUP_ID is not None and DATASET_GROUP_ID != 0:
                selected_group = GROUP_MAPPING.get(DATASET_GROUP_ID, None)
                if selected_group is None:
                    print(f"  ERROR: Invalid DATASET_GROUP_ID={DATASET_GROUP_ID}. Valid values: 0 (all), 1-15 (specific group)")
                    continue
                if dataset_file not in selected_group:
                    continue
            
            for seed in SEEDS:
                # Ensure model is cleared from previous iteration
                if 'model' in locals():
                    del model
                n_samples_full = None
                n_features_full = None
                
                dataset_idx += 1
                dataset_path = category_path / dataset_file
    
                # Check if file exists
                if not dataset_path.exists():
                    print(f"  ERROR: File not found: {dataset_path}")
                    continue
    
                try:
                    data_id = extract_data_id(dataset_file, category)
                    if TEST_DATA_ID is not None and (data_id != TEST_DATA_ID or category != 'Classical'):
                        continue

                    # Prepare save paths for artifact caching
                    save_base = Path(SAVE_PATH)
                    dataset_name = sanitize_filename(dataset_file)
                    dataset_id = f"{dataset_name}_seed{seed}"

                    # Construct mode string for CSV checking
                    if CONFIG['use_ddm']:
                        training_source = 'context' if CONFIG['ddm_train_on_context'] else 'initial'
                        score_variant = f"ddm_{CONFIG['ddm_aggregation']}_{training_source}"
                    else:
                        score_variant = 'dependency_deviation'
                    mode = f'{Path(SAVE_PATH).stem}_{CONFIG["scoring_method"]}_{score_variant}'

                    # Check if results already exist in CSV
                    if not CONFIG['from_scratch'] and SAVE_RESULTS:
                        # Pass dataset_file (with .npz) to match how it's saved in CSV
                        results_exist = check_results_exist_in_csv(OUTPUT_CSV, dataset_file, seed, mode)
                        if results_exist:
                            print(f"\n[{dataset_idx}/{total_datasets}] {category}/{dataset_file} (seed={seed})")
                            print(f"  ✓ Results already exist in CSV - Skipping")
                            continue  # Skip this test
                    elif CONFIG['from_scratch'] or not SAVE_RESULTS:
                        # CSV check was skipped - show why
                        skip_reasons = []
                        if CONFIG['from_scratch']:
                            skip_reasons.append("from_scratch=True")
                        if not SAVE_RESULTS:
                            skip_reasons.append("save_results=False")
                        # Only print this once per dataset (not for each seed)
                        if seed == CONFIG['seeds'][0]:
                            print(f"  [Note] CSV check skipped for {dataset_file} ({', '.join(skip_reasons)})")

                    # Dependency deviation / NLL matrix save path
                    # When DDM is active: save NLL matrix to ddm_nll/
                    # Otherwise: save dependency deviations to dep_dev/
                    if CONFIG['use_ddm']:
                        # Build cache key for global DDM
                        cache_parts = [dataset_id]
                        suffix = 'ctx' if CONFIG['ddm_train_on_context'] else 'init'
                        cache_parts.append(suffix)
                        cache_key = '_'.join(cache_parts)
                        dep_dev_dir = save_base / 'ddm_nll'
                        dep_dev_dir.mkdir(parents=True, exist_ok=True)
                        dep_dev_path = dep_dev_dir / f"{cache_key}.npy"
                    else:
                        dep_dev_dir = save_base / 'dep_dev'
                        dep_dev_dir.mkdir(parents=True, exist_ok=True)
                        dep_dev_path = dep_dev_dir / f"{dataset_id}.npy"

                    # Context deviation save path (for cached scoring normalization)
                    context_dev_dir = save_base / 'context_dev'
                    context_dev_dir.mkdir(parents=True, exist_ok=True)
                    context_dev_path = context_dev_dir / f"{dataset_id}.npy"

                    # Context set save path
                    context_set_dir = save_base / 'context_set'
                    context_set_dir.mkdir(parents=True, exist_ok=True)
                    context_set_path = context_set_dir / f"{dataset_id}.npy"
                    # Initial normal selection save path (pre-downsampling)
                    initial_context_dir = save_base / 'context_initial'
                    initial_context_dir.mkdir(parents=True, exist_ok=True)
                    initial_context_path = initial_context_dir / f"{dataset_id}.npy"

                    # ==================================================================
                    # CACHE HIERARCHY CHECK (TIERED)
                    # ==================================================================

                    FROM_SCRATCH = CONFIG.get('from_scratch', False)

                    # Tier 1: Check if primary artifact (dep_dev) exists
                    primary_artifact_exists = dep_dev_path.exists()

                    # DDM caches contain complete per-dimension NLL values. The
                    # non-DDM scorer additionally requires context deviations.
                    all_artifacts_exist = primary_artifact_exists and (
                        CONFIG['use_ddm'] or context_dev_path.exists()
                    )

                    cache_ready = False

                    # Branch 1: All artifacts exist - full cache load
                    if not FROM_SCRATCH and all_artifacts_exist:
                        print(f"\n[{dataset_idx}/{total_datasets}] {category}/{dataset_file} (seed={seed})")
                        print("  ✓ Found cached artifacts - Skipping model load/train")

                        # Load the cached artifacts
                        dep_deviations = np.load(dep_dev_path)
                        context_deviations = np.load(context_dev_path) if context_dev_path.exists() else None

                        # Load original dataset to get labels
                        data = np.load(dataset_path, allow_pickle=True)
                        X_full, y_full = data['X'], data['y']
                        n_samples_full = X_full.shape[0]
                        n_features_full = X_full.shape[1]

                        # Reconstruct test_indices based on saved selection
                        context_indices = np.load(context_set_path) if context_set_path.exists() else np.array([])
                        initial_indices = np.load(initial_context_path) if initial_context_path.exists() else None
                        test_indices = reconstruct_test_indices_from_saved(len(y_full), context_indices, initial_indices)
                        y = y_full[test_indices]  # Extract test labels

                        X = None  # Not needed for cached scoring
                        model = LeadTabPFN(seed=seed)
                        model.context_indices = context_indices
                        model.initial_context_indices = initial_indices
                        model.context_purity = 1.0 if len(context_indices) else None
                        model.latent_dim = dep_deviations.shape[1]
                        model._context_deviation_cache = context_deviations

                        # Ensure size consistency
                        scores = model.compute_scores_from_cached_artifacts(
                            dep_deviations=dep_deviations
                        )

                        if len(scores) != len(y):
                            print(f"  ✗ Cached artifact size mismatch → invalid cache")
                        else:
                            # Scores already correspond 1:1 to y
                            test_indices = None
                            t_train = 0.0
                            t_pred = 0.0
                            t_load = 0.0
                            t_save = 0.0
                            cache_ready = True
                            print("  ✓ Cached artifacts loaded successfully")

                    # Branch 2: Only primary artifact (dep_dev) exists - compute missing
                    elif not FROM_SCRATCH and primary_artifact_exists and not all_artifacts_exist:
                        print(f"\n[{dataset_idx}/{total_datasets}] {category}/{dataset_file} (seed={seed})")
                        print("  ✓ Found dep_dev - Computing missing artifacts from cache")

                        # Load primary artifact
                        dep_deviations = np.load(dep_dev_path)

                        # Load dataset to get context info
                        data = np.load(dataset_path, allow_pickle=True)
                        X_full, y_full = data['X'], data['y']
                        n_samples_full = X_full.shape[0]
                        n_features_full = X_full.shape[1]
                        context_indices = np.load(context_set_path) if context_set_path.exists() else None
                        initial_indices = np.load(initial_context_path) if initial_context_path.exists() else None
                        if context_indices is None or len(context_indices) == 0:
                            context_indices, initial_indices = generate_context_indices(X_full, y_full, seed)
                            if SAVE_RESULTS:
                                np.save(context_set_path, context_indices)
                                if initial_indices is not None:
                                    np.save(initial_context_path, initial_indices)
                        test_indices = reconstruct_test_indices_from_saved(len(y_full), context_indices, initial_indices)
                        y = y_full[test_indices]

                        # context_dev: Estimate from dep_deviations statistics if missing
                        if not context_dev_path.exists():
                            # Use quantiles of test dep_deviations as proxy for context deviations
                            # Assumption: normal samples have lower deviations
                            # Take bottom 50% of deviations as proxy for context behavior
                            sorted_devs = np.sort(dep_deviations, axis=0)
                            n_proxy = len(sorted_devs) // 2
                            context_dev_proxy = sorted_devs[:n_proxy]
                            print(f"  ⚠ context_dev missing - estimated from dep_dev (n={n_proxy})")
                            # Save for future
                            if SAVE_RESULTS:
                                np.save(context_dev_path, context_dev_proxy)
                            context_deviations = context_dev_proxy
                        else:
                            context_deviations = np.load(context_dev_path) if context_dev_path.exists() else None

                        # Create minimal model for scoring
                        X = None  # Not needed for cached scoring
                        model = LeadTabPFN(seed=seed)
                        model.context_indices = context_indices
                        model.initial_context_indices = initial_indices
                        model.context_purity = 1.0 if len(context_indices) else None
                        model.latent_dim = dep_deviations.shape[1]
                        model._context_deviation_cache = context_deviations

                        # Compute scores from cached artifacts
                        scores = model.compute_scores_from_cached_artifacts(
                            dep_deviations=dep_deviations
                        )

                        # Validate size
                        if len(scores) != len(y):
                            print(f"  ✗ Size mismatch after computing missing artifacts ({len(scores)} vs {len(y)}) → retraining")
                            # Fall through to training
                        else:
                            test_indices = None
                            t_train = 0.0
                            t_pred = 0.0
                            t_load = 0.0
                            t_save = 0.0
                            cache_ready = True
                            print("  ✓ Missing artifacts computed successfully")


                    # Branch 3: Train from scratch (dep_dev missing or FROM_SCRATCH or size mismatch)
                    if FROM_SCRATCH or not cache_ready:
                        print(f"\n[{dataset_idx}/{total_datasets}] {category}/{dataset_file} (seed={seed})")
                        if FROM_SCRATCH:
                            print(f"  Training from scratch (forced)...")
                        else:
                            print(f"  Training new model...")
                        
                        # Load data
                        t_load_start = time.time()
                        data = np.load(dataset_path, allow_pickle=True)
                        X, y = data['X'], data['y']
                        t_load = time.time() - t_load_start
                        
                        # Feature range filtering
                        if TEST_N_FEATURES is not None:
                            n_features = X.shape[1]
                            min_features, max_features = TEST_N_FEATURES
                            if not (min_features <= n_features < max_features):
                                print(f"  Skipping {dataset_file}: n_features={n_features} not in range [{min_features}, {max_features})")
                                continue
                        
                        # Sample count filtering
                        if TEST_N_SAMPLE_CAP is not None:
                            n_samples = X.shape[0]
                            if n_samples > TEST_N_SAMPLE_CAP:
                                print(f"  Skipping {dataset_file}: n_samples={n_samples} exceeds cap {TEST_N_SAMPLE_CAP}")
                                continue
                        
                        model = LeadTabPFN(seed=seed)

                        t_train_start = time.time()
                        model.fit(X, y, dataset_name=dataset_file)
                        t_train = time.time() - t_train_start
                        t_save = 0.0

                        # Predict
                        print(f"  Running prediction...")
                        t_pred_start = time.time()
                        scores, test_indices, dep_deviations = model.predict_score(
                            X, return_intermediate=True
                        )
                        t_pred = time.time() - t_pred_start

                        # Save artifacts
                        if SAVE_RESULTS:
                            model.save_prediction_artifacts(
                                dep_dev_path=str(dep_dev_path),
                                dep_deviations=dep_deviations
                            )
                            if model._context_deviation_cache is not None:
                                np.save(context_dev_path, model._context_deviation_cache)
                            # Save context indices
                            if model.context_indices is not None:
                                np.save(context_set_path, model.context_indices)
                            if getattr(model, "initial_context_indices", None) is not None:
                                np.save(initial_context_path, model.initial_context_indices)
                            print(f"  Cached artifacts")

                    
                    # Common post-processing
                    # Extract metadata
                    if X is not None:
                        n_samples = X.shape[0]
                        n_features = X.shape[1]
                    else:
                        n_samples = n_samples_full if 'n_samples_full' in locals() else y.shape[0]
                        n_features = n_features_full if 'n_features_full' in locals() else 0
                        
                    n_anomaly = int(np.sum(y == 1))
                    anomaly_percentage = 100 * np.mean(y == 1)
                    print(f"  Loaded: n={n_samples}, d={n_features}, anomalies={n_anomaly} ({anomaly_percentage:.2f}%)")

                    # Extract test labels
                    if test_indices is None:
                        # Artifacts case: scores already correspond to all samples in y
                        y_test = y
                    else:
                        y_test = y[test_indices]

                    # Compute metrics on test set only
                    n_test_anomalies = int(np.sum(y_test == 1))
                    test_anomaly_pct = 100 * np.mean(y_test == 1)
                    metrics = compute_metrics(y_test, scores)

                    # Store results
                    if SAVE_RESULTS:
                        # ================================================================
                        # STEP 1: Already computed metrics above
                        # ================================================================

                        # ================================================================
                        # STEP 2: Build result data (metadata + dataset info + metrics + timing)
                        # ================================================================
                        # Mode already constructed earlier in the loop (line 320-323)
                        data = {
                            # Metadata
                            'hostname': platform.node() or 'unknown',
                            'category': category,
                            'dataset': extract_dataset_name(dataset_file, category),
                            'file_name': dataset_file,
                            'data_id': data_id,
                            'seed': seed,
                            'mode': mode,

                            # Dataset dimensions
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'n_anomaly': n_test_anomalies,
                            'anomaly_percentage': test_anomaly_pct,

                            # Context set info
                            'context_size': len(model.context_indices) if model.context_indices is not None else 0,
                            'context_purity': float(model.context_purity) if model.context_purity is not None else 0.0,
                            'n_train': len(model.context_indices) if model.context_indices is not None else 0,
                            'n_test': len(y_test),

                            # Model info
                            'latent_dim': model.latent_dim,

                            # Metrics (4 fields)
                            **metrics,

                            # Timing (5 fields)
                            'train_time': t_train,
                            'predict_time': t_pred,
                            'pred_time_per_sample': t_pred / len(y_test) if len(y_test) > 0 else 0,
                            'save_time': t_save,
                            'load_time': t_load,

                            # Model path (no longer used - models not saved)
                            'model_path': ''
                        }

                        # ================================================================
                        # STEP 3: Extract training metrics (from model.training_metrics)
                        # ================================================================
                        training_metrics = model.training_metrics if hasattr(model, 'training_metrics') and model.training_metrics else {}
                        data.update({
                            'n_epoch': training_metrics.get('n_epoch', ''),
                            'dep_loss_start': training_metrics.get('dep_loss_start', ''),
                            'dep_loss_end': training_metrics.get('dep_loss_end', ''),
                            'recon_loss_start': training_metrics.get('recon_loss_start', ''),
                            'recon_loss_end': training_metrics.get('recon_loss_end', ''),
                            'total_loss_start': training_metrics.get('total_loss_start', ''),
                            'total_loss_end': training_metrics.get('total_loss_end', ''),
                            'lambda_rec': training_metrics.get('lambda_rec', ''),
                        })


                        # ================================================================
                        # STEP 5: Merge with ALL CONFIG parameters (automatic)
                        # ================================================================
                        config_data = flatten_config_for_csv(CONFIG)
                        data.update(config_data)

                        # ================================================================
                        # STEP 5.5: Add timestamp (always last field)
                        # ================================================================
                        data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        # ================================================================
                        # STEP 6: Append to results
                        # ================================================================
                        results.append(data)
                        print(f"  Result data collected ({len(data)} fields)")

                        # Save immediate result
                        if SAVE_RESULTS:
                            # Check if result already exists in CSV to prevent duplicates
                            results_already_exist = check_results_exist_in_csv(
                                OUTPUT_CSV, dataset_file, seed, mode
                            )

                            if results_already_exist:
                                print(f"  ⚠ Result already in CSV - Skipping save to prevent duplicate")
                            else:
                                # Save with schema reconciliation
                                append_to_csv_with_schema_update(data, OUTPUT_CSV)
                                print(f"  Result appended to {OUTPUT_CSV}")

                    # Print summary
                    print(f"\n{'='*80}")
                    print(f"  ✓ DATA={dataset_file.split('.')[0]} SEED={seed}: ROC-AUC={metrics['roc_auc']:.4f}, AP={metrics['ap']:.4f}, "
                          f"PR-AUC={metrics['pr_auc']:.4f}, F1@{test_anomaly_pct:.1f}%={metrics['f1_at_anomaly_pct']:.4f}")
                    context_size = len(model.context_indices) if model.context_indices is not None else 0
                    context_purity = float(model.context_purity) if model.context_purity is not None else 0.0
                    print(f"  Train/Test split: {context_size} train, {len(y_test)} test ({n_test_anomalies} anomalies, {test_anomaly_pct:.1f}%)")
                    print(f"  Context purity: {context_purity:.4f}")
                    print(f"  Time: train={t_train:.1f}s, predict={t_pred:.2f}s")
    
                    # Clean up GPU memory
                    del model
                    
    
                except Exception as e:
                    error_msg = str(e)
                    print(f"  ✗ ERROR: {error_msg}")
    
                    # Clean up GPU memory on error
                    if 'model' in locals():
                        del model
            
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()




 

if __name__ == '__main__':
    main()
