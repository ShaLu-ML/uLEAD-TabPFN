"""
LEAD Configuration File
Centralized configuration for the LEAD anomaly detection model.

This file organizes all hyperparameters and execution settings according to
the three-step architecture of the LEAD model:

STEP 1: Data Preparation & Context Generation
STEP 2: Encoder Training & Latent Representation
STEP 3: Dependency-Based Anomaly Detection & Scoring
"""

from pathlib import Path

# ============================================================================
# STEP 0: GLOBAL & INFRASTRUCTURE
# ============================================================================

BATCH_RUN = False

CONFIG = {
    # --------------------------------------------------------------------------
    # Data Paths & I/O
    # --------------------------------------------------------------------------
    'data_path': None,  # Auto-detected from ADBENCH_DATA_PATH or bundled example data
    'datasets_json': 'datasets_example.json',
    'output_csv': 'results/results.csv',
    'save_path': 'results/lead',

    # --------------------------------------------------------------------------
    # Execution & Reproducibility
    # --------------------------------------------------------------------------
    'test_data_id': None,  # None = all datasets, int = specific dataset ID
    'n_iter': None,  # Limit iterations (None = 5 for Classical datasets, 1 for others)
    
    'save_results': True,
    'from_scratch': True, # If True, ignore cached artifacts/models and retrain

    'dataset_group_id': None, # None = all datasets, int = specific dataset group ID
    'test_n_features': None, # None = all features, list = specific range of features
    'test_n_sample_cap': 50000, # None = all samples, int = specific sample cap
    'log_path': None,  # If None, logs go to SAVE_PATH/run.log in appending mode
    'seeds': [0],
    

    # --------------------------------------------------------------------------
    # Device & Logging
    # --------------------------------------------------------------------------
    'device': 'cuda',
    'verbose': True,
    'pred_batch_size': 4096,  # Batch size for TabPFN predictions: 4096, 8192, 16384, 32768, 40960, 46080

    # ========================================================================
    # STEP 1: DATA PREPARATION & CONTEXT GENERATION
    # ========================================================================

    # --------------------------------------------------------------------------
    # Context Set Generation
    # --------------------------------------------------------------------------
    'context_fraction': 0.5,
    'context_cap': 500,
    'include_context_in_test': False,
    # Test set selection: 'context' excludes only final context set,
    # 'initial_normals' excludes the initial normal sample selection (pre-downsampling)
    'test_set_exclude': 'initial_normals', # context or initial_normals
    'context_clustering_n_clusters': 100,  # Number of clusters for downsampling (None = random sampling only)

    # ========================================================================
    # STEP 2: ENCODER TRAINING & LATENT REPRESENTATION
    # ========================================================================

    # --------------------------------------------------------------------------
    # Latent Dimensionality
    # --------------------------------------------------------------------------
    # Adaptive strategy:
    #   d <= latent_dim_identity_threshold  →  latent_dim = d  (identity, no compression)
    #   d >  latent_dim_identity_threshold  →  latent_dim = min(cap, max(3, floor(min(2.5*sqrt(d), 0.5*d))))
    'latent_dim': None,  # None = adaptive (recommended); set to an int to override
    'latent_dim_identity_threshold': None,  # Feature count below which no compression is applied
    'latent_dim_max_cap': 100,  # Upper bound on compressed latent dimension

    # --------------------------------------------------------------------------
    # Training Dynamics
    # --------------------------------------------------------------------------
    'ae_epochs': 100,  # Paper setting; reduce with --epochs for a quick smoke test
    'ae_warmup_epochs': 5,
    'ae_batch_size': 1024,
    'ae_lr_init': 5e-4,
    'ae_lr_final': 1e-6,

    # --------------------------------------------------------------------------
    # Reconstruction Loss Configuration
    # --------------------------------------------------------------------------
    'ae_dynamic_rec_scaling': True,
    'ae_lambda_rec': 0.00005,  # Fixed weight (if dynamic scaling disabled)
    'ae_lambda_rec_base': 1.0,
    'ae_lambda_rec_factor': 0.3,
    'ae_use_cosine_ramp': True,  # Use cosine ramp for lambda_rec after warmup

    # --------------------------------------------------------------------------
    # Baseline Mode (Skip Encoder)
    # --------------------------------------------------------------------------
    'use_original_features': False, 
    

    # --------------------------------------------------------------------------
    # Scoring Method Selection
    # --------------------------------------------------------------------------
    'scoring_method': 'dependency',

    # --------------------------------------------------------------------------
    # Distributional Dependency Modeling (DDM)
    # --------------------------------------------------------------------------
    'use_ddm': True,  # Enable DDM (Gaussian NLL) instead of point predictions

    # Variance Network Architecture
    'ddm_hidden_dim': 64,
    'ddm_dropout': 0.1,

    # Variance Network Training
    'ddm_epochs': 50,
    'ddm_lr': 1e-3,
    'ddm_weight_decay': 1e-4,
    # DDM variance network training cap (None = use all initial normals)
    'ddm_train_cap': 100000,
    # DDM training data selection
    'ddm_train_on_context': False,  # If True, train variance networks on context set (e.g., 500 samples);
                                     # If False, train on initial normals (larger pre-downsampled set)

    # Variance Stabilization
    'ddm_log_var_min': -10.0,  # Min log-variance (σ² ≥ 4.5e-5)
    'ddm_log_var_max': 5.0,    # Max log-variance (σ² ≤ 148)

    # Score Aggregation
    'ddm_aggregation': 'mean',  # 'mean', 'sum', 'top_k_mean', 'logsumexp'
    'ddm_top_k': None,  # For 'top_k_mean': default = latent_dim // 3

}

# ============================================================================
# AUTO-CONFIGURATION: Data Path Detection
# ============================================================================

def _detect_data_path():
    """
    Return the path to an ADBench-format dataset directory.

    Priority:
    1. ADBENCH_DATA_PATH environment variable (if set)
    2. Bundled example data under ./data/
    3. Edit the return statement below for a hardcoded local path

    The directory should contain subdirectories Classical/, CV_by_ResNet18/, etc.,
    each holding .npz dataset files as distributed by ADBench.
    """
    import os
    env_path = os.environ.get('ADBENCH_DATA_PATH')
    if env_path:
        return Path(env_path)

    bundled_data = Path(__file__).resolve().parent / 'data'
    if bundled_data.exists():
        return bundled_data

    # Optional: set your local ADBench dataset directory here, e.g.:
    #   return Path('/path/to/adbench/datasets/')
    #   return Path('C:/path/to/adbench/datasets/')
    raise ValueError(
        "ADBench data path not configured.\n"
        "Option 1: Set the ADBENCH_DATA_PATH environment variable:\n"
        "    export ADBENCH_DATA_PATH=/path/to/adbench/datasets/\n"
        "Option 2: keep the bundled ./data example directory in this repository.\n"
        "Option 3: edit _detect_data_path() in config.py and add a return statement."
    )


if BATCH_RUN:
    CONFIG['test_data_id'] = None
    CONFIG['n_iter'] = None
    CONFIG['save_results'] = True
    CONFIG['seeds'] = list(range(5))

    CONFIG['from_scratch'] = False  # Set True to retrain from scratch (ignores cached artifacts)


CONFIG['data_path'] = _detect_data_path()
if CONFIG['n_iter'] is not None:
    CONFIG['seeds'] = list(range(CONFIG['n_iter']))



group1 = [
        "10_cover.npz",
        "11_donors.npz",
        "12_fault.npz",
        "13_fraud.npz",
        "14_glass.npz",
        "15_Hepatitis.npz",
        "16_http.npz",
        "17_InternetAds.npz",
        "18_Ionosphere.npz",
        "19_landsat.npz",
        "1_ALOI.npz",
        
    ]


group2 = [
        "20_letter.npz",
        "21_Lymphography.npz",
        "22_magic.gamma.npz",
        "23_mammography.npz",
        "24_mnist.npz",
        "25_musk.npz",
        "26_optdigits.npz",
        "27_PageBlocks.npz",
        "28_pendigits.npz",
        "29_Pima.npz",
        "2_annthyroid.npz",
        "30_satellite.npz",
        "31_satimage-2.npz",
        "32_shuttle.npz",
        "33_skin.npz",
        "34_smtp.npz",
        "35_SpamBase.npz",
    ]


group3 = [
        "36_speech.npz",
        "37_Stamps.npz",
        "38_thyroid.npz",
        "39_vertebral.npz",
        "3_backdoor.npz",
        "40_vowels.npz",
        "41_Waveform.npz"
    ]


group4 = [
        "MNIST-C_brightness.npz",
        "MNIST-C_canny_edges.npz",
        "MNIST-C_dotted_line.npz",
        "MNIST-C_fog.npz",
        "MNIST-C_glass_blur.npz",
        "MNIST-C_identity.npz",
        "MNIST-C_impulse_noise.npz",
        "MNIST-C_motion_blur.npz",
        "MNIST-C_rotate.npz",
        "MNIST-C_scale.npz",
        "MNIST-C_shear.npz",
        "MNIST-C_shot_noise.npz",
        "MNIST-C_spatter.npz",
        "MNIST-C_stripe.npz",
        "MNIST-C_translate.npz",
    ]


group5 = [
        "MNIST-C_zigzag.npz",
        "MVTec-AD_bottle.npz",
        "MVTec-AD_cable.npz",
        "MVTec-AD_capsule.npz",
        "MVTec-AD_carpet.npz",
        "MVTec-AD_grid.npz",
        "MVTec-AD_hazelnut.npz",
        "MVTec-AD_leather.npz",
        "MVTec-AD_metal_nut.npz",
        "MVTec-AD_pill.npz",
        "MVTec-AD_screw.npz",
        "MVTec-AD_tile.npz",
        "MVTec-AD_toothbrush.npz",
        "MVTec-AD_transistor.npz",
        "MVTec-AD_wood.npz",
        "MVTec-AD_zipper.npz"        
    ]

group6 = [
        "CIFAR10_0.npz",
        "CIFAR10_1.npz",
        "CIFAR10_2.npz",
        "CIFAR10_3.npz",
        "CIFAR10_4.npz",
        "CIFAR10_5.npz",
        "CIFAR10_6.npz",
        "CIFAR10_7.npz",
        "CIFAR10_8.npz",
        "CIFAR10_9.npz"
    ] 

group7 = [
        "FashionMNIST_0.npz",
        "FashionMNIST_1.npz",
        "FashionMNIST_2.npz",
        "FashionMNIST_3.npz",
        "FashionMNIST_4.npz"       
    ]

group8 = [
        "SVHN_0.npz",
        "SVHN_1.npz",
        "SVHN_2.npz",
        "SVHN_3.npz",
        "SVHN_4.npz"        
    ]

group9 = [
        "20news_0.npz",
        "20news_1.npz",
        "20news_2.npz",
        "20news_3.npz",
        "20news_4.npz",
        "20news_5.npz"
    ]

group10 = [
        "MVTec-AD_bottle.npz",
        "MVTec-AD_cable.npz",
        "MVTec-AD_capsule.npz",
        "MVTec-AD_carpet.npz",
        "MVTec-AD_grid.npz",
        "MVTec-AD_hazelnut.npz",
        "MVTec-AD_leather.npz",
        "MVTec-AD_metal_nut.npz",
        "MVTec-AD_pill.npz",
        "MVTec-AD_screw.npz",
        "MVTec-AD_tile.npz",
        "MVTec-AD_toothbrush.npz",
        "MVTec-AD_transistor.npz",
        "MVTec-AD_wood.npz",
        "MVTec-AD_zipper.npz"     
    ]


group11 = [
         "42_WBC.npz",
        "43_WDBC.npz",
        "44_Wilt.npz",
        "45_wine.npz",
        "46_WPBC.npz",
        "47_yeast.npz",
        "4_breastw.npz",
        "5_campaign.npz",
        "6_cardio.npz",
        "7_Cardiotocography.npz",
        "8_celeba.npz",
        "9_census.npz"     
    ]


group12 = [        
        "amazon.npz",
        "imdb.npz",
        "yelp.npz"
    ]


group13 = [
        "agnews_0.npz",
        "agnews_1.npz",
        "agnews_2.npz",
        "agnews_3.npz"
    ]

group14 = [  
        "FashionMNIST_5.npz",
        "FashionMNIST_6.npz",
        "FashionMNIST_7.npz",
        "FashionMNIST_8.npz",
        "FashionMNIST_9.npz"
    ]

group15 = [
        "SVHN_5.npz",
        "SVHN_6.npz",
        "SVHN_7.npz",
        "SVHN_8.npz",
        "SVHN_9.npz"
    ]


if __name__ == '__main__':
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
