# -*- coding: utf-8 -*-
"""
uLEAD-TabPFN: Uncertainty-aware Dependency-based Anomaly Detection with TabPFN
Stand-alone version for ADBench integration

Architecture:
    X (d dims) -> Linear Encoder -> z (p dims) -> Dependency Expert -> Anomaly Scores

Key Features:
    - Linear encoder trained with dependency-based loss (dep_decoder mode)
    - Encoder learns latent space where dimensions are predictable from each other
    - TabPFN-based dependency expert operates in latent space to detect anomalies
    - Adaptive latent dimension with identity mapping for low-dimensional data
      and a capped compressed representation for high-dimensional data
    - Double normalization: min-max [0,1] + median/MAD
    - All features treated as numerical (simplified pipeline)

ADBench Interface:
    - __init__(seed, **kwargs): Initialize with seed for reproducibility
    - fit(X_train, y_train): Train encoder on labeled data using context set
    - predict_score(X_test): Return composite conditional NLL anomaly scores
"""

import os
os.environ['tabpfn_DISABLE_TELEMETRY'] = '1'
# Avoid MKL KMeans memory leak warning on Windows by capping threads.
os.environ['OMP_NUM_THREADS'] = '9'

import numpy as np
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from tabpfn import TabPFNClassifier, TabPFNRegressor
from config import CONFIG

# Try to import scipy for rank transformation
try:
    from scipy.stats import rankdata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# Helper Functions
# ============================================================================

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, cached
    return 0.0, 0.0


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def batch_predict(model, X, use_proba=False):
    """
    Perform predictions in batches to reduce memory usage.

    Args:
        model: Trained tabpfn model (classifier or regressor)
        X: Input features for prediction
        use_proba: If True, use predict_proba for classifiers

    Returns:
        predictions: Concatenated predictions
    """
    with torch.no_grad():
        pred_batch_size = CONFIG['pred_batch_size']
        if len(X) <= pred_batch_size:
            if use_proba and hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return model.predict(X)

        # Process in batches
        all_predictions = []
        n_samples = len(X)

        for i in range(0, n_samples, pred_batch_size):
            end_idx = min(i + pred_batch_size, n_samples)
            if hasattr(X, 'iloc'):
                X_batch = X.iloc[i:end_idx]
            else:
                X_batch = X[i:end_idx]

            if use_proba and hasattr(model, 'predict_proba'):
                batch_pred = model.predict_proba(X_batch)
            else:
                batch_pred = model.predict(X_batch)

            all_predictions.append(batch_pred)

            # Clear intermediate memory
            del batch_pred
            if i % (pred_batch_size * 3) == 0:  # Clear cache every 3 batches
                clear_gpu_memory()

        # Concatenate all predictions
        if len(all_predictions[0].shape) > 1:  # 2D array (probabilities)
            result = np.vstack(all_predictions)
        else:  # 1D array (predictions)
            result = np.concatenate(all_predictions)

        # Clear batch predictions from memory
        del all_predictions
        clear_gpu_memory()

        return result


# ============================================================================
# TabPFN Context Manager
# ============================================================================

class TabPFNManager:
    """Context manager for tabpfn model lifecycle with GPU memory management."""

    def __init__(self, model_type='classifier', device='cuda', **kwargs):
        self.model_type = model_type
        self.device = device
        self.kwargs = kwargs
        self.model = None

    def __enter__(self):
        if self.model_type == 'regressor':
            self.model = TabPFNRegressor(device=self.device, **self.kwargs)
        else:
            self.model = TabPFNClassifier(device=self.device, **self.kwargs)
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model is not None:
            del self.model
            self.model = None
        clear_gpu_memory()


# ============================================================================
# Linear Encoder with Dependency-Based Training (PyTorch Module)
# ============================================================================

class LinearAutoencoder(nn.Module):
    """
    Linear encoder for dimensionality reduction with dependency-based training (dep_decoder mode).

    Architecture:
        Encoder: z = W·x + b
        Decoder: x' = W^T·z + c (used for reconstruction loss during training)

    Training Loss (dep_decoder mode):
        L = L_dep + λ_rec·L_recon + λ_orth·L_orth + λ_z·L_z + λ_sparse·L_sparse
        where:
            L_dep = Dependency deviation loss (TabPFN-based predictions in latent space)
            L_recon = Reconstruction loss (MSE or Huber)
    """

    def __init__(self, input_dim, latent_dim, use_batchnorm=False):
        """
        Args:
            input_dim (int): Original feature dimension (d)
            latent_dim (int): Latent dimension (p)
            use_batchnorm (bool): Whether to use batch normalization in encoder
        """
        super(LinearAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batchnorm = use_batchnorm

        # Encoder: z = W·x + b
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)

        # Optional batch normalization for encoder output
        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(latent_dim)
        else:
            self.bn = None

        # Decoder: x' = W^T·z + c
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x):
        """Encode input to latent space."""
        z = self.encoder(x)

        # Optional batch normalization
        if self.use_batchnorm and self.bn is not None:
            if z.size(0) > 1:
                # Check if batch has sufficient variance for stable BatchNorm
                z_var = torch.var(z, dim=0)
                if (z_var > 1e-8).all():
                    z = self.bn(z)  # Apply batch normalization
            else:
                # For batch_size=1, use BatchNorm in eval mode
                self.bn.eval()
                z = self.bn(z)
                self.bn.train()

        return z

    def decode(self, z):
        """Decode latent representation to original space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: x → z → x'"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_dep_deviation_loss(self, Z, tabpfn_reg=None, tabpfn_models=None):
        """
        Compute dependency deviation loss for dep_decoder mode.

        For each latent dimension i:
            - Use ALL other latent dimensions as predictors
            - Fit frozen TabPFN regressor on context set (REUSED across dimensions for speed)
        - Compute prediction deviations
        Return sum of squared deviations across all dimensions.

        Args:
            Z (tensor): Latent codes (n_context, latent_dim)
            tabpfn_reg: Pre-created TabPFN regressor (reused for all dimensions)
            tabpfn_models: Optional list of pre-fitted TabPFN models, one per dimension
            sample_weights (tensor, optional): Sample weights for soft context mode

        Returns:
            loss (tensor): Scalar loss (sum of squared deviations)
        """
        if tabpfn_models is None and tabpfn_reg is None:
            raise ValueError("Either tabpfn_reg or tabpfn_models must be provided")

        device = Z.device
        latent_dim = Z.shape[1]

        # Clip latent codes for TabPFN stability
        Z = torch.clamp(Z, -3, 3)

        # Detach Z for TabPFN (frozen, no backprop through it)
        Z_detached = Z.detach().cpu().numpy()

        # Pre-allocate matrix for all predictions (vectorized aggregation optimization)
        all_predictions = torch.zeros_like(Z)  # (n_context, latent_dim)

        # For each latent dimension
        for target_idx in range(latent_dim):
            # Use ALL other latent dimensions as predictors
            predictor_idx = np.delete(np.arange(latent_dim), target_idx)

            # Extract target and predictors from detached Z for TabPFN
            y_true_np = Z_detached[:, target_idx]
            X_parents_np = Z_detached[:, predictor_idx]

            # Validate data before fitting TabPFN
            if np.isnan(y_true_np).any() or np.isnan(X_parents_np).any():
                # Store zeros as predictions for invalid data
                all_predictions[:, target_idx] = torch.zeros(Z.shape[0], dtype=torch.float32, device=device)
            else:
                if tabpfn_models is not None:
                    # Use pre-fitted model for this dimension (no refit per batch)
                    tabpfn = tabpfn_models[target_idx]
                    y_pred_np = batch_predict(tabpfn, X_parents_np)
                else:
                    # Fit TabPFN regressor and get predictions (reuse same model for all dimensions)
                    tabpfn_reg.fit(X_parents_np, y_true_np)
                    y_pred_np = batch_predict(tabpfn_reg, X_parents_np)

                # Validate TabPFN predictions for NaN
                if np.isnan(y_pred_np).any() or np.isinf(y_pred_np).any():
                    # Store zeros as predictions for invalid predictions
                    all_predictions[:, target_idx] = torch.zeros(Z.shape[0], dtype=torch.float32, device=device)
                else:
                    # Convert predictions to tensor and store (no gradient)
                    all_predictions[:, target_idx] = torch.tensor(y_pred_np, dtype=torch.float32, device=device)

        # Vectorized loss computation: compute deviations for all dimensions at once
        deviations = (Z - all_predictions) ** 2  # (n_context, latent_dim)
        total_loss = torch.mean(deviations)

        return total_loss

    def normalize_latent(self, Z):
        """
        Normalize each latent dimension to unit standard deviation.

        Args:
            Z (tensor): Latent codes (n_samples, latent_dim)

        Returns:
            Z_normalized (tensor): Normalized latent codes
        """
        eps = 1e-6
        Z_std = torch.std(Z, dim=0, keepdim=True)  # (1, latent_dim)

        # Prevent division by very small std
        Z_std_safe = torch.maximum(Z_std, torch.tensor(1e-3, device=Z.device))
        Z_normalized = Z / (Z_std_safe + eps)

        # Replace any NaN or Inf with zeros
        Z_normalized = torch.nan_to_num(Z_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return Z_normalized


# ============================================================================
# Variance Network for Distributional Dependency Modeling (DDM)
# ============================================================================

class VarianceNetwork(nn.Module):
    """
    Variance network for predicting log-variance in DDM.
    Predicts log(σ²) for each latent dimension given other dimensions.

    Architecture: Simple MLP with dropout for regularization.
    Used in distributional dependency modeling (DDM) to estimate
    heteroscedastic uncertainty in latent space dependencies.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        """
        Args:
            input_dim: Number of predictor dimensions (latent_dim - 1)
            hidden_dim: Hidden layer size
            dropout: Dropout rate for regularization
        """
        super(VarianceNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output: log(σ²)
        )
        # Conservative initialization: bias = -4.6 → σ² ≈ 0.01
        # Prevents variance collapse during early training
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.6)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Predictor features (n_samples, input_dim)
        Returns:
            log_var: Log-variance predictions (n_samples,)
        """
        log_var = self.network(x).squeeze(-1)
        return log_var


# ============================================================================
# LeadTabPFN Main Class (ADBench Interface)
# ============================================================================

class LeadTabPFN:
    """
    uLEAD-TabPFN: Uncertainty-aware Dependency-based Anomaly Detection with TabPFN

    Uses dependency-based training (dep_decoder mode) where the encoder learns to create
    a latent space where each dimension is predictable from others using TabPFN.

    ADBench-compatible interface with fit() and predict_score() methods.

    Pipeline:
        1. fit(): Normalize → Generate representative context set → Train encoder (dep_decoder) → Cache latent representations
        2. predict_score(): Normalize → Encode → Compute conditional NLL → Return anomaly scores

    Performance Optimizations:
        - TabPFN model reuse: Create once per epoch/call instead of per-dimension (50-70% speedup)
        - Batch processing: Efficient batched predictions with configurable batch sizes
        - Context caching: Reuse generated context sets and encodings across predictions
    """

    def __init__(self, seed=42):
        """
        Initialize LeadTabPFN anomaly detector.

        All hyperparameters are read from config.CONFIG.
        Only seed is a parameter since it changes per iteration for reproducibility experiments.

        To modify hyperparameters, edit config.py before instantiation.

        Args:
            seed (int): Random seed for reproducibility (default: 42)

        Configuration:
            All other parameters are read from config.CONFIG and organized by the three-step
            architecture of the LEAD model:

            STEP 1 (Data Preparation & Context Generation):
                - context_cap, context_fraction, include_context_in_test

            STEP 2 (Encoder Training & Latent Representation):
                - latent_dim, latent_dim_identity_threshold, latent_dim_max_cap
                - ae_epochs, ae_warmup_epochs, ae_batch_size, ae_lr_init, ae_lr_final
                - ae_lambda_rec, ae_lambda_rec_base, ae_lambda_rec_factor, ae_dynamic_rec_scaling
                - use_original_features

            STEP 3 (Dependency-Based Anomaly Detection & Scoring):
                - scoring_method ('dependency')

            Infrastructure:
                - device, verbose

            See config.py for detailed descriptions and default values.

        Example:
            >>> # Use default CONFIG
            >>> model = LeadTabPFN(seed=42)
            >>> model.fit(X_train, y_train)
            >>> scores = model.predict_score(X_test)

            >>> # Override CONFIG before instantiation
            >>> from config import CONFIG
            >>> CONFIG['ae_epochs'] = 50
            >>> model = LeadTabPFN(seed=42)
        """
        # Set random seeds
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # ====================================================================
        # STEP 1: DATA PREPARATION & CONTEXT GENERATION
        # ====================================================================

        # Context Set Generation
        self.context_cap = CONFIG['context_cap']
        self.context_fraction = CONFIG['context_fraction']
        self.include_context_in_test = CONFIG['include_context_in_test']
        self.test_set_exclude = CONFIG.get('test_set_exclude', 'context')
        self.context_clustering_n_clusters = CONFIG['context_clustering_n_clusters']

        # ====================================================================
        # STEP 2: ENCODER TRAINING & LATENT REPRESENTATION
        # ====================================================================

        # Latent Dimensionality
        self.latent_dim_config = CONFIG['latent_dim']
        self.latent_dim_max_cap = CONFIG['latent_dim_max_cap']
        self.latent_dim_identity_threshold = CONFIG['latent_dim_identity_threshold']

        # Training Dynamics
        self.ae_epochs = CONFIG['ae_epochs']
        self.ae_warmup_epochs = CONFIG['ae_warmup_epochs']
        self.ae_batch_size = CONFIG['ae_batch_size']
        self.ae_lr_init = CONFIG['ae_lr_init']
        self.ae_lr_final = CONFIG['ae_lr_final']

        # Reconstruction Loss
        self.ae_lambda_rec = CONFIG['ae_lambda_rec']
        self.ae_lambda_rec_base = CONFIG['ae_lambda_rec_base']
        self.ae_lambda_rec_factor = CONFIG['ae_lambda_rec_factor']
        self.ae_dynamic_rec_scaling = CONFIG['ae_dynamic_rec_scaling']
        self.ae_use_cosine_ramp = CONFIG['ae_use_cosine_ramp']

        # Baseline Mode
        self.use_original_features = CONFIG['use_original_features']

        # ====================================================================
        # INFRASTRUCTURE
        # ====================================================================

        # Device setup: store as string for TabPFN, convert to torch.device for PyTorch ops
        device = CONFIG['device']
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.device_torch = torch.device(self.device)

        # Logging
        self.verbose = CONFIG['verbose']

        # Validate context_fraction
        if not (0.0 < self.context_fraction <= 1.0):
            raise ValueError(
                f"context_fraction must be in range (0.0, 1.0], got {self.context_fraction}"
            )

        # Model components (will be populated during fit)
        self.autoencoder = None
        self.latent_dim = None
        self.context_indices = None
        self.initial_context_indices = None
        self.context_purity = None
        self.normalization_min = None   # Min-max normalization parameters
        self.normalization_max = None
        self.normalization_median = None  # Median/MAD normalization parameters
        self.normalization_mad = None
        self.dep_matrix = None

        self._Z_context_cache = None  # Cache for context latent codes
        self._context_deviation_cache = None  # Cache for context deviation matrix
        self._tabpfn_models = None  # Cache for fitted TabPFN regressors (one per latent dimension)
        self._variance_networks = None  # Cache for variance networks (DDM)

        # Training data dimensions
        self.n_features = None
        self.n_samples_train = None  # Total samples in training set
        self.test_indices = None  # Indices of test samples (non-context)

        # Training metrics tracking
        self.training_metrics = None  # Will store dict with training history

        if self.verbose:
            print(f"[LeadTabPFN] Initialized with seed={seed}, device={self.device}")

    @staticmethod
    def weighted_median(data, weights):
        """Compute weighted median."""
        is_1d = (data.ndim == 1)
        if is_1d:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        weighted_medians = np.zeros(n_features)

        for j in range(n_features):
            sorted_idx = np.argsort(data[:, j])
            sorted_data = data[sorted_idx, j]
            sorted_weights = weights[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            total_weight = cumsum[-1]
            median_idx = np.searchsorted(cumsum, total_weight / 2.0)
            weighted_medians[j] = sorted_data[median_idx]

        return weighted_medians[0] if is_1d else weighted_medians

    @staticmethod
    def weighted_mad(data, weights, center):
        """Compute weighted median absolute deviation (MAD)."""
        is_1d = (data.ndim == 1)
        if is_1d:
            data = data.reshape(-1, 1)
            center = np.array([center])

        abs_dev = np.abs(data - center)
        mad = LeadTabPFN.weighted_median(abs_dev, weights)
        return mad[0] if is_1d else mad

    def _determine_latent_dim(self, n_features):
        """
        Determine the latent dimension for the encoder using the adaptive strategy.

        If CONFIG['latent_dim'] is set to an integer, that value is used directly.
        Otherwise the paper setting is applied:
          - d <= latent_dim_identity_threshold  →  latent_dim = d  (no compression)
          - d >  latent_dim_identity_threshold  →  latent_dim = latent_dim_max_cap
        """
        if n_features < 2:
            raise ValueError("uLEAD-TabPFN requires at least two input features")

        # User-specified override
        if self.latent_dim_config is not None:
            latent_dim = self.latent_dim_config
            if not isinstance(latent_dim, int) or not 2 <= latent_dim <= n_features:
                raise ValueError(
                    f"latent_dim must be an integer in [2, {n_features}], got {latent_dim}"
                )
            if self.verbose:
                print(f"[LeadTabPFN] Latent dimension: {latent_dim} (user-specified, original: {n_features})")
            return latent_dim

        # Adaptive heuristic
        if n_features <= self.latent_dim_identity_threshold:
            latent_dim = n_features
            if self.verbose:
                print(f"[LeadTabPFN] Latent dimension: {latent_dim} (identity, d={n_features} <= threshold={self.latent_dim_identity_threshold})")
        else:
            latent_dim = min(self.latent_dim_max_cap, n_features)
            if self.verbose:
                print(f"[LeadTabPFN] Latent dimension: {latent_dim} (compressed, d={n_features} > threshold={self.latent_dim_identity_threshold}, cap={self.latent_dim_max_cap})")
        return latent_dim

    def _gen_context_set(self, X, y):
        """Construct the normal-only Representative Context Set (RCS).

        Half of the available normal samples are first assigned to the training
        pool. If that pool exceeds ``context_cap``, K-means partitions it and a
        near-uniform quota is selected from each cluster. Each quota combines
        points nearest to and farthest from the centroid to cover central and
        dispersed regions while respecting the global budget.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be two-dimensional and aligned with y")

        normal_indices = np.flatnonzero(y == 0)
        if len(normal_indices) == 0:
            raise ValueError("uLEAD-TabPFN requires at least one normal sample")

        n_initial = max(1, int(len(normal_indices) * self.context_fraction))
        rng = np.random.default_rng(self.seed)
        initial_indices = rng.choice(normal_indices, size=n_initial, replace=False)

        budget = min(self.context_cap, n_initial)
        if n_initial <= budget:
            context_indices = initial_indices.copy()
        elif self.context_clustering_n_clusters is None:
            context_indices = rng.choice(initial_indices, size=budget, replace=False)
        else:
            X_initial = X[initial_indices]
            n_clusters = min(self.context_clustering_n_clusters, budget, n_initial)
            if n_clusters < 2:
                context_indices = rng.choice(initial_indices, size=budget, replace=False)
            else:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.seed,
                    n_init=10,
                    max_iter=300,
                )
                labels = kmeans.fit_predict(X_initial)
                base_quota, remainder = divmod(budget, n_clusters)
                selected = []

                for cluster_id in range(n_clusters):
                    local_positions = np.flatnonzero(labels == cluster_id)
                    quota = min(
                        len(local_positions),
                        base_quota + (1 if cluster_id < remainder else 0),
                    )
                    if quota == 0:
                        continue

                    distances = np.linalg.norm(
                        X_initial[local_positions] - kmeans.cluster_centers_[cluster_id],
                        axis=1,
                    )
                    ordered = local_positions[np.argsort(distances)]
                    n_near = (quota + 1) // 2
                    n_far = quota - n_near
                    chosen = ordered[:n_near]
                    if n_far:
                        chosen = np.concatenate((chosen, ordered[-n_far:]))
                    selected.extend(initial_indices[chosen].tolist())

                # Small clusters may leave part of the uniform quota unused.
                # Fill the remainder deterministically from the unselected pool.
                selected = list(dict.fromkeys(selected))
                if len(selected) < budget:
                    remaining = np.setdiff1d(
                        initial_indices,
                        np.asarray(selected, dtype=int),
                        assume_unique=False,
                    )
                    fill = rng.choice(remaining, size=budget - len(selected), replace=False)
                    selected.extend(fill.tolist())
                context_indices = np.asarray(selected[:budget], dtype=int)

        context_purity = float(np.mean(y[context_indices] == 0))
        if self.verbose:
            print(
                f"[LeadTabPFN] RCS: {len(context_indices)} samples selected "
                f"from {n_initial} normal training samples"
            )
        return context_indices, context_purity, initial_indices

    def _train_autoencoder(self, X_context):
        """
        Train linear encoder using dependency-based loss (dep_decoder mode).

        The encoder is trained to minimize dependency deviations in the latent space,
        where each latent dimension is predicted from other dimensions using TabPFN.

        Args:
            X_context (ndarray): Context set features (n_context, n_features)
        """
        if self.verbose:
            print(f"[LeadTabPFN] Training encoder (dep_decoder mode)...")

        t_start = time.time()

        n_context, n_features = X_context.shape

        # Convert to tensor
        X_context_tensor = torch.tensor(X_context, dtype=torch.float32, device=self.device_torch)

        # Validate input data
        if torch.isnan(X_context_tensor).any() or torch.isinf(X_context_tensor).any():
            if self.verbose:
                print(f"[LeadTabPFN] WARNING: Input data contains NaN or Inf values, replacing with 0")
            X_context_tensor = torch.nan_to_num(X_context_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize linear encoder (nonlinear AE removed)
        self.autoencoder = LinearAutoencoder(
            input_dim=self.n_features,
            latent_dim=self.latent_dim,
            use_batchnorm=False
        ).to(self.device_torch)
        if self.verbose:
            print(f"  [Architecture] LinearAutoencoder: {self.n_features} → {self.latent_dim}")

        # Optimizer with learning rate scheduling
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_lr_init)

        # Learning rate scheduler (exponential decay from lr_init to lr_final)
        lr_lambda = lambda epoch: np.exp(np.log(self.ae_lr_final / self.ae_lr_init) * epoch / max(self.ae_epochs - 1, 1))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # TabPFN settings for dependency loss (fit once per epoch, reuse for all batches)
        tabpfn_models = None
        tabpfn_manager = TabPFNManager(model_type='regressor', device=self.device,
                                       ignore_pretraining_limits=True)

        # Initialize dynamic reconstruction loss weight
        if self.ae_dynamic_rec_scaling:
            lambda_rec_eff = None  # Will be computed after warmup
            if self.verbose:
                print(f"  [Dynamic Scaling] Enabled (base={self.ae_lambda_rec_base}, factor={self.ae_lambda_rec_factor})")
        else:
            lambda_rec_eff = self.ae_lambda_rec
            if self.verbose:
                print(f"  [Fixed Lambda] lambda_rec={lambda_rec_eff}")

        # Keep at least one full-training epoch when a shortened smoke test is
        # requested with fewer epochs than the paper configuration.
        warmup_epochs = min(self.ae_warmup_epochs, max(self.ae_epochs - 1, 0))

        # Training phases
        if warmup_epochs > 0:
            if self.verbose:
                print(f"  [Training Strategy] Two-phase training:")
                print(f"    Phase 1 (Warmup): Epochs 1-{warmup_epochs} - Reconstruction only")
                print(f"    Phase 2 (Full): Epochs {warmup_epochs+1}-{self.ae_epochs} - All losses")
        else:
            if self.verbose:
                print(f"  [Training Strategy] Single-phase training with all losses from start")

        # Training loop
        batch_size = min(self.ae_batch_size, n_context)
        n_batches = (n_context + batch_size - 1) // batch_size

        # Training metrics tracking
        first_epoch_after_warmup = None
        dep_loss_start = None
        recon_loss_start = None
        total_loss_start = None
        dep_loss_end = None
        recon_loss_end = None
        total_loss_end = None

        self.autoencoder.train()
        for epoch in range(self.ae_epochs):
            epoch_loss = 0.0
            epoch_dep = 0.0
            epoch_recon = 0.0

            # Check if in warmup phase
            is_warmup = (epoch < warmup_epochs)
            tabpfn_models = None

            # Create TabPFN model once per epoch (reuse across all batches and dimensions)
            # This is a major performance optimization: instead of creating 100s of models,
            # we create just ONE per epoch and reuse it for all batches and dimensions
            with tabpfn_manager as tabpfn_reg:
                if not is_warmup:
                    if self.verbose:
                        print(f"  [TabPFN] Fitting dependency models for epoch {epoch+1}...")

                    with torch.no_grad():
                        z_full = self.autoencoder.encode(X_context_tensor)
                        z_full = self.autoencoder.normalize_latent(z_full)
                        z_full = torch.clamp(z_full, -3, 3)
                        if torch.isnan(z_full).any() or torch.isinf(z_full).any():
                            z_full = torch.nan_to_num(z_full, nan=0.0, posinf=0.0, neginf=0.0)
                        Z_full_np = z_full.cpu().numpy()

                    tabpfn_models = []
                    for target_idx in range(self.latent_dim):
                        predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)
                        X_parents_context = Z_full_np[:, predictor_idx]
                        y_target_context = Z_full_np[:, target_idx]

                        tabpfn = TabPFNRegressor(device=self.device, ignore_pretraining_limits=True)
                        tabpfn.fit(X_parents_context, y_target_context)
                        tabpfn_models.append(tabpfn)

                # Shuffle data and weights for each epoch
                perm = torch.randperm(n_context)
                X_shuffled = X_context_tensor[perm]

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, n_context)
                    X_batch = X_shuffled[start_idx:end_idx]


                    # Skip single-sample batches
                    if len(X_batch) < 2:
                        continue

                    optimizer.zero_grad()

                    # Forward pass
                    x_recon, z = self.autoencoder(X_batch)

                    # Normalize latent codes (skip during warmup)
                    z_normalized = self.autoencoder.normalize_latent(z) if not is_warmup else z
                    z_normalized = torch.clamp(z_normalized, -3, 3)

                    # Check for NaN/Inf in latent codes
                    if torch.isnan(z_normalized).any() or torch.isinf(z_normalized).any():
                        if self.verbose:
                            print(f"  WARNING: NaN/Inf in latent codes at epoch {epoch+1}, skipping batch")
                        z_normalized = torch.nan_to_num(z_normalized, nan=0.0, posinf=0.0, neginf=0.0)

                    # Compute reconstruction error (Huber loss with delta=1.0)
                    recon_error = torch.nn.functional.huber_loss(
                        x_recon, X_batch, reduction='none', delta=1.0
                    )

                    # Compute reconstruction loss
                    loss_recon = torch.mean(recon_error)

                    # Compute dependency deviation loss (skip during warmup)
                    if is_warmup:
                        dep_loss = torch.tensor(0.0, device=X_batch.device)
                    else:
                        if torch.isnan(z_normalized).any() or torch.isinf(z_normalized).any():
                            dep_loss = torch.tensor(0.0, device=z_normalized.device)
                        else:
                            dep_loss = self.autoencoder.compute_dep_deviation_loss(
                                z_normalized, tabpfn_reg=tabpfn_reg, tabpfn_models=tabpfn_models
                            )



                    # Dynamic reconstruction loss scaling (at start of full training)
                    if lambda_rec_eff is None and epoch == warmup_epochs and batch_idx == 0:
                        dep_raw = dep_loss.item()
                        recon_raw = loss_recon.item()
                        eps = 1e-8

                        if recon_raw > eps and dep_raw > eps:
                            lambda_rec_eff = (self.ae_lambda_rec_base *
                                            self.ae_lambda_rec_factor *
                                            (dep_raw / (recon_raw + eps)))
                        else:
                            lambda_rec_eff = self.ae_lambda_rec_base

                        if self.verbose:
                            print(f"  [Dynamic Scaling] Calibration at epoch {epoch+1}:")
                            print(f"    dep={dep_raw:.6f}, recon={recon_raw:.6f}, lambda_rec={lambda_rec_eff:.6f}")

                    # Apply cosine ramp to reconstruction weight (only after warmup)
                    lambda_rec_base = lambda_rec_eff if lambda_rec_eff is not None else self.ae_lambda_rec_base
                    if self.ae_use_cosine_ramp and not is_warmup:
                        # lambda_rec(epoch) = lambda_max * 0.5 * (1 + cos(pi * progress))
                        # where progress goes from 0 to 1 over epochs after warmup
                        epochs_after_warmup = epoch - warmup_epochs
                        total_post_warmup = max(self.ae_epochs - warmup_epochs - 1, 1)
                        progress = epochs_after_warmup / total_post_warmup
                        lambda_rec_used = lambda_rec_base * 0.5 * (1 + np.cos(np.pi * progress))
                    else:
                        lambda_rec_used = lambda_rec_base

                    # Combined loss
                    loss = dep_loss + lambda_rec_used * loss_recon

                    # Check for NaN/Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        if self.verbose:
                            print(f"  WARNING: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}, skipping")
                        continue

                    # Backward pass
                    loss.backward()

                    # Check for NaN in gradients
                    gradients_have_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in self.autoencoder.parameters()
                    )
                    if gradients_have_nan:
                        if self.verbose:
                            print(f"  WARNING: NaN in gradients at epoch {epoch+1}, batch {batch_idx}, skipping")
                        optimizer.zero_grad()
                        continue

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Track losses
                    epoch_loss += loss.item() * len(X_batch)
                    epoch_dep += dep_loss.item() * len(X_batch)
                    epoch_recon += loss_recon.item() * len(X_batch)

            if tabpfn_models is not None:
                del tabpfn_models
                tabpfn_models = None
                clear_gpu_memory()

            # Average losses (after all batches in epoch)
            epoch_loss /= n_context
            epoch_dep /= n_context
            epoch_recon /= n_context

            # Track first epoch after warmup for start metrics
            if not is_warmup and first_epoch_after_warmup is None:
                first_epoch_after_warmup = epoch
                dep_loss_start = epoch_dep
                recon_loss_start = epoch_recon
                total_loss_start = epoch_loss

            # Track last epoch for end metrics
            if epoch == self.ae_epochs - 1:
                dep_loss_end = epoch_dep
                recon_loss_end = epoch_recon
                total_loss_end = epoch_loss

            # Update learning rate (after epoch completes)
            scheduler.step()

            # Log progress
            current_lr = optimizer.param_groups[0]['lr']
            phase = "Warmup" if is_warmup else "Full"
            # Compute lambda_rec_used for logging (same logic as in the batch loop)
            lambda_rec_base_log = lambda_rec_eff if lambda_rec_eff is not None else self.ae_lambda_rec_base
            if self.ae_use_cosine_ramp and not is_warmup:
                epochs_after_warmup = epoch - warmup_epochs
                total_post_warmup = max(self.ae_epochs - warmup_epochs - 1, 1)
                progress = epochs_after_warmup / total_post_warmup
                lambda_rec_log = lambda_rec_base_log * 0.5 * (1 + np.cos(np.pi * progress))
            else:
                lambda_rec_log = lambda_rec_base_log

            print(f"  Epoch {epoch+1}/{self.ae_epochs} [{phase}]: "
                  f"total={epoch_loss:.6f}, dep={epoch_dep:.6f}, "
                  f"recon={epoch_recon:.6f}(w:{lambda_rec_log:.6f}), "
                  f"lr={current_lr:.6f}")

        self.autoencoder.eval()

        # Store effective lambda for later use
        self.lambda_rec_eff = lambda_rec_eff if lambda_rec_eff is not None else self.ae_lambda_rec

        # Store training metrics
        self.training_metrics = {
            'n_epoch': self.ae_epochs,
            'dep_loss_start': dep_loss_start if dep_loss_start is not None else 0.0,
            'dep_loss_end': dep_loss_end if dep_loss_end is not None else 0.0,
            'recon_loss_start': recon_loss_start if recon_loss_start is not None else 0.0,
            'recon_loss_end': recon_loss_end if recon_loss_end is not None else 0.0,
            'total_loss_start': total_loss_start if total_loss_start is not None else 0.0,
            'total_loss_end': total_loss_end if total_loss_end is not None else 0.0,
            'lambda_rec': self.lambda_rec_eff
        }

        if self.verbose:
            print(f"[LeadTabPFN] Encoder training completed in {time.time()-t_start:.2f}s")

    def _encode_dataset(self, X_normalized, batch_size=None):
        """
        Encode normalized dataset to latent space with batching.

        Args:
            X_normalized (ndarray): Normalized features (n_samples, n_features)
            batch_size (int, optional): Batch size for encoding. Defaults to CONFIG['pred_batch_size']

        Returns:
            Z (ndarray): Latent codes (n_samples, latent_dim)
        """
        if batch_size is None:
            batch_size = CONFIG['pred_batch_size']

        n_samples = len(X_normalized)

        # Small dataset - process at once
        if n_samples <= batch_size:
            with torch.no_grad():
                X_tensor = torch.tensor(X_normalized, dtype=torch.float32, device=self.device_torch)
                Z = self.autoencoder.encode(X_tensor)
                Z = Z.cpu().numpy()
            return Z

        # Large dataset - batch processing
        Z_list = []
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_normalized[i:end_idx]

            with torch.no_grad():
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=self.device_torch)
                Z_batch = self.autoencoder.encode(X_tensor)
                Z_batch = Z_batch.cpu().numpy()

            Z_list.append(Z_batch)

            # Clear GPU memory every 3 batches
            if i % (batch_size * 3) == 0 and i > 0:
                torch.cuda.empty_cache()

        Z = np.vstack(Z_list)

        # Final cleanup
        del Z_list
        torch.cuda.empty_cache()

        return Z

    def _detect_dependence_trivial(self, Z_context):
        """
        Trivial dependency detection: use all other latent dimensions as parents.

        Args:
            Z_context (ndarray): Context set latent codes (n_context, latent_dim)

        Returns:
            dep_matrix (ndarray): Dependency matrix (latent_dim, latent_dim)
        """
        # For each dimension, use all other dimensions as parents
        dep_matrix = np.ones((self.latent_dim, self.latent_dim))
        np.fill_diagonal(dep_matrix, 0)  # No self-dependency

        if self.verbose:
            print(f"[LeadTabPFN] Using all-to-all dependencies")

        return dep_matrix

    def gen_dep_deviation_latent(self, Z, context_indices, dep_matrix):
        """
        Compute dependency deviations in latent space.

        For each latent dimension, predict from parent dimensions using TabPFN
        trained on context set, then compute prediction deviations.

        Args:
            Z (ndarray): Full dataset latent codes (n_samples, latent_dim)
            context_indices (ndarray): Context set indices
            dep_matrix (ndarray): Dependency matrix (latent_dim, latent_dim)

        Returns:
            deviation_matrix (ndarray): Deviation matrix (latent_dim, n_samples)
        """
        if self.verbose:
            print(f"[LeadTabPFN] Computing dependency deviations in latent space...")

        t_start = time.time()

        n_samples = Z.shape[0]
        deviation_matrix = np.zeros((self.latent_dim, n_samples))

        Z_context = Z[context_indices]

        for target_idx in range(self.latent_dim):
            # Get parent indices
            parent_indices = np.where(dep_matrix[target_idx, :] > 0)[0]

            if len(parent_indices) == 0:
                # No parents, use marginal deviation (median-based)
                median_val = np.median(Z_context[:, target_idx])
                mad_val = np.median(np.abs(Z_context[:, target_idx] - median_val))
                deviations = np.abs(Z[:, target_idx] - median_val) / (mad_val + 1e-8)
            else:
                # Use TabPFN to predict from parents
                X_parents_context = Z_context[:, parent_indices]
                y_target_context = Z_context[:, target_idx]

                X_parents_full = Z[:, parent_indices]

                try:
                    with TabPFNManager(model_type='regressor', device=self.device,
                                     ignore_pretraining_limits=True) as tabpfn_reg:
                        tabpfn_reg.fit(X_parents_context, y_target_context)
                        y_pred = batch_predict(tabpfn_reg, X_parents_full)

                    # Compute absolute deviations
                    deviations = np.abs(Z[:, target_idx] - y_pred)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: TabPFN prediction failed for dimension {target_idx}: {e}")
                    # Fallback to marginal deviation
                    median_val = np.median(Z_context[:, target_idx])
                    mad_val = np.median(np.abs(Z_context[:, target_idx] - median_val))
                    deviations = np.abs(Z[:, target_idx] - median_val) / (mad_val + 1e-8)

            deviation_matrix[target_idx, :] = deviations

        if self.verbose:
            print(f"[LeadTabPFN] Dependency deviations computed in {time.time()-t_start:.2f}s")

        return deviation_matrix

    def compute_anomaly_scores(self, dev_abs, dev_abs_context=None):
        """
        Compute anomaly scores from absolute deviations.

        Args:
            dev_abs (ndarray): Absolute deviations for test samples (n_samples, n_features)
            dev_abs_context (ndarray, optional): Absolute deviations for context samples (n_context, n_features)
                                                If provided, median/AAD are computed from this instead of dev_abs

        Returns:
            anomaly_scores (ndarray): Anomaly scores (n_samples,)
        """
        # Use context statistics if provided, otherwise use test statistics (backward compatible)
        reference_data = dev_abs_context if dev_abs_context is not None else dev_abs
        
        # Compute median and AAD from reference data (context or test)
        medians = np.median(reference_data, axis=0)  # shape: (n_features,)
        deviations_from_median_ref = reference_data - medians
        aad = np.mean(np.abs(deviations_from_median_ref), axis=0)  # shape: (n_features,)

        # Normalize TEST deviations using reference statistics
        deviations_from_median = dev_abs - medians  # shape: (n_samples, n_features)
        normalized_deviations = np.zeros_like(dev_abs)

        # Identify features where AAD > 0 (normal case) vs AAD == 0 (constant feature)
        valid_aad_mask = aad > 0
        zero_aad_mask = ~valid_aad_mask
        
        # For features with AAD > 0, use robust z-score (median/AAD)
        if np.any(valid_aad_mask):
            normalized_deviations[:, valid_aad_mask] = deviations_from_median[:, valid_aad_mask] / aad[valid_aad_mask]
        
        # For features with AAD == 0 (constant in reference), use standard z-score
        # to detect univariate deviations
        if np.any(zero_aad_mask):
            # Compute mean and std from reference data for these features
            means_zero_aad = np.mean(reference_data[:, zero_aad_mask], axis=0)
            stds_zero_aad = np.std(reference_data[:, zero_aad_mask], axis=0)
            
            # Compute absolute z-scores for test data
            # If std is also 0 (truly constant), z-score will be 0 (no deviation detectable)
            std_nonzero_mask = stds_zero_aad > 0
            if np.any(std_nonzero_mask):
                # Extract columns with zero AAD
                zero_aad_cols = dev_abs[:, zero_aad_mask]
                # Compute z-scores: abs((value - mean) / std)
                z_scores = np.abs((zero_aad_cols - means_zero_aad) / np.where(stds_zero_aad > 0, stds_zero_aad, 1.0))
                # Only keep z-scores where std > 0
                z_scores[:, ~std_nonzero_mask] = 0
                normalized_deviations[:, zero_aad_mask] = z_scores

        # Clip negative values to 0 (ReLU)
        normalized_deviations[normalized_deviations < 0] = 0
        
        # Sum across features (axis 1)
        anomaly_scores = np.sum(normalized_deviations, axis=1)
        return anomaly_scores

    def _aggregate_ddm_scores(self, nll_matrix, aggregation='mean'):
        """
        Aggregate per-dimension NLL scores to final anomaly score.

        Args:
            nll_matrix: NLL per dimension (n_samples, latent_dim)
            aggregation: Aggregation method
                - 'mean': Average NLL across dimensions (default)
                - 'sum': Sum NLL across dimensions
                - 'top_k_mean': Average of top-k highest NLL dimensions
                - 'logsumexp': log(Σ exp(NLL_j)) for smooth max

        Returns:
            anomaly_scores: Aggregated scores (n_samples,)
        """
        if aggregation == 'mean':
            return np.mean(nll_matrix, axis=1)

        elif aggregation == 'sum':
            return np.sum(nll_matrix, axis=1)

        elif aggregation == 'top_k_mean':
            k = CONFIG['ddm_top_k']
            if k is None:
                k = max(1, nll_matrix.shape[1] // 3)
            k = min(k, nll_matrix.shape[1])
            sorted_nll = np.sort(nll_matrix, axis=1)[:, -k:]
            return np.mean(sorted_nll, axis=1)

        elif aggregation == 'logsumexp':
            from scipy.special import logsumexp
            return logsumexp(nll_matrix, axis=1)

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def _compute_reconstruction_error(self, X_normalized, batch_size=None):
        """
        Compute per-sample reconstruction error from the autoencoder with batching.

        Args:
            X_normalized (ndarray): Normalized input data (n_samples, n_features)
            batch_size (int, optional): Batch size for processing. Defaults to CONFIG['pred_batch_size']

        Returns:
            recon_errors (ndarray): Per-sample reconstruction errors (n_samples,)
        """
        if batch_size is None:
            batch_size = CONFIG['pred_batch_size']

        n_samples = len(X_normalized)
        self.autoencoder.eval()

        # Small dataset - process at once
        if n_samples <= batch_size:
            with torch.no_grad():
                X_tensor = torch.tensor(X_normalized, dtype=torch.float32, device=self.device_torch)
                X_recon, _ = self.autoencoder(X_tensor)

                # Compute reconstruction error (Huber loss with delta=1.0)
                err = torch.nn.functional.huber_loss(
                    X_recon, X_tensor, reduction='none', delta=1.0
                )

                # Mean over features to get per-sample error
                err = err.mean(dim=1).cpu().numpy()

            return err

        # Large dataset - batch processing
        err_list = []
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_normalized[i:end_idx]

            with torch.no_grad():
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=self.device_torch)
                X_recon, _ = self.autoencoder(X_tensor)

                # Compute reconstruction error (Huber loss with delta=1.0)
                err_batch = torch.nn.functional.huber_loss(
                    X_recon, X_tensor, reduction='none', delta=1.0
                )

                # Mean over features to get per-sample error
                err_batch = err_batch.mean(dim=1).cpu().numpy()

            err_list.append(err_batch)

            # Clear GPU memory every 3 batches
            if i % (batch_size * 3) == 0 and i > 0:
                torch.cuda.empty_cache()

        err = np.concatenate(err_list)

        # Final cleanup
        del err_list
        torch.cuda.empty_cache()

        return err

    def _fit_variance_networks(self, Z_train, tabpfn_models):
        """
        Fit variance networks for DDM using frozen TabPFN mean predictions.

        For each dimension j:
            - Use TabPFN to predict μ_j = f(z_{-j}) [frozen, trained on 500-sample context]
            - Train variance network to predict log(σ²_j) from z_{-j}
            - Minimize Gaussian NLL: (z_j - μ_j)² / σ² + log(σ²)

        Note: By default, variance networks are trained on initial_normals (before downsampling)
              to provide more training data for better variance estimation. This can be changed
              via CONFIG['ddm_train_on_context'] to use the context set (matches TabPFN training data).
              TabPFN models always use the downsampled context set for stability.

        Args:
            Z_train: Latent codes for variance network training (n_train, latent_dim)
                     Source determined by CONFIG['ddm_train_on_context']:
                     - False: initial normals before downsampling (default)
                     - True: context set (downsampled, matches TabPFN)
            tabpfn_models: Cached TabPFN models trained on context set (from self._tabpfn_models)
        """
        if self.verbose:
            print(f"[LeadTabPFN] Training {self.latent_dim} variance networks for DDM...")
            print(f"[LeadTabPFN]   Using {Z_train.shape[0]} training samples")

        self._variance_networks = []
        device = self.device_torch

        for target_idx in range(self.latent_dim):
            predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)
            X_parents = Z_train[:, predictor_idx]
            y_target = Z_train[:, target_idx]

            # Get TabPFN mean predictions (frozen)
            tabpfn = tabpfn_models[target_idx]
            mu_pred = batch_predict(tabpfn, X_parents)
            residuals = y_target - mu_pred

            # Create and train variance network
            variance_net = VarianceNetwork(
                input_dim=len(predictor_idx),
                hidden_dim=CONFIG['ddm_hidden_dim'],
                dropout=CONFIG['ddm_dropout']
            ).to(device)

            optimizer = torch.optim.Adam(
                variance_net.parameters(),
                lr=CONFIG['ddm_lr'],
                weight_decay=CONFIG['ddm_weight_decay']
            )

            variance_net.train()
            n_epochs = CONFIG['ddm_epochs']
            batch_size = min(CONFIG['ae_batch_size'], len(X_parents))

            for epoch in range(n_epochs):
                perm = np.random.permutation(len(X_parents))
                epoch_loss = 0.0

                for batch_start in range(0, len(X_parents), batch_size):
                    batch_end = min(batch_start + batch_size, len(X_parents))
                    batch_idx = perm[batch_start:batch_end]

                    X_batch = torch.tensor(X_parents[batch_idx], dtype=torch.float32, device=device)
                    r_batch = torch.tensor(residuals[batch_idx], dtype=torch.float32, device=device)

                    # Forward pass
                    log_var = variance_net(X_batch)
                    log_var = torch.clamp(log_var, CONFIG['ddm_log_var_min'], CONFIG['ddm_log_var_max'])
                    var = torch.exp(log_var)

                    # Gaussian NLL loss
                    nll = 0.5 * (r_batch ** 2 / (var + 1e-8) + log_var)
                    loss = torch.mean(nll)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(variance_net.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item() * len(batch_idx)

                epoch_loss /= len(X_parents)
                # if self.verbose and (epoch + 1) % 10 == 0:
                #     print(f"  Dim {target_idx}: Epoch {epoch+1}/{n_epochs}, NLL={epoch_loss:.6f}")

            variance_net.eval()
            self._variance_networks.append(variance_net)

        if self.verbose:
            print(f"[LeadTabPFN] Variance networks training completed")

    def fit(self, X_train, y_train, dataset_name=None):
        """
        Train LeadTabPFN on labeled training data.

        Args:
            X_train (ndarray): Training features (n_samples, n_features)
            y_train (ndarray): Training labels (n_samples,) - 0=normal, 1=anomaly
            dataset_name (str, optional): Name of the dataset being trained on
        """
        if self.verbose:
            print(f"\n{'='*60}")
            dataset_info = f" on {dataset_name}" if dataset_name else ""
            n_anom = int(np.sum(y_train == 1))
            n_normal = len(y_train) - n_anom
            print(f"[LeadTabPFN] Training{dataset_info}...")
            print(f"  Samples: {len(X_train)}, Features: {X_train.shape[1]}")
            print(f"  Normal: {n_normal}, Anomalies: {n_anom}")
            print(f"  Anomaly ratio: {np.mean(y_train == 1):.3f}")
            print(f"{'='*60}\n")

        t_start = time.time()

        # Validate input
        if np.isnan(X_train).any():
            if self.verbose:
                print(f"[LeadTabPFN] WARNING: Input contains {np.isnan(X_train).sum()} NaN values, replacing with 0")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        self.n_features = X_train.shape[1]
        self.n_samples_train = X_train.shape[0]

        # Determine latent dimension
        if self.use_original_features:
            self.latent_dim = self.n_features
            if self.verbose:
                print(f"[LeadTabPFN] Using original features (latent_dim={self.n_features}, no encoder)")
        else:
            self.latent_dim = self._determine_latent_dim(self.n_features)

        # Clear GPU memory before starting
        clear_gpu_memory()


        # 3. Median/MAD normalize using context set statistics
        # Generate context set directly from raw training data
        self.context_indices, self.context_purity, self.initial_context_indices = self._gen_context_set(
            X_train, y_train
        )

        # 3.1 Compute test indices
        if self.include_context_in_test:
            # Include context in test set (all samples)
            self.test_indices = np.arange(self.n_samples_train)
            if self.verbose:
                print(f"[LeadTabPFN] Test set: {len(self.test_indices)} samples (including {len(self.context_indices)} context samples)")
        else:
            # Exclude either final context set or initial normal sample selection
            if self.test_set_exclude == 'initial_normals' and self.initial_context_indices is not None:
                exclude_set = set(self.initial_context_indices)
            else:
                exclude_set = set(self.context_indices)
            self.test_indices = np.array([i for i in range(self.n_samples_train) if i not in exclude_set])
            if self.verbose:
                print(f"[LeadTabPFN] Train/Test split: {len(self.context_indices)} train, {len(self.test_indices)} test (semi-supervised)")

        # 4. Double Normalization: Min-Max + Median/MAD
        X_context = X_train[self.context_indices]

        # 4.1 Min-Max Normalization (based on context set)
        self.normalization_min = np.min(X_context, axis=0, keepdims=True)
        self.normalization_max = np.max(X_context, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = self.normalization_max - self.normalization_min
        range_vals[range_vals == 0] = 1.0
        
        # Apply Min-Max to full training set
        X_minmax = (X_train - self.normalization_min) / range_vals
        X_minmax = np.clip(X_minmax, 0, 1)
        
        # Update context set in minmax space
        X_context_minmax = X_minmax[self.context_indices]

        # 4.2 Median/MAD Normalization (based on minmax context set)
        self.normalization_median = np.median(X_context_minmax, axis=0, keepdims=True)
        self.normalization_mad = np.median(np.abs(X_context_minmax - self.normalization_median), axis=0, keepdims=True)
        if self.verbose:
            print(f"[LeadTabPFN] Normalization: Min-Max + Median/MAD (using context set)")

        eps = 1e-8
        X_normalized = (X_minmax - self.normalization_median) / (self.normalization_mad + eps)
        X_normalized = np.clip(X_normalized, -5, 5)

        if self.verbose:
            print(f"  Range after normalization: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")


        # Train encoder (dep_decoder mode) on double-normalized context set
        X_context_normalized = X_normalized[self.context_indices]

        if not self.use_original_features:
            # Train autoencoder on context set
            self._train_autoencoder(X_context_normalized)
        else:
            # Skip autoencoder training when using original features
            self.autoencoder = None
            self.training_metrics = {
                'n_epoch': 0,
                'dep_loss_start': 0.0,
                'dep_loss_end': 0.0,
                'recon_loss_start': 0.0,
                'recon_loss_end': 0.0,
                'total_loss_start': 0.0,
                'total_loss_end': 0.0,
                'lambda_rec': 0.0
            }
            self.lambda_rec_eff = 0.0
            if self.verbose:
                print(f"[LeadTabPFN] Skipping encoder training (use_original_features=True)")
        
        # Clear GPU memory after autoencoder training
        clear_gpu_memory()

        # Encode context set to latent space and cache
        if self.use_original_features:
            # Use normalized features directly as "latent codes"
            Z_context = X_context_normalized
            self._Z_context_cache = Z_context
            if self.verbose:
                print(f"[LeadTabPFN] Using original features as latent space (shape: {Z_context.shape})")
        else:
            Z_context = self._encode_dataset(X_context_normalized)
            self._Z_context_cache = Z_context  # Cache for use in predict_score

        # Select DDM training data based on CONFIG['ddm_train_on_context']
        if CONFIG['use_ddm']:
            if CONFIG['ddm_train_on_context']:
                # Use context set (downsampled, matches TabPFN training data)
                Z_ddm_train = self._Z_context_cache
                if self.verbose:
                    print(f"[LeadTabPFN] Using context set for DDM variance networks: {Z_ddm_train.shape[0]} samples")
            elif self.initial_context_indices is not None:
                # Use initial normals (larger pre-downsampled set)
                X_initial_normals = X_normalized[self.initial_context_indices]
                ddm_cap = CONFIG.get('ddm_train_cap')
                if ddm_cap is not None and len(X_initial_normals) > ddm_cap:
                    rng = np.random.default_rng(self.seed)
                    subset_idx = rng.choice(len(X_initial_normals), size=ddm_cap, replace=False)
                    X_initial_normals = X_initial_normals[subset_idx]
                if self.use_original_features:
                    Z_ddm_train = X_initial_normals
                else:
                    Z_ddm_train = self._encode_dataset(X_initial_normals)
                if self.verbose:
                    print(f"[LeadTabPFN] Using initial normals for DDM variance networks: {Z_ddm_train.shape[0]} samples")
            else:
                # Fallback to context set if initial_context_indices is None
                Z_ddm_train = self._Z_context_cache
                if self.verbose:
                    print(f"[LeadTabPFN] WARNING: initial_context_indices is None, falling back to context set for DDM: {Z_ddm_train.shape[0]} samples")
        else:
            Z_ddm_train = None

        # Detect dependencies (trivial: all-to-all in latent space)
        self.dep_matrix = self._detect_dependence_trivial(Z_context)

        # Cache fitted TabPFN models for dependency prediction (prevents redundant fitting)
        if self.verbose:
            print(f"[LeadTabPFN] Caching TabPFN models for {self.latent_dim} dimensions...")

        self._tabpfn_models = []
        Z_context = self._Z_context_cache

        for target_idx in range(self.latent_dim):
            predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)
            X_train = Z_context[:, predictor_idx]
            y_train = Z_context[:, target_idx]

            from tabpfn import TabPFNRegressor
            tabpfn = TabPFNRegressor(device=self.device, ignore_pretraining_limits=True)
            tabpfn.fit(X_train, y_train)
            self._tabpfn_models.append(tabpfn)

        if self.verbose:
            print(f"[LeadTabPFN] TabPFN models cached successfully")

        # Train variance networks for DDM (if enabled)
        if CONFIG['use_ddm'] and Z_ddm_train is not None:
            self._fit_variance_networks(Z_ddm_train, self._tabpfn_models)

        # Clear GPU memory
        clear_gpu_memory()

        if self.verbose:
            print(f"\n[LeadTabPFN] Training completed in {time.time()-t_start:.2f}s")
            print(f"{'='*60}\n")


    def _compute_latent_deviations(self, Z_samples, Z_context=None):
        """
        Compute dependency deviations for samples in latent space using cached TabPFN models if available.

        For each latent dimension, predict from parent dimensions using TabPFN
        trained on context set, then compute prediction deviations.

        Args:
            Z_samples (ndarray): Latent codes for samples to score (n_samples, latent_dim)
            Z_context (ndarray, optional): Latent codes for context set (n_context, latent_dim).
                                          Only required if cached TabPFN models are not available.

        Returns:
            deviation_matrix (ndarray): Deviation matrix (n_samples, latent_dim)
        """
        n_samples = Z_samples.shape[0]
        deviation_matrix = np.zeros((n_samples, self.latent_dim))

        # Use cached TabPFN models if available (much faster - avoids redundant fitting)
        if self._tabpfn_models is not None:
            for target_idx in range(self.latent_dim):
                predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)
                X_pred = Z_samples[:, predictor_idx]
                y_true = Z_samples[:, target_idx]

                try:
                    # Use pre-fitted model from cache
                    tabpfn = self._tabpfn_models[target_idx]
                    y_pred = batch_predict(tabpfn, X_pred)
                    deviations = np.abs(y_true - y_pred)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Cached TabPFN prediction failed for dim {target_idx}: {e}")
                    # Fallback to marginal deviation
                    if Z_context is not None:
                        median_val = np.median(Z_context[:, target_idx])
                        mad_val = np.median(np.abs(Z_context[:, target_idx] - median_val))
                        deviations = np.abs(y_true - median_val) / (mad_val + 1e-8)
                    else:
                        # No context available, use zeros
                        deviations = np.zeros(n_samples)

                deviation_matrix[:, target_idx] = deviations

            return deviation_matrix

        # Fallback: No cached models available, fit TabPFN on-the-fly (old behavior)
        if Z_context is None:
            raise ValueError("Z_context required when TabPFN models are not cached")

        # Create TabPFN model once and reuse for all latent dimensions
        with TabPFNManager(model_type='regressor', device=self.device,
                         ignore_pretraining_limits=True) as tabpfn_reg:
            for target_idx in range(self.latent_dim):
                # Use all other latent dimensions as predictors
                predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)

                # Prepare training data (context set)
                X_train = Z_context[:, predictor_idx]
                y_train = Z_context[:, target_idx]

                # Prepare test data (samples to score)
                X_pred = Z_samples[:, predictor_idx]
                y_true = Z_samples[:, target_idx]

                try:
                    # Fit and predict with TabPFN
                    tabpfn_reg.fit(X_train, y_train)
                    y_pred = batch_predict(tabpfn_reg, X_pred)

                    # Compute absolute deviations
                    deviations = np.abs(y_true - y_pred)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: TabPFN prediction failed for dimension {target_idx}: {e}")
                    # Fallback to marginal deviation
                    median_val = np.median(Z_context[:, target_idx])
                    mad_val = np.median(np.abs(Z_context[:, target_idx] - median_val))
                    deviations = np.abs(y_true - median_val) / (mad_val + 1e-8)

                deviation_matrix[:, target_idx] = deviations

        return deviation_matrix

    def _compute_latent_nll(self, Z_samples, tabpfn_models, variance_networks):
        """
        Compute Gaussian NLL for samples using DDM.

        For each dimension j:
            μ_j = TabPFN(z_{-j})  [cached, frozen]
            log(σ²_j) = VarianceNet(z_{-j})  [cached]
            NLL_j = 0.5 * ((z_j - μ_j)² / σ²_j + log(σ²_j))

        Args:
            Z_samples: Latent codes for samples (n_samples, latent_dim)
            tabpfn_models: Cached TabPFN models (from self._tabpfn_models)
            variance_networks: Cached variance networks (from self._variance_networks)

        Returns:
            nll_matrix: NLL per dimension (n_samples, latent_dim)
        """
        n_samples = Z_samples.shape[0]
        nll_matrix = np.zeros((n_samples, self.latent_dim))
        device = self.device_torch

        for target_idx in range(self.latent_dim):
            predictor_idx = np.delete(np.arange(self.latent_dim), target_idx)
            X_pred = Z_samples[:, predictor_idx]
            y_true = Z_samples[:, target_idx]

            try:
                # Get mean from TabPFN
                tabpfn = tabpfn_models[target_idx]
                mu_pred = batch_predict(tabpfn, X_pred)

                # Get log-variance from variance network
                variance_net = variance_networks[target_idx]
                variance_net.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_pred, dtype=torch.float32, device=device)
                    log_var = variance_net(X_tensor).cpu().numpy()

                # Stabilize variance
                log_var = np.clip(log_var, CONFIG['ddm_log_var_min'], CONFIG['ddm_log_var_max'])
                var = np.exp(log_var)

                # Compute Gaussian NLL
                residuals = y_true - mu_pred
                nll = 0.5 * (residuals ** 2 / (var + 1e-8) + log_var)
                nll_matrix[:, target_idx] = nll

            except Exception as e:
                if self.verbose:
                    print(f"  Warning: NLL computation failed for dim {target_idx}: {e}")
                nll_matrix[:, target_idx] = 10.0  # Fallback: high anomaly score

        return nll_matrix

    def _rank_transform(self, scores):
        """
        Transform scores to ranks normalized to [0, 1].

        Args:
            scores (ndarray): Raw anomaly scores (n_samples,)

        Returns:
            ranks (ndarray): Rank-normalized scores in [0, 1]
        """
        n = len(scores)
        if n == 0:
            return scores

        if SCIPY_AVAILABLE:
            # Use scipy's rankdata (handles ties better)
            ranks = rankdata(scores, method='average')
        else:
            # Numpy fallback (slightly different tie handling)
            sorted_idx = np.argsort(scores)
            ranks = np.empty(n, dtype=float)
            ranks[sorted_idx] = np.arange(1, n + 1)

        # Normalize to [0, 1]
        normalized_ranks = ranks / n
        return normalized_ranks

    def _cdf_transform(self, scores):
        """
        Transform scores using empirical CDF.

        Args:
            scores (ndarray): Raw anomaly scores (n_samples,)

        Returns:
            cdf_values (ndarray): CDF-transformed scores in [0, 1]
        """
        n = len(scores)
        if n == 0:
            return scores

        # For each score, compute fraction of scores ≤ that value
        cdf_values = np.zeros(n)
        for i in range(n):
            cdf_values[i] = np.sum(scores <= scores[i]) / n

        return cdf_values

    def predict_score(self, X_test, return_intermediate=False):
        """
        Predict anomaly scores for test data.

        Semi-supervised mode: If X_test is the training dataset, returns scores only for
        samples not in the context set (test samples).

        Args:
            X_test (ndarray): Test features (n_samples, n_features)
            return_intermediate (bool): If True, also return the raw dependency deviation matrix

        Returns:
            If return_intermediate=False:
                tuple: (anomaly_scores, test_indices)
                    - anomaly_scores (ndarray): Anomaly scores for test samples (n_test,)
                    - test_indices (ndarray): Indices of scored samples (n_test,)
            If return_intermediate=True:
                tuple: (anomaly_scores, test_indices, dep_deviations)
                    - anomaly_scores (ndarray): Anomaly scores for test samples (n_test,)
                    - test_indices (ndarray): Indices of scored samples (n_test,)
                    - dep_deviations (ndarray or None): Dependency deviation matrix (n_test, latent_dim)
        """
        # Check if model is trained
        if not self.use_original_features and self.autoencoder is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        if self.use_original_features and self._Z_context_cache is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Validate input
        if np.isnan(X_test).any():
            if self.verbose:
                print(f"[LeadTabPFN] WARNING: Test input contains {np.isnan(X_test).sum()} NaN values, replacing with 0")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize intermediate artifact tracking (for caching)
        dep_deviations = None

        # Determine if this is the training dataset (semi-supervised mode)
        is_training_data = (self.n_samples_train is not None and len(X_test) == self.n_samples_train)

        if is_training_data:
            if self.verbose:
                print(f"[LeadTabPFN] Scoring {len(self.test_indices)} test samples")
            # Optimization: Only process test samples
            X_proc = X_test[self.test_indices]
            target_indices = self.test_indices
        else:
            X_proc = X_test
            target_indices = np.arange(len(X_test))

        # Apply Median/MAD normalization
        # 1. Min-Max
        range_vals = self.normalization_max - self.normalization_min
        range_vals[range_vals == 0] = 1.0
        X_minmax = (X_proc - self.normalization_min) / range_vals
        X_minmax = np.clip(X_minmax, 0, 1)
        
        # 2. Median/MAD
        eps = 1e-8
        X_normalized = (X_minmax - self.normalization_median) / (self.normalization_mad + eps)
        X_normalized = np.clip(X_normalized, -5, 5)

        # Compute scores using dependency scoring method
        # 1. Encode to latent space
        if self.use_original_features:
            Z_test = X_normalized  # Use normalized features directly
        else:
            Z_test = self._encode_dataset(X_normalized)
        Z_context = self._Z_context_cache

        # 2. Compute anomaly scores (DDM or standard deviation)
        if CONFIG['use_ddm'] and self._variance_networks is not None:
            # Global DDM scoring
            nll_matrix = self._compute_latent_nll(
                Z_test,
                self._tabpfn_models,
                self._variance_networks
            )

            # Aggregate NLL to anomaly scores
            anomaly_scores = self._aggregate_ddm_scores(
                nll_matrix,
                aggregation=CONFIG['ddm_aggregation']
            )
            # Store NLL matrix for caching (will be saved to ddm_nll/)
            dep_deviations = nll_matrix
        else:
            # Standard: Compute dependency deviations
            deviation_matrix_test = self._compute_latent_deviations(Z_test, Z_context)
            dep_deviations = deviation_matrix_test  # Store for caching

            # Get context deviations for normalization
            deviation_matrix_context = None
            if self._context_deviation_cache is not None:
                deviation_matrix_context = self._context_deviation_cache
            else:
                deviation_matrix_context = self._compute_latent_deviations(Z_context, Z_context)
                self._context_deviation_cache = deviation_matrix_context

            # Compute anomaly scores using median/AAD normalization
            anomaly_scores = self.compute_anomaly_scores(
                deviation_matrix_test,
                dev_abs_context=deviation_matrix_context
            )

        # Return based on flag
        if return_intermediate:
            return anomaly_scores, target_indices, dep_deviations
        else:
            return anomaly_scores, target_indices

    def save_prediction_artifacts(self, dep_dev_path=None, dep_deviations=None):
        """
        Save intermediate prediction artifacts to disk.

        This allows caching expensive TabPFN inference results to avoid
        re-computation on subsequent runs.

        Args:
            dep_dev_path (str, optional): Path to save dependency deviation matrix (.npy)
            dep_deviations (ndarray, optional): Dependency deviation matrix (n_test, latent_dim)
        """
        import numpy as np

        if dep_dev_path is not None and dep_deviations is not None:
            np.save(dep_dev_path, dep_deviations)
            if self.verbose:
                if CONFIG['use_ddm']:
                    print(f"  Saved NLL matrix to {dep_dev_path}")
                else:
                    print(f"  Saved dependency deviations to {dep_dev_path}")

    def compute_scores_from_cached_artifacts(self, dep_deviations=None):
        """
        Compute final anomaly scores from a cached per-dimension score matrix.

        This allows recomputing scores without re-running expensive TabPFN inference.

        Args:
            dep_deviations (ndarray, optional): Cached NLL matrix when DDM is enabled,
                otherwise a dependency-deviation matrix (n_test, latent_dim)

        Returns:
            ndarray: Final anomaly scores (n_test,)

        Raises:
            ValueError: If the cached matrix is missing
        """
        if dep_deviations is None:
            raise ValueError("A cached per-dimension score matrix is required")

        dep_deviations = np.asarray(dep_deviations)
        if dep_deviations.ndim != 2:
            raise ValueError(
                f"Cached score matrix must be two-dimensional, got shape {dep_deviations.shape}"
            )

        if CONFIG['use_ddm']:
            return self._aggregate_ddm_scores(
                dep_deviations,
                aggregation=CONFIG['ddm_aggregation'],
            )

        return self.compute_anomaly_scores(
            dev_abs=dep_deviations,
            dev_abs_context=self._context_deviation_cache
        )


# ============================================================================
# Main Block - Context Set Generation Demo
# ============================================================================

if __name__ == '__main__':
    pass
