"""
Mahalanobis Distance Model —
Fits a multivariate Gaussian on normal CLS embeddings and computes
the Mahalanobis distance as a distributional anomaly signal.
"""
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class MahalanobisModel:
    """
    Fits a multivariate Gaussian N(μ, Σ) on normal embeddings.
    At inference, returns the Mahalanobis distance from the fitted distribution.

    Because 768D covariance inversion is numerically fragile, we use
    PCA whitening (via SVD) to project to a lower-dimensional space first.
    """

    def __init__(self, n_components: int = 128):
        self.n_components  = n_components
        self.mean_: Optional[np.ndarray]     = None      # [n_components]
        self.inv_cov_: Optional[np.ndarray]  = None      # [n_components, n_components]
        self.components_: Optional[np.ndarray] = None    # [n_components, 768]  PCA basis

    def fit(self, embeddings: np.ndarray):
        """
        Fit on normal embeddings [N, D].
        Projects to n_components dimensions via PCA, then fits Gaussian.
        """
        N, D = embeddings.shape
        n_comp = min(self.n_components, N - 1, D)

        # PCA via SVD (zero-mean)
        mu_orig = embeddings.mean(axis=0)
        centered = embeddings - mu_orig
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = Vt[:n_comp]                   # [n_comp, D]

        projected = centered @ self.components_.T         # [N, n_comp]
        self.mean_ = projected.mean(axis=0)
        cov = np.cov(projected, rowvar=False) + 1e-6 * np.eye(n_comp)
        self.inv_cov_ = np.linalg.inv(cov)
        self.mu_orig_ = mu_orig

    def distance(self, embedding: np.ndarray) -> float:
        """
        Mahalanobis distance for a single embedding [D].
        Returns a non-negative float.
        """
        if self.mean_ is None:
            return 0.0
        centered  = embedding - self.mu_orig_
        projected = centered @ self.components_.T         # [n_comp]
        diff      = projected - self.mean_                # [n_comp]
        d2 = diff @ self.inv_cov_ @ diff
        return float(np.sqrt(max(d2, 0.0)))

    def batch_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute distances for [N, D] embeddings. Returns [N]."""
        return np.array([self.distance(e) for e in embeddings])

    def save(self, path: Path = cfg.MAHAL_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            mean=self.mean_,
            inv_cov=self.inv_cov_,
            components=self.components_,
            mu_orig=self.mu_orig_,
        )

    def load(self, path: Path = cfg.MAHAL_FILE):
        data = np.load(str(path))
        self.mean_       = data["mean"]
        self.inv_cov_    = data["inv_cov"]
        self.components_ = data["components"]
        self.mu_orig_    = data["mu_orig"]
        return self
