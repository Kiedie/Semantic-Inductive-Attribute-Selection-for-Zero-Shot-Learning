"""
Semantic Autoencoder (SAE) for Zero-Shot Learning
Implementation compatible with scikit-learn
"""

import numpy as np
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import List, Optional, Union, Tuple
import warnings

from solver.sylvester import SylvesterSolver


def normalize_feature(x: np.ndarray) -> np.ndarray:
    """Normalize features to unit norm."""
    x = x + 1e-10  # avoid division by zero
    if x.ndim == 1:
        return x / np.sqrt(np.sum(x**2))
    else:
        feature_norm = np.sqrt(np.sum(x**2, axis=1))
        return x / feature_norm[:, np.newaxis]


class SAE(BaseEstimator, TransformerMixin):
    """
    Semantic Autoencoder (SAE) for zero-shot learning.
    
    Parameters:
    -----------
    lambda_reg : float, default=500000
        Regularization parameter.
    """
    
    def __init__(self, lambda_reg: float = 500000):
        self.lambda_reg = lambda_reg
        self.X = np.zeros((0, 0))
        self.solver : SylvesterSolver = None

    def _compute_w(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Compute projection matrix using Sylvester equation."""
        if not np.array_equal(X, self.X):
            # Note: Input features changed, recomputing projection matrix
            # Suppressed warning for cleaner output during GA evolution
            self.X = X.copy()
            self.X_T = self.X.T
            B = self.lambda_reg * np.dot(self.X_T, self.X)
            self.solver = SylvesterSolver(B)
        
        S_T = S.T  # k x N
        
        A = np.dot(S_T, S)
        C = (1 + self.lambda_reg) * np.dot(S_T, self.X)

        # The solver is expected to be initialized in fit()
        W = self.solver.solve_fast(A, C)
        return W
    
    def fit(self, X: np.ndarray, S: np.ndarray) -> 'SAE':
        """
        Fit the SAE model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Visual features
        S : array-like of shape (n_samples, n_attributes)
            Semantic attributes. If feature selection is needed, 
            it should be applied to S before passing it to this method.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        X = check_array(X)
        S = check_array(S)

        self.W_ = self._compute_w(X, S)
        self.n_features_in_ = X.shape[1]
        self.n_attributes_out_ = S.shape[1]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project visual features to semantic space."""
        check_is_fitted(self, ['W_'])
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but SAE was trained with {self.n_features_in_} features")
        
        return np.dot(X, normalize_feature(self.W_).T)
    
    def predict(self, X: np.ndarray, prototype_attributes: np.ndarray, prototype_classes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class labels for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test data
        prototype_attributes : array-like of shape (n_classes, n_attributes)
            Semantic attributes for all classes. If feature selection was applied during fit,
            the same selection must be applied to prototype_attributes.
        prototype_classes : array-like, optional
            Class indices corresponding to prototype_attributes
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, ['W_'])
        X = check_array(X)
        
        if prototype_classes is None:
            prototype_classes = np.arange(prototype_attributes.shape[0])
        
        semantic_predicted = self.transform(X)
        
        # Import here to avoid circular import
        from ..metrics.sae_eval import cosine_distance
        
        distances = cosine_distance(semantic_predicted, prototype_attributes)
        closest_indices = np.argmin(distances, axis=1)
        
        return prototype_classes[closest_indices]
    
    def get_W(self):
        """Get the learned projection matrix."""
        check_is_fitted(self, ['W_'])
        return self.W_
