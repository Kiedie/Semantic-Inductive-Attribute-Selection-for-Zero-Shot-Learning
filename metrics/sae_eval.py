"""
Evaluation metrics for Semantic Autoencoder (SAE) for Zero-Shot Learning
"""

import numpy as np
from typing import Tuple, List, Optional, Union


# def normalize_feature(x: np.ndarray) -> np.ndarray:
#     """Normalize features to unit norm."""
#     x = x + 1e-10  # avoid division by zero
#     if x.ndim == 1:
#         return x / np.sqrt(np.sum(x**2))
#     else:
#         feature_norm = np.sqrt(np.sum(x**2, axis=1))
#         return x / feature_norm[:, np.newaxis]
from models.sae import normalize_feature

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute cosine distance between x and y."""
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Handle 1D arrays
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
        
    # Normalize
    x_norm = normalize_feature(x)
    y_norm = normalize_feature(y)
    
    # Compute distance
    similarity = np.dot(x_norm, y_norm.T)
    distance = 1 - similarity
    
    return distance


def compute_boundary(eval_classes: np.ndarray, attributes: np.ndarray, epsilon: float = 3) -> np.ndarray:
    """
    Compute the radius of semantic neighborhoods.
    
    Parameters:
    -----------
    eval_classes : np.ndarray
        Classes to evaluate
    attributes : np.ndarray
        Semantic attributes for all classes
    epsilon : float, default=3
        Scaling factor for boundary computation
        
    Returns:
    --------
    radii : np.ndarray
        Radius for each class in eval_classes
    """
    dist_matrix = cosine_distance(attributes[eval_classes], attributes[eval_classes])
    sorted_indices = np.argsort(dist_matrix, axis=1)
    closest_distances = np.array([dist_matrix[i, sorted_indices[i, 1]] for i in range(dist_matrix.shape[0])])
    radii = closest_distances / epsilon
    return radii


def generate_point_in_neighbourhood(x: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Generate a random point within epsilon radius of x.
    
    Parameters:
    -----------
    x : np.ndarray
        Center point
    epsilon : float
        Radius of the neighborhood
        
    Returns:
    --------
    point : np.ndarray
        A random point within the epsilon-ball around x
    """
    dim = len(x)
    random_vector = np.random.uniform(-1, 1, dim)
    unit_vector = random_vector / np.linalg.norm(random_vector)
    scaled_radius = np.random.uniform(0, epsilon)
    return x + unit_vector * scaled_radius


def aug_semantic_space(eval_classes: np.ndarray, attributes: np.ndarray, 
                      epsilons: np.ndarray, n_points: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points around each semantic attribute.
    
    Parameters:
    -----------
    eval_classes : np.ndarray
        Classes to augment
    attributes : np.ndarray
        Semantic attributes for all classes
    epsilons : np.ndarray
        Radius for each class in eval_classes
    n_points : int, default=10
        Number of points to generate around each attribute
        
    Returns:
    --------
    X : np.ndarray
        Augmented semantic space
    labels : np.ndarray
        Class labels for each point in X
    """
    attributes_to_aug = attributes[eval_classes]
    
    assert len(attributes_to_aug) == len(epsilons), "Attributes and epsilons must have the same length"
    
    X, labels = [], []
    for i, (att, eps) in enumerate(zip(attributes_to_aug, epsilons)):
        X.append(att)
        labels.append(eval_classes[i])
        for _ in range(n_points):
            X.append(generate_point_in_neighbourhood(att, eps))
            labels.append(eval_classes[i])
            
    return np.asarray(X), np.asarray(labels)


def evaluate(model, attributes: np.ndarray, eval_features: np.ndarray,
            eval_classes: np.ndarray, eval_labels: np.ndarray,
            boundary: bool = False, epsilon: float = 3, hitk: int = 1, awa: bool = False) -> Tuple[float, np.ndarray]:
    """
    Evaluate a SAE model on test data.
    
    Parameters:
    -----------
    model : object
        The SAE model with transform method
    attributes : np.ndarray
        Semantic attributes for all classes
    eval_features : np.ndarray
        Test features
    eval_classes : np.ndarray
        Class indices to evaluate
    eval_labels : np.ndarray
        True labels for test features
    boundary : bool, default=False
        Whether to use boundary-based evaluation
    epsilon : float, default=3
        Scaling factor for boundary computation
    hitk : int, default=1
        Number of top predictions to consider
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    predictions : np.ndarray
        Top-k predictions for each test sample
    """
    # Get prototype attributes
    gt_ss = attributes[eval_classes]
    
    # Project test features
    semantic_predicted = model.transform(eval_features)
    
    # Compute distances and predictions
    if boundary:
        return evaluate_with_boundary(
            semantic_predicted, gt_ss, eval_classes, eval_labels, 
            attributes, epsilon, hitk
        )
    else:
        return evaluate_standard(
            semantic_predicted, gt_ss, eval_classes, eval_labels, hitk, awa
        )


def evaluate_standard(semantic_predicted: np.ndarray, semantic_gt: np.ndarray,
                     eval_classes: np.ndarray, eval_labels: np.ndarray,
                     hitk: int = 1, awa: bool = False) -> Tuple[float, np.ndarray]:
    """
    Standard evaluation using cosine distance.
    
    Parameters:
    -----------
    semantic_predicted : np.ndarray
        Predicted semantic vectors
    semantic_gt : np.ndarray
        Ground truth semantic vectors
    eval_classes : np.ndarray
        Class indices to evaluate
    eval_labels : np.ndarray
        True labels for test features
    hitk : int, default=1
        Number of top predictions to consider
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    predictions : np.ndarray
        Top-k predictions for each test sample
    """
    if awa:
        distances = cosine_distance(semantic_predicted, normalize_feature(semantic_gt.transpose()).transpose())
    else:
        distances = cosine_distance(semantic_predicted, semantic_gt)
    
    # Get top-k predictions
    y_hit_k = np.zeros((distances.shape[0], hitk))
    for idx in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[idx, :])[:hitk]
        y_hit_k[idx, :] = eval_classes[sorted_indices]
    
    # Compute accuracy
    correct = sum(1 for idx in range(distances.shape[0]) if eval_labels[idx] in y_hit_k[idx, :])
    accuracy = float(correct) / distances.shape[0] * 100
    
    return accuracy, y_hit_k


def evaluate_with_boundary(semantic_predicted: np.ndarray, semantic_gt: np.ndarray,
                          eval_classes: np.ndarray, eval_labels: np.ndarray,
                          attributes: np.ndarray, epsilon: float = 3, hitk: int = 1) -> Tuple[float, np.ndarray]:
    """
    Evaluation using boundary-based approach.
    
    Parameters:
    -----------
    semantic_predicted : np.ndarray
        Predicted semantic vectors
    semantic_gt : np.ndarray
        Ground truth semantic vectors
    eval_classes : np.ndarray
        Class indices to evaluate
    eval_labels : np.ndarray
        True labels for test features
    attributes : np.ndarray
        Semantic attributes for all classes
    epsilon : float, default=3
        Scaling factor for boundary computation
    hitk : int, default=1
        Number of top predictions to consider
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    predictions : np.ndarray
        Top-k predictions for each test sample
    """
    radii = compute_boundary(eval_classes, attributes, epsilon)
    distances = cosine_distance(semantic_predicted, semantic_gt) - radii
    
    # Get top-k predictions
    y_hit_k = np.zeros((distances.shape[0], hitk))
    for idx in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[idx, :])[:hitk]
        y_hit_k[idx, :] = eval_classes[sorted_indices]
    
    # Compute accuracy
    correct = sum(1 for idx in range(distances.shape[0]) if eval_labels[idx] in y_hit_k[idx, :])
    accuracy = float(correct) / distances.shape[0] * 100
    
    return accuracy, y_hit_k


def evaluate_with_augmentation(model, attributes: np.ndarray, eval_features: np.ndarray,
                              eval_classes: np.ndarray, eval_labels: np.ndarray,
                              epsilon: float = 3, n_points: int = 100, hitk: int = 1) -> Tuple[float, np.ndarray]:
    """
    Evaluation using augmentation-based approach.
    
    Parameters:
    -----------
    model : object
        The SAE model with transform method
    attributes : np.ndarray
        Semantic attributes for all classes
    eval_features : np.ndarray
        Test features
    eval_classes : np.ndarray
        Class indices to evaluate
    eval_labels : np.ndarray
        True labels for test features
    epsilon : float, default=3
        Scaling factor for boundary computation
    n_points : int, default=100
        Number of points to generate around each attribute
    hitk : int, default=1
        Number of top predictions to consider
        
    Returns:
    --------
    accuracy : float
        Classification accuracy
    predictions : np.ndarray
        Top-k predictions for each test sample
    """
    # Compute boundaries and augment semantic space
    radii = compute_boundary(eval_classes, attributes, epsilon)
    gt_ss, semantic_labels = aug_semantic_space(eval_classes, attributes, radii, n_points)
    
    # Project test features
    semantic_predicted = model.transform(eval_features)
    
    # Compute distances
    distances = cosine_distance(semantic_predicted, gt_ss)
    
    # Get top predictions
    y_hit_k = np.zeros((distances.shape[0], hitk))
    for idx in range(distances.shape[0]):
        sorted_indices = np.argsort(distances[idx, :])[:hitk]
        y_hit_k[idx, :] = semantic_labels[sorted_indices]
    
    # Compute accuracy
    correct = sum(1 for idx in range(distances.shape[0]) if eval_labels[idx] in y_hit_k[idx, :])
    accuracy = float(correct) / distances.shape[0] * 100
    
    return accuracy, y_hit_k
