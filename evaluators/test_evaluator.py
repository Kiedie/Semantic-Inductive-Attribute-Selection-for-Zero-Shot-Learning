import numpy as np
import json
from functools import lru_cache
from models.sae import SAE
from metrics.sae_eval import evaluate

class TestEvaluator:
    """
    Evaluates models on test data periodically during evolution.
    Uses an independent SAE instance to avoid contamination.
    """
    def __init__(self, train_dataset, test_seen_dataset, test_unseen_dataset, all_attributes, 
                 eval_frequency=10, boundary=False, epsilon=3.0, hitk=1, use_cache=True, awa=False):
        self.train_dataset = train_dataset
        self.test_seen_dataset = test_seen_dataset
        self.test_unseen_dataset = test_unseen_dataset
        self.all_attributes = all_attributes
        self.eval_frequency = eval_frequency
        self.boundary = boundary
        self.epsilon = epsilon
        self.hitk = hitk
        self.use_cache = use_cache
        self.awa = awa
        
        # Create an independent SAE model for test evaluation
        self.model = SAE()
        
        # Store results
        self.test_results = {
            'generations': [],
            'seen_acc': [],
            'unseen_acc': [],
            'h_mean': []
        }
        
        # Simple cache for evaluations
        if use_cache:
            self._evaluate_cached = lru_cache(maxsize=1000)(self._evaluate_individual)
    
    def should_evaluate(self, generation):
        """Determine if evaluation should be performed at this generation"""
        return generation % self.eval_frequency == 0
    
    def _evaluate_individual(self, individual_tuple):
        """
        Core evaluation logic for a given individual.
        Individual is converted to tuple for hashability.
        
        Returns:
        --------
        dict
            Dictionary containing test metrics
        """
        # Convert tuple back to numpy array
        mask = np.array(individual_tuple, dtype=bool)
        
        # Extract selected attributes
        selected_train_attributes = self.train_dataset.att[:, mask]
        selected_all_attributes = self.all_attributes[:, mask]
        
        # Train the model
        self.model.fit(self.train_dataset.features, selected_train_attributes)
        
        # Evaluate on test_seen
        seen_acc, _ = evaluate(
            model=self.model,
            attributes=selected_all_attributes,
            eval_features=self.test_seen_dataset.features,
            eval_classes=self.test_seen_dataset.classes,
            eval_labels=self.test_seen_dataset.labels,
            boundary=self.boundary,
            epsilon=self.epsilon,
            hitk=self.hitk,
            awa=self.awa
        )
        
        # Evaluate on test_unseen
        unseen_acc, _ = evaluate(
            model=self.model,
            attributes=selected_all_attributes,
            eval_features=self.test_unseen_dataset.features,
            eval_classes=self.test_unseen_dataset.classes,
            eval_labels=self.test_unseen_dataset.labels,
            boundary=self.boundary,
            epsilon=self.epsilon,
            hitk=self.hitk,
            awa=self.awa
        )
        
        # Calculate harmonic mean
        h_mean = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) > 0 else 0
        
        return {
            'seen_acc': seen_acc,
            'unseen_acc': unseen_acc,
            'h_mean': h_mean
        }
    
    def evaluate(self, best_individual, generation):
        """
        Evaluate the best individual on test data.
        
        Parameters:
        -----------
        best_individual : list or numpy.ndarray
            Binary mask of selected features
        generation : int
            Current generation number
            
        Returns:
        --------
        dict
            Dictionary containing test metrics
        """
        # Convert to tuple for hashability
        if not isinstance(best_individual, tuple):
            if isinstance(best_individual, np.ndarray):
                individual_tuple = tuple(best_individual.tolist())
            else:
                individual_tuple = tuple(best_individual)
        else:
            individual_tuple = best_individual
        
        # Use cached version if enabled, otherwise direct evaluation
        if self.use_cache:
            result = self._evaluate_cached(individual_tuple)
        else:
            result = self._evaluate_individual(individual_tuple)
        
        # Store results with generation information
        self.test_results['generations'].append(generation)
        self.test_results['seen_acc'].append(float(result['seen_acc']))
        self.test_results['unseen_acc'].append(float(result['unseen_acc']))
        self.test_results['h_mean'].append(float(result['h_mean']))
        
        return result
    
    def save_results(self, log_dir):
        """Save test evaluation results to file"""
        with open(log_dir / "test_evaluation.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        # Also generate a plot if matplotlib is available
        try:
            self._generate_plot(log_dir)
        except Exception as e:
            print(f"Warning: Could not create test progress plot: {e}")
    
    def _generate_plot(self, log_dir):
        """Generate a plot of test metrics over generations"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        plt.plot(self.test_results['generations'], self.test_results['seen_acc'], 
                 'b-', label='Test Seen Accuracy')
        plt.plot(self.test_results['generations'], self.test_results['unseen_acc'], 
                 'r-', label='Test Unseen Accuracy')
        plt.plot(self.test_results['generations'], self.test_results['h_mean'], 
                 'g-', label='Harmonic Mean')
        
        plt.xlabel('Generation')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Performance Across Generations')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(log_dir / "test_progress_plot.png", dpi=300, bbox_inches='tight')
