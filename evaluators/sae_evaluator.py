import numpy as np
import json
from functools import lru_cache
from models.sae import SAE
from metrics.sae_eval import evaluate

class SAEvaluator:
    """
    A wrapper class to compute fitness for a DEAP-based GA.
    The fitness is the ZSL accuracy of an SAE model trained on a subset of semantic features.
    Maintains separate SAE instances for each fold (or single fold in non-CV mode).
    """
    def __init__(self, folds, all_attributes, max_features=None, error_score=float('nan'), use_cache=True, awa=False, dynamic_fold=False):
        self.folds = folds
        self.all_attributes = all_attributes
        self.error_score = error_score
        self.use_cache = use_cache
        self.max_features = max_features
        self.awa = awa
        self.dynamic_fold = dynamic_fold # E
        
        # Contador de evaluaciones
        self.evaluation_count = 0
        self.log_dir = None  # Se establecerá desde el exterior
        
        # For dynamic fold mode, use only one fold at a time
        if dynamic_fold:
            self.current_fold_idx = 0
            self.sae_models = [SAE()]  # Only one SAE model needed
        else:
            # Create one SAE model per fold
            self.sae_models = [SAE() for _ in range(len(folds))]
        
        # Simple cache for evaluations
        if use_cache:
            self._evaluate_cached = lru_cache(maxsize=10000)(self._evaluate_individual)
    
    def set_log_dir(self, log_dir):
        """Set the log directory to save evaluation count"""
        self.log_dir = log_dir
    
    def save_evaluation_count(self):
        """Save the evaluation count to a JSON file"""
        if self.log_dir:
            eval_data = {
                "total_evaluations": self.evaluation_count,
                "cache_enabled": self.use_cache,
                "cache_info": self._evaluate_cached.cache_info()._asdict() if self.use_cache else None
            }
            
            eval_file = self.log_dir / "num_evals.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
    
    def set_current_fold(self, fold_idx):
        """Set the current fold for dynamic training"""
        if self.dynamic_fold:
            self.current_fold_idx = fold_idx % len(self.folds)
            # Clear cache when changing fold
            if self.use_cache:
                self._evaluate_cached.cache_clear()
    
    def evaluate_all_folds(self, individual):
        """Evaluate individual on all folds and return mean accuracy"""
        try:
            mask = np.array(individual, dtype=bool)
            num_selected = np.sum(mask)
            if num_selected == 0 or (self.max_features is not None and num_selected > self.max_features):
                return 0.0

            fold_accuracies = []
            for i, (train_dataset, val_dataset) in enumerate(self.folds):
                # Select attributes based on the mask
                selected_train_attributes = train_dataset.att[:, mask]
                selected_all_attributes = self.all_attributes[:, mask]

                # Train SAE on this fold
                sae_model = SAE()
                sae_model.fit(train_dataset.features, selected_train_attributes)

                # Evaluate on validation fold
                acc, _ = evaluate(
                    model=sae_model,
                    attributes=selected_all_attributes,
                    eval_features=val_dataset.features,
                    eval_classes=val_dataset.classes,
                    eval_labels=val_dataset.labels,
                    awa=self.awa
                )
                fold_accuracies.append(acc)
            
            return np.mean(fold_accuracies)
            
        except Exception as e:
            print(f"Error during final evaluation: {e}")
            return self.error_score
        
    def _evaluate_individual(self, individual_tuple):
        """
        Actual evaluation function for a given individual (feature mask).
        Individual is converted to tuple for hashability when using cache.
        For cross-validation, it averages results across all folds.
        """
        try:
            # Convert tuple back to numpy array
            mask = np.array(individual_tuple, dtype=bool)
            
            # If no features are selected or exceeds max_features, return 0 fitness
            num_selected = np.sum(mask)
            if num_selected == 0 or (self.max_features is not None and num_selected > self.max_features):
                return 0.0

            if self.dynamic_fold:
                # Use only the current fold
                train_dataset, val_dataset = self.folds[self.current_fold_idx]
                selected_train_attributes = train_dataset.att[:, mask]
                selected_all_attributes = self.all_attributes[:, mask]
                
                # Train SAE on current fold
                self.sae_models[0].fit(train_dataset.features, selected_train_attributes)
                
                # Evaluate on validation fold
                acc, _ = evaluate(
                    model=self.sae_models[0],
                    attributes=selected_all_attributes,
                    eval_features=val_dataset.features,
                    eval_classes=val_dataset.classes,
                    eval_labels=val_dataset.labels,
                    awa=self.awa
                )
                return acc
            else:
                # Evaluate on all folds (standard CV or single fold)
                fold_accuracies = []
                for i, (train_dataset, val_dataset) in enumerate(self.folds):
                    # Select attributes based on the mask
                    selected_train_attributes = train_dataset.att[:, mask]
                    selected_all_attributes = self.all_attributes[:, mask]

                    # Train SAE on this fold
                    self.sae_models[i].fit(train_dataset.features, selected_train_attributes)

                    # Evaluate on validation fold
                    acc, _ = evaluate(
                        model=self.sae_models[i],
                        attributes=selected_all_attributes,
                        eval_features=val_dataset.features,
                        eval_classes=val_dataset.classes,
                        eval_labels=val_dataset.labels,
                        awa=self.awa
                    )
                    fold_accuracies.append(acc)
                
                # Return average accuracy across all folds
                return np.mean(fold_accuracies)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return self.error_score

    def evaluate_fitness(self, individual):
        """
        Calculates the fitness score for a given individual (feature mask).
        Handles caching if enabled.
        """
        # Incrementar contador antes de la evaluación
        self.evaluation_count += 1
        
        # Convert individual to tuple for hashability
        individual_tuple = tuple(individual)
        
        if self.use_cache:
            result = self._evaluate_cached(individual_tuple)
        else:
            result = self._evaluate_individual(individual_tuple)
            
        return (result,)
