import argparse
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from data.datareader import get_data
from models.sae import SAE
from metrics.sae_eval import evaluate

def load_config(log_path: pathlib.Path) -> Dict[str, Any]:
    """Load configuration from log path"""
    config_path = log_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, "r") as f:
        return json.load(f)

def load_best_individual(log_path: pathlib.Path) -> np.ndarray:
    """Load best individual from log path"""
    best_path = log_path / "best_individual.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Best individual file {best_path} not found")
    
    with open(best_path, "r") as f:
        best_data = json.load(f)
    
    return np.array(best_data["mask"], dtype=bool)

def get_generation_files(log_path: pathlib.Path, periodicity: int) -> List[Tuple[int, pathlib.Path]]:
    """Get generation files at specified periodicity"""
    gen_dir = log_path / "generations"
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generations directory not found at {gen_dir}")
    
    # Get all generation files
    gen_files = list(gen_dir.glob("gen_*.json"))
    gen_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    
    # Get files at specified periodicity
    selected_files = []
    for file_path in gen_files:
        gen_number = int(file_path.stem.split("_")[1])
        if gen_number % periodicity == 0 or gen_number == int(gen_files[-1].stem.split("_")[1]):
            selected_files.append((gen_number, file_path))
    
    return selected_files

def get_best_individual_from_generation(gen_file: pathlib.Path) -> np.ndarray:
    """Extract the best individual from a generation file"""
    with open(gen_file, "r") as f:
        population = json.load(f)
    
    # Find the individual with the highest fitness
    best_individual = max(population, key=lambda x: x["fitness"] if x["fitness"] is not None else -float("inf"))
    
    return np.array(best_individual["mask"], dtype=bool)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SAE model with selected features')
    parser.add_argument('--log_path', type=str, help='Path to the log directory from ga_deap.py')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print verbose output')
    parser.add_argument('--boundary', action='store_true', help='Use boundary-based evaluation')
    parser.add_argument('--epsilon', type=float, default=3.0, help='Epsilon for boundary calculation')
    parser.add_argument('--hitk', type=int, default=1, help='Top-k accuracy to compute')
    parser.add_argument('--awa', action='store_true', default=False, help='Use AWA dataset normalization')
    parser.add_argument('--progress', action='store_true', help='Track progress across generations to detect overfitting')
    parser.add_argument('--periodicity', type=int, default=10, help='Evaluate every N generations when using --progress')
    return parser.parse_args()

def evaluate_model(model, selected_all_attributes, test_seen_dataset, test_unseen_dataset, args):
    """Evaluate the model on test seen and unseen datasets"""
    # Evaluate on test_seen
    seen_acc, seen_preds = evaluate(
        model=model,
        attributes=selected_all_attributes,
        eval_features=test_seen_dataset.features,
        eval_classes=test_seen_dataset.classes,
        eval_labels=test_seen_dataset.labels,
        boundary=args.boundary,
        epsilon=args.epsilon,
        hitk=args.hitk,
        awa=args.awa
    )
    
    # Evaluate on test_unseen
    unseen_acc, unseen_preds = evaluate(
        model=model,
        attributes=selected_all_attributes,
        eval_features=test_unseen_dataset.features,
        eval_classes=test_unseen_dataset.classes,
        eval_labels=test_unseen_dataset.labels,
        boundary=args.boundary,
        epsilon=args.epsilon,
        hitk=args.hitk,
        awa=args.awa
    )
    
    # Calculate harmonic mean
    h_mean = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) > 0 else 0
    
    return seen_acc, unseen_acc, h_mean

def plot_progress(generations, seen_accs, unseen_accs, h_means, log_path):
    """Plot performance metrics across generations"""
    plt.figure(figsize=(12, 7))
    
    plt.plot(generations, seen_accs, 'b-', label='Test Seen Accuracy')
    plt.plot(generations, unseen_accs, 'r-', label='Test Unseen Accuracy')
    plt.plot(generations, h_means, 'g-', label='Harmonic Mean')
    
    plt.xlabel('Generation')
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Across Generations')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = log_path / "progress_plot.png"
    plt.savefig(plot_path)
    print(f"Progress plot saved to {plot_path}")
    
    # Display the plot if running in an environment with GUI
    try:
        plt.show()
    except:
        pass

def main():
    # Parse arguments
    args = parse_args()
    log_path = pathlib.Path(args.log_path)
    verbose = args.verbose
    
    # Load configuration and best individual
    if verbose:
        print(f"Loading configuration from {log_path}")
    config = load_config(log_path)
    
    # Create args namespace for get_data
    class Args:
        def __init__(self, config):
            self.dataset = config["dataset"]
            self.preprocessing = config["preprocessing"]
            self.orig_attribute = config["orig_attribute"]
            self.scaler_str = config["scaler"]
            self.seed = config["seed"]
    
    data_args = Args(config)
    
    # Auto-detect AWA2 dataset and set awa parameter accordingly
    args.awa = True if data_args.dataset == "AWA2" else False
    
    # Load dataset
    if verbose:
        print(f"Loading dataset {config['dataset']}")
    
    path_data = pathlib.Path(f"../ZSL-preprocessing/data/{config['dataset']}")
    train_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(
        path_data, data_args, trainval=True, preprocessing=config["preprocessing"]
    )
    
    if args.awa:
        from models.sae import normalize_feature
        train_dataset.features = normalize_feature(train_dataset.features)
        if verbose:
            print("AWA2 dataset: Applied normalize_feature to train_dataset.features")
    
    
    if args.progress:
        # Track progress across generations
        if verbose:
            print(f"Tracking progress across generations with periodicity {args.periodicity}")
        
        generation_files = get_generation_files(log_path, args.periodicity)
        
        if verbose:
            print(f"Found {len(generation_files)} generations to evaluate")
        
        generations = []
        seen_accs = []
        unseen_accs = []
        h_means = []
        
        # Load SAE outside the loop to take advantage of model reusability
        model = SAE()
        
        for gen_number, gen_file in generation_files:
            if verbose:
                print(f"\nEvaluating generation {gen_number}...")
            
            # Get best individual from this generation
            best_mask = get_best_individual_from_generation(gen_file)
            
            # Extract selected attributes
            selected_train_attributes = train_dataset.att[:, best_mask]
            selected_all_attributes = attribute[:, best_mask]
            
            # Train the model
            if verbose:
                print(f"Training model with {sum(best_mask)} features...")
            
            model.fit(train_dataset.features, selected_train_attributes)
            
            # Evaluate model
            seen_acc, unseen_acc, h_mean = evaluate_model(
                model, selected_all_attributes, test_seen_dataset, test_unseen_dataset, args
            )
            
            generations.append(gen_number)
            seen_accs.append(seen_acc)
            unseen_accs.append(unseen_acc)
            h_means.append(h_mean)
            
            if verbose:
                print(f"Generation {gen_number}: Seen: {seen_acc:.2f}%, Unseen: {unseen_acc:.2f}%, H-mean: {h_mean:.2f}%")
        
        # Save progress results
        progress_results = {
            "generations": generations,
            "test_seen_accuracy": seen_accs,
            "test_unseen_accuracy": unseen_accs,
            "harmonic_mean": h_means
        }
        
        with open(log_path / "progress_results.json", "w") as f:
            json.dump(progress_results, f, indent=2)
        
        if verbose:
            print(f"\nProgress results saved to {log_path / 'progress_results.json'}")
            
        # Plot progress
        try:
            plot_progress(generations, seen_accs, unseen_accs, h_means, log_path)
        except Exception as e:
            print(f"Warning: Could not create progress plot: {e}")
            
    else:
        # Regular evaluation of final best individual
        best_mask = load_best_individual(log_path)
        
        if verbose:
            print(f"Best individual loaded: {sum(best_mask)} features selected")
            print(f"Selected feature indices: {np.where(best_mask)[0]}")
        
        # Apply feature mask
        selected_train_attributes = train_dataset.att[:, best_mask]
        selected_all_attributes = attribute[:, best_mask]
        
        # Train the model
        if verbose:
            print("Training SAE model...")
        
        model = SAE()
        model.fit(train_dataset.features, selected_train_attributes)
        
        # Evaluate model
        seen_acc, unseen_acc, h_mean = evaluate_model(
            model, selected_all_attributes, test_seen_dataset, test_unseen_dataset, args
        )
        
        # Print results
        print("\n----- Results -----")
        print(f"Test Seen Accuracy: {seen_acc:.2f}%")
        print(f"Test Unseen Accuracy: {unseen_acc:.2f}%")
        print(f"Harmonic Mean: {h_mean:.2f}%")
        
        # Save results
        results = {
            "test_seen_accuracy": float(seen_acc),
            "test_unseen_accuracy": float(unseen_acc),
            "harmonic_mean": float(h_mean),
            "selected_features_count": int(sum(best_mask)),
            "selected_feature_indices": [int(i) for i in np.where(best_mask)[0]],
            "evaluation_settings": {
                "boundary": args.boundary,
                "epsilon": args.epsilon,
                "hitk": args.hitk
            }
        }
        
        with open(log_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"Results saved to {log_path / 'evaluation_results.json'}")

if __name__ == "__main__":
    main()
