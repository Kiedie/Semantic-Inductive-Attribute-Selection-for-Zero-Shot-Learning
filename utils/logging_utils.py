import csv
import datetime
import json
import pathlib
import uuid


# --- Logging Functions ---
def create_log_directory(log_path=None, continue_path=None):
    """Create a unique directory for logging this GA run"""
    if continue_path:
        # If continuing from previous run, use the same directory but append "_continued"
        log_dir = pathlib.Path(continue_path + "_continued")
    else:
        # Create a new directory for a fresh run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Use provided log path or default if none
        base_log_dir = pathlib.Path(log_path) if log_path else pathlib.Path("logs")
        log_dir = base_log_dir / f"ga_run_{timestamp}_{unique_id}"
    
    # Create main directory and subdirectories
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "generations").mkdir(exist_ok=True)
    
    return log_dir

def save_config(log_dir, args, dataset_info):
    """Save configuration parameters to a JSON file"""
    # Handle both argparse.Namespace and dict objects
    if not isinstance(args, dict):
        args = vars(args)
        
    config = {
        "dataset": args["dataset"],
        "seed": args["seed"],
        "preprocessing": args["preprocessing"],
        "orig_attribute": args["orig_attribute"],
        "scaler": args["scaler_str"],
        "ga_params": {
            "population_size": args["population_size"],
            "generations": args["generations"],
            "crossover_probability": args["crossover_probability"],
            "mutation_probability": args["mutation_probability"],
            "tournament_size": args["tournament_size"],
            "elitism": args["elitism"],
            "max_features": args["max_features"],
            "algorithm": args["algorithm"],
            "all_1": args["all_1"],
            "include_hais_solutions": args["include_hais_solutions"],
            "n_jobs": args["n_jobs"],
            "dynamic_train": args["dynamic_train"],
            "init-uniform": args["init_uniform"],
            "fold_change_frequency": args["fold_change_frequency"]
        },
        "dataset_info": dataset_info
    }
    
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

def log_population(log_dir, generation, population, stats):
    """Save population data for a given generation"""
    gen_dir = log_dir / "generations"
    
    # Extract population data
    pop_data = []
    for i, ind in enumerate(population):
        pop_data.append({
            "id": i,
            "mask": [int(bit) for bit in ind],
            "fitness": ind.fitness.values[0] if ind.fitness.valid else None,
            "num_features": sum(ind)
        })
    
    # Save population data
    with open(gen_dir / f"gen_{generation:03d}.json", "w") as f:
        json.dump(pop_data, f, indent=2)
    
    # Return stats for adding to statistics.csv
    return stats

def save_final_results(log_dir, best_individual, logbook, pop):
    """Save the final results and best individual"""
    # Save best individual
    best_data = {
        "mask": [int(bit) for bit in best_individual],
        "fitness": best_individual.fitness.values[0],
        "num_features": sum(best_individual),
        "feature_indices": [i for i, bit in enumerate(best_individual) if bit]
    }
    
    with open(log_dir / "best_individual.json", "w") as f:
        json.dump(best_data, f, indent=2)
    
    # Save logbook
    with open(log_dir / "logbook.json", "w") as f:
        # Convert logbook to serializable format
        logbook_data = [dict(chapter) for chapter in logbook]
        json.dump(logbook_data, f, indent=2)
    
    # Save final population
    final_pop = []
    for i, ind in enumerate(pop):
        final_pop.append({
            "id": i,
            "mask": [int(bit) for bit in ind],
            "fitness": ind.fitness.values[0] if ind.fitness.valid else None,
            "num_features": sum(ind)
        })
    
    with open(log_dir / "final_population.json", "w") as f:
        json.dump(final_pop, f, indent=2)

def write_statistics_csv(log_dir, stats_history):
    """Write all statistics to a CSV file"""
    with open(log_dir / "statistics.csv", "w", newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Generation"]
        if stats_history and len(stats_history) > 0:
            header.extend(stats_history[0].keys())
        writer.writerow(header)
        
        # Write data
        for gen, stats in enumerate(stats_history):
            row = [gen]
            row.extend(stats.values())
            writer.writerow(row)
