import random
import pathlib
import numpy as np
import multiprocessing
import os
import json
import datetime
import uuid
import csv
from copy import deepcopy
import argparse
import glob
import time

from deap import base, creator, tools, algorithms
from deap.algorithms import varOr


from data.datareader import get_data
from data.split import SplitIntoFolds
from models.sae import SAE
from metrics.sae_eval import evaluate

# Import evaluators from their new location
from evaluators.sae_evaluator import SAEvaluator
from evaluators.test_evaluator import TestEvaluator

# Import logging utilities
from utils.logging_utils import (
    create_log_directory, 
    save_config, 
    log_population,
    save_final_results,
    write_statistics_csv,
)

# Import population utilities
from utils.population_utils import (
    load_previous_run,
    create_population_from_data
)


# Import the modified eaMuPlusLambdaWithLog from eamllog
from algorithms.eamllog import eaMuPlusLambdaWithLog

from viz.plots import visualize_ga_run

# --- Main GA execution ---
if __name__ == '__main__':
    # --- DEAP Setup ---
    # Create a fitness class that maximizes a single objective
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Create an individual class that is a list with the FitnessMax fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(description='Genetic Algorithm for SAE Feature Selection')
    
    # Dataset parameters
    data_group = parser.add_argument_group('Dataset Parameters')
    data_group.add_argument('--dataset', type=str, default='CUB', help='Dataset name (default: CUB)')
    data_group.add_argument('--awa', action='store_true', default=False,help='Use AWA dataset normalization')
    data_group.add_argument('--preprocessing', action='store_true', help='Apply preprocessing to features')
    data_group.add_argument('--orig-attribute', action='store_true', default=False, help='Use original attributes')
    data_group.add_argument('--scaler-str', type=str, default='StandardScaler', help='Scaler to use (default: StandardScaler)')
    
    # GA parameters
    ga_group = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_group.add_argument('--population-size', type=int, default=50, help='Population size (default: 50)')
    ga_group.add_argument('--all-1', action='store_true', default=False, help='Use all features in initial population')
    ga_group.add_argument('--init-uniform', action='store_true', default=False, help='Initialize population uniformly from 35% to 100% active features')
    ga_group.add_argument('--generations', type=int, default=100, help='Number of generations (default: 100)')
    ga_group.add_argument('--crossover-probability', type=float, default=0.2, help='Crossover probability (default: 0.2)')
    ga_group.add_argument('--mutation-probability', type=float, default=0.8, help='Mutation probability (default: 0.8)')
    ga_group.add_argument('--tournament-size', type=int, default=3, help='Tournament selection size (default: 3)')
    ga_group.add_argument('--elitism', action='store_true', default=True, help='Use elitism')
    ga_group.add_argument('--max-features', type=int, default=None, help='Maximum number of features to select (default: None)')
    ga_group.add_argument('--keep-top-k', type=int, default=1, help='Number of top individuals to keep (default: 1)')
    ga_group.add_argument('--algorithm', type=str, default='eaMuPlusLambda', choices=['eaMuPlusLambda', 'eaSimple'], 
                           help='GA algorithm to use (default: eaMuPlusLambda)')
    ga_group.add_argument('--refit', action='store_true', help='Refit model with best features')
    ga_group.add_argument('--include-hais-solutions', action='store_true', default=False, help='Include HAIS solutions in the population')

    # Cross-validation parameters
    cv_group = parser.add_argument_group('Cross-validation Parameters')
    cv_group.add_argument('--cv', action='store_true', help='Use cross-validation')
    cv_group.add_argument('--n-splits', type=int, default=5, help='Number of cross-validation folds (default: 5)')
    cv_group.add_argument('--dynamic-train', action='store_true', default=False, help='Use dynamic training with changing folds')
    cv_group.add_argument('--fold-change-frequency', type=int, default=10, help='Change fold every N generations (default: 10)')
    
    # Execution parameters
    exec_group = parser.add_argument_group('Execution Parameters')
    exec_group.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs, -1 for all cores (default: 1)')
    exec_group.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    exec_group.add_argument('--error-score', type=float, default=float('nan'), help='Score to return if an error occurs (default: NaN)')
    exec_group.add_argument('--use-cache', action='store_true', default=True, help='Cache evaluation results')
    exec_group.add_argument('--no-cache', action='store_true', default=False, help='Cache evaluation results')
    exec_group.add_argument('--verbose', action='store_true', default=True, help='Print verbose output')
    exec_group.add_argument('--log-path', type=str, default='logs', 
                            help='Path to directory for saving logs (default: ./logs)')
    
    # Add argument for continuing from previous run
    exec_group.add_argument('--continue-from', type=str, help='Path to a previous run log directory to continue from')
    
    # Add test evaluation parameters
    test_group = parser.add_argument_group('Test Evaluation Parameters')
    test_group.add_argument('--test-eval', action='store_true', help='Periodically evaluate on test data')
    test_group.add_argument('--test-frequency', type=int, default=10, help='Test evaluation frequency (every N generations)')
    test_group.add_argument('--boundary', action='store_true', help='Use boundary-based evaluation')
    test_group.add_argument('--epsilon', type=float, default=3.0, help='Epsilon for boundary calculation')
    test_group.add_argument('--hitk', type=int, default=1, help='Top-k accuracy to compute')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Variables to keep track of continuation
    start_gen = 0
    previous_population = None
    
    # Esto es un pequeño parche que pongo para no usar la caché ya que da problemas si ejeuto n-jobs > 1
    if args.no_cache:
        args.use_cache = False
    
    # Check if continuing from previous run
    if args.continue_from:
        try:
            prev_config, prev_population_data, prev_gen_number = load_previous_run(args.continue_from)
            
            # Copy args for later modification check
            old_args = deepcopy(args)
            
            # Update arguments from loaded config if requested
            if args.verbose:
                print(f"Continuing from previous run: {args.continue_from}")
                print(f"Previous run completed {prev_gen_number} generations")
            
            # Set start generation for continuation
            start_gen = prev_gen_number
            previous_population = prev_population_data
            
            # Update dataset from config
            args.dataset = prev_config["dataset"]
            args.preprocessing = prev_config["preprocessing"]
            args.orig_attribute = prev_config["orig_attribute"]
            args.scaler_str = prev_config["scaler"]
            
            # Update GA parameters
            args.population_size = prev_config["ga_params"]["population_size"]
            args.crossover_probability = prev_config["ga_params"]["crossover_probability"]
            args.mutation_probability = prev_config["ga_params"]["mutation_probability"]
            args.tournament_size = prev_config["ga_params"]["tournament_size"]
            args.elitism = prev_config["ga_params"]["elitism"]
            args.max_features = prev_config["ga_params"]["max_features"]
            args.algorithm = prev_config["ga_params"]["algorithm"]
            
            # Update cross-validation parameters
            args.seed = prev_config["seed"]
            args.cv = prev_config["dataset_info"]["cv"]
            args.n_splits = prev_config["dataset_info"]["n_splits"]
            
            # Check if args have changed comparing args to 
            if args.verbose:
                for key, value in vars(args).items():
                    if key in vars(old_args) and vars(old_args)[key] != value:
                        print(f"Updated argument: {key} changed from {vars(old_args)[key]} to {value}")
            

        except Exception as e:
            print(f"Error loading previous run: {e}")
            print("Starting a new run instead.")
            args.continue_from = None
    
    # --- Random seed ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Auto-detect AWA2 dataset and set awa parameter accordingly
    args.awa = True if args.dataset == "AWA2" else False

    # --- Data Loading ---
    path_data = pathlib.Path(f"data/{args.dataset}")
    trainval_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(path_data, args, trainval=True, preprocessing=args.preprocessing)
    
    # Apply normalize_feature for AWA2 dataset
    if args.awa:
        from models.sae import normalize_feature
        trainval_dataset.features = normalize_feature(trainval_dataset.features)
        if args.verbose:
            print("AWA2 dataset: Applied normalize_feature to trainval_dataset.features")
    
    # --- Create Log Directory ---
    log_dir = create_log_directory(args.log_path, args.continue_from)
    if args.verbose:
        print(f"Logging GA run to {log_dir}")
    
    # Save configuration information
    dataset_info = {
        "num_features": trainval_dataset.features.shape[1],
        "num_attributes": trainval_dataset.att.shape[1],
        "train_size": len(trainval_dataset.features),
        "cv": args.cv,
        "n_splits": args.n_splits if args.cv else 0
    }
    save_config(log_dir, args, dataset_info)
    
    # --- Evaluator Setup ---
    if args.cv or args.dynamic_train:
        # Create folds using the split_via_classes_per_folds method
        splitter = SplitIntoFolds(n_splits=args.n_splits, show=args.verbose, seed=args.seed)
        folds = list(splitter.split_via_classes_per_folds(
            features=trainval_dataset.features,
            labels=trainval_dataset.labels,
            attribute=attribute,
            preprocessing=args.preprocessing
        ))
        
        # Apply normalize_feature for AWA2 dataset to each fold
        if args.awa:
            from models.sae import normalize_feature
            normalized_folds = []
            for i, (train_fold, val_fold) in enumerate(folds):
                train_fold.features = normalize_feature(train_fold.features)
                #val_fold.features = normalize_feature(val_fold.features)
                normalized_folds.append((train_fold, val_fold))
            folds = normalized_folds
            if args.verbose:
                print(f"AWA2 dataset: Applied normalize_feature to all {len(folds)} CV folds")
        
        if args.verbose:
            print(f"Using {args.n_splits}-fold cross-validation")
            if args.dynamic_train:
                print(f"Dynamic training enabled: changing fold every {args.fold_change_frequency} generations")
    else:
        # Use separate train and validation sets - create a single "fold"
        train_dataset, val_dataset, _, _, attribute = get_data(path_data, args, trainval=False, preprocessing=args.preprocessing)
        
        # Apply normalize_feature for AWA2 dataset to train and validation datasets
        if args.awa:
            from models.sae import normalize_feature
            train_dataset.features = normalize_feature(train_dataset.features)
            #val_dataset.features = normalize_feature(val_dataset.features)
            if args.verbose:
                print("AWA2 dataset: Applied normalize_feature to train_dataset.features and val_dataset.features")
        
        # Create a list with a single fold (train_dataset, val_dataset)
        folds = [(train_dataset, val_dataset)]
        
        if args.verbose:
            print("Using train/validation split as a single fold")
    
    # Create evaluator with folds (same code for both CV and non-CV)
    evaluator = SAEvaluator(
        folds=folds,
        all_attributes=attribute,
        max_features=args.max_features,
        error_score=args.error_score,
        use_cache=args.use_cache,
        awa=args.awa,
        dynamic_fold=args.dynamic_train
    )
    
    # Set log directory for evaluation counter
    evaluator.set_log_dir(log_dir)
    
    test_evaluator = None
    if args.test_eval:
        # Create test evaluator if test evaluation is enabled
        test_evaluator = TestEvaluator(
            train_dataset=trainval_dataset,
            test_seen_dataset=test_seen_dataset,
            test_unseen_dataset=test_unseen_dataset,
            all_attributes=attribute,
            eval_frequency=args.test_frequency,
            boundary=args.boundary,
            epsilon=args.epsilon,
            hitk=args.hitk,
            awa=args.awa
        )

    # --- Toolbox Setup ---
    toolbox = base.Toolbox()
    
    # Set up parallel processing if n_jobs != 1
    if args.n_jobs != 1:
        pool_size = multiprocessing.cpu_count() if args.n_jobs == -1 else args.n_jobs
        pool = multiprocessing.Pool(processes=pool_size)
        toolbox.register("map", pool.map)
    
    # Attribute generator: generates 0 or 1
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    # Individual: a list of 0s/1s of length equal to number of attributes
    num_attributes = trainval_dataset.att.shape[1]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_attributes)
    # Population: a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluator.evaluate_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)  # Uniform crossover with 0.5 probability per bit
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/num_attributes)  # 1/L probability to flip each bit
    toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)

    # Create population - either new or from previous run
    if previous_population:
        # Create population from loaded data
        pop = create_population_from_data(toolbox, previous_population)
        if args.verbose:
            print(f"Continuing with population of {len(pop)} individuals")
    else:
        # Create a new population
        if args.init_uniform:
            # Initialize population uniformly from 35% to 100% active features
            pop = []
            proportion_min = 0.35
            
            if args.verbose:
                print(f"Initializing population uniformly from {proportion_min*100:.0f}% to 100% active features")
            
            for i in range(args.population_size):
                # Calculate proportion for this individual (uniform distribution)
                proportion = proportion_min + (1.0 - proportion_min) * i / (args.population_size - 1)
                n_active = int(proportion * num_attributes)
                
                # Create individual with random active features
                individual = creator.Individual([0] * num_attributes)
                active_indices = np.random.choice(num_attributes, n_active, replace=False)
                for idx in active_indices:
                    individual[idx] = 1
                
                # Invalidate fitness so it gets evaluated
                del individual.fitness.values
                pop.append(individual)
                
                if args.verbose:
                    print(f"Individual {i+1}: {n_active}/{num_attributes} features ({proportion*100:.1f}%)")
            if np.sum(pop[-1]) != num_attributes:
                raise ValueError("Last individual does not have all attributes activated, check initialization logic")
        else:
            # Standard random initialization
            pop = toolbox.population(n=args.population_size)
    
    # Add an individual with all attributes activated (only for new populations)
    if args.all_1 and not previous_population and len(pop) > 0:
        # Create individual with all attributes set to 1
        all_attributes_individual = creator.Individual([1] * num_attributes)
        # Replace the first individual in the population
        pop[-1] = all_attributes_individual
        # Invalidate fitness so it gets evaluated
        del pop[-1].fitness.values
        if args.verbose:
            print(f"Added individual with all {num_attributes} attributes activated to population")

    if args.include_hais_solutions:
        # Load HAIS solutions from the specified directory
        hais_solutions_dir = pathlib.Path(f"data/hais_solutions/{args.dataset}")
        hais_files         = hais_solutions_dir / "solutions.json"

        if not hais_files.exists():
            raise ValueError(f"HAIS solutions file not found: {hais_files}")
        else:
            with open(hais_files, 'r') as f:
                hais_solutions = json.load(f)
                
                # Convert existing population to set of tuples for efficient lookup
                existing_solutions = set()
                for ind in pop:
                    existing_solutions.add(tuple(ind))
                
                # Add HAIS solutions that are not already in the population
                added_count = 0
                for solution_name, mask in hais_solutions.items():
                    mask_tuple = tuple(mask)
                    if mask_tuple not in existing_solutions:
                        hais_individual = creator.Individual(mask)
                        # Invalidate fitness so it gets evaluated
                        del hais_individual.fitness.values
                        pop.append(hais_individual)
                        existing_solutions.add(mask_tuple)
                        added_count += 1
                
                if args.verbose:
                    print(f"Added {added_count} HAIS solutions to population from {len(hais_solutions)} available solutions")
                
                

    # To store the best individuals
    hof = tools.HallOfFame(args.keep_top_k)
    
    # Update hall of fame with existing population if continuing
    if previous_population:
        for ind in pop:
            if ind.fitness.valid:
                hof.update([ind])
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if args.verbose:
        if previous_population:
            print(f"Continuing DEAP Genetic Algorithm from generation {start_gen}")
        else:
            print("Starting DEAP Genetic Algorithm for feature selection...")
        
        if args.dynamic_train:
            print(f"Dynamic training mode: Using {len(folds)} folds, changing every {args.fold_change_frequency} generations")
    
    # List to store statistics for each generation
    stats_history = []
    
    # Run the GA using the selected algorithm
    if args.algorithm == 'eaMuPlusLambda' and args.elitism:
        # Use our modified version with built-in logging
        # Note: The algorithm assumes we're starting from generation 0
        # We'll need to adjust the total generations if continuing
        remaining_gens = args.generations - start_gen
        
        pop, logbook, stats_history = eaMuPlusLambdaWithLog(
            pop, toolbox, 
            mu=args.population_size, 
            lambda_=args.population_size, 
            cxpb=args.crossover_probability, 
            mutpb=args.mutation_probability, 
            ngen=remaining_gens, 
            stats=stats, 
            halloffame=hof, 
            verbose=args.verbose,
            log_dir=log_dir,
            test_evaluator=test_evaluator,
            evaluator=evaluator if args.dynamic_train else None,
            dynamic_train=args.dynamic_train,
            fold_change_frequency=args.fold_change_frequency,
            num_folds=len(folds)
        )
    else:
        # Default to eaSimple with manual logging
        for gen in range(args.generations):
            # Dynamic fold change for dynamic training
            if args.dynamic_train and gen % args.fold_change_frequency == 0:
                current_fold_idx = (gen // args.fold_change_frequency) % len(folds)
                evaluator.set_current_fold(current_fold_idx)
                if args.verbose:
                    print(f"Gen {gen}: Switched to fold {current_fold_idx}/{len(folds)-1}")
            
            # Evolve population for one generation
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, 
                cxpb=args.crossover_probability, 
                mutpb=args.mutation_probability, 
                ngen=1, 
                stats=stats, 
                halloffame=hof, 
                verbose=False
            )
            
            # Log this generation
            if logbook:
                stats_history.append(deepcopy(logbook[-1]))
                log_population(log_dir, gen, pop, logbook[-1])
            
            # Print progress if verbose
            if args.verbose and gen % 10 == 0:
                print(f"Generation {gen}/{args.generations} - Best fitness: {hof[0].fitness.values[0] if hof else 'N/A'}")

            # Add test evaluation to eaSimple path
            if test_evaluator is not None and test_evaluator.should_evaluate(gen) and hof:
                test_metrics = test_evaluator.evaluate(hof[0], gen)
                if args.verbose:
                    print(f"Gen {gen} Test: Seen={test_metrics['seen_acc']:.2f}%, " +
                          f"Unseen={test_metrics['unseen_acc']:.2f}%, " +
                          f"H-mean={test_metrics['h_mean']:.2f}%")
        
        # Final test evaluation for eaSimple path
        if test_evaluator is not None and hof:
            test_evaluator.save_results(log_dir)
    
    # Close multiprocessing pool if it exists
    if args.n_jobs != 1:
        pool.close()
        pool.join()

    # --- Results ---
    best_individual = hof[0]
    best_mask = np.array(best_individual, dtype=bool)
    
    # Final evaluation on all folds for accurate fitness
    if args.dynamic_train and len(folds) > 1:
        final_fitness = evaluator.evaluate_all_folds(best_individual)
        # Update the best individual's fitness with the all-folds evaluation
        best_individual.fitness.values = (final_fitness,)
        if args.verbose:
            print(f"Final evaluation on all folds: {final_fitness:.2f}%")
    
    if args.verbose:
        print("\n--- Results ---")
        print(f"Best individual (feature mask): \n{np.array(best_individual)}")
        print(f"Best validation accuracy: {best_individual.fitness.values[0]:.2f}%")
        print(f"Number of selected features: {np.sum(best_mask)}")
        print(f"Selected feature indices: {np.where(best_mask)[0]}")
    
    # Save final results - stats_history is already populated by eaMuPlusLambdaWithLog
    save_final_results(log_dir, best_individual, logbook, pop)
    write_statistics_csv(log_dir, stats_history)
    
    # Save evaluation count
    evaluator.save_evaluation_count()
    
    if args.verbose:
        print(f"\nComplete log saved to: {log_dir}")
        print(f"Total evaluations performed: {evaluator.evaluation_count}")
    
    # Add visualization generation
    try:
        print("\nGenerating visualizations...")
        visualize_ga_run(str(log_dir), check_continuations=args.continue_from is not None)  
        print("Visualizations generated successfully!")
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        print("You can generate visualizations later by running: python -m viz.plots " + str(log_dir))
    
    # Add a section in the final results printout to show the best test performance if available
    if args.test_eval and test_evaluator and test_evaluator.test_results['h_mean']:
        best_idx = np.argmax(test_evaluator.test_results['h_mean'])
        best_gen = test_evaluator.test_results['generations'][best_idx]
        best_seen = test_evaluator.test_results['seen_acc'][best_idx]
        best_unseen = test_evaluator.test_results['unseen_acc'][best_idx]
        best_hmean = test_evaluator.test_results['h_mean'][best_idx]
        
        print("\n--- Best Test Performance (Gen {}) ---".format(best_gen))
        print(f"Test Seen Accuracy: {best_seen:.2f}%")
        print(f"Test Unseen Accuracy: {best_unseen:.2f}%")
        print(f"Harmonic Mean: {best_hmean:.2f}%")

