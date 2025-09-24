import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from tqdm import tqdm
from copy import deepcopy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Custom color palette
palette = sns.color_palette("mako_r", 6)

def load_ga_logs(log_path, check_continuations=True):
    """
    Load all relevant GA log files from the specified directory.
    If the path ends with "_continued", also load data from the original run.
    
    Parameters:
    -----------
    log_path : str
        Path to the GA log directory
    check_continuations : bool
        Whether to check for and load continued runs
    
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    log_path = Path(log_path)
    
    # Check if the directory exists
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_path}")
    
    # Initialize the data dictionary
    data = {
        'config': None,
        'logbook': None,
        'best_individual': None,
        'final_population': None,
        'generations': [],
        'statistics': None,
        'run_paths': [str(log_path)]  # Track all paths that were used
    }
    
    # Load configuration
    config_path = log_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)
    
    # Load logbook
    logbook_path = log_path / "logbook.json"
    if logbook_path.exists():
        with open(logbook_path, 'r') as f:
            data['logbook'] = json.load(f)
    
    # Load best individual
    best_path = log_path / "best_individual.json"
    if best_path.exists():
        with open(best_path, 'r') as f:
            data['best_individual'] = json.load(f)
    
    # Load final population
    final_pop_path = log_path / "final_population.json"
    if final_pop_path.exists():
        with open(final_pop_path, 'r') as f:
            data['final_population'] = json.load(f)
    
    # Load statistics
    stats_path = log_path / "statistics.csv"
    if stats_path.exists():
        data['statistics'] = pd.read_csv(stats_path)
    
    # Load generation data
    gen_dir = log_path / "generations"
    if gen_dir.exists():
        gen_files = sorted(glob.glob(str(gen_dir / "gen_*.json")))
        
        print(f"Loading {len(gen_files)} generation files from {log_path}...")
        for file in tqdm(gen_files):
            with open(file, 'r') as f:
                gen_data = json.load(f)
                data['generations'].append(gen_data)
    
    # Check if this is a continued run and load the original data if it is
    if check_continuations and str(log_path).endswith("_continued"):
        # Extract original path by removing "_continued" suffix
        original_path = str(log_path).rstrip("_continued")
        print(f"Detected continued run. Loading original data from {original_path}...")
        
        try:
            # Load the original data (which might itself be a continuation)
            original_data = load_ga_logs(original_path, check_continuations=True)
            
            # Merge the data
            data = merge_ga_logs(original_data, data)
        except Exception as e:
            print(f"Error loading original data: {e}")
            print("Continuing with only the current data.")
    
    return data

def merge_ga_logs(first_data, second_data):
    """
    Merge two GA log data dictionaries to create a continuous optimization history.
    
    Parameters:
    -----------
    first_data : dict
        First GA log data (the earlier run)
    second_data : dict
        Second GA log data (the later/continued run)
    
    Returns:
    --------
    dict
        Merged GA log data
    """
    # Create a new dictionary for the merged data
    merged_data = {
        'config': second_data['config'],  # Use the latest config
        'best_individual': second_data['best_individual'],  # Use the latest best individual
        'final_population': second_data['final_population'],  # Use the latest final population
        'generations': [],
        'logbook': [],
        'statistics': None,
        'run_paths': first_data['run_paths'] + second_data['run_paths']  # Combine run paths
    }
    
    # Get the last generation number from the first run
    last_gen = 0
    if first_data['logbook']:
        last_gen = first_data['logbook'][-1]['gen']
    
    # Merge generations data with adjusted generation numbers
    merged_data['generations'] = first_data['generations'].copy()
    
    # Add generations from second data with adjusted generation numbers
    for gen_idx, gen_data in enumerate(second_data['generations']):
        # Add generations from second run
        merged_data['generations'].append(gen_data)
    
    # Merge logbook data with adjusted generation numbers
    if first_data['logbook']:
        merged_data['logbook'] = first_data['logbook'].copy()
    
    if second_data['logbook']:
        # Create copy of second logbook
        second_logbook = deepcopy(second_data['logbook'])
        
        # Adjust generation numbers
        for entry in second_logbook:
            entry['gen'] += last_gen + 1
        
        # Add to merged logbook
        if merged_data['logbook']:
            merged_data['logbook'].extend(second_logbook)
        else:
            merged_data['logbook'] = second_logbook
    
    # Merge statistics data
    if first_data['statistics'] is not None and second_data['statistics'] is not None:
        # Create a copy of first statistics
        merged_stats = first_data['statistics'].copy()
        
        # Get the second statistics and adjust Generation column
        second_stats = second_data['statistics'].copy()
        if 'Generation' in second_stats.columns:
            second_stats['Generation'] += last_gen + 1
        
        # Concatenate the dataframes
        merged_data['statistics'] = pd.concat([merged_stats, second_stats], ignore_index=True)
    elif first_data['statistics'] is not None:
        merged_data['statistics'] = first_data['statistics'].copy()
    elif second_data['statistics'] is not None:
        merged_data['statistics'] = second_data['statistics'].copy()
    
    return merged_data

def plot_fitness_progress(data, title=None, figsize=(12, 6), mark_continuations=True):
    """Plot fitness progression over generations"""
    if data['logbook'] is None:
        print("No logbook data found.")
        return
    
    # Extract fitness data from logbook
    generations = [entry['gen'] for entry in data['logbook']]
    max_fitness = [entry['max'] for entry in data['logbook']]
    avg_fitness = [entry['avg'] for entry in data['logbook']]
    std_fitness = [entry['std'] for entry in data['logbook']]
    min_fitness = [entry['min'] for entry in data['logbook']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot max, avg, min fitness
    ax.plot(generations, max_fitness, '-', color=palette[0], linewidth=2, label='Max Fitness')
    ax.plot(generations, avg_fitness, '-', color=palette[2], linewidth=2, label='Avg Fitness')
    ax.plot(generations, min_fitness, '-', color=palette[4], linewidth=2, label='Min Fitness')
    
    # Add standard deviation band
    ax.fill_between(
        generations, 
        [avg - std for avg, std in zip(avg_fitness, std_fitness)],
        [avg + std for avg, std in zip(avg_fitness, std_fitness)],
        alpha=0.2, color=palette[2], label='±1 Std Dev'
    )
    
    # Mark the boundaries between different runs if requested
    if mark_continuations and 'run_paths' in data and len(data['run_paths']) > 1:
        # Get unique run paths
        run_paths = data['run_paths']
        
        # Find where each continuation starts
        continuation_gens = []
        last_gen = 0
        
        for path_idx, path in enumerate(run_paths[:-1]):  # Skip the last one
            # Get the number of generations in this run
            if '_continued' in path:
                # This is already a continuation, skip
                continue
                
            # Check how many generations this run had
            path = Path(path)
            gen_files = sorted(glob.glob(str(path / "generations" / "gen_*.json")))
            num_gens = len(gen_files)
            
            # Add the boundary
            last_gen += num_gens
            continuation_gens.append(last_gen)
        
        # Draw vertical lines at each continuation point
        for gen in continuation_gens:
            if gen < len(generations):
                ax.axvline(x=generations[gen], color='red', linestyle='--', alpha=0.7)
                ax.text(generations[gen], ax.get_ylim()[0] + 0.01, 'Continued', rotation=90, 
                        verticalalignment='bottom', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Accuracy %)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Fitness Progression Over Generations')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Annotate final values
    ax.annotate(f'Final max: {max_fitness[-1]:.2f}%', 
                xy=(generations[-1], max_fitness[-1]),
                xytext=(generations[-1] - len(generations) * 0.15, max_fitness[-1]), 
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_feature_selection_matrix(data):
    """
    Create a matrix showing feature selection frequency across generations.
    
    Returns:
    --------
    numpy.ndarray
        Matrix of shape (num_generations, num_features) with values between 0 and 1
    """
    if not data['generations']:
        print("No generation data found.")
        return None
    
    # Get dimensions
    num_generations = len(data['generations'])
    if num_generations == 0:
        return None
    
    num_features = len(data['generations'][0][0]['mask'])
    
    # Create matrix
    selection_matrix = np.zeros((num_generations, num_features))
    
    # Fill matrix with selection frequencies
    for gen_idx, generation in enumerate(data['generations']):
        population_size = len(generation)
        for feature_idx in range(num_features):
            # Count how many individuals have this feature selected
            feature_count = sum(ind['mask'][feature_idx] for ind in generation)
            # Calculate frequency
            selection_matrix[gen_idx, feature_idx] = feature_count / population_size
    
    return selection_matrix

def plot_feature_selection_heatmap(data, figsize=(16, 10), title=None, mark_continuations=True):
    """Plot heatmap of feature selection frequency over generations"""
    selection_matrix = create_feature_selection_matrix(data)
    
    if selection_matrix is None:
        print("Could not create feature selection matrix.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap from white to dark blue
    colors = [(1, 1, 1), palette[0]]  # White to first color in our palette
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Plot heatmap
    heatmap = sns.heatmap(
        selection_matrix.T,
        cmap=cmap,
        ax=ax,
        cbar_kws={'label': 'Selection Frequency'},
        xticklabels=50,  # Show only every 50th generation
        yticklabels=20   # Show only every 20th feature
    )
    
    # Mark the boundaries between different runs if requested
    if mark_continuations and 'run_paths' in data and len(data['run_paths']) > 1:
        # Get unique run paths
        run_paths = data['run_paths']
        
        # Find where each continuation starts
        continuation_gens = []
        last_gen = 0
        
        for path_idx, path in enumerate(run_paths[:-1]):  # Skip the last one
            # Get the number of generations in this run
            if '_continued' in path:
                # This is already a continuation, skip
                continue
                
            # Check how many generations this run had
            path = Path(path)
            gen_files = sorted(glob.glob(str(path / "generations" / "gen_*.json")))
            num_gens = len(gen_files)
            
            # Add the boundary
            last_gen += num_gens
            continuation_gens.append(last_gen)
        
        # Draw vertical lines at each continuation point
        for gen in continuation_gens:
            if gen < selection_matrix.shape[0]:
                ax.axvline(x=gen, color='red', linestyle='--', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Generation')
    ax.set_ylabel('Feature Index')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Feature Selection Frequency Across Generations')
    
    plt.tight_layout()
    return fig

def plot_final_feature_selection(data, top_n=50, figsize=(14, 8), title=None):
    """Plot bar chart of feature selection frequency in final population"""
    if data['final_population'] is None:
        print("No final population data found.")
        return
    
    # Extract feature selection data
    num_features = len(data['final_population'][0]['mask'])
    feature_counts = np.zeros(num_features)
    
    for ind in data['final_population']:
        for i, selected in enumerate(ind['mask']):
            feature_counts[i] += selected
    
    # Calculate frequency
    feature_freq = feature_counts / len(data['final_population'])
    
    # Sort features by frequency
    sorted_indices = np.argsort(feature_freq)[::-1]  # Descending order
    
    # Take top N features
    top_indices = sorted_indices[:top_n]
    top_freq = feature_freq[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bar chart
    bars = ax.bar(np.arange(len(top_indices)), top_freq, color=palette[0], alpha=0.7)
    
    # Add best individual's selected features
    if data['best_individual'] is not None:
        best_mask = data['best_individual']['mask']
        for i, idx in enumerate(top_indices):
            if best_mask[idx]:
                bars[i].set_color(palette[3])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=palette[0], alpha=0.7, label='Not in Best Solution'),
            Patch(facecolor=palette[3], alpha=0.7, label='In Best Solution')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Add labels and title
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Selection Frequency')
    ax.set_xticks(np.arange(len(top_indices)))
    ax.set_xticklabels([str(idx) for idx in top_indices], rotation=90)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Top {top_n} Most Frequently Selected Features in Final Population')
    
    plt.tight_layout()
    return fig

def analyze_population_diversity(data):
    """Calculate diversity metrics for each generation"""
    if not data['generations']:
        print("No generation data found.")
        return None
    
    num_generations = len(data['generations'])
    
    # Initialize metrics
    metrics = {
        'generation': list(range(num_generations)),
        'avg_features': np.zeros(num_generations),
        'std_features': np.zeros(num_generations),
        'unique_solutions': np.zeros(num_generations),
        'hamming_diversity': np.zeros(num_generations),
    }
    
    # Calculate metrics for each generation
    for gen_idx, generation in enumerate(data['generations']):
        # Number of features selected in each individual
        feature_counts = [sum(ind['mask']) for ind in generation]
        
        # Average and std of features
        metrics['avg_features'][gen_idx] = np.mean(feature_counts)
        metrics['std_features'][gen_idx] = np.std(feature_counts)
        
        # Number of unique solutions
        unique_masks = set(tuple(ind['mask']) for ind in generation)
        metrics['unique_solutions'][gen_idx] = len(unique_masks)
        
        # Hamming diversity (average pairwise distance)
        if len(generation) > 1:
            hamming_sum = 0
            count = 0
            for i in range(len(generation)):
                for j in range(i + 1, len(generation)):
                    # Calculate Hamming distance
                    hamming_sum += sum(a != b for a, b in zip(generation[i]['mask'], generation[j]['mask']))
                    count += 1
            metrics['hamming_diversity'][gen_idx] = hamming_sum / count if count > 0 else 0
    
    return pd.DataFrame(metrics)

def plot_population_diversity(data, figsize=(16, 10)):
    """Plot population diversity metrics"""
    # Calculate diversity metrics
    diversity_df = analyze_population_diversity(data)
    
    if diversity_df is None:
        print("Could not calculate diversity metrics.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # 1. Average number of features
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(diversity_df['generation'], diversity_df['avg_features'], color=palette[0], linewidth=2)
    ax1.fill_between(
        diversity_df['generation'],
        diversity_df['avg_features'] - diversity_df['std_features'],
        diversity_df['avg_features'] + diversity_df['std_features'],
        alpha=0.2, color=palette[0]
    )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Avg. Features Selected')
    ax1.set_title('Average Number of Features Selected')
    
    # 2. Unique solutions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(diversity_df['generation'], diversity_df['unique_solutions'], color=palette[1], linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Count')
    ax2.set_title('Number of Unique Solutions')
    
    # 3. Hamming diversity
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(diversity_df['generation'], diversity_df['hamming_diversity'], color=palette[2], linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Avg. Hamming Distance')
    ax3.set_title('Population Diversity (Hamming Distance)')
    
    # 4. Distribution of feature counts in the final population
    ax4 = fig.add_subplot(gs[1, 1])
    if data['final_population']:
        feature_counts = [sum(ind['mask']) for ind in data['final_population']]
        sns.histplot(feature_counts, kde=True, color=palette[3], ax=ax4)
        ax4.set_xlabel('Number of Features Selected')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Selected Features (Final Population)')
    
    plt.tight_layout()
    return fig

def plot_feature_coselection(data, top_n=50, figsize=(14, 12)):
    """Plot heatmap of feature co-selection in final population"""
    if data['final_population'] is None or len(data['final_population']) == 0:
        print("No final population data found.")
        return
    
    # Extract masks from final population
    masks = np.array([ind['mask'] for ind in data['final_population']])
    
    # Get the most frequently selected features
    feature_counts = masks.sum(axis=0)
    top_indices = np.argsort(feature_counts)[::-1][:top_n]
    
    # Create correlation matrix for top features
    top_masks = masks[:, top_indices]
    
    # Calculate variances to identify features with zero variance
    variances = np.var(top_masks, axis=0)
    zero_var_indices = np.where(variances == 0)[0]
    
    # If all features have zero variance, return
    if len(zero_var_indices) == len(top_indices):
        print("Warning: All top features have zero variance. Cannot calculate correlations.")
        return
    
    # Calculate correlation with warnings suppressed
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = np.corrcoef(top_masks.T)
    
    # Replace NaN values with 0 in correlation matrix
    correlation = np.nan_to_num(correlation)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        correlation,
        cmap='coolwarm',
        ax=ax,
        xticklabels=[str(idx) for idx in top_indices],
        yticklabels=[str(idx) for idx in top_indices],
        vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'}
    )
    
    # Add labels and title
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Index')
    ax.set_title(f'Feature Co-selection Patterns (Top {top_n} Features)')
    
    plt.tight_layout()
    return fig

def plot_feature_fitness_correlation(data, top_n=30, figsize=(14, 6)):
    """Plot correlation between feature selection and fitness"""
    if data['final_population'] is None or len(data['final_population']) == 0:
        print("No final population data found.")
        return
    
    # Extract data
    population = data['final_population']
    
    # Create mask matrix and fitness vector
    masks = []
    fitness = []
    for ind in population:
        if ind['fitness'] is not None:  # Skip individuals with no fitness
            masks.append(ind['mask'])
            fitness.append(ind['fitness'])
    
    if not masks:
        print("No individuals with valid fitness found.")
        return
    
    # Convert to numpy arrays
    masks = np.array(masks)
    fitness = np.array(fitness)
    
    # Check if fitness has variation
    if np.std(fitness) == 0:
        print("Warning: All individuals have the same fitness. Cannot calculate correlations.")
        return
    
    # Calculate correlation between each feature and fitness
    num_features = masks.shape[1]
    correlations = np.zeros(num_features)
    
    # Check for features with zero variance
    feature_variances = np.var(masks, axis=0)
    
    for i in range(num_features):
        # Skip correlation calculation for zero-variance features
        if feature_variances[i] == 0:
            correlations[i] = 0
            continue
            
        # Calculate correlation with warnings suppressed
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(masks[:, i], fitness)
            correlations[i] = corr[0, 1] if not np.isnan(corr[0, 1]) else 0
    
    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    top_indices = sorted_indices[:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap for positive/negative correlation
    colors = [palette[0] if c > 0 else palette[4] for c in correlations[top_indices]]
    
    # Plot bar chart
    bars = ax.bar(np.arange(len(top_indices)), correlations[top_indices], color=colors)
    
    # Add best individual's selected features
    if data['best_individual'] is not None:
        best_mask = data['best_individual']['mask']
        for i, idx in enumerate(top_indices):
            if best_mask[idx]:
                # Highlight bar with edge
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
    
    # Add labels and title
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Correlation with Fitness')
    ax.set_xticks(np.arange(len(top_indices)))
    ax.set_xticklabels([str(idx) for idx in top_indices], rotation=90)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title('Correlation Between Feature Selection and Fitness')
    
    plt.tight_layout()
    return fig

def plot_convergence_analysis(data, figsize=(14, 8)):
    """Plot convergence analysis metrics"""
    if not data['generations'] or data['best_individual'] is None:
        print("Missing generation data or best individual.")
        return
    
    # Extract best solution
    best_mask = np.array(data['best_individual']['mask'])
    
    # Calculate metrics for each generation
    num_generations = len(data['generations'])
    
    # Initialize metrics
    metrics = {
        'generation': list(range(num_generations)),
        'best_feature_match': np.zeros(num_generations),  # % of best features already discovered
        'false_positives': np.zeros(num_generations),     # % of non-best features selected
        'fitness_gap': np.zeros(num_generations)          # gap between best fitness and average
    }
    
    # Calculate metrics for each generation
    for gen_idx, generation in enumerate(data['generations']):
        # Calculate average mask in current population
        avg_mask = np.zeros(len(best_mask))
        for ind in generation:
            avg_mask += np.array(ind['mask'])
        avg_mask /= len(generation)
        
        # Best feature match: average selection rate of best features
        best_indices = np.where(best_mask == 1)[0]
        if len(best_indices) > 0:
            metrics['best_feature_match'][gen_idx] = np.mean(avg_mask[best_indices])
        
        # False positives: average selection rate of non-best features
        non_best_indices = np.where(best_mask == 0)[0]
        if len(non_best_indices) > 0:
            metrics['false_positives'][gen_idx] = np.mean(avg_mask[non_best_indices])
        
        # Fitness gap
        if data['logbook'] and gen_idx < len(data['logbook']):
            logbook_entry = data['logbook'][gen_idx]
            metrics['fitness_gap'][gen_idx] = logbook_entry['max'] - logbook_entry['avg']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot feature convergence
    ax1.plot(metrics['generation'], metrics['best_feature_match'], 
             color=palette[1], linewidth=2, label='Best Feature Match')
    ax1.plot(metrics['generation'], metrics['false_positives'], 
             color=palette[4], linewidth=2, label='False Positives')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Selection Rate')
    ax1.set_title('Feature Convergence Analysis')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot fitness gap
    ax2.plot(metrics['generation'], metrics['fitness_gap'], 
             color=palette[2], linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Gap')
    ax2.set_title('Fitness Gap (Best - Average)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_evolutionary_feature_correlation(data, top_n=30, figsize=(14, 6)):
    """
    Plot correlation between feature selection and fitness across the entire evolutionary process.
    
    Parameters:
    -----------
    data : dict
        GA log data containing generations data
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    if not data['generations']:
        print("No generation data found.")
        return
    
    # Extract all individuals across all generations
    all_masks = []
    all_fitness = []
    
    for gen_idx, generation in enumerate(data['generations']):
        for individual in generation:
            if individual['fitness'] is not None:  # Skip individuals with no fitness
                all_masks.append(individual['mask'])
                all_fitness.append(individual['fitness'])
    
    if not all_masks:
        print("No individuals with valid fitness found.")
        return
    
    # Convert to numpy arrays
    all_masks = np.array(all_masks)
    all_fitness = np.array(all_fitness)
    
    # Check if fitness has variation
    if np.std(all_fitness) == 0:
        print("Warning: All individuals have the same fitness. Cannot calculate correlations.")
        return
    
    # Calculate correlation between each feature and fitness
    num_features = all_masks.shape[1]
    correlations = np.zeros(num_features)
    
    # Check for features with zero variance
    feature_variances = np.var(all_masks, axis=0)
    
    for i in range(num_features):
        # Skip correlation calculation for zero-variance features
        if feature_variances[i] == 0:
            correlations[i] = 0
            continue
            
        # Calculate correlation with warnings suppressed
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(all_masks[:, i], all_fitness)
            correlations[i] = corr[0, 1] if not np.isnan(corr[0, 1]) else 0
    
    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    top_indices = sorted_indices[:top_n]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap for positive/negative correlation
    colors = [palette[0] if c > 0 else palette[4] for c in correlations[top_indices]]
    
    # Plot bar chart
    bars = ax.bar(np.arange(len(top_indices)), correlations[top_indices], color=colors)
    
    # Add labels and title
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Correlation with Fitness')
    ax.set_xticks(np.arange(len(top_indices)))
    ax.set_xticklabels([str(idx) for idx in top_indices], rotation=90)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title('Evolutionary Correlation Between Feature Selection and Fitness')
    
    plt.tight_layout()
    return fig

def visualize_ga_run(log_path, output_dir=None, check_continuations=True):
    """Generate and save all visualizations for a GA run"""
    # Load data
    print(f"Loading data from {log_path}...")
    data = load_ga_logs(log_path, check_continuations=check_continuations)
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(log_path, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Print information about continuations
    if 'run_paths' in data and len(data['run_paths']) > 1:
        print(f"Merged data from {len(data['run_paths'])} runs:")
        for path in data['run_paths']:
            print(f"  - {path}")
    
    print("Generating visualizations...")
    
    # 1. Fitness Progress
    print("  - Fitness progress...")
    fig1 = plot_fitness_progress(data, mark_continuations=True)
    if fig1:
        fig1.savefig(os.path.join(output_dir, "1_fitness_progress.png"), dpi=300, bbox_inches='tight')
    
    # 2. Feature Selection Heatmap
    print("  - Feature selection heatmap...")
    fig2 = plot_feature_selection_heatmap(data, mark_continuations=True)
    if fig2:
        fig2.savefig(os.path.join(output_dir, "2_feature_selection_heatmap.png"), dpi=300, bbox_inches='tight')
    
    # 3. Final Feature Selection
    print("  - Final feature selection...")
    fig3 = plot_final_feature_selection(data)
    if fig3:
        fig3.savefig(os.path.join(output_dir, "3_final_feature_selection.png"), dpi=300, bbox_inches='tight')
    
    # 4. Population Diversity
    print("  - Population diversity...")
    fig4 = plot_population_diversity(data)
    if fig4:
        fig4.savefig(os.path.join(output_dir, "4_population_diversity.png"), dpi=300, bbox_inches='tight')
    
    # 5. Feature Co-selection
    print("  - Feature co-selection patterns...")
    fig5 = plot_feature_coselection(data)
    if fig5:
        fig5.savefig(os.path.join(output_dir, "5_feature_coselection.png"), dpi=300, bbox_inches='tight')
    
    # 6. Feature-Fitness Correlation
    print("  - Feature-fitness correlation...")
    fig6 = plot_feature_fitness_correlation(data)
    if fig6:
        fig6.savefig(os.path.join(output_dir, "6_feature_fitness_correlation.png"), dpi=300, bbox_inches='tight')
    
    # 7. Convergence Analysis
    print("  - Convergence analysis...")
    fig7 = plot_convergence_analysis(data)
    if fig7:
        fig7.savefig(os.path.join(output_dir, "7_convergence_analysis.png"), dpi=300, bbox_inches='tight')
    
    # 8. Evolutionary Feature-Fitness Correlation
    print("  - Evolutionary feature-fitness correlation...")
    fig8 = plot_evolutionary_feature_correlation(data)
    if fig8:
        fig8.savefig(os.path.join(output_dir, "8_evolutionary_feature_correlation.png"), dpi=300, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")
    return data
