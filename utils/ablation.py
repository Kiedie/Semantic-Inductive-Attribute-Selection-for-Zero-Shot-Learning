from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from scipy.stats import binomtest, spearmanr  # binomtest es la nueva función
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import jaccard_score
import sys

warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)



def load_convergence_data(dataset, 
                          read_dir,
                          base_dir="logs",
                          algorithm="eaMuPlusLambda", 
                          max_runs = None, 
                          verbose = False):
    """
    Cargar datos de ejecuciones desde logs/{dataset}/convergence/ga_run_*
    """
    if verbose:
        print(f"Cargando datos para {dataset}...")

    convergence_dir = read_dir

    if not convergence_dir.exists():
        print(f"No existe: {convergence_dir}")
        return None
    
    # Buscar directorios ga_run_*
    ga_run_dirs = sorted(list(convergence_dir.glob("ga_run_*")))
    if max_runs is not None:
        ga_run_dirs = ga_run_dirs[:max_runs]
    
    if verbose:
        print(f"Encontrados: {len(ga_run_dirs)} runs")
    
    if len(ga_run_dirs) == 0:
        print("❌ No hay directorios ga_run_*")
        return None
    
    best_individuals = []
    all_statistics = []
    successful_runs = 0
    
    # Procesar cada run
    for i, run_dir in enumerate(ga_run_dirs, 1):
        if verbose:
            print(f"  [{i:2d}] {run_dir.name}...", end=" ")

        try:
            # Cargar mejor individuo
            best_file = run_dir / "best_individual.json"
            if not best_file.exists():
                print("❌ Sin best_individual.json")
                continue
                
            with open(best_file, 'r') as f:
                data = json.load(f)
            
            # Buscar máscara binaria
            mask = None
            for field in ['mask', 'individual', 'features', 'solution']:
                if field in data:
                    mask = np.array(data[field], dtype=int)
                    break
            
            if mask is None:
                print("❌ Sin máscara")
                continue
            
            # Cargar estadísticas
            stats_file = run_dir / "statistics.csv"
            stats_df = None
            if stats_file.exists():
                try:
                    stats_df = pd.read_csv(stats_file)
                except:
                    stats_df = None
            
            # Guardar datos
            best_individuals.append(mask)
            all_statistics.append(stats_df)
            successful_runs += 1
            
            if verbose:
                print("✅")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:20]}")
    
    if successful_runs == 0:
        print("❌ No hay ejecuciones exitosas")
        return None
    
    n_attributes = len(best_individuals[0])
    
    if verbose:
        print(f"\n✅ Datos cargados:")
        print(f"  - Runs exitosos: {successful_runs}")
        print(f"  - Atributos: {n_attributes}")
    
    return {
        'best_individuals': best_individuals,
        'all_statistics': all_statistics,
        'successful_runs': successful_runs,
        'dataset': dataset,
        'n_attributes': n_attributes
    }

def load_generation_population(run_dir, generation):
    """
    Cargar población completa de una generación específica
    
    Args:
        run_dir: Path al directorio del run (e.g., ga_run_20250727_024711_30468a01)
        generation: Número de generación (0-300)
        
    Returns:
        list: Lista de máscaras binarias de todos los individuos de la población
        None si hay error
    """
    gen_file = run_dir / "generations" / f"gen_{generation:03d}.json"
    
    if not gen_file.exists():
        return None
    
    try:
        with open(gen_file, 'r') as f:
            population_data = json.load(f)
        
        # Extraer solo las máscaras de todos los individuos
        population_masks = []
        for individual in population_data:
            if 'mask' in individual:
                mask = np.array(individual['mask'], dtype=int)
                population_masks.append(mask)
        
        return population_masks
    
    except Exception as e:
        print(f"Error cargando {gen_file}: {e}")
        return None

# PARTE 1: ANÁLISIS DE FRECUENCIAS DE SELECCIÓN DE ATRIBUTOS
# ==========================================================
def analyze_attribute_frequencies(convergence_data, verbose = True):
    """
    Analizar frecuencias de selección de atributos
    
    Args:
        convergence_data: Datos cargados de convergencia
        
    Returns:
        dict: Resultados del análisis de frecuencias
    """
    if verbose:
        print("Analizando frecuencias de selección de atributos...")
    
    if convergence_data is None:
        print("❌ No hay datos de convergencia disponibles")
        return None
    
    best_individuals = convergence_data['best_individuals']
    n_runs = convergence_data['successful_runs']
    n_attributes = convergence_data['n_attributes']
    
    # Convertir a array numpy para facilitar cálculos
    individuals_matrix = np.array(best_individuals)  # Shape: (n_runs, n_attributes)
    
    # Calcular frecuencias de selección (suma por columnas)
    selection_frequencies = np.sum(individuals_matrix, axis=0)  # Shape: (n_attributes,)
    
    # Calcular estadísticas descriptivas
    stats = {
        'frequencies': selection_frequencies,
        'n_runs': n_runs,
        'n_attributes': n_attributes,
        
        # Estadísticas de tendencia central
        'mean_frequency': np.mean(selection_frequencies),           # Media (puede ser engañosa)
        'median_frequency': np.median(selection_frequencies),       # Mediana (más robusta)
        'mode_frequency': np.bincount(selection_frequencies).argmax(),  # Moda
        
        # Estadísticas de dispersión
        'std_frequency': np.std(selection_frequencies),
        'min_frequency': np.min(selection_frequencies),
        'max_frequency': np.max(selection_frequencies),
        
        # Percentiles útiles
        'p25_frequency': np.percentile(selection_frequencies, 25),  # Q1
        'p75_frequency': np.percentile(selection_frequencies, 75),  # Q3
        'p90_frequency': np.percentile(selection_frequencies, 90),  # Atributos más importantes
        
        # Categorización de atributos
        'never_selected': np.sum(selection_frequencies == 0),                   # Atributos irrelevantes
        'always_selected': np.sum(selection_frequencies == n_runs),             # Atributos críticos
        'frequently_selected': np.sum(selection_frequencies >= n_runs * 0.75),  # 75% de las veces
        'rarely_selected': np.sum(selection_frequencies <= n_runs * 0.25),      # 25% de las veces
        
        # Métricas derivadas útiles
        'avg_attributes_per_run': np.sum(selection_frequencies) / n_runs,       # Atributos promedio por ejecución
        'selection_sparsity': np.sum(selection_frequencies == 0) / n_attributes, # % atributos nunca usados
    }
    
    if verbose:
        print(f"✅ Frecuencias calculadas:")
        print(f"  - Total de atributos: {n_attributes}")
        print(f"  - Frecuencia media: {stats['mean_frequency']:.2f} ± {stats['std_frequency']:.2f}")
        print(f"  - Frecuencia mediana: {stats['median_frequency']:.1f} (más robusta que la media)")
        print(f"  - Rango: [{stats['min_frequency']} - {stats['max_frequency']}]")
        print(f"  - Percentiles (Q1/Q3): [{stats['p25_frequency']:.1f} - {stats['p75_frequency']:.1f}]")
        print(f"")
        print(f"  📊 INTERPRETACIÓN CLAVE:")
        print(f"  - Atributos promedio por run: {stats['avg_attributes_per_run']:.1f}")
        print(f"  - Sparsity (% nunca usados): {stats['selection_sparsity']*100:.1f}%")
        print(f"")
        print(f"  - Nunca seleccionados: {stats['never_selected']} ({stats['never_selected']/n_attributes*100:.1f}%)")
        print(f"  - Siempre seleccionados: {stats['always_selected']} ({stats['always_selected']/n_attributes*100:.1f}%)")
        print(f"  - Frecuentemente seleccionados (≥75%): {stats['frequently_selected']}")
        print(f"  - Raramente seleccionados (≤25%): {stats['rarely_selected']}")
        
    return stats

def visualize_frequency_distribution(frequency_stats, plots_dir):
    """Visualize frequency distribution"""
    if frequency_stats is None:
        return
        
    frequencies = frequency_stats['frequencies']
    n_runs = frequency_stats['n_runs']
    
    # Create plots directory
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Frequency histogram
    axes[0,0].hist(frequencies, bins=range(n_runs+2), alpha=0.7, edgecolor='black', align='left')
    axes[0,0].set_xlabel('Selection Frequency', fontsize=16)
    axes[0,0].set_ylabel('Number of Attributes',fontsize=16)
    #axes[0,0].set_title('Selection Frequency Distribution')
    axes[0,0].set_xticks(range(0, n_runs+1, 2))  # Show every 2nd tick (0, 2, 4, 6, ...)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axvline(frequency_stats['mean_frequency'], color='red', linestyle='--', label=f'Mean: {frequency_stats["mean_frequency"]:.1f}')
    
    # Add interpretive annotations
    never_selected = frequency_stats.get('never_selected', np.sum(frequencies == 0))
    always_selected = frequency_stats.get('always_selected', np.sum(frequencies == n_runs))
    
    # Add text annotations for key insights
    max_height = max(np.histogram(frequencies, bins=range(n_runs+2))[0])
    if never_selected > 0:
        axes[0,0].annotate(f'{never_selected} attributes\nnever selected', 
                          xy=(0, np.sum(frequencies == 0)), 
                          xytext=(5, max_height*0.8),
                          arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                          fontsize=9, ha='center', color='red')
    
    if always_selected > 0:
        axes[0,0].annotate(f'{always_selected} attributes\nalways selected', 
                          xy=(n_runs, np.sum(frequencies == n_runs)), 
                          xytext=(n_runs-5, max_height*0.8),
                          arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                          fontsize=9, ha='center', color='green')
    
    axes[0,0].legend()
    
    # 2. Top 20 most frequent attributes
    top_indices = np.argsort(frequencies)[-20:]
    axes[0,1].barh(range(20), frequencies[top_indices])
    axes[0,1].set_xlabel('Selection Frequency')
    axes[0,1].set_ylabel('Attributes (Top 20)')
    axes[0,1].set_title('Top 20 Most Selected Attributes')
    axes[0,1].set_yticks(range(20))
    axes[0,1].set_yticklabels([f'Attr_{i}' for i in top_indices])
    axes[0,1].set_xticks(range(0, n_runs+1, 2))  # Force integer ticks on X axis
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Distribution by frequency ranges
    ranges = ['0', '1-25%', '26-50%', '51-75%', '76-99%', '100%']
    counts = [
        np.sum(frequencies == 0),
        np.sum((frequencies > 0) & (frequencies <= n_runs * 0.25)),
        np.sum((frequencies > n_runs * 0.25) & (frequencies <= n_runs * 0.5)),
        np.sum((frequencies > n_runs * 0.5) & (frequencies <= n_runs * 0.75)),
        np.sum((frequencies > n_runs * 0.75) & (frequencies < n_runs)),
        np.sum(frequencies == n_runs)
    ]
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    axes[1,0].bar(ranges, counts, color=colors, alpha=0.7)
    axes[1,0].set_xlabel('Frequency Range (%)')
    axes[1,0].set_ylabel('Number of Attributes')
    axes[1,0].set_title('Distribution by Frequency Ranges')
    axes[1,0].grid(True, alpha=0.3)
    
    # Add values above bars
    for i, count in enumerate(counts):
        if count > 0:
            axes[1,0].text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    # 4. Sorted frequencies (ranking plot)
    sorted_frequencies = np.sort(frequencies)[::-1]  # Descending
    axes[1,1].plot(range(len(sorted_frequencies)), sorted_frequencies, 'b-', alpha=0.7)
    axes[1,1].set_xlabel('Attribute Ranking')
    axes[1,1].set_ylabel('Selection Frequency')
    axes[1,1].set_title('Sorted Frequencies (Ranking)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(n_runs * 0.5, color='red', linestyle='--', label='50%')
    axes[1,1].axhline(n_runs * 0.75, color='green', linestyle='--', label='75%')
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save complete figure
    complete_path = plots_dir / "frequency_analysis_complete.png"
    plt.savefig(complete_path, dpi=300, bbox_inches='tight')
    if verbose:
        print(f"💾 Complete analysis saved: {complete_path}")
    
    # Show the complete figure
    plt.show()
    
    # Now create and save individual plots
    print(f"📊 Saving individual plots to: {plots_dir}")
    
    # Individual plot 1: Frequency histogram
    fig1 = plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=range(n_runs+2), alpha=0.7, edgecolor='black', align='left')
    plt.xlabel('Selection Frequency (out of 20 runs)')
    plt.ylabel('Number of Attributes')
    plt.title('Selection Frequency Distribution')
    plt.xticks(range(0, n_runs+1, 2))  # Show every 2nd tick (0, 2, 4, 6, ...)
    plt.grid(True, alpha=0.3)
    plt.axvline(frequency_stats['mean_frequency'], color='red', linestyle='--', label=f'Mean: {frequency_stats["mean_frequency"]:.1f}')
    plt.legend()
    hist_path = plots_dir / "frequency_histogram.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Histogram: {hist_path}")
    
    # Individual plot 2: Top 20 attributes
    fig2 = plt.figure(figsize=(10, 8))
    plt.barh(range(20), frequencies[top_indices])
    plt.xlabel('Selection Frequency')
    plt.ylabel('Attributes (Top 20)')
    plt.title('Top 20 Most Selected Attributes')
    plt.yticks(range(20), [f'Attr_{i}' for i in top_indices])
    plt.xticks(range(0, n_runs+1, 2))  # Force integer ticks on X axis
    plt.grid(True, alpha=0.3)
    top20_path = plots_dir / "top20_attributes.png"
    plt.savefig(top20_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Top 20: {top20_path}")
    
    # Individual plot 3: Frequency ranges
    fig3 = plt.figure(figsize=(10, 6))
    plt.bar(ranges, counts, color=colors, alpha=0.7)
    plt.xlabel('Frequency Range (%)')
    plt.ylabel('Number of Attributes')
    plt.title('Distribution by Frequency Ranges')
    plt.grid(True, alpha=0.3)
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    # Add interpretive annotations for never/always selected
    never_selected = frequency_stats.get('never_selected', np.sum(frequencies == 0))
    always_selected = frequency_stats.get('always_selected', np.sum(frequencies == n_runs))
    
    if never_selected > 0:
        plt.annotate(f'{never_selected} attributes\nnever selected', 
                    xy=(0, counts[0]), 
                    xytext=(0.5, max(counts)*0.8),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=9, ha='center', color='red')
    
    if always_selected > 0:
        plt.annotate(f'{always_selected} attributes\nalways selected', 
                    xy=(5, counts[5]), 
                    xytext=(4.5, max(counts)*0.8),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                    fontsize=9, ha='center', color='green')
    
    ranges_path = plots_dir / "frequency_ranges_distribution.png"
    plt.savefig(ranges_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Ranges: {ranges_path}")
    
    # Individual plot 4: Ranking plot
    fig4 = plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_frequencies)), sorted_frequencies, 'b-', alpha=0.7)
    plt.xlabel('Attribute Ranking')
    plt.ylabel('Selection Frequency')
    plt.title('Sorted Frequencies (Ranking)')
    plt.grid(True, alpha=0.3)
    plt.axhline(n_runs * 0.5, color='red', linestyle='--', label='50%')
    plt.axhline(n_runs * 0.75, color='green', linestyle='--', label='75%')
    plt.legend()
    ranking_path = plots_dir / "frequency_ranking.png"
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Ranking: {ranking_path}")
        
        print(f"✅ All frequency plots saved successfully!")



# PARTE 1.5: TESTS BINOMIALES CON CORRECCIÓN BONFERRONI
# ===================================================
def perform_binomial_tests(frequency_results, alpha=0.05):
    """
    Realizar tests binomiales para determinar significancia estadística
    
    Args:
        frequency_results: Resultados del análisis de frecuencias
        alpha: Nivel de significancia (default: 0.05)
        
    Returns:
        dict: Resultados de los tests estadísticos
    """
    print("🧮 PARTE 2: Realizando tests binomiales con corrección Bonferroni...")
    
    if frequency_results is None:
        print("❌ No hay resultados de frecuencias disponibles")
        return None
    
    frequencies = frequency_results['frequencies']
    n_runs = frequency_results['n_runs']
    n_attributes = frequency_results['n_attributes']
    
    print(f"  - Atributos a testear: {n_attributes}")
    print(f"  - Número de ejecuciones: {n_runs}")
    print(f"  - Nivel de significancia: {alpha}")
    
    # Calcular p-valores para cada atributo
    p_values = []
    
    for i, freq in enumerate(frequencies):
        # Test binomial: H0: p = 0.5 (selección aleatoria)
        # H1: p > 0.5 (selección sistemática)
        # Usamos test de una cola (greater) porque buscamos selección por encima del azar
        result = binomtest(freq, n_runs, p=0.5, alternative='greater')
        p_values.append(result.pvalue)
    
    p_values = np.array(p_values)
    
    # Aplicar corrección Bonferroni
    print("🔧 Aplicando corrección Bonferroni...")
    rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, 
        alpha=alpha, 
        method='bonferroni'
    )
    
    # Calcular estadísticas
    n_significant = np.sum(rejected)
    significant_indices = np.where(rejected)[0]
    
    print(f"✅ Tests completados:")
    print(f"  - Atributos significativos: {n_significant}/{n_attributes} ({n_significant/n_attributes*100:.1f}%)")
    print(f"  - Alpha corregido (Bonferroni): {alpha_bonf:.6f}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'attribute_id': range(n_attributes),
        'selection_frequency': frequencies,
        'selection_rate': frequencies / n_runs,
        'p_value_original': p_values,
        'p_value_bonferroni': p_adjusted,
        'is_significant': rejected
    })
    
    # Ordenar por frecuencia descendente
    results_df = results_df.sort_values('selection_frequency', ascending=False)
    
    print(f"\n📊 TOP 10 ATRIBUTOS MÁS SIGNIFICATIVOS:")
    print(results_df.head(10)[['attribute_id', 'selection_frequency', 'selection_rate', 'p_value_bonferroni', 'is_significant']].to_string(index=False))
    
    return {
        'results_df': results_df,
        'n_significant': n_significant,
        'significant_indices': significant_indices,
        'alpha_original': alpha,
        'alpha_bonferroni': alpha_bonf,
        'p_values_original': p_values,
        'p_values_adjusted': p_adjusted
    }
    
    
    
# PARTE 2: ANÁLISIS DE CONVERGENCIA DE FITNESS POBLACIONAL
# ========================================================
def analyze_fitness_convergence(convergence_data, verbose = True):
    """
    Analizar convergencia de fitness a lo largo de las generaciones
    
    Args:
        convergence_data: Datos cargados de convergencia
        
    Returns:
        dict: Resultados del análisis de convergencia
    """
    if verbose:
        print("Analizando convergencia de fitness poblacional...")
    
    if convergence_data is None:
        print("❌ No hay datos de convergencia disponibles")
        return None
    
    all_statistics = convergence_data['all_statistics']
    successful_runs = convergence_data['successful_runs']
    
    # Filtrar ejecuciones con estadísticas válidas
    valid_stats = [stats for stats in all_statistics if stats is not None]
    
    if len(valid_stats) == 0:
        print("❌ No hay estadísticas de fitness disponibles")
        return None
    
    if verbose:
        print(f"  - Ejecuciones con estadísticas: {len(valid_stats)}/{successful_runs}")
    
    # Extraer fitness por generación
    fitness_histories = []
    
    for i, stats_df in enumerate(valid_stats):
        # Buscar columna de fitness (pueden tener nombres diferentes)
        fitness_col = None
        for col in ['max', 'Max', 'best', 'Best', 'fitness', 'Fitness']:
            if col in stats_df.columns:
                fitness_col = col
                break
        
        if fitness_col is None:
            print(f"⚠️  Run {i+1}: No se encontró columna de fitness")
            continue
            
        fitness_history = stats_df[fitness_col].values
        fitness_histories.append(fitness_history)
    
    if len(fitness_histories) == 0:
        print("❌ No se pudieron extraer historias de fitness")
        return None
    
    # Determinar longitud común (mínima)
    min_generations = min(len(history) for history in fitness_histories)
    
    if verbose: 
        print(f"  - Generaciones analizadas: {min_generations}")
    
    # Truncar todas las historias a la longitud común
    fitness_histories = [history[:min_generations] for history in fitness_histories]
    
    # Convertir a array numpy
    fitness_matrix = np.array(fitness_histories)  # Shape: (n_runs, n_generations)
    
    # Calcular estadísticas por generación
    mean_fitness = np.mean(fitness_matrix, axis=0)
    std_fitness = np.std(fitness_matrix, axis=0)
    min_fitness = np.min(fitness_matrix, axis=0)
    max_fitness = np.max(fitness_matrix, axis=0)
    
    # Calcular métricas de convergencia
    final_fitness = fitness_matrix[:, -1]
    initial_fitness = fitness_matrix[:, 0]
    improvement = final_fitness - initial_fitness
    
    # Detectar generación de convergencia (cuando mejora < 1% del mejor fitness)
    convergence_generations = []
    for run_fitness in fitness_matrix:
        best_fitness = np.max(run_fitness)
        threshold = best_fitness * 0.01  # 1% del mejor fitness
        
        # Buscar última mejora significativa
        for gen in range(len(run_fitness)-1, 0, -1):
            if run_fitness[gen] - run_fitness[gen-1] > threshold:
                convergence_generations.append(gen + 1)
                break
        else:
            convergence_generations.append(0)  # Convergió desde el inicio
    
    mean_convergence_gen = np.mean(convergence_generations)
    
    if verbose:
        print(f"✅ Análisis de convergencia completado:")
        print(f"  - Fitness inicial medio: {np.mean(initial_fitness):.4f} ± {np.std(initial_fitness):.4f}")
        print(f"  - Fitness final medio: {np.mean(final_fitness):.4f} ± {np.std(final_fitness):.4f}")
        print(f"  - Mejora promedio: {np.mean(improvement):.4f}")
        print(f"  - Generación de convergencia media: {mean_convergence_gen:.1f}")
        
    return {
        'fitness_matrix': fitness_matrix,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'min_fitness': min_fitness,
        'max_fitness': max_fitness,
        'final_fitness': final_fitness,
        'initial_fitness': initial_fitness,
        'improvement': improvement,
        'convergence_generations': convergence_generations,
        'mean_convergence_gen': mean_convergence_gen,
        'n_generations': min_generations,
        'n_valid_runs': len(fitness_histories)
    }
    
def visualize_fitness_convergence(fitness_results, plots_dir, verbose = True):
    """Visualizar resultados de convergencia de fitness"""
    if fitness_results is None:
        return
        
    fitness_matrix = fitness_results['fitness_matrix']
    mean_fitness = fitness_results['mean_fitness']
    std_fitness = fitness_results['std_fitness']
    generations = range(fitness_results['n_generations'])
    
    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Curvas de convergencia individuales + promedio
    for i, run_fitness in enumerate(fitness_matrix):
        axes[0,0].plot(generations, run_fitness, alpha=0.3, color='blue', linewidth=0.5)
    
    axes[0,0].plot(generations, mean_fitness, 'red', linewidth=2, label='Mean')
    axes[0,0].fill_between(generations, 
                          mean_fitness - std_fitness, 
                          mean_fitness + std_fitness, 
                          alpha=0.2, color='red', label='±1 SD')
    axes[0,0].set_xlabel('Generation')
    axes[0,0].set_ylabel('Fitness')
    axes[0,0].set_title(f'Fitness Convergence ({fitness_results["n_valid_runs"]} runs)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Distribución de fitness final
    axes[0,1].hist(fitness_results['final_fitness'], bins=15, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(np.mean(fitness_results['final_fitness']), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(fitness_results["final_fitness"]):.4f}')
    axes[0,1].set_xlabel('Final Fitness')
    axes[0,1].set_ylabel('Number of Runs')
    axes[0,1].set_title('Final Fitness Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Mejora por ejecución
    run_numbers = range(1, len(fitness_results['improvement']) + 1)  # Start from 1
    axes[1,0].bar(run_numbers, fitness_results['improvement'])
    axes[1,0].axhline(np.mean(fitness_results['improvement']), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(fitness_results["improvement"]):.4f}')
    axes[1,0].set_xlabel('Run Number')
    axes[1,0].set_ylabel('Fitness Improvement (Final - Initial)')
    axes[1,0].set_title('Fitness Improvement per Run')
    axes[1,0].set_xticks(run_numbers)  # Force integer ticks on X axis
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Generaciones de convergencia
    axes[1,1].hist(fitness_results['convergence_generations'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(fitness_results['mean_convergence_gen'], color='red', linestyle='--', 
                     label=f'Mean: {fitness_results["mean_convergence_gen"]:.1f}')
    axes[1,1].set_xlabel('Convergence Generation')
    axes[1,1].set_ylabel('Number of Runs')
    axes[1,1].set_title('Convergence Generation Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save complete figure
    complete_path = plots_dir / "fitness_convergence_complete.png"
    plt.savefig(complete_path, dpi=300, bbox_inches='tight')
    print(f"💾 Complete fitness analysis saved: {complete_path}")
    
    # Show the complete figure
    plt.show()
    
    # Now create and save individual plots
    print(f"📊 Saving individual fitness plots to: {plots_dir}")
    
    # Individual plot 1: Convergence curves
    fig1 = plt.figure(figsize=(10, 6))
    for i, run_fitness in enumerate(fitness_matrix):
        plt.plot(generations, run_fitness, alpha=0.3, color='blue', linewidth=0.5)
    
    plt.plot(generations, mean_fitness, 'red', linewidth=2, label='Mean')
    plt.fill_between(generations, 
                    mean_fitness - std_fitness, 
                    mean_fitness + std_fitness, 
                    alpha=0.2, color='red', label='±1 SD')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Convergence ({fitness_results["n_valid_runs"]} runs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    convergence_path = plots_dir / "fitness_convergence_curves.png"
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Convergence curves: {convergence_path}")
    
    # Individual plot 2: Final fitness distribution
    fig2 = plt.figure(figsize=(10, 6))
    plt.hist(fitness_results['final_fitness'], bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(fitness_results['final_fitness']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(fitness_results["final_fitness"]):.4f}')
    plt.xlabel('Final Fitness')
    plt.ylabel('Number of Runs')
    plt.title('Final Fitness Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    distribution_path = plots_dir / "final_fitness_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Final fitness distribution: {distribution_path}")
    
    # Individual plot 3: Improvement per run
    fig3 = plt.figure(figsize=(10, 6))
    run_numbers = range(1, len(fitness_results['improvement']) + 1)  # Start from 1
    plt.bar(run_numbers, fitness_results['improvement'])
    plt.axhline(np.mean(fitness_results['improvement']), color='red', linestyle='--', 
               label=f'Mean: {np.mean(fitness_results["improvement"]):.4f}')
    plt.xlabel('Run Number')
    plt.ylabel('Fitness Improvement (Final - Initial)')
    plt.title('Fitness Improvement per Run')
    plt.xticks(run_numbers)  # Force integer ticks on X axis
    plt.legend()
    plt.grid(True, alpha=0.3)
    improvement_path = plots_dir / "fitness_improvement_per_run.png"
    plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Fitness improvement per run: {improvement_path}")
    
    # Individual plot 4: Convergence generations
    fig4 = plt.figure(figsize=(10, 6))
    plt.hist(fitness_results['convergence_generations'], bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(fitness_results['mean_convergence_gen'], color='red', linestyle='--', 
               label=f'Mean: {fitness_results["mean_convergence_gen"]:.1f}')
    plt.xlabel('Convergence Generation')
    plt.ylabel('Number of Runs')
    plt.title('Convergence Generation Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    conv_gen_path = plots_dir / "convergence_generations_distribution.png"
    plt.savefig(conv_gen_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"   ✅ Convergence generations: {conv_gen_path}")
        
        print(f"✅ All fitness convergence plots saved successfully!")



# PARTE 3: FUNCIONES DE SOPORTE PARA ANÁLISIS DE DIVERSIDAD
# =========================================================
def calculate_jaccard_diversity(population):
    """
    Calcular diversidad promedio de una población usando distancia Jaccard
    
    Args:
        population: Lista de vectores binarios (máscaras)
        
    Returns:
        float: Diversidad promedio (0 = idénticos, 1 = máxima diversidad)
    """
    if len(population) < 2:
        return 0.0
    
    population = np.array(population)
    n_individuals = len(population)
    
    total_distance = 0.0
    n_pairs = 0
    
    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            # Calcular intersección y unión
            intersection = np.sum(population[i] & population[j])
            union = np.sum(population[i] | population[j])
            
            if union == 0:
                # Ambos individuos son vectores cero
                jaccard_sim = 1.0
            else:
                jaccard_sim = intersection / union
            
            # Convertir similitud a distancia
            jaccard_dist = 1.0 - jaccard_sim
            total_distance += jaccard_dist
            n_pairs += 1
    
    return total_distance / n_pairs if n_pairs > 0 else 0.0

def get_population_stats(population):
    """
    Obtener estadísticas básicas de una población
    
    Args:
        population: Lista de vectores binarios
        
    Returns:
        dict: Estadísticas de la población
    """
    if len(population) == 0:
        return None
    
    population = np.array(population)
    
    # Estadísticas básicas
    population_size = len(population)
    n_attributes = len(population[0])
    
    # Número de atributos seleccionados por individuo
    features_per_individual = np.sum(population, axis=1)
    
    # Frecuencias de selección de atributos en la población
    attribute_frequencies = np.sum(population, axis=0)
    
    return {
        'population_size': population_size,
        'n_attributes': n_attributes,
        'mean_features_per_individual': np.mean(features_per_individual),
        'std_features_per_individual': np.std(features_per_individual),
        'min_features': np.min(features_per_individual),
        'max_features': np.max(features_per_individual),
        'attribute_frequencies': attribute_frequencies,
        'most_selected_attributes': np.argsort(attribute_frequencies)[-10:]  # Top 10
    }

def calculate_hamming_diversity(population):
    """
    Calcular diversidad promedio de una población usando distancia Hamming

    Args:
        population: Lista de vectores binarios (máscaras)

    Returns:
        float: Diversidad promedio (0 = idénticos, 1 = máxima diversidad)
    """
    if len(population) < 2:
        return 0.0

    population = np.array(population)
    n_individuals = len(population)
    n_attributes = population.shape[1]

    total_distance = 0.0
    n_pairs = 0

    for i in range(n_individuals):
        for j in range(i + 1, n_individuals):
            hamming_dist = np.sum(population[i] != population[j]) / n_attributes
            total_distance += hamming_dist
            n_pairs += 1

    return total_distance / n_pairs if n_pairs > 0 else 0.0


# --- NIVEL 1: DIVERSIDAD ENTRE MEJORES INDIVIDUOS DE DIFERENTES EJECUCIONES ---
def analyze_best_individuals_diversity(convergence_data, verbose = True):
    """
    Nivel 1: Analizar diversidad entre los mejores individuos de diferentes ejecuciones
    
    Args:
        convergence_data: Datos de convergencia cargados
        
    Returns:
        dict: Resultados del análisis de diversidad entre mejores individuos
    """
    if verbose:
        print("NIVEL 1: Analizando diversidad entre mejores individuos...")
    
    if convergence_data is None:
        print("❌ No hay datos de convergencia disponibles")
        return None
    
    best_individuals = convergence_data['best_individuals']
    n_runs = convergence_data['successful_runs']
    n_attributes = convergence_data['n_attributes']
    
    if len(best_individuals) < 2:
        print("❌ Se necesitan al menos 2 individuos para calcular diversidad")
        return None
    if verbose:
        print(f"  - Analizando {n_runs} mejores individuos")
        print(f"  - Cada individuo tiene {n_attributes} atributos")
    
    # Calcular diversidad poblacional
    diversity_score = calculate_hamming_diversity(best_individuals)
    #diversity_score = calculate_jaccard_diversity(best_individuals)
    similarity_score = 1.0 - diversity_score
    
    # Calcular matriz de distancias entre todos los pares
    distance_matrix = np.zeros((n_runs, n_runs))
    similarity_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(n_runs):
            if i != j:
                # Calcular distancia Hamming normalizada
                hamming_dist = np.sum(np.array(best_individuals[i]) != np.array(best_individuals[j])) / len(best_individuals[i])
                hamming_sim = 1.0 - hamming_dist
                
                similarity_matrix[i, j] = hamming_sim
                distance_matrix[i, j] = hamming_dist
                
                

            
            else:
                similarity_matrix[i, j] = 1.0  # Un individuo es idéntico a sí mismo
                distance_matrix[i, j] = 0.0
    
    # Estadísticas de las distancias
    upper_triangle = distance_matrix[np.triu_indices(n_runs, k=1)]  # Solo triángulo superior
    
    mean_distance = np.mean(upper_triangle)
    std_distance = np.std(upper_triangle)
    min_distance = np.min(upper_triangle)
    max_distance = np.max(upper_triangle)
    
    # Encontrar pares más similares y más diferentes
    min_idx = np.unravel_index(np.argmin(distance_matrix + np.eye(n_runs) * 10), distance_matrix.shape)
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    
    # Análisis de similitud por atributos
    best_matrix = np.array(best_individuals)
    attribute_consensus = np.mean(best_matrix, axis=0)  # Frecuencia de selección por atributo
    high_consensus_attrs = np.where(attribute_consensus >= 0.8)[0]  # Seleccionados en ≥80% runs
    low_consensus_attrs = np.where(attribute_consensus <= 0.2)[0]   # Seleccionados en ≤20% runs
    variable_attrs = np.where((attribute_consensus > 0.2) & (attribute_consensus < 0.8))[0]
    
    if verbose:
        print(f"✅ Análisis Nivel 1 completado:")
        print(f"  - Diversidad promedio: {diversity_score:.4f}")
        print(f"  - Similitud promedio: {similarity_score:.4f}")
        print(f"  - Distancia mín/máx: {min_distance:.4f} / {max_distance:.4f}")
        print(f"  - Desviación estándar: {std_distance:.4f}")
        print(f"")
        print(f"  📊 CONSENSO DE ATRIBUTOS:")
        print(f"  - Alto consenso (≥80%): {len(high_consensus_attrs)} atributos")
        print(f"  - Bajo consenso (≤20%): {len(low_consensus_attrs)} atributos")
        print(f"  - Consenso variable (20-80%): {len(variable_attrs)} atributos")
    
    # Interpretación automática
    if diversity_score >= 0.7:
        interpretation = "🟢 ALTA DIVERSIDAD - Múltiples soluciones óptimas diferentes"
        recommendation = "El GA encuentra diversos óptimos válidos. Considera analizar qué hace únicas a cada solución."
    elif diversity_score >= 0.3:
        interpretation = "🟡 DIVERSIDAD MODERADA - Familia de soluciones relacionadas"
        recommendation = "Hay patrones comunes pero también diferencias. Analiza atributos de consenso variable."
    else:
        interpretation = "🔴 BAJA DIVERSIDAD - Convergencia a solución similar"
        recommendation = "El GA converge consistentemente. Verifica si esta es realmente la solución óptima."
    
    if verbose:
        print(f"")
        print(f"  🎯 INTERPRETACIÓN: {interpretation}")
        print(f"  💡 RECOMENDACIÓN: {recommendation}")
    
    return {
        'diversity_score': diversity_score,
        'similarity_score': similarity_score,
        'distance_matrix': distance_matrix,
        'similarity_matrix': similarity_matrix,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'most_similar_pair': min_idx,
        'most_different_pair': max_idx,
        'attribute_consensus': attribute_consensus,
        'high_consensus_attrs': high_consensus_attrs,
        'low_consensus_attrs': low_consensus_attrs,
        'variable_attrs': variable_attrs,
        'interpretation': interpretation,
        'recommendation': recommendation,
        'n_runs': n_runs,
        'n_attributes': n_attributes
    }
    
def visualize_best_individuals_diversity(nivel1_results,plots_dir):
    """Visualizar resultados del análisis Nivel 1"""
    if nivel1_results is None:
        return
    
    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Matriz de distancias Hamming
    im1 = axes[0,0].imshow(nivel1_results['distance_matrix'], cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[0,0].set_title('Hamming Distance Matrix\n(Best Individuals)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Run')
    axes[0,0].set_ylabel('Run')
    
    # Añadir valores en cada celda para mayor claridad
    for i in range(nivel1_results['n_runs']):
        for j in range(nivel1_results['n_runs']):
            if i != j:  # No mostrar diagonal
                axes[0,0].text(j, i, f'{nivel1_results["distance_matrix"][i,j]:.2f}', 
                             ha='center', va='center', fontsize=8)
    
    plt.colorbar(im1, ax=axes[0,0], label='Hamming Distance')
    
    # 2. Distance distribution
    upper_triangle = nivel1_results['distance_matrix'][np.triu_indices(nivel1_results['n_runs'], k=1)]
    axes[0,1].hist(upper_triangle, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0,1].axvline(nivel1_results['mean_distance'], color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {nivel1_results["mean_distance"]:.3f}')
    axes[0,1].set_xlabel('Hamming Distance')
    axes[0,1].set_ylabel('Number of Pairs')
    axes[0,1].set_title('Distance Distribution\nBetween Best Individuals', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Attribute consensus
    attribute_consensus = nivel1_results['attribute_consensus']
    axes[1,0].plot(range(len(attribute_consensus)), attribute_consensus, 
                   'b-', alpha=0.7, linewidth=1)
    axes[1,0].axhline(0.8, color='green', linestyle='--', alpha=0.8, label='High consensus (≥80%)')
    axes[1,0].axhline(0.2, color='red', linestyle='--', alpha=0.8, label='Low consensus (≤20%)')
    axes[1,0].fill_between(range(len(attribute_consensus)), 0.2, 0.8, alpha=0.1, color='orange', 
                          label='Variable consensus')
    axes[1,0].set_xlabel('Attribute Index')
    axes[1,0].set_ylabel('Selection Frequency')
    axes[1,0].set_title('Attribute Consensus\nAmong Best Individuals', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(0, 1)
    
    # 4. Key metrics summary
    axes[1,1].axis('off')
    
    # Create summary text
    summary_text = [
        f"📊 LEVEL 1 SUMMARY - DIVERSITY BETWEEN BEST INDIVIDUALS",
        f"",
        f"🎯 Main Metrics:",
        f"  • Average diversity: {nivel1_results['diversity_score']:.4f}",
        f"  • Average similarity: {nivel1_results['similarity_score']:.4f}",
        f"  • Distance range: [{nivel1_results['min_distance']:.3f} - {nivel1_results['max_distance']:.3f}]",
        f"",
        f"🔍 Attribute Consensus:",
        f"  • High consensus (≥80%): {len(nivel1_results['high_consensus_attrs'])} attributes",
        f"  • Low consensus (≤20%): {len(nivel1_results['low_consensus_attrs'])} attributes", 
        f"  • Variable consensus: {len(nivel1_results['variable_attrs'])} attributes",
        f"",
        f"🎯 Interpretation:",
        f"{nivel1_results['interpretation']}",
        f"",
        f"💡 Recommendation:",
        f"{nivel1_results['recommendation'][:60]}..."
    ]
    
    axes[1,1].text(0.05, 0.95, '\n'.join(summary_text), 
                   transform=axes[1,1].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save complete figure
    complete_path = plots_dir / "best_individuals_diversity_complete.png"
    plt.savefig(complete_path, dpi=300, bbox_inches='tight')
    print(f"💾 Complete diversity analysis saved: {complete_path}")
    
    # Show the complete figure
    plt.show()
    
    # Now create and save individual plots
    print(f"📊 Saving individual diversity plots to: {plots_dir}")
    
    # Individual plot 1: Distance matrix
    fig1 = plt.figure(figsize=(10, 8))
    im = plt.imshow(nivel1_results['distance_matrix'], cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.title('Hamming Distance Matrix\n(Best Individuals)', fontsize=14, fontweight='bold')
    plt.xlabel('Run Number')
    plt.ylabel('Run Number')
    
    # Add values in each cell
    for i in range(nivel1_results['n_runs']):
        for j in range(nivel1_results['n_runs']):
            if i != j:  # Don't show diagonal
                plt.text(j, i, f'{nivel1_results["distance_matrix"][i,j]:.2f}', 
                        ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, label='Hamming Distance')
    matrix_path = plots_dir / "hamming_distance_matrix.png"
    plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Distance matrix: {matrix_path}")
    
    # Individual plot 2: Distance distribution
    fig2 = plt.figure(figsize=(10, 6))
    upper_triangle = nivel1_results['distance_matrix'][np.triu_indices(nivel1_results['n_runs'], k=1)]
    plt.hist(upper_triangle, bins=15, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(nivel1_results['mean_distance'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nivel1_results["mean_distance"]:.3f}')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Number of Pairs')
    plt.title('Distance Distribution\n(Between Best Individuals)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    distribution_path = plots_dir / "distance_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Distance distribution: {distribution_path}")
    
    # Individual plot 3: Attribute consensus
    fig3 = plt.figure(figsize=(12, 6))
    attribute_consensus = nivel1_results['attribute_consensus']
    plt.plot(range(len(attribute_consensus)), attribute_consensus, 
            'b-', alpha=0.7, linewidth=1)
    plt.axhline(0.8, color='green', linestyle='--', alpha=0.8, label='High consensus (≥80%)')
    plt.axhline(0.2, color='red', linestyle='--', alpha=0.8, label='Low consensus (≤20%)')
    plt.fill_between(range(len(attribute_consensus)), 0.2, 0.8, alpha=0.1, color='orange', 
                    label='Variable consensus')
    plt.xlabel('Attribute Index')
    plt.ylabel('Selection Frequency')
    plt.title('Attribute Consensus\n(Among Best Individuals)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    consensus_path = plots_dir / "attribute_consensus.png"
    plt.savefig(consensus_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Attribute consensus: {consensus_path}")
    
    # Individual plot 4: Summary metrics (text-based visualization)
    fig4 = plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    summary_text_individual = [
        f"📊 DIVERSITY ANALYSIS - LEVEL 1",
        f"Best Individuals Across {nivel1_results['n_runs']} Independent Runs",
        f"",
        f"🎯 Main Metrics:",
        f"  • Average diversity: {nivel1_results['diversity_score']:.4f}",
        f"  • Average similarity: {nivel1_results['similarity_score']:.4f}",
        f"  • Distance range: [{nivel1_results['min_distance']:.3f} - {nivel1_results['max_distance']:.3f}]",
        f"  • Standard deviation: {nivel1_results['std_distance']:.4f}",
        f"",
        f"🔍 Attribute Consensus:",
        f"  • High consensus (≥80%): {len(nivel1_results['high_consensus_attrs'])} attributes",
        f"  • Low consensus (≤20%): {len(nivel1_results['low_consensus_attrs'])} attributes", 
        f"  • Variable consensus: {len(nivel1_results['variable_attrs'])} attributes",
        f"  • Total attributes: {len(attribute_consensus)} attributes",
        f"",
        f"🎯 Interpretation:",
        f"{nivel1_results['interpretation']}",
        f"",
        f"💡 Recommendation:",
        f"{nivel1_results['recommendation']}"
    ]
    
    plt.text(0.05, 0.95, '\n'.join(summary_text_individual), 
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    summary_path = plots_dir / "diversity_summary_metrics.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    if verbose:
        print(f"   ✅ Summary metrics: {summary_path}")
        
        print(f"✅ All diversity plots saved successfully!")
 
    
# --- NIVEL 2: DIVERSIDAD DENTRO DE ÚLTIMA GENERACIÓN DE CADA EJECUCIÓN
def analyze_final_generation_diversity(convergence_data, 
                                       dataset, 
                                       base_dir='logs', 
                                       algorithm="eaMuPlusLambda",
                                       verbose=True):
    """
    Nivel 2: Analizar diversidad dentro de la última generación (gen_300) de cada ejecución
    
    Args:
        convergence_data: Datos de convergencia
        dataset: Nombre del dataset
        base_dir: Directorio base
        algorithm: Algoritmo usado
        
    Returns:
        dict: Resultados del análisis de diversidad por generación final
    """
    if verbose:
        print("NIVEL 2: Analizando diversidad en generaciones finales...")
    
    if convergence_data is None:
        print("❌ No hay datos de convergencia disponibles")
        return None

    convergence_dir = Path(base_dir) / dataset / "convergence" / algorithm
    ga_run_dirs = sorted(list(convergence_dir.glob("ga_run_*")))
    
    if verbose:
        print(f"  - Analizando última generación (gen_300) de {len(ga_run_dirs)} ejecuciones")
    
    run_diversities = []
    run_stats = []
    successful_analyses = 0
    
    for i, run_dir in enumerate(ga_run_dirs, 1):
        
        if verbose:
            print(f"  [{i:2d}] {run_dir.name}...", end=" ")
        
        # Cargar población de la última generación (gen_300)
        final_population = load_generation_population(run_dir, 299)
        
        if final_population is None:
            print("❌ Sin gen_300.json")
            continue
        
        if len(final_population) < 2:
            print("❌ Población muy pequeña")
            continue
        
        # Calcular diversidad de esta población
        diversity = calculate_hamming_diversity(final_population)
        run_diversities.append(diversity)
        
        # Obtener estadísticas de la población
        pop_stats = get_population_stats(final_population)
        pop_stats['run_index'] = i - 1
        pop_stats['diversity'] = diversity
        run_stats.append(pop_stats)
        
        successful_analyses += 1
        if verbose:
            print(f"✅ (div: {diversity:.3f}, pop: {len(final_population)})")
    
    if successful_analyses == 0:
        print("❌ No se pudieron analizar generaciones finales")
        return None
    
    # Calcular estadísticas agregadas
    run_diversities = np.array(run_diversities)
    
    mean_diversity = np.mean(run_diversities)
    std_diversity = np.std(run_diversities)
    min_diversity = np.min(run_diversities)
    max_diversity = np.max(run_diversities)
    
    # Estadísticas de tamaño de población
    population_sizes = [stats['population_size'] for stats in run_stats]
    mean_pop_size = np.mean(population_sizes)
    
    # Estadísticas de características por individuo
    mean_features_per_run = [stats['mean_features_per_individual'] for stats in run_stats]
    overall_mean_features = np.mean(mean_features_per_run)
    
    if verbose:
        print(f"")
        print(f"✅ Análisis Nivel 2 completado:")
        print(f"  - Ejecuciones analizadas: {successful_analyses}")
        print(f"  - Diversidad promedio: {mean_diversity:.4f} ± {std_diversity:.4f}")
        print(f"  - Rango de diversidad: [{min_diversity:.4f} - {max_diversity:.4f}]")
        print(f"  - Tamaño promedio de población: {mean_pop_size:.1f}")
        print(f"  - Características promedio por individuo: {overall_mean_features:.1f}")
    
    # Identificar ejecuciones con diversidad extrema
    high_diversity_runs = np.where(run_diversities >= np.percentile(run_diversities, 75))[0]
    low_diversity_runs = np.where(run_diversities <= np.percentile(run_diversities, 25))[0]
    
    if verbose:
        print(f"")
        print(f"  📊 ANÁLISIS POR CUARTILES:")
        print(f"  - Runs con alta diversidad (Q4): {len(high_diversity_runs)} ejecuciones")
        print(f"  - Runs con baja diversidad (Q1): {len(low_diversity_runs)} ejecuciones")
    
    # Interpretación automática
    if mean_diversity >= 0.6:
        interpretation = "🟢 ALTA DIVERSIDAD FINAL - Población mantiene exploración"
        recommendation = "Excelente! El GA mantiene diversidad hasta el final. Considera aumentar generaciones."
    elif mean_diversity >= 0.3:
        interpretation = "🟡 DIVERSIDAD MODERADA - Balance exploración/explotación"
        recommendation = "Buen equilibrio. El GA converge pero mantiene cierta diversidad exploratoria."
    else:
        interpretation = "🔴 BAJA DIVERSIDAD FINAL - Posible convergencia prematura"
        recommendation = "¡Cuidado! Población muy homogénea. Considera ajustar operadores de mutación/crossover."
    
    # Análisis de consistencia entre runs
    diversity_cv = std_diversity / mean_diversity if mean_diversity > 0 else 0
    if diversity_cv <= 0.2:
        consistency = "🟢 ALTA CONSISTENCIA - Comportamiento uniforme entre runs"
    elif diversity_cv <= 0.5:
        consistency = "🟡 CONSISTENCIA MODERADA - Alguna variabilidad entre runs"
    else:
        consistency = "🔴 BAJA CONSISTENCIA - Gran variabilidad entre runs"
    
    if verbose:
        print(f"")
        print(f"  🎯 INTERPRETACIÓN: {interpretation}")
        print(f"  🔄 CONSISTENCIA: {consistency}")
        print(f"  💡 RECOMENDACIÓN: {recommendation}")
    
    return {
        'run_diversities': run_diversities,
        'run_stats': run_stats,
        'mean_diversity': mean_diversity,
        'std_diversity': std_diversity,
        'min_diversity': min_diversity,
        'max_diversity': max_diversity,
        'diversity_cv': diversity_cv,
        'high_diversity_runs': high_diversity_runs,
        'low_diversity_runs': low_diversity_runs,
        'mean_pop_size': mean_pop_size,
        'overall_mean_features': overall_mean_features,
        'successful_analyses': successful_analyses,
        'interpretation': interpretation,
        'consistency': consistency,
        'recommendation': recommendation
    }

def visualize_final_generation_diversity(nivel2_results, plots_dir,verbose=True):
    """Visualizar resultados del análisis Nivel 2"""
    if nivel2_results is None:
        return
    
    # Create plots directory
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    run_diversities = nivel2_results['run_diversities']
    run_stats = nivel2_results['run_stats']
    
    # 1. Distribución de diversidades por ejecución
    axes[0,0].hist(run_diversities, bins=15, alpha=0.7, edgecolor='black', color='lightblue')
    axes[0,0].axvline(nivel2_results['mean_diversity'], color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {nivel2_results["mean_diversity"]:.3f}')
    axes[0,0].axvline(nivel2_results['mean_diversity'] - nivel2_results['std_diversity'], 
                     color='orange', linestyle=':', alpha=0.8, label='±1 SD')
    axes[0,0].axvline(nivel2_results['mean_diversity'] + nivel2_results['std_diversity'], 
                     color='orange', linestyle=':', alpha=0.8)
    axes[0,0].set_xlabel('Diversity in Final Generation')
    axes[0,0].set_ylabel('Number of Runs')
    axes[0,0].set_title('Diversity Distribution\nin Final Generations', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Diversidad por ejecución (serie temporal)
    run_indices = range(1, len(run_diversities) + 1)  # Start from 1 for better readability
    axes[0,1].plot(run_indices, run_diversities, 'o-', alpha=0.7, markersize=6)
    axes[0,1].axhline(nivel2_results['mean_diversity'], color='red', linestyle='--', 
                     label='Mean')
    axes[0,1].fill_between(run_indices, 
                          nivel2_results['mean_diversity'] - nivel2_results['std_diversity'],
                          nivel2_results['mean_diversity'] + nivel2_results['std_diversity'],
                          alpha=0.2, color='red', label='±1 SD')
    axes[0,1].set_xlabel('Run Number')
    axes[0,1].set_ylabel('Final Diversity')
    axes[0,1].set_title('Final Diversity per Run', fontweight='bold')
    axes[0,1].set_xticks(range(1, len(run_diversities) + 1, max(1, len(run_diversities)//10)))  # Force integer ticks
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Relación: Diversidad vs Tamaño de Población
    pop_sizes = [stats['population_size'] for stats in run_stats]
    axes[0,2].scatter(pop_sizes, run_diversities, alpha=0.6, s=50)
    axes[0,2].set_xlabel('Final Population Size')
    axes[0,2].set_ylabel('Final Diversity')
    axes[0,2].set_title('Diversity vs Population Size', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    # Calcular correlación
    from scipy.stats import pearsonr
    if len(pop_sizes) > 1:
        corr, p_val = pearsonr(pop_sizes, run_diversities)
        axes[0,2].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                      transform=axes[0,2].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Relación: Diversidad vs Características promedio por individuo
    mean_features = [stats['mean_features_per_individual'] for stats in run_stats]
    axes[1,0].scatter(mean_features, run_diversities, alpha=0.6, s=50, color='green')
    axes[1,0].set_xlabel('Average Features per Individual')
    axes[1,0].set_ylabel('Final Diversity')
    axes[1,0].set_title('Diversity vs Sparsity', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Calcular correlación
    if len(mean_features) > 1:
        corr2, p_val2 = pearsonr(mean_features, run_diversities)
        axes[1,0].text(0.05, 0.95, f'r = {corr2:.3f}\np = {p_val2:.3f}', 
                      transform=axes[1,0].transAxes, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. Comparación de cuartiles
    q1_runs = nivel2_results['low_diversity_runs']
    q4_runs = nivel2_results['high_diversity_runs']
    
    if len(q1_runs) > 0 and len(q4_runs) > 0:
        q1_diversities = run_diversities[q1_runs]
        q4_diversities = run_diversities[q4_runs]
        
        box_data = [q1_diversities, q4_diversities]
        labels = [f'Q1 (Low)\nn={len(q1_runs)}', f'Q4 (High)\nn={len(q4_runs)}']
        
        bp = axes[1,1].boxplot(box_data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        axes[1,1].set_ylabel('Final Diversity')
        axes[1,1].set_title('Quartile Comparison', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient data\nfor quartiles', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Quartile Comparison', fontweight='bold')
    
    # 6. Resumen de métricas
    axes[1,2].axis('off')
    
    summary_text = [
        f"📊 LEVEL 2 SUMMARY - FINAL GENERATION DIVERSITY",
        f"",
        f"🎯 Main Metrics:",
        f"  • Average diversity: {nivel2_results['mean_diversity']:.4f} ± {nivel2_results['std_diversity']:.4f}",
        f"  • Range: [{nivel2_results['min_diversity']:.3f} - {nivel2_results['max_diversity']:.3f}]",
        f"  • Coefficient of variation: {nivel2_results['diversity_cv']:.3f}",
        f"",
        f"📈 Population Statistics:",
        f"  • Average size: {nivel2_results['mean_pop_size']:.1f} individuals",
        f"  • Average features: {nivel2_results['overall_mean_features']:.1f}",
        f"  • Analyzed runs: {nivel2_results['successful_analyses']}",
        f"",
        f"🎯 Interpretation:",
        f"{nivel2_results['interpretation']}",
        f"",
        f"🔄 Consistency:",
        f"{nivel2_results['consistency']}",
        f"",
        f"💡 Recommendation:",
        f"{nivel2_results['recommendation'][:50]}..."
    ]
    
    axes[1,2].text(0.05, 0.95, '\n'.join(summary_text), 
                   transform=axes[1,2].transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Save complete figure
    complete_path = plots_dir / "final_generation_diversity_complete.png"
    plt.savefig(complete_path, dpi=300, bbox_inches='tight')
    print(f"💾 Complete Level 2 analysis saved: {complete_path}")
    
    # Show the complete figure
    plt.show()
    
    # Now create and save individual plots
    print(f"📊 Saving individual Level 2 diversity plots to: {plots_dir}")
    
    # Individual plot 1: Diversity distribution
    fig1 = plt.figure(figsize=(10, 6))
    plt.hist(run_diversities, bins=15, alpha=0.7, edgecolor='black', color='lightblue')
    plt.axvline(nivel2_results['mean_diversity'], color='red', linestyle='--', linewidth=2,
               label=f'Mean: {nivel2_results["mean_diversity"]:.3f}')
    plt.axvline(nivel2_results['mean_diversity'] - nivel2_results['std_diversity'], 
               color='orange', linestyle=':', alpha=0.8, label='±1 SD')
    plt.axvline(nivel2_results['mean_diversity'] + nivel2_results['std_diversity'], 
               color='orange', linestyle=':', alpha=0.8)
    plt.xlabel('Diversity in Final Generation')
    plt.ylabel('Number of Runs')
    plt.title('Diversity Distribution in Final Generations', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    distribution_path = plots_dir / "diversity_distribution.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Diversity distribution: {distribution_path}")
    
    # Individual plot 2: Diversity per run
    fig2 = plt.figure(figsize=(12, 6))
    run_indices = range(1, len(run_diversities) + 1)
    plt.plot(run_indices, run_diversities, 'o-', alpha=0.7, markersize=6)
    plt.axhline(nivel2_results['mean_diversity'], color='red', linestyle='--', label='Mean')
    plt.fill_between(run_indices, 
                    nivel2_results['mean_diversity'] - nivel2_results['std_diversity'],
                    nivel2_results['mean_diversity'] + nivel2_results['std_diversity'],
                    alpha=0.2, color='red', label='±1 SD')
    plt.xlabel('Run Number')
    plt.ylabel('Final Diversity')
    plt.title('Final Diversity per Run', fontweight='bold')
    plt.xticks(range(1, len(run_diversities) + 1, max(1, len(run_diversities)//10)))  # Force integer ticks
    plt.legend()
    plt.grid(True, alpha=0.3)
    per_run_path = plots_dir / "diversity_per_run.png"
    plt.savefig(per_run_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Diversity per run: {per_run_path}")
    
    # Individual plot 3: Diversity vs Population size
    fig3 = plt.figure(figsize=(10, 6))
    pop_sizes = [stats['population_size'] for stats in run_stats]
    plt.scatter(pop_sizes, run_diversities, alpha=0.6, s=50)
    plt.xlabel('Final Population Size')
    plt.ylabel('Final Diversity')
    plt.title('Diversity vs Population Size', fontweight='bold')
    plt.grid(True, alpha=0.3)
    if len(pop_sizes) > 1:
        corr, p_val = pearsonr(pop_sizes, run_diversities)
        plt.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    vs_population_path = plots_dir / "diversity_vs_population.png"
    plt.savefig(vs_population_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Diversity vs population: {vs_population_path}")
    
    # Individual plot 4: Diversity vs Sparsity
    fig4 = plt.figure(figsize=(10, 6))
    mean_features = [stats['mean_features_per_individual'] for stats in run_stats]
    plt.scatter(mean_features, run_diversities, alpha=0.6, s=50, color='green')
    plt.xlabel('Average Features per Individual')
    plt.ylabel('Final Diversity')
    plt.title('Diversity vs Sparsity', fontweight='bold')
    plt.grid(True, alpha=0.3)
    if len(mean_features) > 1:
        corr2, p_val2 = pearsonr(mean_features, run_diversities)
        plt.text(0.05, 0.95, f'r = {corr2:.3f}\np = {p_val2:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    vs_sparsity_path = plots_dir / "diversity_vs_sparsity.png"
    plt.savefig(vs_sparsity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Diversity vs sparsity: {vs_sparsity_path}")
    
    # Individual plot 5: Quartile comparison
    fig5 = plt.figure(figsize=(8, 6))
    q1_runs = nivel2_results['low_diversity_runs']
    q4_runs = nivel2_results['high_diversity_runs']
    
    if len(q1_runs) > 0 and len(q4_runs) > 0:
        q1_diversities = run_diversities[q1_runs]
        q4_diversities = run_diversities[q4_runs]
        
        box_data = [q1_diversities, q4_diversities]
        labels = [f'Q1 (Low)\nn={len(q1_runs)}', f'Q4 (High)\nn={len(q4_runs)}']
        
        bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        plt.ylabel('Final Diversity')
        plt.title('Quartile Comparison', fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor quartiles', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Quartile Comparison', fontweight='bold')
    
    quartiles_path = plots_dir / "quartile_comparison.png"
    plt.savefig(quartiles_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Quartile comparison: {quartiles_path}")
    
    # Individual plot 6: Summary metrics
    fig6 = plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    summary_text_individual = [
        f"📊 DIVERSITY ANALYSIS - LEVEL 2",
        f"Final Generation Diversity Across {nivel2_results['successful_analyses']} Runs",
        f"",
        f"🎯 Main Metrics:",
        f"  • Average diversity: {nivel2_results['mean_diversity']:.4f} ± {nivel2_results['std_diversity']:.4f}",
        f"  • Range: [{nivel2_results['min_diversity']:.3f} - {nivel2_results['max_diversity']:.3f}]",
        f"  • Coefficient of variation: {nivel2_results['diversity_cv']:.3f}",
        f"",
        f"📈 Population Statistics:",
        f"  • Average population size: {nivel2_results['mean_pop_size']:.1f} individuals",
        f"  • Average features per individual: {nivel2_results['overall_mean_features']:.1f}",
        f"  • Successfully analyzed runs: {nivel2_results['successful_analyses']}",
        f"",
        f"🎯 Interpretation:",
        f"{nivel2_results['interpretation']}",
        f"",
        f"🔄 Consistency:",
        f"{nivel2_results['consistency']}",
        f"",
        f"💡 Recommendation:",
        f"{nivel2_results['recommendation']}"
    ]
    
    plt.text(0.05, 0.95, '\n'.join(summary_text_individual), 
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    summary_path = plots_dir / "level2_summary_metrics.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
    if verbose:
        print(f"   ✅ Summary metrics: {summary_path}")
    
        print(f"✅ All Level 2 diversity plots saved successfully!")




# -- Nivel 3: Diversidad entre generaciones

def analyze_temporal_diversity_evolution(dataset_name, logs_path ,verbose = True):
    """
    Analiza la evolución temporal de la diversidad a lo largo de las generaciones
    leyendo los datos reales de los archivos JSON usando distancia Hamming
    
    Args:
        dataset_name: Nombre del dataset
        
    Returns:
        dict: Resultados del análisis temporal de diversidad
    """
    import json
    import glob
    from scipy.stats import pearsonr, spearmanr
    from sklearn.cluster import KMeans
    from scipy.signal import savgol_filter
    
    print("🎯 NIVEL 3: Analyzing temporal diversity evolution...")
    
    # Path to convergence data
    convergence_base_path = logs_path
    
    if not convergence_base_path.exists():
        print(f"⚠️  Convergence data not found at: {convergence_base_path}")
        return None
    
    # Find all run directories
    run_dirs = list(convergence_base_path.glob("ga_run_*"))
    run_dirs = sorted(run_dirs)[:20]  # Take first 20 runs
    
    print(f"   📁 Found {len(run_dirs)} run directories")
    
    if len(run_dirs) == 0:
        print("⚠️  No run directories found")
        return None
    
    # Detect number of generations from first run
    n_runs = len(run_dirs)
    n_generations = 0
    if n_runs > 0:
        first_gen_dir = run_dirs[0] / "generations"
        if first_gen_dir.exists():
            gen_files = sorted(list(first_gen_dir.glob("gen_*.json")))
            n_generations = len(gen_files)
    if n_generations == 0:
        print("❌ No generation files found in first run.")
        return None

    diversity_matrix = np.zeros((n_runs, n_generations))
    successful_runs = 0

    # Process each run
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"   🔄 Processing run {run_idx + 1}/{n_runs}: {run_dir.name}")
        generations_dir = run_dir / "generations"
        if not generations_dir.exists():
            print(f"     ⚠️  No generations directory found")
            continue

        run_diversity = []
        successful_gens = 0

        # Process each generation
        for gen_idx in range(n_generations):
            gen_file = generations_dir / f"gen_{gen_idx:03d}.json"
            if not gen_file.exists():
                run_diversity.append(0.0)
                continue
            try:
                with open(gen_file, 'r') as f:
                    generation_data = json.load(f)
                if not generation_data or len(generation_data) < 2:
                    run_diversity.append(0.0)
                    continue
                n_individuals = len(generation_data)
                total_distance = 0
                pair_count = 0
                for i in range(n_individuals):
                    for j in range(i + 1, n_individuals):
                        try:
                            mask_i = generation_data[i]['mask']
                            mask_j = generation_data[j]['mask']
                            mask_i_array = np.array(mask_i)
                            mask_j_array = np.array(mask_j)
                            hamming_distance = np.sum(mask_i_array != mask_j_array) / len(mask_i_array)
                            total_distance += hamming_distance
                            pair_count += 1
                        except (KeyError, IndexError, TypeError):
                            continue
                if pair_count > 0:
                    avg_diversity = total_distance / pair_count
                    successful_gens += 1
                else:
                    avg_diversity = 0.0
                run_diversity.append(avg_diversity)
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                run_diversity.append(0.0)
                continue
        # Store run diversity
        # Si alguna run tiene menos generaciones, rellena con ceros
        if len(run_diversity) < n_generations:
            run_diversity += [0.0] * (n_generations - len(run_diversity))
        diversity_matrix[run_idx, :] = run_diversity
        if successful_gens > 50:
            successful_runs += 1
        print(f"     ✅ Processed {successful_gens}/{n_generations} generations")
    
    print(f"   ✅ Successfully processed {successful_runs}/{n_runs} runs")
    
    if successful_runs == 0:
        print("⚠️  No runs with sufficient data found")
        return None
    
    # 1. ESTADÍSTICAS AGREGADAS POR GENERACIÓN
    generation_means = np.mean(diversity_matrix, axis=0)
    generation_stds = np.std(diversity_matrix, axis=0)
    generation_q25 = np.percentile(diversity_matrix, 25, axis=0)
    generation_q75 = np.percentile(diversity_matrix, 75, axis=0)
    
    """
    # 2. DETECCIÓN DE PUNTOS CRÍTICOS
    max_diversity_gens = []
    convergence_gens = []
    convergence_velocities = []
    
    convergence_threshold = 0.1  # Umbral de convergencia
    
    for run_idx in range(n_runs):
        run_diversity = diversity_matrix[run_idx, :]
        
        # Skip runs with all zeros
        if np.sum(run_diversity) == 0:
            max_diversity_gens.append(0)
            convergence_gens.append(299)
            convergence_velocities.append(0.0)
            continue
        
        # Generación de máxima diversidad
        max_gen = np.argmax(run_diversity)
        max_diversity_gens.append(max_gen)
        
        # Generación de convergencia (primera vez que baja del umbral)
        conv_gen = None
        for gen in range(len(run_diversity)):
            if run_diversity[gen] < convergence_threshold:
                conv_gen = gen
                break
        convergence_gens.append(conv_gen if conv_gen is not None else 299)
        
        # Velocidad de convergencia (pendiente de los primeros 50 puntos)
        if len(run_diversity) > 10:
            # Usar los primeros 50 puntos para calcular velocidad inicial
            end_point = min(50, len(run_diversity))
            x = np.arange(end_point)
            y = run_diversity[:end_point]
            
            if np.std(y) > 0.001:  # Solo si hay variación real
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    convergence_velocities.append(abs(slope))
                except:
                    convergence_velocities.append(0.0)
            else:
                convergence_velocities.append(0.0)
        else:
            convergence_velocities.append(0.0)
    
    # 3. ANÁLISIS DE FASES
    initial_phase = diversity_matrix[:, :50]    # Gen 1-50
    middle_phase = diversity_matrix[:, 50:200]  # Gen 51-200
    final_phase = diversity_matrix[:, 200:]     # Gen 201-300
    
    phase_analysis = {
        'initial': {
            'mean': np.mean(initial_phase),
            'std': np.std(initial_phase),
            'max': np.max(initial_phase),
            'min': np.min(initial_phase)
        },
        'middle': {
            'mean': np.mean(middle_phase), 
            'std': np.std(middle_phase),
            'max': np.max(middle_phase),
            'min': np.min(middle_phase)
        },
        'final': {
            'mean': np.mean(final_phase),
            'std': np.std(final_phase), 
            'max': np.max(final_phase),
            'min': np.min(final_phase)
        }
    }
    
    # 4. CLUSTERING DE EJECUCIONES
    try:
        # Usar primeras 100 generaciones para clustering
        clustering_data = diversity_matrix[:, :100]
        
        # Solo usar filas que no sean todo ceros
        non_zero_rows = np.sum(clustering_data, axis=1) > 0
        valid_runs = np.where(non_zero_rows)[0]
        
        if len(valid_runs) > 3:
            valid_data = clustering_data[valid_runs]
            
            # Aplicar suavizado para reducir ruido
            smoothed_data = np.array([savgol_filter(row, min(5, len(row)), 2) for row in valid_data])
            
            n_clusters = min(4, len(valid_runs))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels_valid = kmeans.fit_predict(smoothed_data)
            
            # Map back to all runs
            cluster_labels = np.zeros(n_runs, dtype=int)
            cluster_labels[valid_runs] = cluster_labels_valid
            
            execution_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in execution_clusters:
                    execution_clusters[label] = []
                execution_clusters[label].append(i)
        else:
            execution_clusters = {0: list(range(n_runs))}
            cluster_labels = [0] * n_runs
            
    except Exception as e:
        print(f"   ⚠️ Clustering failed: {e}")
        execution_clusters = {0: list(range(n_runs))}
        cluster_labels = [0] * n_runs
    
    # 5. CORRELACIONES TEMPORALES
    # Correlación inicial vs final
    initial_diversity = diversity_matrix[:, 0]
    final_diversity = diversity_matrix[:, -1]
    
    # Filter out zero values for correlation
    valid_mask = (initial_diversity > 0) & (final_diversity >= 0)
    
    if np.sum(valid_mask) > 3 and np.std(initial_diversity[valid_mask]) > 0 and np.std(final_diversity[valid_mask]) > 0:
        initial_final_corr, initial_final_p = pearsonr(initial_diversity[valid_mask], final_diversity[valid_mask])
    else:
        initial_final_corr, initial_final_p = 0.0, 1.0
    
    # 6. INTERPRETACIÓN AUTOMÁTICA
    avg_initial_diversity = np.mean(initial_diversity[initial_diversity > 0]) if np.sum(initial_diversity > 0) > 0 else 0
    avg_final_diversity = np.mean(final_diversity)
    valid_conv_gens = [g for g in convergence_gens if g < 299 and g > 0]
    avg_convergence_gen = np.mean(valid_conv_gens) if valid_conv_gens else 299
    
    if avg_convergence_gen < 50:
        convergence_speed = "Very Fast"
    elif avg_convergence_gen < 100:
        convergence_speed = "Fast"
    elif avg_convergence_gen < 200:
        convergence_speed = "Moderate"
    else:
        convergence_speed = "Slow"
    
    if avg_initial_diversity > 0:
        diversity_loss = (avg_initial_diversity - avg_final_diversity) / avg_initial_diversity * 100
    else:
        diversity_loss = 0
    
    if diversity_loss > 80:
        diversity_pattern = "Strong convergence - high diversity loss"
    elif diversity_loss > 50:
        diversity_pattern = "Moderate convergence - balanced exploration/exploitation"
    else:
        diversity_pattern = "Weak convergence - maintained diversity"
    
    interpretation = f"{convergence_speed} convergence with {diversity_pattern.lower()}"
    
    # Recomendación
    if avg_convergence_gen < 30:
        recommendation = "Consider increasing population diversity or reducing selection pressure"
    elif diversity_loss < 30:
        recommendation = "Consider increasing selection pressure for better convergence"
    else:
        recommendation = "Good balance between exploration and exploitation"
    """
    results = {
        # Datos base
        'diversity_matrix': diversity_matrix,
        'generation_means': generation_means,
        'generation_stds': generation_stds,
        'generation_q25': generation_q25,
        'generation_q75': generation_q75,
        
        # Puntos críticos
        #'max_diversity_gens': max_diversity_gens,
        #'convergence_gens': convergence_gens,
        #'convergence_velocities': convergence_velocities,
        #'avg_convergence_gen': avg_convergence_gen,
        #'convergence_speed': convergence_speed,
        #
        ## Análisis de fases
        #'phase_analysis': phase_analysis,
        #
        ## Clustering
        #'execution_clusters': execution_clusters,
        #'cluster_labels': cluster_labels,
        #'n_clusters': len(execution_clusters),
        #
        ## Correlaciones
        #'initial_final_corr': initial_final_corr,
        #'initial_final_p': initial_final_p,
        #
        ## Métricas resumen
        #'avg_initial_diversity': avg_initial_diversity,
        #'avg_final_diversity': avg_final_diversity,
        #'diversity_loss_percent': diversity_loss,
        #'diversity_pattern': diversity_pattern,
        #
        ## Interpretación
        #'interpretation': interpretation,
        #'recommendation': recommendation,
        
        # Metadata
        'n_runs': n_runs,
        'successful_runs': successful_runs,
        'n_generations': n_generations,
        #'convergence_threshold': convergence_threshold
    }
    
    if verbose:
        print(f"✅ Level 3 analysis completed:")
        print(f"   - Successful runs analyzed: {successful_runs}/{n_runs}")
        #print(f"   - Average initial diversity: {avg_initial_diversity:.4f}")
        #print(f"   - Average final diversity: {avg_final_diversity:.4f}")
        #print(f"   - Average convergence generation: {avg_convergence_gen:.1f}")
        #print(f"   - Diversity loss: {diversity_loss:.1f}%")
        #print(f"   - Convergence pattern: {convergence_speed}")
        #print(f"   - Found {len(execution_clusters)} execution clusters")
    
    return results


def generate_simplified_plots(output_analyze_temporal_diversity_evolution_function, plots_dir, 
                              dataset_name):
    """
    Genera solo el mapa de calor y la gráfica de evolución de diversidad
    a partir de los resultados de analyze_temporal_diversity_evolution
    """
    nivel3_results = output_analyze_temporal_diversity_evolution_function
    # Configurar directorio de plots
    #plots_dir = Path(f"logs/{dataset_name}/convergence/plots/diversity/simplified")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Generando plots simplificados para {dataset_name}")
    print(f"📁 Directorio: {plots_dir}")
    
    # Extraer datos de diversidad temporal
    diversity_matrix = nivel3_results['diversity_matrix']
    n_gens      = len(diversity_matrix[0])
    generations = np.arange(len(diversity_matrix[0]))
    
    
    print(f"📊 Matriz de diversidad: {diversity_matrix.shape} (runs x generaciones)")
    
    # 1. MAPA DE CALOR
    plt.figure(figsize=(15, 8))
    
    # Crear el heatmap
    sns.heatmap(diversity_matrix, 
                cmap='viridis', 
                cbar_kws={'label': 'Hamming Diversity'},
                xticklabels=50,  # Mostrar cada 50 generaciones
                yticklabels=False)  # No mostrar labels de runs
    
    plt.title('Diversity Evolution Heatmap', fontweight='bold', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Run ID', fontsize=12)
    
    # Guardar heatmap
    heatmap_path = plots_dir / "diversity_evolution_heatmap.png"
    plt.savefig(heatmap_path, dpi=n_gens, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Heatmap guardado: {heatmap_path}")
    
    # 2. GRÁFICA DE EVOLUCIÓN DE DIVERSIDAD ACROSS ALL RUNS
    plt.figure(figsize=(14, 8))
    
    # Calcular estadísticas por generación
    mean_diversity = np.mean(diversity_matrix, axis=0)
    std_diversity = np.std(diversity_matrix, axis=0)
    
    # Gráfica principal con media y banda de confianza
    plt.plot(generations, mean_diversity, 
             color='darkblue', linewidth=3, label='Mean Diversity', zorder=3)
    
    plt.fill_between(generations, 
                     mean_diversity - std_diversity,
                     mean_diversity + std_diversity,
                     alpha=0.3, color='lightblue', label='±1 Std Dev', zorder=2)
    
    # Mostrar algunas líneas individuales para contexto
    n_sample_runs = min(8, len(diversity_matrix))
    for i in range(n_sample_runs):
        plt.plot(generations, diversity_matrix[i], 
                alpha=0.4, linewidth=0.8, color='gray', zorder=1)
    
    plt.title('Diversity Evolution Across All Runs', fontweight='bold', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Hamming Diversity', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Añadir estadísticas como texto
    final_mean = mean_diversity[-1]
    final_std = std_diversity[-1]
    initial_mean = mean_diversity[0]
    
    #stats_text = f'Initial: {initial_mean:.4f}\\nFinal: {final_mean:.4f}±{final_std:.4f}'
    #plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
    #        verticalalignment='top', fontsize=10,
    #       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Guardar gráfica de evolución
    evolution_path = plots_dir / "diversity_evolution_across_runs.png"
    plt.savefig(evolution_path, dpi=n_gens, bbox_inches='tight')
    plt.show()
    print(f"   ✅ Evolución guardada: {evolution_path}")
    
    # Resumen de estadísticas
    print(f"\\n📈 Estadísticas de diversidad:")
    print(f"   • Runs analizados: {len(diversity_matrix)}")
    print(f"   • Generaciones: {len(generations)}")
    print(f"   • Diversidad inicial promedio: {initial_mean:.4f}")
    print(f"   • Diversidad final promedio: {final_mean:.4f} ± {final_std:.4f}")
    
    results = {
        'heatmap_path': heatmap_path,
        'evolution_path': evolution_path,
        'diversity_matrix': diversity_matrix,
        'mean_diversity': mean_diversity,
        'std_diversity': std_diversity,
        'final_mean': final_mean,
        'final_std': final_std,
        'initial_mean': initial_mean
    }
    
    print(f"✅ Plots simplificados generados exitosamente")
    return results

    
    
    