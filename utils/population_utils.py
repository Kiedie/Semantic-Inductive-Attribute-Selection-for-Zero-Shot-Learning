import json
import pathlib
import glob
from deap import creator, tools

def load_previous_run(continue_from):
    """Load configuration and population from a previous run"""
    continue_path = pathlib.Path(continue_from)
    
    if not continue_path.exists():
        raise FileNotFoundError(f"Path {continue_path} does not exist")
    
    # Load configuration
    config_path = continue_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Try to load final population or the last generation
    population = None
    gen_number = 0
    
    # First, try to load final_population.json
    final_pop_path = continue_path / "final_population.json"
    if final_pop_path.exists():
        with open(final_pop_path, "r") as f:
            population_data = json.load(f)
        
        # Load logbook to determine the number of generations
        logbook_path = continue_path / "logbook.json"
        if logbook_path.exists():
            with open(logbook_path, "r") as f:
                logbook_data = json.load(f)
                gen_number = len(logbook_data)
    else:
        # If no final_population.json, find the latest generation file
        gen_files = sorted(glob.glob(str(continue_path / "generations" / "gen_*.json")))
        if gen_files:
            last_gen_file = gen_files[-1]
            gen_number = int(last_gen_file.split("_")[-1].split(".")[0])
            
            with open(last_gen_file, "r") as f:
                population_data = json.load(f)
        else:
            raise FileNotFoundError(f"No population data found in {continue_path}")
    
    print(f"Loaded population from generation {gen_number}")
    
    return config, population_data, gen_number

def create_population_from_data(toolbox, population_data):
    """Create DEAP individuals from loaded population data"""
    population = []
    for ind_data in population_data:
        # Create individual from mask
        ind = creator.Individual(ind_data["mask"])
        # Set fitness if available
        if ind_data["fitness"] is not None:
            ind.fitness.values = (ind_data["fitness"],)
        
        population.append(ind)
    
    return population
