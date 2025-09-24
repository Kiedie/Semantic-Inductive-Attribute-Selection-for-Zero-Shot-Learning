"""Este fichero es una ampliación del HAIS
- Versión 0: 
    Experimentación del HAIS

- Version 1:
    Creación de un ranking aleatorio de los atributos
    - Ejecución con CV
    - Ejecución sin CV 
    El objetivo sería comparar:
        - Versión cero con la ejecución con CV para comprobar que la selección de atributos con los algoritmos es eficaz+
        - La ejecución del ranking aleatorio con y sin CV para comprobar que la CV es efectiva.
    

- Versión 2:
    Entrenar el modelo normalmente.
    Para cada attributo semántico de test crear un entorno semántico ampliado.
    Evaluar el modelo usando la frontera más cercana. 
    El entorno se crea tomando la distancia con su vecino más cercando y se calcula
        el radio del entorno dividiendo dicha distancia por un factor de escala llamado epsilon.
    Se hace una búsqueda de epsilon para encontrar el mejor valor. 

"""


import scipy.io 
import numpy as np
import pandas as pd
import pathlib
import random
import time
from preprocess.featureselectors import *
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from data.datareader import Dataset, StackedDataset, gen_ss_from_data, load_data_from_path, get_data, scaler_data, get_scaler
from models.sae import SAE, normalize_feature
from evaluators.sae_evaluator import SAEvaluator
from metrics.sae_eval import evaluate
import os
from data.split import SplitIntoFolds
import csv
import json
# ============================================================================= # 
# ============================= GLOBALS VARIABLES ============================= # 
# ============================================================================= # 
DECIMAL_NUMBERS        = 3
FOLDS                  = 5

scaler_str             = 'Standard'     # ['Standard','MinMax']
orig_attribute         = False # Attributos originales u otros
debug                  = True 
SHOW                   = True

step                     = 5  # Paso de atributos a seleccionar [a, a+step, a+2*step, ...]
hitk = 1    


FS_NAMES = ['RandomForest', 'RF', 'LR', 'LogisticRegression', 'SVC', 'SupportVectorClassifier']
DB_NAMES = ['CUB', 'SUN', 'FLO', 'AWA2','APY']
ZSL_NAMES = ['Sae','SAE']
VERSIONS = [0,1,2,3]

# ============================================================================= # 
# ============================= ARGUMENTS & PATHS ============================= # 
# ============================================================================= #
parser = argparse.ArgumentParser(description='Hais')
parser.add_argument('-d', '--dataset', type = str, default='CUB', choices = DB_NAMES,  help='DataSet Name')
parser.add_argument('-v','--version',  type = int, default=0,     choices = VERSIONS,  help='Version')
parser.add_argument('-md', '--mode',   type = str, default='RF',  choices = FS_NAMES,  help='Type of Feature Selection Embedded Algorithm')
parser.add_argument('-m', '--model',   type = str, default='Sae', choices = ZSL_NAMES, help='ZSL model')  

parser.add_argument('-s', '--seed',       type = int, default=42,         help='Random Seed')
parser.add_argument('-sc','--scaler_str', type = str, default='Standard', help='Scaler')
parser.add_argument('-stp','--step',      type = int, default=1,          help='Step')
parser.add_argument('-oa','--orig_attribute', action='store_true', default=False, help='Original attributes')
parser.add_argument('-vb','--verbose',        action='store_true', default=True,  help='Verbose')
args = parser.parse_args()


#class Args:
#    def __init__(self):
#        self.dataset                = 'CUB'
#        self.seed                   = 42
#        self.verbose                = True
#        self.mode                   = 'rfe'
#        self.percentage_test_seen   = 0.6
#        self.percentage_test_unseen = 0.3
#        self.orig_attribute         = False 
#        self.debug                  = True
#        self.SHOW                   = True
#        self.version                = 1 # 0: HAIS, 1: lo que comentó Isaac 
#        self.step                   = 1 # Atributos que evaluamos en cada paso. Puede ser de 5 en 5 o en 1 en 1 para hacerlo más exhaustivo
#        self.INFO                   = f"HAIS_v1"
#args = Args()

random.seed(args.seed)
np.random.seed(args.seed)

print("="*80)
print(f"RFS AMPLIATION - EXPERIMENT CONFIGURATION")
print("="*80)
print(f"Dataset: {args.dataset}")
print(f"Version: {args.version}")
print(f"Mode: {args.mode}")
print(f"Model: {args.model}")
print(f"Random Seed: {args.seed}")
print(f"Step: {args.step}")
print("="*80)

# ============================================================================= # 
# ============================= DIRECTORY CREATION ============================= # 
# ============================================================================= # 

def ensure_dir_exists(path):
    """Ensure that a directory exists, create it if it doesn't"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Creating directory: {path}")

def save_experiment_config():
    """Save experiment configuration to a JSON file"""
    config = {
        'experiment_info': {
            'dataset': args.dataset,
            'version': args.version,
            'mode': args.mode,
            'model': args.model,
            'seed': args.seed,
            'step': args.step,
            'scaler_str': args.scaler_str,
            'orig_attribute': args.orig_attribute,
            'verbose': args.verbose,
            'decimal_numbers': DECIMAL_NUMBERS,
            'folds': FOLDS
        },
        'directories': {
            'results_dir': str(results_dir),
            'data_path': str(path_data)
        },
        'output_files': {}
    }
    
    # Add version-specific output files
    if args.version == 0:
        config['output_files'] = {
            'selection_results': str(path_results_selection),
            'thresholds_results': str(path_results_thresholds),
            'masks_json': str(path_masks_json)
        }
    elif args.version == 1:
        config['output_files'] = {
            'selection_results': str(path_results_selection),
            'thresholds_results': str(path_results_thresholds),
            'random_no_cv_results': str(path_results_random_no_cv),
            'random_no_cv_masks': str(path_masks_random_no_cv),
            'masks_json': str(path_masks_json)
        }
    elif args.version == 2:
        config['output_files'] = {
            'no_cv_results': str(path_results_no_cv),
            'no_cv_masks': str(path_masks_no_cv)
        }
    elif args.version == 3:
        config['output_files'] = {
            'selection_results': str(path_results_selection),
            'thresholds_results': str(path_results_thresholds),
            'masks_json': str(path_masks_json)
        }
    
    # Save configuration file
    config_file = results_dir / 'experiment_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Experiment configuration saved: {config_file}")
    return config_file

# Set the paths with better organization
path_data = pathlib.Path(f'../ZSL-preprocessing/data/{args.dataset}')

# Create a more structured directory organization
base_results_dir = pathlib.Path('results')
version_dir = base_results_dir / f'HAIS_v{args.version}_time'
dataset_dir = version_dir / args.dataset

# For versions 0 and 1, also organize by mode
if args.version in [0, 1]:
    results_dir = dataset_dir / args.mode
else:
    results_dir = dataset_dir

# Ensure all necessary directories exist
ensure_dir_exists(base_results_dir)
ensure_dir_exists(version_dir)
ensure_dir_exists(dataset_dir)
ensure_dir_exists(results_dir)

# Print directory structure information
print("\nDIRECTORY STRUCTURE:")
print(f"Results directory: {results_dir}")

# Set file paths based on version
if args.version == 0:
    path_results_selection = results_dir / f'{args.mode}_selection.csv'
    path_results_thresholds = results_dir / f'{args.mode}_thresholds.csv'
    path_masks_json = results_dir / f'{args.mode}_masks.json'
elif args.version == 1:
    path_results_selection = results_dir / f'{args.dataset}_selection.csv'
    path_results_thresholds = results_dir / f'{args.dataset}_thresholds.csv' 
    path_results_random_no_cv = results_dir / f'{args.dataset}_RandomRankingWithoutCV_results.csv'
    path_masks_random_no_cv = results_dir / f'{args.dataset}_RandomRankingWithoutCV_masks.csv'
    path_masks_json = results_dir / f'{args.mode}_masks.json'
elif args.version == 2:
    path_results_no_cv = results_dir / f'{args.mode}_NoCV_results.csv'
    path_masks_no_cv = results_dir / f'{args.mode}_NoCV_masks.csv'
elif args.version == 3:
    path_results_selection = results_dir / f'Aggregation_selection.csv'
    path_results_thresholds = results_dir / f'Aggregation_thresholds.csv'
    path_masks_json = results_dir / f'Aggregation_masks.json'

# Save experiment configuration
config_file = save_experiment_config()

# Print file paths information
print("\nOUTPUT FILES:")
print(f"Experiment config: {config_file}")
if args.version == 0:
    print(f"Selection results: {path_results_selection}")
    print(f"Thresholds results: {path_results_thresholds}")
    print(f"Masks JSON: {path_masks_json}")
elif args.version == 1:
    print(f"Selection results: {path_results_selection}")
    print(f"Thresholds results: {path_results_thresholds}")
    print(f"Random ranking (no CV) results: {path_results_random_no_cv}")
    print(f"Random ranking (no CV) masks: {path_masks_random_no_cv}")
    print(f"Masks JSON: {path_masks_json}")
elif args.version == 2:
    print(f"No CV results: {path_results_no_cv}")
    print(f"No CV masks: {path_masks_no_cv}")
elif args.version == 3:
    print(f"Selection results: {path_results_selection}")
    print(f"Thresholds results: {path_results_thresholds}")
    print(f"Masks JSON: {path_masks_json}")
print("="*80)

# ============================================================================= # 
# =============================== DATA READING ================================ # 
# ============================================================================= # 
print(f"Reading the data {args.dataset}...")
training_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(path_data    = path_data,
                                                                               args         = args,
                                                                               trainval     = True)


awa = True if args.dataset == 'AWA2' else False


splitter  = SplitIntoFolds(n_splits = 5, show = args.verbose , seed = 42)
folds = list(splitter.split_via_classes_per_folds(features       = training_dataset.features,
                                                 labels         = training_dataset.labels,
                                                 attribute      = attribute,
                                                 preprocessing  = True))


# Apply normalize_feature for AWA2 dataset to each fold
if awa:
    normalized_folds = []
    for i, (train_fold, val_fold) in enumerate(folds):
        train_fold.features = normalize_feature(train_fold.features)
        #val_fold.features = normalize_feature(val_fold.features)
        normalized_folds.append((train_fold, val_fold))
    folds = normalized_folds

vattributes = np.arange(25, attribute.shape[1], args.step)
if attribute.shape[1] % args.step != 0:
    vattributes = np.append(vattributes, attribute.shape[1])

# ============================================================================= # 
# ============================= TIMING SETUP ================================= # 
# ============================================================================= # 
# Initialize timing data structure
timing_data = {
    'total_time': 0.0,
    'ranking_time': {
        'total': 0.0,
        'per_fold': [],
        'average': 0.0
    },
    'fold_times': [],
    'average_fold_time': 0.0,
    'feature_selection_time': {
        'total': 0.0,
        'per_fold': [],
        'average': 0.0
    },
    'consensus_time': 0.0
}

# Start total time measurement
start_total = time.time()

conteo                      = np.zeros(attribute.shape[1])  # Conteo de las veces que se selecciona un atributo a lo largo del cross-validation
best_folds_configuration    = []                            # Mejor configurración de atributos seleccionados en cada uno de los folds
df_results                  = {}                            # Resultados de cada uno de los folds



############################################################################################################
#                                       VERSIÓN 0 & 1: RFS                                                #
############################################################################################################

if args.version == 0 or args.version == 1: # RFS
    # PART 1: Cross-Validation + FS 
    for i, (training, val) in enumerate(folds): 
        start_fold = time.time()
        if args.verbose:
            print(f"Fold {i+1}...")
        
        # VERSION 0
        if args.version==0:
            # 1. Computing the ranking via FS embedded algorithm across the training (psedo-seen data)
            start_ranking = time.time()
            if args.mode in ['RandomForest','RF']:
                estimator, name = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-3), 'RandomForestClassifier'
            elif args.mode in ['SVC','SupportVectorClassifier']:
                estimator, name = SVC(kernel='linear', random_state=42), 'SVC'
            elif args.mode in ['LR','LogisticRegression']:
                estimator, name = LogisticRegression(random_state=42, n_jobs=-3), 'LogisticRegression'
            else:
                raise ValueError(f"Model {args.mode} not implemented")
            # Training the feature selection model
            print(f"Making FS Ranking with {args.mode}...")
            selector  = EmbeddingSimpleFeatureSelector(estimator,name)
            selector.fit(training)
            ranking   = selector.get_ranking()
            fold_ranking_time = time.time() - start_ranking
            timing_data['ranking_time']['per_fold'].append(fold_ranking_time)
        # VERSION 1
        else:
            # 1. Making a random ranking
            start_ranking = time.time()
            ranking = np.random.permutation(range(attribute.shape[1]))
            fold_ranking_time = time.time() - start_ranking
            timing_data['ranking_time']['per_fold'].append(fold_ranking_time)
        
        #2. Selecting the optimal number of features via the Validation (pseudo-unseen data)
        start_selection = time.time()
        fold_i_tracking = {'Attributes': [], 'Train Acc':[], 'Val Acc':[], 'Mask':[]}
        for nv in vattributes:
            mask = ranking <= nv
            sae = SAE()
            # Apply mask to attributes if provided
            if mask is not None:
                selected_attributes = training.att[:, mask]
            else:
                selected_attributes = training.att
            sae.fit(training.features, selected_attributes)
            
            # Evaluate using sae_eval functions
            if mask is not None:
                eval_attributes = attribute[:, mask]
            else:
                eval_attributes = attribute
            acc_tr, _ = evaluate(sae, eval_attributes, training.features, training.classes, training.labels,awa=awa)
            acc_val, _ = evaluate(sae, eval_attributes, val.features, val.classes, val.labels,awa=awa)
            acc_tr     = round(acc_tr, DECIMAL_NUMBERS)
            acc_val    = round(acc_val,DECIMAL_NUMBERS)
            fold_i_tracking['Attributes']   .append(nv)
            fold_i_tracking['Train Acc']    .append(acc_tr)
            fold_i_tracking['Val Acc']      .append(acc_val)
            fold_i_tracking['Mask']         .append(mask)
            
            if args.verbose:
                print(f"nv: {nv} || Train Acc: {acc_tr} || Val Acc: {acc_val}")
        
        fold_selection_time = time.time() - start_selection
        timing_data['feature_selection_time']['per_fold'].append(fold_selection_time)
        fold_i_tracking = pd.DataFrame(fold_i_tracking).sort_values(by='Val Acc', ascending=False)
        best_folds_configuration.append(fold_i_tracking.iloc[0])
        
        # Record fold time
        fold_time = time.time() - start_fold
        timing_data['fold_times'].append(fold_time)
    
    # Calculate average fold time
    timing_data['average_fold_time'] = sum(timing_data['fold_times']) / len(timing_data['fold_times'])
    
    # Calculate ranking time statistics
    timing_data['ranking_time']['total'] = sum(timing_data['ranking_time']['per_fold'])
    timing_data['ranking_time']['average'] = timing_data['ranking_time']['total'] / len(timing_data['ranking_time']['per_fold'])
    
    # Calculate feature selection time statistics
    timing_data['feature_selection_time']['total'] = sum(timing_data['feature_selection_time']['per_fold'])
    timing_data['feature_selection_time']['average'] = timing_data['feature_selection_time']['total'] / len(timing_data['feature_selection_time']['per_fold'])
        
    # Make the consensus (comuting threshold): An element-wise sum of the best configurations extracted from each fold 
    start_consensus = time.time()
    print("Making the consensus...")
    for res in best_folds_configuration:
        conteo = conteo + np.asarray(res['Mask']).astype(int)
    # Turn the best results of each fold into a DataFrame
    for i in range(len(best_folds_configuration)):
        df_results[i] = best_folds_configuration[i].to_dict()
    df_results = pd.DataFrame(df_results).T

    # PART 2: Diving into the threshold
    test_results = {'Cut':[],'Attributes':[],'Train Acc':[],'Test Seen Acc':[],'Test Unseen Acc':[]}
    j = 1
    cadena = str(FOLDS)
    
    data_to_json = {'Dataset': args.dataset,'Algorithm': args.mode}
    thresholds = {}
    print("Executing the thresholds loop...")
    for i in tqdm(range(FOLDS+1),desc="Threshold Loop"):
        
        if i != 0:
            cadena = cadena+'-'+str(FOLDS - i)
        
        test_results['Cut'].append(cadena)
        test_results['Attributes'].append(np.where(conteo>=FOLDS-i)[0].shape[0])
        mejores = np.where(conteo>=FOLDS-i)[0]
        thresholds[FOLDS-i] = list(map(int,mejores)) 
        if mejores.shape[0] > 0:
            # Fit the model
            sae = SAE()
            # Apply mask to attributes
            selected_attributes = training_dataset.att[:, mejores]
            sae.fit(training_dataset.features, selected_attributes)
            
            # Evaluate using sae_eval functions
            eval_attributes = attribute[:, mejores]
            acc_tr, _     = evaluate(sae, eval_attributes, training_dataset.features, training_dataset.classes, training_dataset.labels,awa=awa)
            acc_seen, _   = evaluate(sae, eval_attributes, test_seen_dataset.features, test_seen_dataset.classes, test_seen_dataset.labels,awa=awa)
            acc_unseen, _ = evaluate(sae, eval_attributes, test_unseen_dataset.features, test_unseen_dataset.classes, test_unseen_dataset.labels,awa=awa)
            test_results['Train Acc'].append(round(acc_tr,DECIMAL_NUMBERS))
            test_results['Test Seen Acc'].append(round(acc_seen,DECIMAL_NUMBERS))
            test_results['Test Unseen Acc'].append(round(acc_unseen,DECIMAL_NUMBERS))
        else:
            test_results['Train Acc'].append(0)
            test_results['Test Seen Acc'].append(0)
            test_results['Test Unseen Acc'].append(0)
    
    # Record consensus time
    timing_data['consensus_time'] = time.time() - start_consensus
    
    print("="*50)
    print("SAVING RESULTS...")
    print("="*50)
    data_to_json['Thresholds'] = thresholds
    
    # Save JSON masks
    with open(path_masks_json, 'w') as json_file:
        json.dump(data_to_json, json_file, indent=4)
    print(f"✓ Masks JSON saved: {path_masks_json}")
    
    # Save CSV results
    pd.DataFrame(test_results).to_csv(path_results_thresholds, index = False)
    print(f"✓ Thresholds results saved: {path_results_thresholds}")
    
    df_results.to_csv(path_results_selection, index = False)
    print(f"✓ Selection results saved: {path_results_selection}")

    if args.version == 1:
        # Now, from the random ranking done, we conduct to evaluate the features without CV in order to check
        # if the crss-validation is really necessary 
        results = {
            'Attributes':[],
            'Test Seen':[],
            'Test Unseen':[],
        }
        training_dataset,val_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(path_data    = path_data,
                                                                                args         = args,
                                                                                trainval     = False)

        if awa:
            training_dataset.features = normalize_feature(training_dataset.features)

        results_selection = {'Attributes':[], 'Val':[],'Mask':[]}
        df_masks = {'Attributes':[], 'Masks':[]}
        
        for nv in vattributes:
            mask = ranking <= nv
            sae = SAE()
            # Apply mask to attributes
            selected_attributes = training_dataset.att[:, mask]
            sae.fit(training_dataset.features, selected_attributes)
            
            # Evaluate using sae_eval functions
            eval_attributes = attribute[:, mask]
            acc_val, _ = evaluate(sae, eval_attributes, val_dataset.features, val_dataset.classes, val_dataset.labels,awa=awa)
            results_selection['Attributes'].append(nv)
            results_selection['Val'].append(round(acc_val,DECIMAL_NUMBERS))
            results_selection['Mask'].append(mask)
        results_selection = pd.DataFrame(results_selection).sort_values(by='Val', ascending=False)

    
        training_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(path_data    = path_data,
                                                                                    args         = args,
                                                                                    trainval     = True)
        
        if awa:
            training_dataset.features = normalize_feature(training_dataset.features)
        
        for index, row in results_selection.iterrows():
            sae = SAE()
            mask = row['Mask']
            # Apply mask to attributes
            selected_attributes = training_dataset.att[:, mask]
            sae.fit(training_dataset.features, selected_attributes)
            
            # Evaluate using sae_eval functions
            eval_attributes = attribute[:, mask]
            acc_seen, _ = evaluate(sae, eval_attributes, test_seen_dataset.features, test_seen_dataset.classes, test_seen_dataset.labels,awa=awa)
            acc_unseen, _ = evaluate(sae, eval_attributes, test_unseen_dataset.features, test_unseen_dataset.classes, test_unseen_dataset.labels,awa=awa)
            results['Attributes'].append(row['Attributes'])
            results['Test Seen'].append(round(acc_seen,DECIMAL_NUMBERS))
            results['Test Unseen'].append(round(acc_unseen,DECIMAL_NUMBERS))
            df_masks['Attributes'].append(row['Attributes'])
            df_masks['Masks'].append(row['Mask'])

        print("\n" + "="*50)
        print("SAVING RANDOM RANKING RESULTS (WITHOUT CV)...")
        print("="*50)
        
        results = pd.DataFrame(results).to_csv(path_results_random_no_cv, index=False)
        print(f"✓ Random ranking results saved: {path_results_random_no_cv}")
        
        df_masks = pd.DataFrame(df_masks).to_csv(path_masks_random_no_cv, index=False)
        print(f"✓ Random ranking masks saved: {path_masks_random_no_cv}")
        print("="*50)

############################################################################################################
#                                       VERSIÓN 2: RFS SIN CV                                           #
############################################################################################################

elif args.version == 2: # RFS sin Cross-Validation
    print("="*80)
    print("VERSIÓN 2: RFS SIN CROSS-VALIDATION")
    print("="*80)
    
    # Cargar datos sin CV (trainval=False para obtener train/val por separado)
    training_dataset, val_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(
        path_data=path_data, args=args, trainval=False
    )
    
    # Aplicar normalización si es AWA2
    if awa:
        training_dataset.features = normalize_feature(training_dataset.features)
    
    # 1. Computing the ranking via FS embedded algorithm usando TODO el conjunto de entrenamiento
    if args.mode in ['RandomForest','RF']:
        estimator, name = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-3), 'RandomForestClassifier'
    elif args.mode in ['SVC','SupportVectorClassifier']:
        estimator, name = SVC(kernel='linear', random_state=42), 'SVC'
    elif args.mode in ['LR','LogisticRegression']:
        estimator, name = LogisticRegression(random_state=42, n_jobs=-3), 'LogisticRegression'
    else:
        raise ValueError(f"Model {args.mode} not implemented")
    
    print(f"Making FS Ranking with {args.mode} (without CV)...")
    start_ranking = time.time()
    selector = EmbeddingSimpleFeatureSelector(estimator, name)
    selector.fit(training_dataset)
    ranking = selector.get_ranking()
    # For version 2, store as simple value (no folds)
    timing_data['ranking_time'] = {'total': time.time() - start_ranking, 'per_fold': [], 'average': 0.0}
    
    # 2. Selecting the optimal number of features via Validation (sin CV)
    results_selection = {'Attributes': [], 'Val Acc': [], 'Mask': []}
    df_masks = {'Attributes': [], 'Masks': []}
    
    print("Evaluating different numbers of attributes...")
    start_selection = time.time()
    for nv in vattributes:
        mask = ranking <= nv
        sae = SAE()
        
        # Apply mask to attributes
        selected_attributes = training_dataset.att[:, mask]
        sae.fit(training_dataset.features, selected_attributes)
        
        # Evaluate using sae_eval functions
        eval_attributes = attribute[:, mask]
        acc_val, _ = evaluate(sae, eval_attributes, val_dataset.features, val_dataset.classes, val_dataset.labels, awa=awa)
        
        results_selection['Attributes'].append(nv)
        results_selection['Val Acc'].append(round(acc_val, DECIMAL_NUMBERS))
        results_selection['Mask'].append(mask)
        df_masks['Attributes'].append(nv)
        df_masks['Masks'].append(mask)
        
        if args.verbose:
            print(f"nv: {nv} || Val Acc: {round(acc_val, DECIMAL_NUMBERS)}")
    
    # For version 2, store as simple value (no folds)
    timing_data['feature_selection_time'] = {'total': time.time() - start_selection, 'per_fold': [], 'average': 0.0}
    
    # 3. Ordenar por precisión de validación y seleccionar el mejor
    results_selection = pd.DataFrame(results_selection).sort_values(by='Val Acc', ascending=False)
    best_config = results_selection.iloc[0]
    
    print(f"\nBest configuration: {best_config['Attributes']} attributes with {best_config['Val Acc']} Val Acc")
    
    # 4. Evaluación final en test con la mejor configuración
    # Recargar datos con trainval=True para usar todo el training para el modelo final
    training_dataset, test_seen_dataset, test_unseen_dataset, attribute = get_data(
        path_data=path_data, args=args, trainval=True
    )
    
    if awa:
        training_dataset.features = normalize_feature(training_dataset.features)
    
    # Usar la mejor máscara
    best_mask = best_config['Mask']
    sae = SAE()
    selected_attributes = training_dataset.att[:, best_mask]
    sae.fit(training_dataset.features, selected_attributes)
    
    # Evaluación final
    eval_attributes = attribute[:, best_mask]
    acc_tr, _ = evaluate(sae, eval_attributes, training_dataset.features, training_dataset.classes, training_dataset.labels, awa=awa)
    acc_seen, _ = evaluate(sae, eval_attributes, test_seen_dataset.features, test_seen_dataset.classes, test_seen_dataset.labels, awa=awa)
    acc_unseen, _ = evaluate(sae, eval_attributes, test_unseen_dataset.features, test_unseen_dataset.classes, test_unseen_dataset.labels, awa=awa)
    
    # Crear resultado final
    final_results = {
        'Attributes': [best_config['Attributes']],
        'Train Acc': [round(acc_tr, DECIMAL_NUMBERS)],
        'Test Seen Acc': [round(acc_seen, DECIMAL_NUMBERS)],
        'Test Unseen Acc': [round(acc_unseen, DECIMAL_NUMBERS)]
    }
    
    print("="*50)
    print("SAVING VERSION 2 RESULTS...")
    print("="*50)
    
    # Guardar resultados
    pd.DataFrame(results_selection).to_csv(path_results_no_cv, index=False)
    print(f"✓ No CV results saved: {path_results_no_cv}")
    
    pd.DataFrame(df_masks).to_csv(path_masks_no_cv, index=False)
    print(f"✓ No CV masks saved: {path_masks_no_cv}")
    
    # Mostrar resultados finales
    print(f"\nFINAL RESULTS:")
    print(f"Best Attributes: {best_config['Attributes']}")
    print(f"Train Accuracy: {round(acc_tr, DECIMAL_NUMBERS)}")
    print(f"Test Seen Accuracy: {round(acc_seen, DECIMAL_NUMBERS)}")
    print(f"Test Unseen Accuracy: {round(acc_unseen, DECIMAL_NUMBERS)}")
    print("="*50)




############################################################################################################
#                                       VERSIÓN 3: HAIS CON LA AGRAGCION                                           #
############################################################################################################

elif args.version == 3: # HAIS Aggregation - Version 3 Unique
    # PART 1: Cross-Validation + FS with 3 algorithms
    for i, (training, val) in enumerate(folds): 
        start_fold = time.time()
        if args.verbose:
            print(f"Fold {i+1}...")
        
        # 1. Computing the ranking via FS embedded algorithm across the training (psedo-seen data)
        start_ranking = time.time()

        estimator_rf, name_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-3), 'RandomForestClassifier'
        estimator_svc, name_svc = SVC(kernel='linear', random_state=42), 'SVC'
        estimator_lr, name_lr = LogisticRegression(random_state=42, n_jobs=-3), 'LogisticRegression'
        
        # Training the feature selection model
        print(f"Making FS Ranking with RF, SVC, and LR...")
        selector_rf  = EmbeddingSimpleFeatureSelector(estimator_rf, name_rf)
        selector_svc = EmbeddingSimpleFeatureSelector(estimator_svc, name_svc)
        selector_lr  = EmbeddingSimpleFeatureSelector(estimator_lr, name_lr)

        selector_rf.fit(training)
        selector_svc.fit(training)
        selector_lr.fit(training)

        ranking_rf   = selector_rf.get_ranking()
        ranking_svc  = selector_svc.get_ranking()
        ranking_lr   = selector_lr.get_ranking()

        fold_ranking_time = time.time() - start_ranking
        timing_data['ranking_time']['per_fold'].append(fold_ranking_time)

        
        #2. Selecting the optimal number of features via the Validation (pseudo-unseen data)
        start_selection = time.time()
        
        # For each algorithm, find its best configuration in this fold
        fold_algorithms_results = []
        for algo_name, ranking in [('RF', ranking_rf), ('SVC', ranking_svc), ('LR', ranking_lr)]:
            fold_algo_tracking = {'Algorithm': [], 'Fold': [], 'Attributes': [], 'Train Acc': [], 'Val Acc': [], 'Mask': []}
            
            for nv in vattributes:
                mask = ranking <= nv
                sae = SAE()
                # Apply mask to attributes if provided
                if mask is not None:
                    selected_attributes = training.att[:, mask]
                else:
                    selected_attributes = training.att
                sae.fit(training.features, selected_attributes)
                
                # Evaluate using sae_eval functions
                if mask is not None:
                    eval_attributes = attribute[:, mask]
                else:
                    eval_attributes = attribute
                acc_tr, _ = evaluate(sae, eval_attributes, training.features, training.classes, training.labels,awa=awa)
                acc_val, _ = evaluate(sae, eval_attributes, val.features, val.classes, val.labels,awa=awa)
                acc_tr     = round(acc_tr, DECIMAL_NUMBERS)
                acc_val    = round(acc_val,DECIMAL_NUMBERS)
                fold_algo_tracking['Algorithm'].append(algo_name)
                fold_algo_tracking['Fold'].append(i+1)
                fold_algo_tracking['Attributes'].append(nv)
                fold_algo_tracking['Train Acc'].append(acc_tr)
                fold_algo_tracking['Val Acc'].append(acc_val)
                fold_algo_tracking['Mask'].append(mask)
                
                if args.verbose:
                    print(f"{algo_name} - nv: {nv} || Train Acc: {acc_tr} || Val Acc: {acc_val}")
            
            # Get best configuration for this algorithm in this fold
            fold_algo_df = pd.DataFrame(fold_algo_tracking).sort_values(by='Val Acc', ascending=False)
            best_config = fold_algo_df.iloc[0]
            fold_algorithms_results.append(best_config)
        
        # Add all algorithm results from this fold to the global list
        best_folds_configuration.extend(fold_algorithms_results)
        
        fold_selection_time = time.time() - start_selection
        timing_data['feature_selection_time']['per_fold'].append(fold_selection_time)
        
        # Record fold time
        fold_time = time.time() - start_fold
        timing_data['fold_times'].append(fold_time)
    
    # Calculate average fold time
    timing_data['average_fold_time'] = sum(timing_data['fold_times']) / len(timing_data['fold_times'])
    
    # Calculate ranking time statistics
    timing_data['ranking_time']['total'] = sum(timing_data['ranking_time']['per_fold'])
    timing_data['ranking_time']['average'] = timing_data['ranking_time']['total'] / len(timing_data['ranking_time']['per_fold'])
    
    # Calculate feature selection time statistics
    timing_data['feature_selection_time']['total'] = sum(timing_data['feature_selection_time']['per_fold'])
    timing_data['feature_selection_time']['average'] = timing_data['feature_selection_time']['total'] / len(timing_data['feature_selection_time']['per_fold'])
        
    # Make the consensus (comuting threshold): An element-wise sum of the best configurations extracted from each fold 
    start_consensus = time.time()
    print("Making the consensus...")
    for res in best_folds_configuration:
        conteo = conteo + np.asarray(res['Mask']).astype(int)
    # Turn the best results of each fold into a DataFrame
    for i in range(len(best_folds_configuration)):
        df_results[i] = best_folds_configuration[i].to_dict()
    df_results = pd.DataFrame(df_results).T

    # PART 2: Diving into the threshold
    test_results = {'Cut':[],'Attributes':[],'Train Acc':[],'Test Seen Acc':[],'Test Unseen Acc':[]}
    j = 1
    total_configurations = 3 * FOLDS  # 3 algorithms × number of folds
    cadena = str(total_configurations)
    
    data_to_json = {'Dataset': args.dataset,'Algorithm': 'Aggregation_RF_SVC_LR'}
    thresholds = {}
    print("Executing the thresholds loop...")
    for i in tqdm(range(total_configurations+1),desc="Threshold Loop"):
        
        if i != 0:
            cadena = cadena+'-'+str(total_configurations - i)
        
        test_results['Cut'].append(cadena)
        test_results['Attributes'].append(np.where(conteo>=total_configurations-i)[0].shape[0])
        mejores = np.where(conteo>=total_configurations-i)[0]
        thresholds[total_configurations-i] = list(map(int,mejores)) 
        if mejores.shape[0] > 0:
            # Fit the model
            sae = SAE()
            # Apply mask to attributes
            selected_attributes = training_dataset.att[:, mejores]
            sae.fit(training_dataset.features, selected_attributes)
            
            # Evaluate using sae_eval functions
            eval_attributes = attribute[:, mejores]
            acc_tr, _     = evaluate(sae, eval_attributes, training_dataset.features, training_dataset.classes, training_dataset.labels,awa=awa)
            acc_seen, _   = evaluate(sae, eval_attributes, test_seen_dataset.features, test_seen_dataset.classes, test_seen_dataset.labels,awa=awa)
            acc_unseen, _ = evaluate(sae, eval_attributes, test_unseen_dataset.features, test_unseen_dataset.classes, test_unseen_dataset.labels,awa=awa)
            test_results['Train Acc'].append(round(acc_tr,DECIMAL_NUMBERS))
            test_results['Test Seen Acc'].append(round(acc_seen,DECIMAL_NUMBERS))
            test_results['Test Unseen Acc'].append(round(acc_unseen,DECIMAL_NUMBERS))
        else:
            test_results['Train Acc'].append(0)
            test_results['Test Seen Acc'].append(0)
            test_results['Test Unseen Acc'].append(0)
    
    # Record consensus time
    timing_data['consensus_time'] = time.time() - start_consensus
    
    print("="*50)
    print("SAVING RESULTS...")
    print("="*50)
    data_to_json['Thresholds'] = thresholds
    
    # Save JSON masks
    with open(path_masks_json, 'w') as json_file:
        json.dump(data_to_json, json_file, indent=4)
    print(f"✓ Masks JSON saved: {path_masks_json}")
    
    # Save CSV results
    pd.DataFrame(test_results).to_csv(path_results_thresholds, index = False)
    print(f"✓ Thresholds results saved: {path_results_thresholds}")
    
    df_results.to_csv(path_results_selection, index = False)
    print(f"✓ Selection results saved: {path_results_selection}")



# ============================================================================= # 
# ============================= TIMING RESULTS ============================== # 
# ============================================================================= # 

# Record total time
timing_data['total_time'] = time.time() - start_total

# Save timing results to JSON file
time_file = results_dir / 'time.json'
with open(time_file, 'w') as f:
    json.dump(timing_data, f, indent=4)

print(f"✓ Timing results saved: {time_file}")
        
# ============================================================================= # 
# ============================= EXPERIMENT SUMMARY ============================= # 
# ============================================================================= # 

def print_experiment_summary():
    """Print a summary of the completed experiment"""
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Version: {args.version}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Random Seed: {args.seed}")
    
    if args.version == 0:
        print(f"\nVersion 0 - RFS with {args.mode}")
        print("Files generated:")
        print(f"  - Selection results: {path_results_selection}")
        print(f"  - Thresholds results: {path_results_thresholds}")
        print(f"  - Masks JSON: {path_masks_json}")
        
    elif args.version == 1:
        print(f"\nVersion 1 - Random Ranking Comparison")
        print("Files generated:")
        print(f"  - Selection results (with CV): {path_results_selection}")
        print(f"  - Thresholds results (with CV): {path_results_thresholds}")
        print(f"  - Random ranking results (without CV): {path_results_random_no_cv}")
        print(f"  - Random ranking masks (without CV): {path_masks_random_no_cv}")
        print(f"  - Masks JSON: {path_masks_json}")
        
    elif args.version == 2:
        print(f"\nVersion 2 - RFS without Cross-Validation")
        print("Files generated:")
        print(f"  - No CV results: {path_results_no_cv}")
        print(f"  - No CV masks: {path_masks_no_cv}")
        
    elif args.version == 3:
        print(f"\nVersion 3 - RFS Aggregation with RF, SVC, and LR")
        print("Files generated:")
        print(f"  - Selection results: {path_results_selection}")
        print(f"  - Thresholds results: {path_results_thresholds}")
        print(f"  - Masks JSON: {path_masks_json}")
        
    print(f"\nTiming Information:")
    print(f"  - Total execution time: {timing_data['total_time']:.2f} seconds")
    print(f"  - Ranking time (total): {timing_data['ranking_time']['total']:.2f} seconds")
    if timing_data['ranking_time']['per_fold']:
        print(f"  - Ranking time (average per fold): {timing_data['ranking_time']['average']:.2f} seconds")
    if timing_data['fold_times']:
        print(f"  - Average fold time: {timing_data['average_fold_time']:.2f} seconds")
    print(f"  - Feature selection time (total): {timing_data['feature_selection_time']['total']:.2f} seconds")
    if timing_data['feature_selection_time']['per_fold']:
        print(f"  - Feature selection time (average per fold): {timing_data['feature_selection_time']['average']:.2f} seconds")
    if timing_data['consensus_time'] > 0:
        print(f"  - Consensus time: {timing_data['consensus_time']:.2f} seconds")
    print(f"  - Timing details saved: {time_file}")
        
    print(f"\nAll results and masks stored in: {results_dir}")
    print("="*80)

# Call the summary function
print_experiment_summary()
