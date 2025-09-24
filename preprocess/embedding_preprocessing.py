"""
This file outputs the semantic spaces preprocessing according to the arguments given in the command line
into the folder '../data/processed semantic spaces/embedding/{args.dataset}/{args.mode}/' 

Clases:
    -

Funciones:
    - 
"""



import scipy.io 
import os
import numpy as np
import pandas as pd
import pathlib
import typing
import argparse
import numpy.typing as npt
from sklearn import preprocessing
from sklearn.svm import SVC 
from auto_tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import List, Tuple, Dict, Any, Callable, Type
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from collections import Counter
import random 
from data.datareader import *
from models.SAE import *
from data.split import SplitIntoFolds
from featureselectors import EmbeddingSimpleFeatureSelector
from sklearn.linear_model import LogisticRegression

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def scaler_datasets(dataset_training:    Dataset, 
                    dataset_test_seen:   Dataset, 
                    dataset_test_unseen: Dataset):
    """Make a StandardScaler transformation to the datasets. 

    Args:
        dataset_training    (np.ndarray): 
        dataset_test_seen   (np.ndarray): 
        dataset_test_unseen (np.ndarray): 
    """
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataset_training.features)
    
    dataset_training.features    = scaler.transform(dataset_training.features)
    dataset_test_seen.features   = scaler.transform(dataset_test_seen.features)
    dataset_test_unseen.features = scaler.transform(dataset_test_unseen.features)


# ============================================================================= # 
# ============================= GLOBALS VARIABLES ============================= # 
# ============================================================================= # 
CONTRIBUTION = 'PATTERN RECOGNICTION'
EXPERIMENT   = 'SimpleFeatureSelection'

DECIMAL_NUMBERS  = 3
FOLDS            = 5
scaler_str       = 'Standard'     # ['Standard','MinMax']
orig_attribute   = False # Attributos originales u otros
DEBUG            = True 
SHOW             = True 
hitk             = 1



# ============================================================================= # 
# ============================= ARGUMENTS & PATHS ============================= # 
# ============================================================================= # 
parser = argparse.ArgumentParser(description=CONTRIBUTION)
parser.add_argument('-d',   '--dataset', type = str,  default = 'CUB', help='DataSet Name', choices=['CUB', 'SUN', 'FLO', 'AWA2'])
parser.add_argument('-rss', '--redundant_ss', type = bool, default = True, help='Using Redundant Semantic Space or not')
parser.add_argument('-s',   '--seed',    type = int,  default = 42, help='Random Seed')
parser.add_argument('-stp', '--step',    type = int,  default = 5, help='Step of attributes to select')
parser.add_argument('-v',   '--verbose', type = str2bool, default = True, help='Verbose Mode')
parser.add_argument('-m',   '--mode',    type = str,  choices = ["RandomForest","SVC",'SAE','LR'] ,help='Mo Feature Selection')
parser.add_argument('-sm',   '--sae_mode',type = str, default="norm" ,choices = ["mean","median","norm"] ,help='Way in which coefficients are obtained')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)


dir_data     = pathlib.Path('../data/')
path_data    = dir_data / args.dataset 

dir_results  = pathlib.Path(f'Results/{CONTRIBUTION}/{args.dataset}/{EXPERIMENT}')
path_results_training = dir_results / f'Embedded_SS_{args.mode}_training.csv' if args.mode != 'SAE' else dir_results / f'SimpleFS_{args.mode}_{args.sae_mode}_training.csv'
path_results_test     = dir_results / f'Embedded_SS_{args.mode}_test.csv' if args.mode != 'SAE' else dir_results / f'SimpleFS_{args.mode}_{args.sae_mode}_test.csv'


if not os.path.exists(dir_results):
    os.makedirs(dir_results)
    print(f"Directory {dir_results} does not exist, but was created ")
    
    
    
# ============================================================================= # 
# =============================== DATA READING ================================ # 
# ============================================================================= # 



matcontent, att_splits = load_data_from_path(path_data, show = False)

### 1. Read Data From file and scale attributes

if args.dataset != 'FLO':
    original_att                = att_splits['original_att'].T.astype('float')  # Attributos originales
    attribute                   = att_splits['att'].T.astype('float')           # Atributos que no se de donde salen, pero son los que usan <=== IMPORTANTE
    attribute = original_att if orig_attribute else attribute
else:
    attribute                   = att_splits['att'].T.astype('float') 
attribute = scaler_data(get_scaler(scaler_str),attribute)

# Indixes 
test_unseen_loc             = att_splits['test_unseen_loc'].squeeze() - 1
test_seen_loc               = att_splits['test_seen_loc'].squeeze()   - 1
trainval_loc                = att_splits['trainval_loc'].squeeze()    - 1  # En caso de no usar validación, aquí estara todo el entrenamiento
train_loc                   = att_splits['train_loc'].squeeze()       - 1  # Esto es si usamos validacion, ver comentario arriba
val_loc                     = att_splits['val_loc'].squeeze()         - 1  # Esto es si usamos validacion, ver comentario arriba

# Instance features and label
feature =  matcontent['features'].T                       # Feature Matrix (Number of instances x Number of features)
labels  =  matcontent['labels'].astype(int).squeeze() - 1 # Ponemos que sea entero, eliminamos la dimension extra y con el '-1' movemos el rango de [1,200] a [0,199]

random.shuffle(trainval_loc)
trainval_classes = np.unique(labels[trainval_loc])

### 2. Generate Semantic Spaces (Number_instances x Number_attributes). Some rows are repeated
training_ss    = gen_ss_from_data(labels[trainval_loc], attribute) if args.redundant_ss else attribute[trainval_classes]
test_seen_ss   = gen_ss_from_data(labels[test_seen_loc], attribute) if args.redundant_ss else attribute[np.unique(labels[test_seen_loc])]
test_unseen_ss = gen_ss_from_data(labels[test_unseen_loc], attribute) if args.redundant_ss else attribute[np.unique(labels[test_unseen_loc])]

### 3. Generate Datasets objects which we will work later.
training_dataset = Dataset(features = feature[trainval_loc],
                            labels   = labels[trainval_loc],
                            att      = training_ss,
                            mode     = 'train')

test_seen_dataset = Dataset(features = feature[test_seen_loc],
                            labels   = labels[test_seen_loc],
                            att      = test_seen_ss,
                            mode     = 'test_seen')

test_unseen_dataset = Dataset(features= feature[test_unseen_loc],
                            labels  = labels[test_unseen_loc],
                            att     = test_unseen_ss,
                            mode    = 'test_unseen')


scaler_datasets(training_dataset.features, test_seen_dataset.features, test_unseen_dataset.features)




# ============================================================================= # 
# =========================== CROSS FOLD VALIDATION =========================== # 
# ============================================================================= # 

classes_per_fold = int(trainval_classes.shape[0] / FOLDS)
acarreo          = trainval_classes.shape[0] % FOLDS
folds            = SplitIntoFolds(FOLDS,show = args.verbose).split_via_classes_per_folds(feature[trainval_loc], labels[trainval_loc], attribute, preprocessing = True)


vattributes = np.arange(25, attribute.shape[1], args.step)
#vattributes = np.array([45,47])
if attribute.shape[1] % args.step != 0:
    vattributes = np.append(vattributes, attribute.shape[1])

if args.mode == 'RandomForest':
    estimator, name = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-3), 'RandomForestClassifier'
elif args.mode == 'SVC':
    estimator, name = SVC(kernel='linear', random_state=42), 'SVC'
elif args.mode == 'SAE':
    estimator, name = Sae(), 'SAE'
elif args.mode in ['LR', 'LogisticRegression']:
    estimator, name = LogisticRegression(random_state=42, max_iter=1000), 'LogisticRegression'


df_results = {}                             # Dataframe con los mejores resultados de cada fold
conteo = np.zeros(attribute.shape[1])       # Conseso de las soluciones de los folds
best_folds_configuration = []               # Best attributes configuration for each fold

for i, fold in enumerate(folds):

    if args.verbose:
        print(f"Fold {i+1}...")

    training_data, val_data = fold.datasets['train'], fold.datasets['val']
    
    fold_i_info = {'Attributes': [], 'Train Acc':[], 'Val Acc':[], 'Mask':[]}
    
    selector  = EmbeddingSimpleFeatureSelector(estimator,name,args.sae_mode)
    selector.fit(training_data)
    ranking  = selector.get_ranking()
        
    for nv in vattributes: 
        mask = ranking <= nv
        sae  = Sae(mask = mask)        
        sae.fit(training_data.features, training_data.att)
        
        acc_tr,  _ = sae.evaluate(attribute, training_data.features, training_data.classes, training_data.labels)
        acc_val, _ = sae.evaluate(attribute, val_data.features, val_data.classes, val_data.labels)
        acc_tr     = round(acc_tr, DECIMAL_NUMBERS)
        acc_val    = round(acc_val,DECIMAL_NUMBERS)
        
        fold_i_info['Attributes'].append(nv)
        fold_i_info['Train Acc'].append(acc_tr)
        fold_i_info['Val Acc'].append(acc_val)
        fold_i_info['Mask'].append(mask)
        
        if args.verbose:
            print(f"nv: {nv} || Train Acc: {acc_tr} || Val Acc: {acc_val}")

    fold_i_info = pd.DataFrame(fold_i_info).sort_values(by = 'Val Acc', ascending = False)
    best_folds_configuration.append(fold_i_info.iloc[0])
    

# Hacemos el consenso de las soluciones 
for res in best_folds_configuration:
    conteo = conteo + np.asarray(res['Mask']).astype(int)
# Convertimos los resultados a un dataframe
for i in range(len(best_folds_configuration)):
    df_results[i] = best_folds_configuration[i].to_dict()
df_results = pd.DataFrame(df_results).T






# ============================================================================= # 
# ================================    TEST    ================================= # 
# ============================================================================= # 

test_results = {'Cut':[],'Attributes':[],'Train Acc':[],'Test Seen Acc':[],'Test Unseen Acc':[]}
j = 1
cadena = str(FOLDS)

for i in range(FOLDS+1):
    
    if i != 0:
        cadena = cadena+'-'+str(FOLDS - i)
    
    test_results['Cut'].append(cadena)
    test_results['Attributes'].append(np.where(conteo>=FOLDS-i)[0].shape[0])
    mejores = np.where(conteo>=FOLDS-i)[0]
    
    if mejores.shape[0] > 0:
        
        # Fit the model
        sae = Sae(mask = mejores)
        sae.fit(training_dataset.features, training_dataset.att)
        
        # Evaluate
        acc_tr, _     = sae.evaluate(attribute, training_dataset.features, training_dataset.classes, training_dataset.labels)
        acc_seen, _   = sae.evaluate(attribute, test_seen_dataset.features, test_seen_dataset.classes, test_seen_dataset.labels)
        acc_unseen, _ = sae.evaluate(attribute, test_unseen_dataset.features, test_unseen_dataset.classes, test_unseen_dataset.labels) 
        
        test_results['Train Acc'].append(round(acc_tr,DECIMAL_NUMBERS))
        test_results['Test Seen Acc'].append(round(acc_seen,DECIMAL_NUMBERS))
        test_results['Test Unseen Acc'].append(round(acc_unseen,DECIMAL_NUMBERS))
    else:
        test_results['Train Acc'].append(0)
        test_results['Test Seen Acc'].append(0)
        test_results['Test Unseen Acc'].append(0)
    

pd.DataFrame(test_results).to_csv(path_results_test, index = False)
df_results.to_csv(path_results_training, index = False)