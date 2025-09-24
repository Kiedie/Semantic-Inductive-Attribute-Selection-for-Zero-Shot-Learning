"""
This file contains the implementation of the Dataset class and functions related to loading and preprocessing data.
The Dataset class is a container for a dataset, which includes features, labels, attributes, and the mode of the dataset. 
It provides methods for reducing the semantic space, obtaining information about the dataset, and scaling the features using different scalers.
The file also includes functions for loading data from a directory, splitting the data into training and test sets, and generating folds for cross-validation. 
These functions make use of the Dataset class to store and manipulate the data.

Main functionalities:

- Dataset class: A container for a dataset with features, labels, attributes, and mode.

- load_data_from_path: Loads data from a directory and returns the contents of the res101.mat and att_splits.mat files.

- load_data: Loads and preprocesses data from a directory, including splitting into training and test sets.

- get_data: This is the main function which loads the data, preparing it for training and test. It returns the objects ready.

- get_folds_unseen: Generates folds for the data and labels, based on the number of unseen classes.

- get_folds_seen: Generates folds for the data and labels, based on the number of seen classes.

- stacked_folds: Combines folds of seen and unseen datasets into StackedDataset objects.

Note: The code provided in the selection is not repeated in the docstring.
"""

import scipy.io 
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pathlib
import typing
from typing import List, Tuple, Dict, Any, Callable, Type
import numpy.typing as npt
import random 
import argparse
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from collections import Counter
import os

Scaler = Type[BaseEstimator]
Arg    = type[argparse.Namespace]

seed=42

random.seed(seed)


class Dataset:
    """
    This is the container for the dataset. It contains the features, labels, attributes and the mode of the dataset.
    """
    
    def __init__(self, 
                features : np.ndarray, 
                labels   : np.ndarray, 
                att      : np.ndarray, 
                mode     : str = 'train'):
        
        """Constructor. It initializes the dataset with the features, labels, attributes and mode.
        Take into account that the 'att' argument must be a matrix with dimension (Number_of_instances, Number_of_attributes). 
        Then, giving that the attributes are repeated for the same class, a new semantic space without repetition is created 'reduce_att'. 
        Therefore two semantic spaces are avaible:
        
        
        
        - att (Number_of_instances, Number_of_attributes) : The labels of this matrix are 'labels' arguments
        - reduce_att: (Number_of_classes, Number_of_attributes) : The labels if this matrix are the unique classes in the dataset, 'self.classes' 

        Raises:
            ValueError: A mode different from 'train', 'test_seen', 'test_unseen' or 'val' has been selected.
        """
        
        
        if mode not in ['train', 'test_seen','test_unseen','val']:
            raise ValueError(f"Mode {mode} not found. Please, select 'train', 'test' or 'val'")
        
        self.features                = features
        self.labels                  = labels 
        self.att                     = att # SS => instances x number of attributes 
        self.classes                 = np.unique(labels)
        self.mode                    = mode
        self.att_reduced             = self._reduce_semantic_space()
        self.device                  = 'cpu' # By default the device is cpu
    
    def to(self, device:str)->None:
        """Set the device where the data is going to be stored. It can be 'cpu' or 'cuda'"""
        import torch
        self.device = device
        
        self.features    = torch.tensor(self.features, dtype = torch.float32).to(device)
        self.labels      = torch.tensor(self.labels,dtype = torch.long).to(device)
        self.att         = torch.tensor(self.att,dtype = torch.float32).to(device)
        self.att_reduced = torch.tensor(self.att_reduced,dtype = torch.float32).to(device)
        self.classes     = torch.tensor(self.classes,dtype = torch.long).to(device)
        
    def to_tensor(self):
        import torch
        """Transform the data to torch.Tensor"""
        self.features    = torch.tensor(self.features).to(self.device)
        self.labels      = torch.tensor(self.labels).to(self.device)
        self.att         = torch.tensor(self.att).to(self.device)
        self.att_reduced = torch.tensor(self.att_reduced).to(self.device)
        self.classes     = torch.tensor(self.classes).to(self.device)
    
    def to_numpy(self):
        import torch
        """Transform the data to numpy"""
        
        if self.device != 'cpu':
            self.features    = self.features.cpu().numpy()
            self.labels      = self.labels.cpu().numpy()
            self.att         = self.att.cpu().numpy()
            self.att_reduced = self.att_reduced.cpu().numpy()
            self.classes     = self.classes.cpu().numpy()
        else:
            self.features    = self.features.numpy()
            self.labels      = self.labels.numpy()
            self.att         = self.att.numpy()
            self.att_reduced = self.att_reduced.numpy()
            self.classes     = self.classes.numpy()
    
    def _reduce_semantic_space(self)->np.ndarray:
        """Reduce the semantic space to only the unique classes in the dataset.add()

        Returns:
            np.ndarray: Reduced semantic spaces
        """
        import torch 
        reduce_ss = [] 
        for label in self.classes:
            index = np.where(self.labels == label)[0][0]
            reduce_ss.append(self.att[index])
            
        if isinstance(self.att, torch.Tensor):
            return torch.stack(reduce_ss)
        else:
            return np.asarray(reduce_ss)
        
        # Esta función la puedo mejorar haciendo
        # attr_reduced = self.att[(torch/np.unique(self.labels))]
        
    def mask(self, mask):
        """ 
        Modify the semantic spaece by applying a mask. 
        The mask can be a list|np.ndarray|torch.Tensor with the indices 
        """
        
        self.att = self.att[:,mask]
        self.att_reduced = self.att_reduced[:,mask]
    
    def info(self):
        print(f"self.mode: {self.mode}")
        print(f"self.features.shape: {self.features.shape}")
        print(f"self.labels.shape: {self.labels.shape}")
        print(f"self.att.shape (Semantic Space): {self.att.shape}")
        print(f"self.att_reduced.shape: {self.att_reduced.shape}")
        print(f"self.classes.shape: {self.classes.shape}")

class StackedDataset:
    """
    Another container for the datasets. It usually contains a dict with the next information:
    {train, val, test_seen, test_unseen}
    """
    def __init__(self, datasets:Dict[str, Dataset]):
        self.datasets = datasets
        self.keys     = datasets.keys()



########################################
##########    NORMALIZATION   ##########
########################################

def l2_normalization(att:np.ndarray)->np.ndarray:
    "Normalization L2 of the attributes matrix"
    norm = np.sqrt(np.power(att,2).sum(axis=1))[:,np.newaxis]
    return att / norm


def get_scaler(scaler_str: str = 'Standard')-> Scaler:
    "Return the scaler Scale the data with the scaler function. It is expected that the scaler function had been previously initialized and came from sklearn.preprocessing"
    
    if scaler_str in {'Standard', 'StandardScaler'}:
        scaler = preprocessing.StandardScaler(copy=True)
    elif scaler_str == 'MinMax':
        scaler = preprocessing.MinMaxScaler(copy=True)
    else:
        raise ValueError(f"Scaler {scaler} not found. Please, select either 'Standard or StandardScaler' or 'MinMax'")
    
    return scaler 


def scaler_data(scaler:Scaler, training:np.ndarray, *args)-> Tuple[np.ndarray, List[np.ndarray]]:
    "Scale the data with the scaler function. It is expected that the scaler function has been previously initialized and came from sklearn.preprocessing"
    
    scaler.fit(training)
    
    training_ret = scaler.transform(training)
    ret = []
    for arg in args:
        if arg.ndim == 1:
            arg = arg.reshape(-1, 1)
        ret.append(scaler.transform(arg))
    if len(ret) == 0:
        return training_ret
    else:
        return training_ret, *ret


def scaler_datasets_features(dataset_training    : Dataset, 
                            dataset_test_seen   : Dataset, 
                            dataset_test_unseen : Dataset,
                            dataset_val         : Dataset = None):
    """Escala los conjuntos de datos usando StandardScaler"""
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataset_training.features)
    dataset_training.features = scaler.transform(dataset_training.features)
    dataset_test_seen.features = scaler.transform(dataset_test_seen.features)
    dataset_test_unseen.features = scaler.transform(dataset_test_unseen.features)
    
    if dataset_val is not None:
        dataset_val.features = scaler.transform(dataset_val.features)



########################################
#####      MAPPING FUNCTIONS    ########
########################################

def map_label(label:np.ndarray, classes:np.ndarray)->np.ndarray:
    """
    Maps the labels in the 'label' array to their corresponding indices in the 'classes' array.

    Parameters:
    label (numpy.ndarray): Array of labels to be mapped.
    classes (numpy.ndarray): Array of unique classes.

    Returns:
    numpy.ndarray: Array of mapped labels.
    """
    """
    Tienes el conjunto de etiquetas de entrenamiento y un vector que te dice
    cuales son las claes vistas. Entonces está función te devuelve un vector que contiene
    los índices que corresponden a las etiquetas de entrenamiento en el vector de clases vistas
    Por ejemplo, si tienes la etiqueta 196, entonces se devolverá el núymero 146, ya que la posición
    146 del vector de clases vistas es contiene la etiqueta 196
    """
    mapped_label = np.zeros(label.shape)
    for i in range(classes.shape[0]):
        mapped_label[label == classes[i]] = i
    return mapped_label
    

def get_data_from_label(data:np.ndarray,labels:np.ndarray,label:int)->np.ndarray:
    """
    Get the instances belonging to a certain label from the dataset you specify in the arguments.

    Parameters:
    data (numpy.ndarray): Dataset.
    labels (numpy.ndarray): Labels of the dataset.
    label (int): Label to be extracted.

    Returns:
    numpy.ndarray: Data from the specified label.
    """
    return np.asarray([data[i] for i in range(len(labels)) if labels[i] == label])





########################################
########     SS OBTAINER      ##########
########################################

def gen_ss_from_data(labels:np.ndarray, attributes:np.ndarray) -> np.ndarray:
    """
    Generate a Semantic Space from the labels and attributes. 
    The attributes matrix has de form (Number_of_instances, Number_of_attributes)
    
    - Note: It should be noted that each row represents a class and each columns represents a vector of atributtes.
            Therefore, several columns are repeated due to instances with the same label. 
    
    Parameters:
    labels (numpy.ndarray): Vector de etiquetas.
    attributes (numpy.ndarray): Matriz de atributos.
    
    Returns:
    numpy.ndarray: Espacio semántico.
    """
    return np.asarray([attributes[label] for label in labels])




########################################
#####      LOADING FUNCTIONS    ########
########################################

def load_data_from_path(path:pathlib.Path, show: bool = True)-> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Carga los datos de un directorio. El directorio tiene que contener dentro los archivos res101.mat y att_splits.mat. 
    Cuando los cargas son dos diccionarios y corresponden con las características del dataset extraidas de ResNet101, con las notaciones y divisiones del dataset.

    Args:
        path (pathlib.Path): Ruta del directorio que contiene los archivos res101.mat y att_splits.mat.
        show (bool, optional): Verbose mode. Defaults to True.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: Contenido del fichero res101.mat y att_splits.mat.
    """
    
    if not os.path.exists(path):
        raise Exception(f"Error: La carpeta de datos en {path} no existe.")
    else:
        if show:
            print(f"La carpeta de datos en {path} existe.")
    
    matcontent = scipy.io.loadmat(path / 'res101.mat')
    att_splits = scipy.io.loadmat(path / 'att_splits.mat')
    
    if show:
        print("========== res101.mat ==========")
        print(f"Keys:     {matcontent.keys()}")
        print(f"Features: {matcontent['features'].shape}")
        print(f"Labels:   {matcontent['labels'].shape}")
        
        print("\n\n========== att_split.mat ==========")
        print(f"Keys: {att_splits.keys()}")
        print(f"allclasses_names: {att_splits['allclasses_names'].shape}")
        print(f"att:              {att_splits['att'].shape}")
        print(f"original_att:     {att_splits['original_att'].shape}")
        print(f"test_unseen:      {att_splits['test_unseen_loc'].shape}")    
        print(f"test_seen:        {att_splits['test_seen_loc'].shape}")
        print(f"trainval:         {att_splits['trainval_loc'].shape}")
        print(f"train:            {att_splits['train_loc'].shape}")
        print(f"val:              {att_splits['val_loc'].shape}")
    
    return matcontent, att_splits


   
########################################
##     MAIN DATA LOADING FUNCTIONS    ##
########################################

# Esta función divide los datos en entrenamiento (training y validacion ) y test. 
# No creo que la vaya a usar porque a la hora de hacer cross-validation no vamos a poder aplicarla. En su lugar incluyo otra.
def load_data(path:pathlib.Path,
              percentage_test_unseen:int = 0.3,
              percentage_test_seen:int   = 0.6, 
              scaler_str:str           = 'Standard',
              preprocessing_opt:bool   = True,
              orig_attribute:bool      = False, 
              show:bool                = True
              )->Dict[str,np.ndarray]:
    
    # Load the data
    matcontent, att_splits = load_data_from_path(path, show = False )

    """
    Comentar que en los datos de este datasets viene una partición establecida para train y val. Sin embargo, la partición de 'val' no contempla
    clases Unseen. Por tanto no la vamos usar. Crearemos una partición de validación para clases Seen y Unseen a partir de la partición de test. 
    Como conjunto de entrenamiento usaremos el que viene dado por los índices de la variable 'trainval_loc' en att_splits. 
    Los índices de las varaibles 'train_loc' y 'val_loc' no los vamos a usar.
    
    IMPORTANTE: Esta función no tiene en cuenta cuando TRAINVAL = FAlse, esto es, el conjunto de entrenamiento y validación lo junta en uno. No los separa.
    """
    # Get the data from att_splits dictionary
    allclasses_names            = att_splits['allclasses_names']
    original_att                = att_splits['original_att'].T.astype('float')  # Attributos originales
    attribute                   = att_splits['att'].T.astype('float')           # Atributos que no se de donde salen, pero son los que usan <=== IMPORTANTE
    # Indixes of the instances in the train, test and validation sets
    test_unseen_loc             = att_splits['test_unseen_loc'].squeeze() - 1
    test_seen_loc               = att_splits['test_seen_loc'].squeeze() - 1
    trainval_loc                = att_splits['trainval_loc'].squeeze() - 1      # En caso de no usar validación, aquí estara todo el entrenamiento
    train_loc                   = att_splits['train_loc'].squeeze() - 1         # Esto es si usamos validacion, ver comentario arriba
    val_loc                     = att_splits['val_loc'].squeeze() - 1           # Esto es si usamos validacion, ver comentario arriba
    # Get the dta from matcontent dictionary
    feature =  matcontent['features'].T                       # Feature Matrix (Number of instances x Number of features)
    labels  =  matcontent['labels'].astype(int).squeeze() - 1 # Ponemos que sea entero, eliminamos la dimension extra y con el '-1' movemos el rango de [1,200] a [0,199]



    # Obtaining train and test data
    train_feature         = feature[trainval_loc]
    test_seen_feature     = feature[test_seen_loc]
    test_unseen_feature   = feature[test_unseen_loc]

    # Obtaining train and test labels
    train_label          = labels[trainval_loc]
    test_unseen_label    = labels[test_unseen_loc]
    test_seen_label      = labels[test_seen_loc]


    unseenclasses = np.unique(test_unseen_label)
    seenclasses   = np.unique(train_label)
    
    
    ##########################################################################################################################################
    ############################################### PARTICIÓN EN VALIDACIÓN Y TEST  ##########################################################                    
    ##########################################################################################################################################


    ##############################################
    # Partición TEST_SEEN & VAL_SEEN
    ##############################################
    """ 
    Hacemos una partición del conjunto de test_seen en dos partes, una para validación y otra para test.
    Usamos la función train_test_split de sklearn para hacer la partición de manera estratificada. 
    """
    # Particionamos el conjunto de test_seen en dos partes, una para validación y otra para test
    val_seen_feature, test_seen_feature, val_seen_label, test_seen_label = train_test_split(test_seen_feature, 
                                                                                            test_seen_label, 
                                                                                            test_size = percentage_test_seen, 
                                                                                            random_state=seed,
                                                                                            stratify=test_seen_label)

    # Imprimo información sobre tamaños
    if show:
        print(f"val_seen_feature: {val_seen_feature.shape}")
        print(f"test_seen_feature: {test_seen_feature.shape}")
        print(f"val_seen_label: {val_seen_label.shape}")
        print(f"test_seen_label: {test_seen_label.shape}")
        print(f"Counter(test_seen_label_feature): {Counter(test_seen_label)}")
        print(f"Counter(val_seen_label_feature): {Counter(val_seen_label)}")



    ##############################################
    # Partición TEST_UNSEEN & VAL_UNSEEN
    ##############################################
    """
    Hacemos una partición del conjunto de test_unseen en dos partes, una para validación y otra para test.
    El array test_unseen_label contiene las etiquetas de manera contigua, no está desordenado. 
    """
    # Calculamos el numero total de clases no vistas y seleccionamos el porcentage para validacion 
    total_classes = np.unique(test_unseen_label).shape[0]

    percentage_test_unseen = 0.3

    n_classes_to_test = int(total_classes*percentage_test_unseen)
    n_classes_to_val  = unseenclasses.shape[0] - n_classes_to_test

    random_classes_to_test = np.asarray(random.sample(unseenclasses.tolist(),n_classes_to_test))
    random_classes_to_val  = np.asarray([i for i in unseenclasses if i not in random_classes_to_test])
    random.shuffle(random_classes_to_val)

    index_val  = [i for i in range(len(test_unseen_label)) if test_unseen_label[i] in random_classes_to_val]
    index_test = [i for i in range(len(test_unseen_label)) if test_unseen_label[i] in random_classes_to_test]

    val_unseen_feature  = test_unseen_feature[index_val]
    val_unseen_label    = test_unseen_label[index_val]
    test_unseen_feature = test_unseen_feature[index_test]
    test_unseen_label   = test_unseen_label[index_test]

    val_unseen_classes  = np.unique(val_unseen_label)
    test_unseen_classes = np.unique(test_unseen_label)




    if show:
        print(f"Unseen Clases: {np.unique(test_unseen_label)}")
        print(f"Size: {total_classes}")
        print(f"val_unseen_classes: {np.unique(val_unseen_classes).shape}") 
        print(f"test_unseen_classes: {np.unique(test_unseen_classes).shape}")
        # Imprimo los tamaños 
        print(f"val_unseen_label: {val_unseen_label.shape}")
        print(f"val_unseen_feature: {val_unseen_feature.shape}")
        print(f"test_unseen_label: {test_unseen_label.shape}")
        print(f"test_unseen_feature: {test_unseen_feature.shape}")


    assert all([i in val_unseen_classes for i in val_unseen_label]), "Error: There are labels in val_unseen_label_split that are not in val_unseen_classes"
    assert all([i in test_unseen_classes for i in test_unseen_label]), "Error: There are labels in test_unseen_label_split that are not in test_unseen_classes"   




    ##########################################################################################################################################
    ####################################################### PREPROCESSING 1 ##################################################################
    ##########################################################################################################################################



    # Choosing the attributes we want to use 
    attribute = original_att if orig_attribute else attribute


    # Applying the scaler to the data if necessary
    if preprocessing_opt:
        scaler = get_scaler(scaler_str)
        
        scaler.fit(train_feature)
        train_feature           = scaler.transform(train_feature)
        val_seen_feature        = scaler.transform(val_seen_feature)
        val_unseen_feature      = scaler.transform(val_unseen_feature)
        test_seen_feature       = scaler.transform(test_seen_feature)
        test_unseen_feature     = scaler.transform(test_unseen_feature)
        
        
        
        # Scaler and transfer attributes
        scaler = get_scaler(scaler_str)
        attribute = scaler.fit_transform(attribute)


    # Espacios semanticos asociados a los conjuntos de X_train, test_seen y test_unseeen. 
    # Estos SS serán matrices con tantas filas como ejemplos y columnas como atributos
    # Los vectores de atributios (columnas) se repiten para aquellas instancias que tengan la misma clase
    S_train          = gen_ss_from_data(train_label,attribute)
    S_test_seen      = gen_ss_from_data(test_seen_label,attribute)
    S_test_unseen    = gen_ss_from_data(test_unseen_label,attribute)
    S_val_seen       = gen_ss_from_data(val_seen_label,attribute)
    S_val_unseen     = gen_ss_from_data(val_unseen_label,attribute)
    semantic_gt      = gen_ss_from_data(np.unique(test_seen_label),attribute)
    S_gt_training    = gen_ss_from_data(np.unique(train_label),attribute)
    S_gt_val_seen    = gen_ss_from_data(np.unique(val_seen_label),attribute)
    S_gt_val_unseen  = gen_ss_from_data(np.unique(val_unseen_label),attribute)
    S_gt_test_seen   = gen_ss_from_data(np.unique(test_seen_label),attribute)
    S_gt_test_unseen = gen_ss_from_data(np.unique(test_unseen_label),attribute)



    if show:
        print(f"S_train: {S_train.shape}")
        print(f"S_test_seen: {S_test_seen.shape}")
        print(f"S_test_unseen: {S_test_unseen.shape}")
        print(f"S_val_seen: {S_val_seen.shape}")
        print(f"S_val_unseen: {S_val_unseen.shape}")
        print(f"semantic_gt: {semantic_gt.shape}")
        print(f"S_gt_training: {S_gt_training.shape}")
        print(f"S_gt_val_seen: {S_gt_val_seen.shape}")
        print(f"S_gt_val_unseen: {S_gt_val_unseen.shape}")
        print(f"S_gt_test_seen: {S_gt_test_seen.shape}")
        print(f"S_gt_test_unseen: {S_gt_test_unseen.shape}")
        
    ret = {
        'train_feature':train_feature,
        'train_label':train_label,
        'S_train':S_train,
        'test_seen_feature':test_seen_feature,
        'test_seen_label':test_seen_label,
        'S_test_seen':S_test_seen,
        'test_unseen_feature':test_unseen_feature,
        'test_unseen_label':test_unseen_label,
        'S_test_unseen':S_test_unseen,
        'val_seen_feature':val_seen_feature,
        'val_seen_label':val_seen_label,
        'S_val_seen':S_val_seen,
        'val_unseen_feature':val_unseen_feature,
        'val_unseen_label':val_unseen_label,
        'S_val_unseen':S_val_unseen,
        'semantic_gt':semantic_gt,
        'S_gt_training':S_gt_training,
        'S_gt_val_seen':S_gt_val_seen,
        'S_gt_val_unseen':S_gt_val_unseen,
        'S_gt_test_seen':S_gt_test_seen,
        'S_gt_test_unseen':S_gt_test_unseen,
        'attribute':attribute
    }
    
    return ret
        
        
        
def get_data(path_data              : str, 
             args                   : Arg,
             preprocessing          : bool = False,
             trainval               : bool = True) -> Tuple[Dataset, Dataset, Dataset, np.ndarray]:
    
    """
    Retrieves data from the specified path and prepares it for training and testing.
    ----------
    Args:
        path_data (str): 
            The path to the data.
            
        preprocessing (bool), default False_
            Wheter to apply preprocessing to the semantic space and images features or not. This argumets concerns the split for cross-validation. There the val and train data are disjoint.add()
            Therefore, we cannot apply this a previous preprocessing to the semantic space as a whole. 
                    
        args (Arg):     
            The arguments for data processing.
            
        trainval (bool, optional): 
            Whether to include the validation data into the training data. Default, True. 
    ----------
    Returns:
        Tuple[Dataset, Dataset, Dataset, np.ndarray]: 
            A tuple containing the training dataset, test seen dataset, test unseen dataset, and attribute array.
    ----------        
    Notes:
        If trainval is True, four elements will be outputted. Otherwise, five elements will be outputted.

    """

    matcontent, att_splits = load_data_from_path(path_data, show = False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Attributes  
    if args.dataset != 'FLO':
        original_att                = att_splits['original_att'].T.astype('float')  # Attributos originales
        attribute                   = att_splits['att'].T.astype('float')           # Atributos que no se de donde salen, pero son los que usan <=== IMPORTANTE
        attribute = original_att if args.orig_attribute else attribute
    else:
        attribute                   = att_splits['att'].T.astype('float') 
    
    if preprocessing:
        attribute = scaler_data(get_scaler(),attribute)


    # Indixes 
    test_unseen_loc             = att_splits['test_unseen_loc'].squeeze() - 1
    test_seen_loc               = att_splits['test_seen_loc'].squeeze()   - 1
    trainval_loc                = att_splits['trainval_loc'].squeeze()    - 1  # En caso de no usar validación, estos son todos los datos de entrenamiento
    train_loc                   = att_splits['train_loc'].squeeze()       - 1  # Si usamos validación, estos serán ahora los datos de entrenamiento => NO LO USAMOS
    val_loc                     = att_splits['val_loc'].squeeze()         - 1  # Si usamos validación, estos serán los datos de validación => NO LO USAMOS

    # Instance features and label
    feature =  matcontent['features'].T                       # Feature Matrix (Number of instances x Number of features)
    labels  =  matcontent['labels'].astype(int).squeeze() - 1 # Ponemos que sea entero, eliminamos la dimension extra y con el '-1' movemos el rango de [1,200] a [0,199]


    # Generate the semantic Spaces for Testing instances
    test_seen_ss   = gen_ss_from_data(labels[test_seen_loc], attribute) 
    test_unseen_ss = gen_ss_from_data(labels[test_unseen_loc], attribute) 

    test_seen_dataset = Dataset(features = feature[test_seen_loc],
                                        labels   = labels[test_seen_loc],
                                        att      = test_seen_ss,
                                        mode     = 'test_seen')

    test_unseen_dataset = Dataset(features= feature[test_unseen_loc],
                                labels  = labels[test_unseen_loc],
                                att     = test_unseen_ss,
                                mode    = 'test_unseen')

    if trainval:
        random.shuffle(trainval_loc)
        
        trainval_classes = np.unique(labels[trainval_loc])
        training_ss      = gen_ss_from_data(labels[trainval_loc], attribute) 
        
        # Inside Dataset object the classes are computing via np.unique(labels)
        training_dataset = Dataset(features = feature[trainval_loc],
                                    labels   = labels[trainval_loc],
                                    att      = training_ss,
                                    mode     = 'train')

        
        if preprocessing:
            scaler_datasets_features(training_dataset, test_seen_dataset, test_unseen_dataset)

        return training_dataset, test_seen_dataset, test_unseen_dataset, attribute
    # Esta parte de aquí vamos a evitar usarla porque nosotros hacemos nuestra propia partición de los datos, sobre todo si usamos cross-validation.
    else: 
        random.shuffle(val_loc)
        random.shuffle(train_loc)
        
        train_classes = np.unique(labels[train_loc])
        val_classes   = np.unique(labels[val_loc])
        
        training_ss = gen_ss_from_data(labels[train_loc], attribute) 
        val_ss      = gen_ss_from_data(labels[val_loc], attribute) 
        
        # Inside Dataset object the classes are computing via np.unique(labels)
        training_dataset = Dataset(features = feature[train_loc],
                                    labels   = labels[train_loc],
                                    att      = training_ss,
                                    mode     = 'train')
        
        val_dataset = Dataset(features = feature[val_loc],
                                labels   = labels[val_loc],
                                att      = val_ss,
                                mode     = 'val')
        
        if preprocessing:
            scaler_datasets_features(training_dataset, test_seen_dataset, test_unseen_dataset, val_dataset)

        
        return training_dataset, val_dataset, test_seen_dataset, test_unseen_dataset, attribute
        