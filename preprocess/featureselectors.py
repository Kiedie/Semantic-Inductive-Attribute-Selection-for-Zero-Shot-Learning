"""
Este módulo define un conjunto de clases y funciones para la selección de características 

Clases:
    - EmbeddingSimpleFeatureSelector: => Embedding
        Implementa una selección de características EMBEBIDAS usando los siguiente modelos auxiliares:
            - SVC
            - RandomForestClassifier
            - Sae (soportado con varios nombres {sae, SAE, Sae}) con sus variantes {mean, median, norm}
            - LogisticRegression
            
    - SequentialSelector: => Wrapper
        Implementa el algoritmo 'Sequential Backward Selection' hecho desde cero. No uso la biblioteca de mxltend.
        El SBS se puede hacer con los siguiente modelos auxiliares:
            - SVC
            - Sae (soportado con varios nombres {sae, SAE, Sae}) y sus variantes {mean, median, norm}
            
    - CustomSequentialFeatureSelector: => Wrapper 
        Implementa una selección de características secuencial ya sea SBS o SFS según la biblioteca mlxtend.
        Es un algoritmo Wrapper y se pueden usar cualquier clasificador que esté programado según los estándares de sklearn.

Funciones:
    Para cada modelo, se implementa un método 'fit' y 'get_ranking' para ajustar el modelo y obtener el ranking de características
    El resto de funciones son privadas.
    
    Recomendable mirar la descripción de cada clase para saber cómo se usa.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from data.datareader import Dataset

from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from typing import List, Dict
from sklearn.neighbors import KDTree

from models.sae import SAE

import typing
from typing import List, Tuple, Dict, Any, Callable, Type
import numpy.typing as npt
from sklearn.base import BaseEstimator
Scaler = Type[BaseEstimator]


class EmbeddingSimpleFeatureSelector:
    
    _supported_estimators = ['SVC','RandomForestClassifier','LR','LoggisticRegression','Sae','SAE','sae']
    _supported_mode_sae   = ['mean','median','norm']

# Private    
    def __init__(self, estimator, name, mode_sae = 'mean'):
        
        if mode_sae not in self._supported_mode_sae:
            raise ValueError(f"Mode {mode_sae} not supported. Supported modes are {self._supported_mode_sae}")
              
        self.estimator = estimator
        self.name      = name
        self.ranking   = None
        self.mode_sae  = mode_sae
        
        
    def _ranking_svc(self, X, y):
        
        self.estimator.fit(X,y)
        
        importances = np.square(self.estimator.coef_).mean(axis=0)
        
        importances = pd.DataFrame({'Index of attribute':np.arange(X.shape[1]),'Importance': importances}).sort_values('Importance',ascending = False)
        importances['Ranking'] = [i for i in range(1,importances.shape[0]+1)]
        
        self.ranking = np.zeros(X.shape[1])
        for index, rank in zip(importances['Index of attribute'],importances['Ranking']):
            self.ranking[index] = rank
                
                
    def _ranking_rf(self, X, y):
        self.estimator.fit(X,y)
        
        importances = self.estimator.feature_importances_
        importances = pd.DataFrame({'Index of attribute':np.arange(X.shape[1]),'Importance': importances}).sort_values('Importance',ascending = False)
        importances['Ranking'] = [i for i in range(1,importances.shape[0]+1)]
        
        self.ranking = np.zeros(X.shape[1])
        for index, rank in zip(importances['Index of attribute'],importances['Ranking']):
            self.ranking[index] = rank
        
            
    def _ranking_sae(self, dataset_train):
    
        
        self.estimator.fit(dataset_train.features, dataset_train.att)
        W = self.estimator.get_W()
        
        if self.mode_sae == 'mean':
            self.ranking = np.argsort(np.mean(np.abs(W), axis = 1))
        elif self.mode_sae == 'median':
            self.ranking = np.argsort(np.median(np.abs(W), axis = 1))
        elif self.mode_sae == 'norm':
            self.ranking = np.argsort(np.linalg.norm(W, axis = 1))
            
    
    def _ranking_lr(self,dataset_train):
        
        self.estimator.fit(dataset_train.att, dataset_train.labels)
        coeff = np.mean(np.abs(self.estimator.coef_), axis=0)
        self.ranking = np.argsort(coeff)
        
# Public
    def fit(self, dataset_train):
        """
        Fit the models. If they are SVC or RandomForest, X and y parameters are use. Otherwise, dataset_train is used
        """
        if isinstance(self.estimator, SVC):
            self._ranking_svc(dataset_train.att,dataset_train.labels)
        elif isinstance(self.estimator, RandomForestClassifier):
            self._ranking_rf(dataset_train.att,dataset_train.labels)
        #elif isinstance(self.estimator, Sae):
        #    self._ranking_sae(dataset_train)
        elif isinstance(self.estimator, LogisticRegression):
            self._ranking_lr(dataset_train)
        else:
            raise ValueError(f"Estimator {self.name} not supported. Supported estimators are {self._supported_estimators}")

        
    def get_ranking(self):
        if self.ranking is None:
            raise ValueError('You must fit the model first')
        else:
            return self.ranking
        

class TrackingData:
    def __init__(self):
        self.metrics = []  # Array para almacenar las métricas
        self.masks = []    # Matriz de máscaras (lista de listas)

    def add_entry(self, metric, mask):
        self.metrics.append(metric)
        self.masks.append(mask)

    def get_tracking(self):
        return {'metrics': np.array(self.metrics), 'masks': np.array(self.masks)}
    
    def sort_by_metrics(self):
        # Combinar métricas y máscaras en una lista de tuplas
        combined = list(zip(self.metrics, self.masks))
        
        # Ordenar las tuplas por la métrica (primer elemento de cada tupla) de mayor a menor
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        
        # Separar las métricas y máscaras ya ordenadas
        self.metrics, self.masks = zip(*combined_sorted)
        
        # Convertir de nuevo las máscaras a listas (ya que zip devuelve tuplas)
        self.metrics = list(self.metrics)
        self.masks = list(self.masks)


 
class SequentialSelector:
    
    NAME = ['sklearn', 'sae']
    
    def __init__(self, estimator, 
                 name:       str, 
                 step:       int        = None, 
                 metrica:    callable   = None, 
                 descending: bool       = True, 
                 debug:      bool       = False):
        
        if name not in self.NAME:
            raise ValueError(f"Name {name} not supported. Supported names are {self.NAME}")
        
        self.estimator = estimator
        self.name       = name
        self.ranking    = []
        self.debug      = debug
        # Solo nos hace falta lo de arriba, esto de abajo no nos hace tanta falta
        self.step       = step
        self.metrica    = metrica
        self.descending = descending
        
        self. tracking = TrackingData()
        
    
    def _backward_selection_svc(self, 
                                data_train : Dataset,
                                data_val   : Dataset,
                                attribute  : np.ndarray = None):
        features = list(range(data_train.att.shape[1]))
        
        if self.debug:
            print("Total number of features: ", len(features))
            
        i = 0
        
        mask = np.ones(len(features)).astype(bool)
        
        with tqdm(total = len(features)) as pbar:
            while(len(features) > 1):
                if self.debug:
                    print(f"========== Iteration {i} ==========")
                    i+=1
                    
                worst_acc  = float("inf")
                worst_feat = None
                
                for feat in features:
                    mask[feat] = False
                    self.estimator.fit(data_train.att[:,mask], data_train.labels)
                    y_pred = self.estimator.predict(data_val.att[:,mask])
                    acc = self.metrica(data_val.labels, y_pred)
                    if acc < worst_acc:
                        worst_acc = acc
                        worst_feat = feat
                    mask[feat] = True
                    if self.debug:
                        print(f"Studing feature {feat} => Accuracy: {acc} || Current candidate {worst_feat} => Accuracy: {worst_acc}")
                
                if self.debug:
                    print(f"Removing feature {worst_feat} with accuracy {worst_acc}")        
                
                mask[worst_feat] = False
                features.remove(worst_feat)
                self.ranking.append(worst_feat)
                pbar.update(1) # Update the progress bar
            
        assert len(features) == 1, "Something went wrong, length of features should be 1"
        
        self.ranking.append(features[0])
        self.ranking = self.ranking[::-1]
        
        return self.ranking
        
        
    def _backward_selection_sae(self, 
                                data_train : Dataset,
                                data_val   : Dataset,
                                attribute  : np.ndarray,):
                    
        """
        This method implements SBS using SAE as the estimator. 
        ----------
        Argumentts:
            - data_train[Dataset]
                The training data
            - data_val[Dataset]
                The validation data
            - attribute[np.ndarray]
                The semantic space
        ----------
        Return:
            - ranking[List]
                The ranking of the feature ordered by the importance level. Better importance to worst importance.
            - tracking[Dict]
                A dictionary with the accuracy as the key and the mask as the value. 
                It is ordered in descending order by the accuracy.
                The dictionary has as many elements as the number of features.
                The mask corresponds to the set of features except the one which is removed. 
        
        """            
        features = list(range(data_train.att.shape[1]))
        mask     = np.ones(len(features)).astype(bool)
        
        if self.debug:
            print("Total number of features: ", len(features))
        
        iteration = 0
        
        if self.debug:
                print(f"========== Iteration {iteration} ==========")
                iteration+=1
        
        with tqdm(total = len(features)) as pbar:
            while(len(features) > 1):
                highest_acc  = float("-inf")
                worst_feat = None
                
                for feat in features:
                    mask[feat] = False
                    
                    self.estimator.set_mask(mask = mask)
                    self.estimator.fit(data_train.features, data_train.att)
                    acc, _ = self.estimator.evaluate(attribute, data_val.features, data_val.classes, data_val.labels)
                                
                    # The worst feature is the one whose removal leads to the least reduction (or largest improvement) in performance.
                    # Thus, We look for the feature that has the highest accuracy.
                    if  acc > highest_acc:
                        highest_acc = acc
                        worst_feat = feat
                        
                    mask[feat] = True
                    if self.debug:
                        print(f"Studing feature {feat} => Accuracy: {acc} || Current candidate {worst_feat} => Accuracy: {highest_acc}")
                
                mask[worst_feat] = False
                features.remove(worst_feat)
                
                self.tracking.add_entry(highest_acc, mask.copy())
                self.ranking.insert(0,worst_feat)
                
                if self.debug:
                    print(f"Removing feature {worst_feat} with accuracy {highest_acc}")        
                
                pbar.update(1) # Update the progress bar
            
            
        assert len(features) == 1, "Something went wrong, length of features should be 1"
        
        self.ranking.insert(0,features[0])    # Include the last
        self.tracking.sort_by_metrics()
        
        
        return self.ranking, self.tracking
    

    def fit(self, data_train : Dataset, data_val : Dataset, attribute):
        return self._backward_selection_sae(data_train, data_val, attribute)    

    def get_ranking(self):
        return self.ranking
    
    def get_tracking(self):
        return self.tracking
        
    def get_indexes_form_mask(self, mask):
        '''
        Given a boolean mask, it returns the indexes of the True values.
        '''
        return [i for i, m in enumerate(mask) if m]


class CustomSequentialFeatureSelector:
    """
    This class is a custom implementation of the SequentialFeatureSelector class of mlxtend
    
    How to use it:
    - (1) Create an instance of this class
    - (2) Call the method `fit` with the data:
        - X: Features, it must be a numpy array matrix
        - y: Labels, it must be a numpy array matrix
    - (3) Call either the method `get_ranking` to get the ranking of the features or `get_best_selection` to get the best selection of features
    
    Example:
    >>> sss = CustomSequentialFeatureSelector(knn,cv=0, k_features = X.shape[1], verbose = True)
    >>> sss.fit(X,y)
    >>> ranking = sss.get_ranking()
    >>> selection = sss.get_best_selection()
    
    You must note that the output of the method `get_best_selection` is a dictionary with the following structure:
    [
        {'features_idx': [1,2,3], 'score': 0.9},
    ]
    'Features idx' is a list with the indexes of the features that are selected and 'score' contains the score of the model with those features. 
    It is important to remark that the features are ordered by the score of the model. Thus, the best one is the first element of the list.
    
    Additionally, the output of the method `get_ranking` is a list of the features ordered by the order of selection. 
    """
    
    def __init__(self, estimator, 
                 k_features = None,  # It is recommended not to change it.
                 forward    = True , 
                 floating   = False, 
                 verbose    = True, 
                 scoring    = 'accuracy', 
                 cv         = 0,
                 n_jobs     = -1):
        
        """
        Constructor
        
        Parameters:
            - estimator (object): The estimator that will be used to fit the model
            - k_features (int): The number of features that we want to select. By default, it is the number of features of the dataset
            - forward (bool): If True, then SFS will be performed. If False, then SBS will be performed.
            - floating (bool): If True, then SFFS and SBFS will be performed.
            - verbose (bool): If True, then the model will print the results of the selection
            - scoring (str or callable): STR o Callable de sklearn, si es str llama a la función get_scorer de sklearn.metrics
            - cv (int): Number of cross-validation folds
            - n_jobs (int): Number of jobs to run in parallel
        """
        
        self.k_features = k_features
        self.forward    = forward
        self.floating   = floating
        self.verbose    = verbose
        self.scoring    = scoring
        self.cv         = cv
        self.estimator  = estimator
        self.n_jobs     = n_jobs

    
    
    
    def fit(self, X, y):
        """Fit the model with the data.
        Parameters:
            X (array-like): The input data.
            y (array-like): The target values.
        Returns:
            self: The fitted model.
        """
        # If we do not specify the number of features, we set it to the number of features of the dataset
        if self.k_features is None:         
            self.k_features = X.shape[1] 
            
        # Initialize de SFS model of mlxtend
        self.sfs        = SFS(estimator  = self.estimator, 
                              k_features = self.k_features , 
                              forward    = self.forward, 
                              floating   = self.floating, 
                              verbose    = self.verbose, 
                              scoring    = self.scoring, 
                              cv         = self.cv,
                              n_jobs     = self.n_jobs)
        # Fit the model
        self.sfs.fit(X,y)
        # Compute the attribute selection and the ranking of the features
        self.training_selection, self.ranking = self._compute_selection_and_ranking()
        
        return self

    
    
    def get_ranking(self) -> List:
        """Return the ranking of the features 

        Returns:
            List: Orderer list by score of dictionaries containing the features indexes and the score of the model
        """
        return self.ranking
    
    
    def get_best_selection(self) -> Dict:
        """Return the best selection of features obtained using the Cross-Validation to measure the performance

        Returns:
            Dict: List in which the indexes represents the feature and the value the ranking of the feature
                Value of the dictionary:
                    - 'features_idx' (list of int): Indices of the selected features.
                    - 'score' (float): The average score of the subset.
        """
        return self.training_selection[0]
    
    
    def plot(self):
        plot_sfs(self.sfs.get_metric_dict(), kind='std_dev')
        plt.show()
    

    def _compute_selection_and_ranking(self):
        """
        Compute the best feature selection and a ranking for the Sequential Feature Selector (SFS) model.

        This method processes the subsets obtained from the SFS model, constructs a list of selected features 
        along with their respective scores, and calculates a ranking for each feature based on its selection order.

        Returns:
            tuple:
                - training_selection (list of dict): A list of dictionaries where each dictionary contains:
                    - 'features_idx' (list of int): Indices of the selected features.
                    - 'score' (float): The average score of the subset.
                - ranking (numpy.ndarray): A 1D numpy array where each element represents the rank of the 
                corresponding feature. The rank is determined by the order of feature selection across 
                the subsets.

        Example:
            >>> training_selection, ranking = self._compute_selection_and_ranking()
            >>> print(training_selection)
            [{'features_idx': [0, 1], 'score': 0.8}, {'features_idx': [2], 'score': 0.7}]
            >>> print(ranking)
            [1. 1. 2.]

        Notes:
            - The `self.sfs.subsets_` is expected to be a dictionary where keys are subset lengths and values 
            are dictionaries containing 'feature_idx' (indices of selected features) and 'avg_score' (average score).
            - The ranking is updated only for features that have not been analyzed previously.

        Raises:
            AttributeError: If `self.sfs.subsets_` or `X.shape` is not defined.

        """
        training_selection = []
        ranking = np.zeros(self.k_features)
        analyzed_features = set()

        for i, (subset_length, subset) in enumerate(self.sfs.subsets_.items(), start=1):
            # Append to training_selection
            training_selection.append({
                'features_idx': list(subset['feature_idx']),
                'score': subset['avg_score']
            })

            # Update ranking
            selected_features = subset['feature_idx']
            for feature in selected_features:
                if feature not in analyzed_features:
                    ranking[feature] = i
                    analyzed_features.add(feature)

        # Sort training_selection by score in descending order
        training_selection = sorted(training_selection, key=lambda x: x['score'], reverse=True)

        return training_selection, ranking
    
    
    
    def _compute_training_selection(self)-> List[Dict]:
        """Compute the best feature selection once the model has been fitted.
        
        Return:
            List[Dict]: List in which the indexes represents the feature and the value the ranking of the feature
                Value of the dictionary:
                    - 'features_idx' (list of int): Indices of the selected features.
                    - 'score' (float): The average score of the subset.
        """
        training_selection = []
        for k,v in self.sfs.subsets_.items():
            training_selection.append({
                    'features_idx': list(v['feature_idx']),
                    'score': v['avg_score']
                })
        training_selection = sorted(training_selection, key = lambda x: x['score'], reverse = True)
        return training_selection
        

    def _compute_ranking(self):
        
        ranking = np.zeros(self.k_features)
        analyzed_features = set()

        for i, (subset_length, subset) in enumerate(self.sfs.subsets_.items(), start=1):
            selected_features = subset['feature_idx']
            for feature in selected_features:
                if feature not in analyzed_features:
                    ranking[feature] = i
                    analyzed_features.add(feature)

        return ranking 
    
    

class RFE:
    
    estimators_available = ['SAE']
    
    def __init__(self, 
                 estimator: BaseEstimator,
                 estimator_name: str,
                 n_features_to_select: int,
                 step: int = 1, 
                 verbose: bool = False):
        
        if estimator_name not in self.estimators_available:
            raise ValueError(f"Estimator {estimator_name} not available. Please use one of the following: {self.estimators_available}")
        
        self.estimator              = estimator
        self.estimator_name         = estimator_name
        self.step                   = step
        self.verbose                = verbose
        self.n_features_to_select   = n_features_to_select
        
        self.ranking                = []   # Arraty de enteros que indica la posiciones de los rankings. # ESTA ES LA IMPORTANTE
        
        #self.support                = None # Array con las mascaras de los atributos seleccionados. No se que hacer con esto
        
        self.n_features_            = None # Número de atributos
        self.classes_               = None # Clases de los datos
        
        
    def get_ranking(self):        
        return self.ranking
            
        
        
    def _fit_SAE(self, features: npt.ArrayLike, gt_att: npt.ArrayLike , y: npt.ArrayLike, lamb: int = 500000):
        """Fit the SAE algorithm

        Args:
            features (npt.ArrayLike): Training data which is the Semantic Space
            gt_att   (npt.ArrayLike): Ground truth attributes. It must be a matrix of shape (n_features, n_attributes)
            y        (npt.ArrayLike): Labels 
        """
        debug = False 
        self.n_features_ = gt_att.shape[1]
        self.classes_    = np.unique(y)
        
        mask = np.ones(self.n_features_, dtype=bool) # Firstly, all attributes are true
        
        if debug:
            print(f"Number of features: {self.n_features_}")
            print(f"Number of classes: {self.classes_.shape[0]}")
            print(f"Number of attributes: {self.n_features_}")

        # Create the SAE model
        while ( self.n_features_ >= self.n_features_to_select):
            # Fit the modelit+=1
            W = Sae(features.T, gt_att[:,mask].T,lamb)

            axis = 1 if W.shape[0] == self.n_features_ else 0
            if debug:
                print(f"Shape of W: {W.shape}")
                print(f"Axis: {axis}")
            # Getting and removing the indexes with the lowest values
            # Mean -> abs -> argsort lowest to highest -> select the worst features by means of step parameter
            index_to_remove_relative       = np.argsort(np.abs(np.mean(W,axis = axis)))[0]
            
            # Hacemos la correspondencia entre el indice calculado y el real
            real_index_to_remove = 0
            contador = -1
            # Recorremos la mascara con todos los atributos
            for i in range(len(mask)):
                # Contamos los atributos que no han sido eliminados
                if mask[i]:
                    contador +=1
                # Si el contador es igual al indice relativo, hemos encontrado el indice real
                if contador == index_to_remove_relative:
                    real_index_to_remove = i
                    break
                    
            
            if mask[real_index_to_remove] == False:
                raise ValueError(f"OJITO, el RFE lo estas haciendo mal. Has cogido un atributo para eliminar que ya estaba eliminado. En concreto, alguno de {real_index_to_remove}")
            
            mask[real_index_to_remove] = False
            if debug:
                print(f"Indexes to remove: {real_index_to_remove}")
            
            # Add this indexes to the ranking list
            self.ranking.append(real_index_to_remove)
        
            self.n_features_ -= self.step
            if debug:
                print(f"Number of features: {self.n_features_}")
        # Greatest to lowest indexes
        self.ranking = self.ranking[::-1]
        
        if self.verbose:
            print(f"Ranking: {self.ranking}")
        
        
        
    def fit(self, features: npt.ArrayLike, gt_att: npt.ArrayLike , y: npt.ArrayLike, lamb: int = 500000):
        
        if self.estimator_name == 'SAE':
            self._fit_SAE(features, gt_att, y, lamb)


class ReliefF(object):

    """Feature selection using data-mined expert knowledge.
    
    Based on the ReliefF algorithm as introduced in:
    
    Kononenko, Igor et al. Overcoming the myopia of inductive learning algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55
    
    This implementation is a slightly modification from https://github.com/gitter-badger/ReliefF
    """
    
    def __init__(self, n_neighbors=100, n_features_to_keep=10):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_neighbors: int (default: 100)
            The number of neighbors to consider when assigning feature importance scores.
            More neighbors results in more accurate scores, but takes longer.

        Returns
        -------
        None

        """
        
        self.feature_scores = None
        self.top_features = None
        self.tree = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep
    
    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        None

        """
        self.feature_scores = np.zeros(X.shape[1])
        self.tree = KDTree(X)

        for source_index in range(X.shape[0]):
            distances, indices = self.tree.query(X[source_index].reshape(1, -1), k=self.n_neighbors + 1)

            # First match is self, so ignore it
            for neighbor_index in indices[0][1:]:
                similar_features = X[source_index] == X[neighbor_index]
                label_match = y[source_index] == y[neighbor_index]

                # If the labels match, then increment features that match and decrement features that do not match
                # Do the opposite if the labels do not match
                if label_match:
                    self.feature_scores[similar_features] += 1.
                    self.feature_scores[~similar_features] -= 1.
                else:
                    self.feature_scores[~similar_features] += 1.
                    self.feature_scores[similar_features] -= 1.
        
        self.top_features = np.argsort(self.feature_scores)[::-1]
        
        self.ranking = dict(zip(self.top_features, sorted(self.feature_scores,reverse=True)))
        
    def get_ranking(self):
        """
        Return the feature ranking done by ReliefF
        
        Return[Dict]: 
            Dictionary with the feature indexes as keys and the scores as values
        """        
        return self.ranking
        
        
    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return X[:, self.top_features[self.n_features_to_keep]]