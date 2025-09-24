import random 
import numpy as np

from .datareader import Dataset, StackedDataset , gen_ss_from_data, scaler_data, get_scaler

class SplitIntoFolds:
    def __init__(self, 
                 n_splits : int, 
                 show     : bool = False, 
                 seed     : int  = 42):
        
        
        self.n_splits = n_splits
        self.show     = show
        self.seed     = seed
    
        
    def split_via_classes_per_folds(self, 
                                     features:np.ndarray, 
                                     labels: np.ndarray, 
                                     attribute: np.ndarray, 
                                     preprocessing: bool = True):
        """
        Hace los splits de los datos en base a las clases que quieras que tenga cada folds. Si hay un acarreo, esto es, si no se puede dividir exactamente el número
        de clases entre el número de folds, se añadirán las clases restantes al último fold.
        ----------
        Observations:
            - Aproximadamente habrá el mismo número de clases en los folds
            - No garantiza el balanceo del número de instancias por clase entre los folds. **TENGA CUIDADO**
            - Inductivo: Las clases no vistas del espacio semántico de validacioon no estan en el entrenamiento. 
        ----------
        Args:
            features (np.ndarray): 
                Data (It should be the training data. Not the whole dataset with train and test data.)
            
            labels (np.ndarray): 
                Labels of the data
            
            attribute (np.ndarray): 
                Semantic spaces. A matrix with dimension (Number of classes x Number of attributes)
            
            preprocessing (bool, optional): 
                Scaling the semantic spaces using Standard Scaler. Defaults to True.
        """
        

        classes                       = np.unique(labels)
        classes_for_val_in_each_fold  = int( len(classes) / self.n_splits )
        acarreo                       = len(classes) % self.n_splits
        random.shuffle(classes)
        
        folds = []
        
        assert self.n_splits <= len(classes), "Number of folds exceeds the number of classes."

        for i in range(self.n_splits):
            
            # Get the classes for validation and training
            start = i * classes_for_val_in_each_fold
            end   = ( i + 1 ) * classes_for_val_in_each_fold if i!= self.n_splits - 1 else (i + 1) * classes_for_val_in_each_fold + acarreo
            
            val_classes      = classes[start:end]                              # Pseudo Unseen
            training_classes = [x for x in classes if x not in val_classes]    # Seen
            
            # Get the indexes
            val_loc     = np.where(np.isin(labels,val_classes))
            train_loc   = np.where(np.isin(labels,training_classes))
            
            assert np.intersect1d(val_loc, train_loc).shape[0] == 0, f"Error: Problems found in creating folds {i}. Val and training classes are not disjoint"
            assert np.intersect1d(val_classes, training_classes).shape[0] == 0, f"Error: Problems found in creating folds {i}. Val and training classes are not disjoint"
            assert all([i not in training_classes for i in labels[val_loc]]), f"Error: Problems found in creating folds {i}"
            assert all([i not in val_classes for i in labels[train_loc]]), f"Error: Problems found in creating folds {i}"
            
            # Get the features and label and attributes
            training_features = features[train_loc]
            training_labels   = labels[train_loc]
            
            val_features      = features[val_loc]
            val_labels        = labels[val_loc]
            
            # Scaling the data if necessary
            if preprocessing:
                training_features, val_features = scaler_data(get_scaler('Standard'), training_features, val_features)
                
            training_ss = gen_ss_from_data(training_labels, attribute) 
            val_ss      = gen_ss_from_data(val_labels, attribute)
            # Add the data to the folds
            training_dataset = Dataset(training_features, training_labels, training_ss, mode = 'train')
            val_dataset      = Dataset(val_features, val_labels, val_ss, mode = 'val')
            
            """ 
            The scaling of the semantic spaces must be thoroughly checked.
            - First, we cannot scale the attribute spaces as a whole.
            - Second, we cannot scale the redundant attribute space and then, reducing it to the not redundant one as the scaling
              will be different. Imagine there are many instances of a class, thereby, the scaling will perjudicate the class which has less instances. 
            
            Having said that, we must scale the two type of semantic spaces separately. 
            """
            if preprocessing:
                training_ss_redundant, val_ss_redundant = scaler_data(get_scaler('Standard'), 
                                                                      training_dataset.att, 
                                                                      val_dataset.att)
                
                training_dataset.att = training_ss_redundant
                val_dataset.att      = val_ss_redundant
                
                training_ss_reduced, val_ss_reduced = scaler_data(get_scaler('Standard'), 
                                                                  training_dataset.att_reduced, 
                                                                  val_dataset.att_reduced)
                
                training_dataset.att_reduced = training_ss_reduced 
                val_dataset.att_reduced      = val_ss_reduced
            
            
            
            
            if self.show:
                print(f"Fold {i+1} || Val Classes {np.unique(val_labels).shape} - {val_labels.shape} instances || Training Classes {np.unique(training_labels).shape} - {training_labels.shape} instances")
            
            #yield StackedDataset({'train': training_dataset, 'val': val_dataset})
            yield training_dataset, val_dataset
            
            #folds.append( StackedDataset({'train': training_dataset, 'val':val_dataset}) )
        #return folds