"""Código adaptado de https://github.com/hoseong-kim/sae-pytorch"""

import numpy as np
import scipy 
from typing import List


def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat

def distCosine(x, y):
	axis = 0 if x.ndim==1 and y.ndim==1 else 1
	xx = np.sum(x**2, axis=axis)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=axis)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist

def distCosine(x, y):
    
    axis = 0 if x.ndim==1 and y.ndim==1 else 1
    # Calcular la norma (magnitud) de cada fila (vector) en x y y
    xx = np.linalg.norm(x, axis=axis, keepdims=True)
    yy = np.linalg.norm(y, axis=axis, keepdims=True)
    
    # Normalizar los vectores, asegurándonos de evitar la división por cero
    x_normalized = np.divide(x, xx, out=np.zeros_like(x), where=xx != 0)
    y_normalized = np.divide(y, yy, out=np.zeros_like(y), where=yy != 0)
    
    # Calcular la similitud del coseno
    cosine_similarity = np.dot(x_normalized, y_normalized.T)
    
    # Clip para asegurar que los valores estén en el rango válido [-1, 1]
    #cosine_similarity = np.clip(cosine_similarity, -1, 1)
    
    # Calcular la distancia del coseno
    dist = 1 - cosine_similarity
    
    return dist

def generate_point_in_neighbourhood(x, epsilon):
    """
    Parameters:
    -----------
    x: np.array
        Punto alrededor del cual generamos el punto en la bola
    epsilon: float
        Radio de la bola
    """
    # Dimensión del espacio
    dim = len(x)
    # Generar un vector aleatorio en [-1, 1]^dim
    vector_aleatorio = np.random.uniform(-1, 1, dim)
    # Normalizar el vector aleatorio para que tenga norma 1
    vector_unitario = vector_aleatorio / np.linalg.norm(vector_aleatorio)
    # Escalar el vector para que esté dentro de la bola de radio epsilon
    radio_escalado = np.random.uniform(0, epsilon)
    punto_y = x + vector_unitario * radio_escalado
    return punto_y

def compute_boundary(eval_classes, attributes, epsilon):
	"""Calcula el radio que tendrá los entornos de los atributos semánticos asociados a las clases 'eval_classes'.
	Para cada clase se considera el atributo semántico correspondiente. Se calcula la distnancia entre todos los atributos semánticos y se toma el más cercano.add()
	Finalmente la distnacia con el más cercano se divide por un factor de escala 'epsilon' para obtener el radio del entorno. 

	Parámetros:
	----------
	eval_classes : (np.ndarray or list)
		Clases a las que se les calculará el radio del entorno.add()
	
	attributes : (np.ndarray)
		Espacio semántico (Ground Truth). Todo el espacio semántico de todas las clases.add()
	
	epsilon : (int, optional)
		Factor de escala usado para la creación del entorno. Defaults to 3.

	Retorno:
	--------
		umbral_entornos (np.ndarray): Vector con los radios de los entornos de las clases 'eval_classes'.
	"""
	 # Creamos matriz de distancias entre todos los atributos. Como hay 10 atributos, esta matriz será 10x10. 
	dist_among_att       = distCosine(attributes[eval_classes],attributes[eval_classes])
	# Ordenamos las distancias de menor a mayor por columnas
	# Para cada atributo (fila) tendremos un vector de índices con la posición de los atributos más cercanos
	indexes_closest_att    = np.argsort(dist_among_att, axis = 1) 

	# Ahora calculamos la clase más cercana a cada calse
	closest_classes_indexes   = np.zeros(dist_among_att.shape[0],dtype='int')
	closest_classes_distances = np.zeros(dist_among_att.shape[0])

	for clase in range(dist_among_att.shape[0]):
		# La clase más cercana siempre es ella misma, que es índice 0, por eso buscamos en la posición 1.
		closest_classes_indexes[clase]   = indexes_closest_att[clase][1]                          # Indice de la clase más cercana a la clase idx
		closest_classes_distances[clase] = dist_among_att[clase][closest_classes_indexes[clase]]  # Distancia de la clase idx con su más cercana 
		
	# Radio del entorno de cada clase
	umbral_entornos = closest_classes_distances/epsilon

	return umbral_entornos

def aug_semantic_space(eval_classes, attributes, epsilons, n_points = 10):
    """
    Genera puntos alrededor de cada atributo semántico
    
    Parameters:
    -----------
    attributes: np.array
        Atributos semánticos
    epsilons: np.array
        Radio de la bola para cada atributo semántico
    n_points: int
        Número de puntos a generar alrededor de cada atributo semántico
    """
    attributes_to_aug      = attributes[eval_classes]
    n_attributes           = attributes.shape[0]
    
    assert len(attributes_to_aug) == len(epsilons), "Attributes and epsilons must have the same length"
    
    X, labels = [], []
    for i, (att,eps) in enumerate(zip(attributes_to_aug,epsilons)):
        X.append(att)
        labels.append(eval_classes[i])
        for _ in range(n_points):
            X.append(generate_point_in_neighbourhood(att, eps))
            labels.append(eval_classes[i])
            
        
        for j in range(n_points):
            dist = distCosine(att, X[-1-j]) # -1, -2, -3, ... -n_points*i
            assert dist <= eps, f"Distancia {dist} mayor que el radio de la bola {eps}"
        
    # Comprobamos que todo va bien
    assert len(X) == len(labels)
    
            
    return np.asarray(X), np.asarray(labels)


class Sae:
    
	def __init__(self, 
              	mask : List[bool]   = None, 
              	lambd: int 			= 500000):
		"""
		Constructor de la clase. El valor de lambda es el mismo que los autores utilizan en el artículo original.

		Parámetros:
		----------
		mask : List[bool], opcional
			Vector que indica los atributos a tener en cuenta. Esto lo usamos para la selección de atributos. Por defecto es None.
		
		lambd : int, opcional
			Valor del regularizador. Por defecto es 500000.
		"""
		self.lambd = lambd
		self.mask = mask

	def set_mask(self, mask: List[bool]):
		self.mask = mask

	def _compute_w(self,x, s):
		"""
		Calcula la matriz de proyección utilizando la ecuación de Sylvester.

		Parámetros:
		----------
		x : np.ndarray
			Matriz de datos, d x N.
		
		s : np.ndarray
			Matriz semántica, k x N.

		Retorna:
		--------
		W : np.ndarray
			Matriz de proyección, k x d.
		"""
		A = np.dot(s, s.transpose())
		B = self.lambd * np.dot(x, x.transpose())
		C = (1+self.lambd) * np.dot(s, x.transpose())
		w = scipy.linalg.solve_sylvester(A,B,C)
		return w

	def fit(self, 
		training_features: np.ndarray , 
		training_att     : np.ndarray , 
		training_labels  : np.ndarray = None, 
		training_classes : np.ndarray = None,
  		mask 		     : np.ndarray = None):
		"""
  		Calcula la matriz W del modelo SAE. 
    
		Dentro del contexto de la selección de características del espacio semántico podemos llamar a esta función de dos maneras distintas. 
		
		1. Usando al argumento 'mask' para seleccionar los atributos del espacio semántico que se tendrán en cuenta. 
		>>> sae = SAE(...)
		>>> mask = [True, False, ..., False, True]
		>>> sae.fit(training_features, training_att, ..., mask)
		Internamente se seleccionarán los atributos de training_att correspondientes a las posiciones True del vector mask.
  
		2. Sin usar el argumento 'mask' para seleccionar los atributos del espacio semántico que se tendrán en cuenta.
		En este caso la más cara deberá ir incorporada dentro del argumento 'training_att'.add()
		>>> sae = SAE(...)
		>>> mask = [True, False, ..., False, True]
		>>> sae.fit(training_features, training_att[mask],...)

		Args:
			training_features (np.ndarray) Matriz con las imágenes vectorizadas. 
			training_att (np.ndarray): _description_. 
			training_labels (np.ndarray, optional): Etiquetas de entrenamiento. Defaults to None.
			training_classes (np.ndarray, optional): Clases. Defaults to None.
			mask (Lit[Bool], optional): Solo si usamos selección de características. Este argumento restringe los atributos a usar del espacio semántico. Defaults to None.
		"""
		if self.mask is None:
			self.W = self._compute_w(training_features.T, training_att.T) 
		else:
			self.W = self._compute_w(training_features.T, training_att[:,self.mask].T) 

	def _zsl_acc(self, semantic_predicted, semantic_gt, hitk, eval_classes, eval_labels):
			"""
			Calcula la precisión basado en la similitud entre predicciones semánticas y clases de prueba.

			Este método mide la precisión del modelo de Zero-Shot Learning al comparar las predicciones del espacio semántico con las clases objetivo,
			utilizando la distancia del coseno para encontrar las clases más cercanas. La precisión se determina si la etiqueta verdadera del ejemplo
			se encuentra entre las k clases más cercanas.

			Parámetros:
			----------
			semantic_predicted : np.ndarray
				Las predicciones del modelo en el espacio semántico.
			
			semantic_gt : np.ndarray
				Las características semánticas reales correspondientes a las etiquetas de prueba.
			
			hitk : int
				El número de clases más cercanas (k) que se considerarán al calcular la precisión.
			
			eval_classes : np.ndarray
				Un array que contiene los identificadores de las clases que se evalúan durante la prueba.
			
			eval_labels : np.ndarray
				Las etiquetas reales correspondientes a los ejemplos de prueba.

			Retorno:
			--------
			zsl_accuracy : float
				La precisión del modelo ZSL, calculada como el porcentaje de ejemplos cuya etiqueta verdadera está entre las k clases más cercanas.
			
			y_hit_k : np.ndarray
				Un array que contiene las k clases más cercanas para cada ejemplo de prueba.
			"""
			

			# Calculamos la distancia entre el espacio semántico predicho y el espacio semántico real
			dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
			# Obtenemos las k clases más cercanas	
			y_hit_k = np.zeros((dist.shape[0], hitk))
			# Para cada ejemplo
			for idx in range(0, dist.shape[0]):
				# Ordenamos las distancias de mayor a menor y obtenemos los índices
				sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
				y_hit_k[idx,:] = eval_classes[sorted_id[0:hitk]]
				
			n = 0
			for idx in range(0, dist.shape[0]):
				if eval_labels[idx] in y_hit_k[idx,:]:
					n = n + 1
			zsl_accuracy = float(n) / dist.shape[0] * 100
			return zsl_accuracy, y_hit_k

	def _zsl_acc_via_boundary(self, semantic_predicted, semantic_gt, hitk, attributes, eval_classes, eval_labels, epsilon):
		"""Evalua el modelo usando ls distancia a la frontera más cercana de los atributos semánticos asociados a las clases 'eval_classes'.

		Args:
			semantic_predicted (_type_): _description_
			semantic_gt (_type_): _description_
			hitk (_type_): _description_
			attributes (_type_): _description_
			eval_classes (_type_): _description_
			eval_labels (_type_): _description_
			epsilon (_type_): _description_

		Returns:
			Tuple: accuracy, y_hit_k
		"""
  
		# Calculamos los umbrales para cada att semántico asocaido a las clases de `eval_classes`
		umbral_entornos = compute_boundary(eval_classes = eval_classes, 
                                           	     attributes   = attributes, 
                                                 epsilon      = epsilon)
  

		distancias = distCosine(semantic_predicted, semantic_gt) - umbral_entornos
  
		y_hit_k = np.zeros((distancias.shape[0],hitk))
    
		for instance in range(distancias.shape[0]):
			y_hit_k[instance,:] = eval_classes[np.argsort(distancias[instance,:])[:hitk]]
			# El código de debajo equivale al de arriba, por temas de eficiencia dejamos el de arriba, el de abajo es más explicativo.
			#sorted_id = sorted(
			#    range(len(distancias[instance,:])),         # Lista con la distancia del elemento a todas las clases  
			#    key = lambda k: distancias[instance,:][k],  # Ordenamos por el valor de la distancia
			#    reverse = False                             # De menor a mayor
			#)
			#
			#y_hit_k[instance,:] = eval_classes[sorted_id[:hitk]]
			
		acc = np.sum([1 if eval_labels[instance] in y_hit_k[instance] else 0 for instance in range(distancias.shape[0])]) / distancias.shape[0] * 100
  
		return acc, y_hit_k

	def _zsl_acc_via_augmentation(self, attributes, eval_features, eval_labels, eval_classes, semantic_predicted, hitk, epsilon,n_points):
		
		umbrals_unseen  = compute_boundary(eval_classes, attributes, epsilon )
		gt_ss, semantic_labels = aug_semantic_space(eval_classes , attributes, umbrals_unseen, n_points)

		semantic_predicted = np.dot(eval_features, normalizeFeature(self.W).transpose())

		dist = 1 - distCosine(semantic_predicted, normalizeFeature(gt_ss.transpose()).transpose())
		y_hit_k = np.zeros((dist.shape[0],hitk))

		for idx in range(0, dist.shape[0]):
			# Ordenamos las distancias de mayor a menor y obtenemos los índices
			sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
			y_hit_k[idx] = semantic_labels[sorted_id[0:hitk]]

		acc = np.sum(
					[1 if eval_labels[idx] == y_hit_k[idx] else 0 for idx in range(dist.shape[0])]
				).astype(float) / dist.shape[0] * 100

		return acc, y_hit_k

	def evaluate(self,
				attributes:	   np.ndarray,
				eval_features: np.ndarray,
				eval_classes:  np.ndarray,
				eval_labels:   np.ndarray,
    			boundary:      bool = False,
				aug: 		   bool = False,
				epsilon:	   int  = 3,
				n_points:      int  = 100,
				hitk:          int  = 1):

		"""Hay que tener en cuenta que la mascara debe de ser oblgatoria pasarla para evaluar el modelo. 

		Parameters:
		-----------
		boundary : bool, optional
			Indica si se va a usar la distancia a la frontera más cercana de los atributos semánticos asociados a las clases 'eval_classes'. Defaults to False.
  
  		epsilon : int, optional
			Factor de escala usado para la creación de entorno. Para que sea tenido en cuenta se tiene que cumplir boundary = True. Defaults to 3.
   
		hitk : int, optional
			El número de clases más cercanas (k) que se considerarán al calcular la precisión. Defaults to 1.

		Raises:
		----------
			ValueError: _description_

		Returns:
		----------
			_type_: _description_
		"""

		if not hasattr(self, 'W'):
			raise ValueError("After evaluation, the model must be trained. Please, call fit method.")
  
		if aug and boundary:
			raise ValueError("Augmentation and boundary cannot be used at the same time.")
  
		if self.mask is None:
			gt_ss = attributes[eval_classes]
		else:
			gt_ss = attributes[eval_classes][:,self.mask]
  
		semantic_predicted = np.dot(eval_features, normalizeFeature(self.W).transpose())
		
		if boundary:
			#print("Using the boundary method")
			[zsl_accuracy, y_hit_k] = self._zsl_acc_via_boundary(semantic_predicted, gt_ss, hitk, attributes, eval_classes, eval_labels, epsilon)
		elif aug:
			#print("Using the augmentation method")
			[zsl_accuracy, y_hit_k] = self._zsl_acc_via_augmentation(attributes, eval_features, eval_labels, eval_classes, semantic_predicted, hitk, epsilon, n_points)
		else:
			#print("Using the original method")
			[zsl_accuracy, y_hit_k] = self._zsl_acc(semantic_predicted, gt_ss, hitk, eval_classes,eval_labels)
		
		return zsl_accuracy, y_hit_k       	
  
	def get_W(self):
		if not hasattr(self, 'W'):
			raise ValueError("After evaluation, the model must be trained. Please, call fit method.")
		else:
			return self.W





