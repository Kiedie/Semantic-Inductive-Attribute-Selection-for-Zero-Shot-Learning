 
Estos datasets de aquí no son los originales sino unos CUSTOMS extraídos de https://github.com/akshitac8/tfvaegan
El paper en el que se basa el repositorio es el siguiente: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670477.pdf
Las particiones están basadas en el paper de "Zero-shot learning a comprehensive evaluation of the good, the bad and the ugly, 2018)

===========================================
====== INSTRUCCIONES GITHUB             ===
===========================================
========== CUSTOM DATASETS ==========

1. Download the custom dataset images in the datsets folder.
2. Use a pre-defined RESNET101 as feature extractor. For example, you can a have look here (https://github.com/akshitac8/Generative_MLZSL/tree/main/datasets/extract_features)
3. Extract features from the pre-defined RESNET101 and save the features in the dictionary format with keys 'features', 'image_files', 'labels'.
4. Save the dictionary in a .mat format using,
```
import scipy.io as io
io.savemat('temp',feat)
```


Comentan en el paper que extraen las características y los embeddings:

```Visual features and embeddings: We extract the average-pooled feature instances of size 2048
from the ImageNet-1K [6] pre-trained ResNet-101 [12]. For semantic embeddings, we use the class-level
attributes for CUB (312-d), SUN (102-d) and AWA2 (85-d). For FLO, fine-grained visual descriptions of
image are used to extract 1024-d embeddings from a character-based CNN-RNN [32]```
