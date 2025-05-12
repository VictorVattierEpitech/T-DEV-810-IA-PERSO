# T-DEV-810

## Contexte et objectif
Ce projet vise à construire un **classifieur multi‐labels** pour détecter plusieurs pathologies thoraciques à partir de radiographies de thorax.

Voici les différentes pathologies à identifier : 
- No Finding
- Enlarged Cardiomediastinum
- Cardiomegaly
- Lung Opacity
- Lung Lesion
- Edema
- Consolidation
- Pneumonia
- Atelectasis
- Pneumothorax
- Pleural Effusion
- Pleural Other
- Fracture
- Support Devices

## Tâche de classification
- **Type** : classification **multi‐labels** (une image peut appartenir à plusieurs classes).  
- **Entrée** : image radiographique (224×224 pixels, RGB)  
- **Sortie** : vecteur de 14 scores/logits, un par pathologie, convertis en probabilités via une sigmoid.  
- **Décision** : on applique un seuil (par défaut 0.5 ou optimisé par pathologie) pour chaque probabilité afin de produire un vecteur binaire de pré­dic­tions.

## Dataset : CheXpert
- **Source** : Stanford CheXpert (Irvin et al., 2019), grand jeu de données de radiographies thoraciques annotées de façon semi‐automatique.  
- **Taille** :  
  - Environ **224 316** images d’entraînement (version complète),  
  - Ici downsampler pour limiter la taille du dataset (+ 400Go normalement)
  - Lien vers le dataset utilisé : https://www.kaggle.com/datasets/ashery/chexpert
- **Étiquettes** : 14 catégories cliniques (ex. “No Finding”, “Pneumonia”, “Edema”…).  
- **Valeurs des labels** :  
  - `1` = pathologie présente  
  - `0` = pathologie absente  
  - `-1` = incertain (annotation douteuse)  
- **Particularités** :  
  - Jeu fortement déséquilibré (certaines pathologies très rares)  
  - Étiquettes produites par extraction automatique de rapports radiologiques, d’où la présence de cas “incertains” à gérer par masquage ou mapping.

## Étude de nos données

## Modèle et architecture (model.py)


## Entrainement (config.py et train.py)



## Evaluation

## Inférence