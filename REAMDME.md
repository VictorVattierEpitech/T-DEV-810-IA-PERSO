

## Pourquoi ces choix ?

1. **DenseNet-121 comme backbone**

   * **Dense Connectivity** : chaque couche reçoit en entrée les sorties de **toutes** les couches précédentes.
   * **Gradient Flow** renforcé et **réduction** du risque de vanishing gradients.
   * **Paramètres partagés** et **feature reuse** améliorent l’efficacité, idéal pour des tâches médicales où les motifs sont subtils.

2. **Head MLP sur-mesure**

   * Le backbone fournit un embedding riche (`num_ftrs ≈ 1024`).
   * Pour ajuster à 14 classes, on **diminue** d’abord `num_ftrs → num_ftrs//2` pour alléger la tête et **ajouter** non-linéarité.
   * **Deux couches** (au lieu d’une simple sortie linéaire) augmentent la capacité du modèle à combiner finement les features extraites.

3. **Dropout 0.5**

   * Taux élevé pour **casser** les corrélations internes et **forcer** le réseau à apprendre des représentations redondantes.
   * Très utile sur des datasets de taille intermédiaire pour éviter l’overfitting.

4. **ReLU(inplace=True)**

   * Activation **efficace** (rapidité & stabilité numérique).
   * L’argument `inplace=True` économise de la mémoire en écrasant l’entrée, pratique sur GPU limité.