import numpy as np

# Dictionnaire avec le nombre d'images positives par classe
count_pos = {
    "Support Devices": 116001,
    "Lung Opacity": 105581,
    "Pleural Effusion": 86187,
    "Edema": 52246,
    "Atelectasis": 33376,
    "Cardiomegaly": 27000,
    "No Finding": 22381,
    "Pneumothorax": 19448,
    "Consolidation": 14783,
    "Enlarged Cardiomediastinum": 10798,
    "Lung Lesion": 9186,
    "Fracture": 9040,
    "Pneumonia": 6039,
    "Pleural Other": 3523,
}

# Calcul de la moyenne des effectifs
mean_count = np.mean(list(count_pos.values()))

# Dictionnaire des poids pour chaque classe
class_weights = {key: mean_count / float(count) for key, count in count_pos.items()}

print(class_weights)