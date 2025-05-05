# Liste des 14 pathologies (classes du dataset)
pathology_labels = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

# nb classes
num_classes = len(pathology_labels)

# Hyperparamètres d'entraînement
global_epochs = 10
fine_tuning_epochs = 5
learning_rate_global = 1e-4
learning_rate_fine_tuning = 1e-4
batch_size = 250

# paths vers les CSV issus du spliting
train_csv = "CheXpert-v1.0-small/train.csv"
valid_csv = "CheXpert-v1.0-small/valid.csv"
test_csv  = "CheXpert-v1.0-small/test.csv"

images_dir = "."
