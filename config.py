TRAIN_CSV = "CheXpert-v1.0-small/train.csv"
VAL_CSV   = "CheXpert-v1.0-small/valid.csv"
TEST_CSV  = "CheXpert-v1.0-small/test.csv"
IMG_DIR   = "."

# Pathologies
CLASSES = [
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
    "Support Devices",
]

# Stratégie de mapping pour les labels incertains (-1)
# 'ones' => map to positive, 'zeros' => map to negative, 'ignore' => keep -1 (mask)
UNCERTAIN_LABEL_MAPPING = {
    "Consolidation": "ones",
    "Pneumonia":     "ones",
    "Edema":         "ones",
    "Pleural Effusion": "ones",
    "Atelectasis":   "zeros",
    "Pneumothorax":  "zeros",
    # Les autres seront ignorés par défaut
}

# Entraînement
EPOCHS       = 10
BATCH_SIZE   = 300
NUM_WORKERS  = 12
LOG_INTERVAL = 20

# Optimizer & Scheduler
LR            = 1e-4
MAX_LR        = 1e-3
WEIGHT_DECAY  = 1e-5

# Focal Loss
FOCAL_GAMMA   = 2.0