# split_csv.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemin vers le CSV complet
full_csv_path = "CheXpert-v1.0-small/full.csv"

# Lecture du CSV complet
df = pd.read_csv(full_csv_path)

# Première séparation : extraire le test (15% du total)
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

# Seconde séparation : découper le train/validation
# Pour obtenir validation = 15% du total, parmi les 85% restants, le ratio validation sera 15 / 85 ≈ 0.1765.
train_df, valid_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)

# Sauvegarde des fichiers CSV
train_df.to_csv("CheXpert-v1.0-small/train.csv", index=False)
valid_df.to_csv("CheXpert-v1.0-small/valid.csv", index=False)
test_df.to_csv("CheXpert-v1.0-small/test.csv", index=False)

print("Splitting terminé : train.csv, valid.csv et test.csv ont été générés.")
