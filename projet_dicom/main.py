"""
Projet de Classification Chest X-Ray

Structure:
- extract_labels.py : Extraction des métadonnées DICOM
- config.py : Configuration du projet
- dataset.py : Classe Dataset PyTorch
- models_arch.py : Implémentations des modèles (ResNet18, SimpleCNN)
- trainer.py : Classe Trainer pour l'entraînement
- main.py : Script principal
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, NUM_EPOCHS
from dataset import ChestXRayDataset
from trainer import Trainer
from models_arch import SimpleCNN

def create_data_splits():
    """Crée les splits train/val/test à partir du CSV des labels"""
    
    # Lire les métadonnées
    labels_df = pd.read_csv("dicom_labels.csv")
    patient_ids = labels_df['patient_id'].unique().tolist()
    
    print(f"✅ {len(patient_ids)} patients trouvés")
    print(f"Patients: {patient_ids}")
    
    # Créer un dictionnaire de labels
    # À adapter selon vos classes réelles!
    labels_dict = {}
    for i, patient_id in enumerate(patient_ids):
        # Exemple simplifié : alterner entre classe 0 et 1
        labels_dict[patient_id] = i % 2
    
    print(f"\nLabels assignés:")
    for patient_id, label in labels_dict.items():
        print(f"  {patient_id}: Classe {label}")
    
    # Split train/val/test
    train, temp = train_test_split(patient_ids, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42)
    val, test = train_test_split(temp, test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT), random_state=42)
    
    print(f"\nSplits:")
    print(f"  Train: {len(train)} patients")
    print(f"  Val: {len(val)} patients")
    print(f"  Test: {len(test)} patients")
    
    return train, val, test, labels_dict

def main():
    print("="*60)
    print("🚀 CLASSIFICATION CHEST X-RAY")
    print("="*60)
    
    # 1. Créer les splits
    train_ids, val_ids, test_ids, labels_dict = create_data_splits()
    
    # 2. Créer les datasets
    print("\n📦 Création des datasets...")
    train_dataset = ChestXRayDataset(train_ids, labels_dict=labels_dict)
    val_dataset = ChestXRayDataset(val_ids, labels_dict=labels_dict)
    test_dataset = ChestXRayDataset(test_ids, labels_dict=labels_dict)
    
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    # 3. Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialiser le trainer
    print("\n🤖 Initialisation du modèle...")
    trainer = Trainer(model_name="simple_cnn", num_classes=2)
    
    # 5. Entraîner
    print("\n⏳ Entraînement...")
    trainer.train(train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    print("\n✅ Entraînement terminé!")
    print("📊 Résultats sauvegardés dans le dossier 'runs/'")

if __name__ == "__main__":
    main()
