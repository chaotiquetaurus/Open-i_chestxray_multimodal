# 🫁 Projet Classification Chest X-Ray

Classification d'images chest X-ray avec PyTorch en utilisant vos données DICOM.

## 📁 Structure

```
projet_dicom/
├── fragment_data_set/          # Vos données DICOM
├── processed_data/             # Données traitées
├── models/                     # Modèles sauvegardés
├── runs/                       # Logs et historiques
├── extract_labels.py           # Extraction des métadonnées
├── config.py                   # Configuration du projet
├── dataset.py                  # Dataset PyTorch
├── models_arch.py              # Architectures (ResNet18, SimpleCNN)
├── trainer.py                  # Trainer pour entraînement
├── main.py                     # Script principal
├── requirements.txt            # Dépendances
└── dicom_labels.csv            # Métadonnées extraites
```

## 🚀 Démarrage rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Extraire les métadonnées DICOM
```bash
python extract_labels.py
```

### 3. Entraîner le modèle
```bash
python main.py
```

## 📊 Métadonnées actuelles

✅ **12 images** de **5 patients** détectées
- Format: CR (Radiographies)
- Dimensions: 2828x2320 ou 2991x2992 pixels
- Labels: À définir dans `main.py`

## 🎯 À personnaliser

1. **Labels de classification** dans `main.py`:
   ```python
   labels_dict[patient_id] = classe  # 0=Normal, 1=Pathologie, etc.
   ```

2. **Architecture du modèle** dans `main.py`:
   ```python
   trainer = Trainer(model_name="resnet18")  # ou "simple_cnn"
   ```

3. **Hyperparamètres** dans `config.py`:
   ```python
   BATCH_SIZE = 4
   LEARNING_RATE = 1e-4
   NUM_EPOCHS = 50
   ```

## 📈 Sortie du modèle

Les résultats sont sauvegardés dans:
- `models/best_model_*.pth` : Meilleur modèle
- `runs/history_*.json` : Historique d'entraînement

## 💡 Modèles disponibles

- **SimpleCNN**: CNN 4 couches légère et rapide
- **ResNet18**: Modèle pré-entraîné plus puissant

Choisissez selon vos ressources!
