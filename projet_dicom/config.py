from pathlib import Path

# Chemins
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "fragment_data_set"
PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "runs"

# Créer les dossiers s'ils n'existent pas
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Modèle
IMAGE_SIZE = (512, 512)  # Standardiser les images à cette taille
NUM_CLASSES = 2  # Adapter selon vos classes (Normal, Pathologie, etc.)

print("✅ Configuration initialisée")
