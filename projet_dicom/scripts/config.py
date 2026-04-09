"""Configuration du preprocessing"""

from pathlib import Path

# Chemins
PROJECT_DIR = Path(".")
DATA_DIR = PROJECT_DIR / "fragment_data_set"
XML_DIR = PROJECT_DIR / "NLMCXR_reports" / "ecgen-radiology"
CSV_FILE = PROJECT_DIR / "dataset_labeled.csv"
OUTPUT_CSV = PROJECT_DIR / "dataset_labeled_enriched.csv"
DICOM_OUTPUT_DIR = PROJECT_DIR / "dicom_preprocessed"

# Créer les dossiers s'ils n'existent pas
for directory in [DICOM_OUTPUT_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration preprocessing DICOM
IMAGE_SIZE = (256, 256)

# Windows radiologiques (Hounsfield Units)
LUNG_WINDOW = {"center": -150, "width": 400}
MEDIASTINAL_WINDOW = {"center": 150, "width": 400}
