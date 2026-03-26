import pydicom
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import IMAGE_SIZE, DATA_DIR

class ChestXRayDataset(Dataset):
    """Dataset pour les images chest X-ray"""
    
    def __init__(self, patient_ids, labels_dict=None, transform=None, image_size=IMAGE_SIZE):
        self.patient_ids = patient_ids
        self.labels_dict = labels_dict or {}
        self.transform = transform
        self.image_size = image_size
        self.dcm_files = self._collect_dcm_files()
    
    def _collect_dcm_files(self):
        """Collecte tous les fichiers DICOM"""
        dcm_files = []
        for patient_id in self.patient_ids:
            patient_path = DATA_DIR / patient_id
            for dcm_file in sorted(patient_path.rglob("*.dcm")):
                dcm_files.append((dcm_file, patient_id))
        return dcm_files
    
    def __len__(self):
        return len(self.dcm_files)
    
    def __getitem__(self, idx):
        dcm_path, patient_id = self.dcm_files[idx]
        
        # Lire le fichier DICOM
        ds = pydicom.dcmread(dcm_path)
        
        # Extraire les pixels
        image_array = ds.pixel_array
        
        # Normaliser entre 0 et 1
        if image_array.max() > 0:
            image_array = image_array / image_array.max()
        
        # Redimensionner
        image_resized = cv2.resize(image_array.astype(np.float32), self.image_size)
        
        # Convertir en tensor (grayscale -> 1 canal)
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).float()
        
        # Appliquer les transformations si disponibles
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Obtenir le label
        label = self.labels_dict.get(patient_id, 0)
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': patient_id,
            'filename': dcm_path.name
        }

class ImageTransforms:
    """Transformations d'images"""
    
    @staticmethod
    def get_augmentations():
        """Retourne les augmentations pour l'entraînement"""
        return [
            # Flip horizontal
            lambda x: torch.flip(x, dims=[-1]) if torch.rand(1) > 0.5 else x,
            # Rotation légère (simulée via affine)
            lambda x: x,  # À mettre en œuvre avec torchvision si nécessaire
        ]
