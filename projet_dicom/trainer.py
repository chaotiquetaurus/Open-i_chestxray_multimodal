import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, MODELS_DIR, LOGS_DIR
from models_arch import ChestXRayClassifier, SimpleCNN
from dataset import ChestXRayDataset

class Trainer:
    def __init__(self, model_name="simple_cnn", num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialiser le modèle
        if model_name == "resnet18":
            self.model = ChestXRayClassifier(num_classes=num_classes)
        else:
            self.model = SimpleCNN(num_classes=num_classes)
        
        self.model = self.model.to(self.device)
        
        # Loss et optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def train_epoch(self, train_loader):
        """Entraîner une époque"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Valider le modèle"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS):
        """Entraîner le modèle"""
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Entraînement
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Sauvegarder le meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
                print("✅ Modèle sauvegardé!")
        
        # Sauvegarder l'historique
        self.save_history()
    
    def save_model(self):
        """Sauvegarder le modèle"""
        model_path = MODELS_DIR / f"best_model_{self.timestamp}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss
        }, model_path)
        print(f"Modèle sauvegardé: {model_path}")
    
    def save_history(self):
        """Sauvegarder l'historique"""
        history_path = LOGS_DIR / f"history_{self.timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Historique sauvegardé: {history_path}")

if __name__ == "__main__":
    print("Trainer module ready")
