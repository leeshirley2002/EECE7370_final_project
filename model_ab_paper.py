import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# ============ CONFIGURATION ============
IMG_SIZE = 50  # Paper uses 50x50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# ============ DATASET CLASS (LAZY LOADING) ============
class RegionDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        img = torch.FloatTensor(img).permute(2, 0, 1)
        label = torch.LongTensor([label])
        
        return img, label.squeeze()

# ============ GET FILE PATHS ============
def get_file_paths(data_path, region):
    """
    region: 'leftEye', 'rightEye', 'nose', or 'eyes' (combines left+right)
    """
    file_paths = []
    labels = []
    
    if region == 'eyes':
        regions = ['leftEye', 'rightEye']
    else:
        regions = [region]
    
    for r in regions:
        # Original (real) - label 0
        folder = os.path.join(data_path, 'Features', 'original', r)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_paths.append(os.path.join(folder, filename))
                labels.append(0)
        
        # Manipulated (fake) - label 1
        folder = os.path.join(data_path, 'Features', 'manipulated', r)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_paths.append(os.path.join(folder, filename))
                labels.append(1)
    
    return file_paths, labels

# ============ MODEL A: 12-layer CNN with Batch Norm (Paper Section 4.2.1) ============
class ModelA(nn.Module):
    """
    Paper Model A: 12 layers with Batch Normalization
    Input: 50x50x3
    Three blocks: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
    """
    def __init__(self):
        super(ModelA, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: Real, Fake
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# ============ MODEL B: 6-layer CNN without Batch Norm (Paper Section 4.2.2) ============
class ModelB(nn.Module):
    """
    Paper Model B: 6 layers without Batch Normalization
    Input: 50x50x3
    Three blocks: Conv -> ReLU -> MaxPool -> Dropout
    """
    def __init__(self):
        super(ModelB, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: Real, Fake
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

# ============ TRAINING FUNCTION ============
def train(data_path, region, model_type, epochs):
    print(f"\n{'='*50}")
    print(f"Training Model {model_type.upper()} for {region.upper()}")
    print(f"{'='*50}")
    
    # Load data
    print("Loading file paths...")
    file_paths, labels = get_file_paths(data_path, region)
    print(f"Total images: {len(file_paths)}")
    print(f"Real: {labels.count(0)}, Fake: {labels.count(1)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = RegionDataset(X_train, y_train)
    test_dataset = RegionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if model_type == 'a':
        model = ModelA().to(device)
    else:
        model = ModelB().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    model_name = f"model_{model_type}_{region}_best.pth"
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_name)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")
    print(f"Model saved as {model_name}")
    
    return best_acc

# ============ MAIN ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Model A/B from paper')
    parser.add_argument('--data', type=str, default='exported_data', help='Path to data folder')
    parser.add_argument('--region', type=str, required=True, choices=['eyes', 'leftEye', 'rightEye', 'nose'], 
                        help='Region to train on')
    parser.add_argument('--model', type=str, default='a', choices=['a', 'b'], 
                        help='Model type: a (with BatchNorm) or b (without)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    train(args.data, args.region, args.model, args.epochs)