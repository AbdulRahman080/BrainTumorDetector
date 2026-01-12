#!/usr/bin/env python
# coding: utf-8

# In[146]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import Counter


# In[136]:


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)


# In[137]:


path = '/Users/abdulrahman/Downloads/brain_tumor_dataset'


# In[138]:


os.listdir(path)


# In[144]:


plt.figure(figsize=(12, 10))
classes = os.listdir(path)

img_count = 0
for cls in ['healthy', 'pituitary', 'glioma', 'meningioma']:
    cls_path = os.path.join(path, cls)
    for img in os.listdir(cls_path)[:5]:
        img_count += 1
        img_path = os.path.join(cls_path, img)
        image = plt.imread(img_path)
        plt.subplot(4, 5, img_count)
        plt.imshow(image)
        plt.title(cls)
        plt.axis("off")

plt.tight_layout()
plt.show()


# In[139]:


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transforms = transform
        self.image_paths = []
          self.labels = []
        
        self.class_to_idx = {
            'healthy': 0,
            'pituitary': 1,
            'glioma': 2,
            'meningioma': 3
        }
        
        self._load_dataset()
    
    def _load_dataset(self):
        for condition in ['healthy', 'pituitary', 'glioma', 'meningioma']:
            condition_path = os.path.join(self.root_dir, condition)
            if not os.path.isdir(condition_path):
                continue
            label = self.class_to_idx[condition]
            for img_name in os.listdir(condition_path):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(condition_path, img_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
         
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# In[140]:


dataset = BrainTumorDataset(path)


# In[141]:


total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size]
)


# In[90]:


def compute_mean_std(loader):
    mean = torch.zeros(1)
    std = torch.zeros(1)
    total_pixels = 0
    
    with torch.no_grad():
        for images, _ in loader:
            batch_pixels = images.numel() / images.size(1)
            total_pixels += batch_pixels

            mean += images.sum(dim=[0, 2, 3])
            std += (images ** 2).sum(dim=[0, 2, 3])
    
    mean /= total_pixels
    std = (std / total_pixels - mean ** 2).sqrt()
    
    return mean, std


# In[91]:


temp_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset.dataset.transform = temp_transforms

temp_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False
)

mean, std = compute_mean_std(temp_loader)

print('Mean: ', mean)
print('STD: ', std)


# In[92]:


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean = mean, std = std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# In[93]:


train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform


# In[94]:


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# In[95]:


class BrainTumorClassification(nn.Module):
    def __init__(self, num_classes=4):
        
        super(BrainTumorClassification, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        


# In[96]:


model = BrainTumorClassification()


# In[97]:


labels = train_dataset.dataset.labels

counts = Counter(labels)
total = sum(counts.values())

class_weights = [total/counts[i] for i in range(len(counts))]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print('Class Weights: ', class_weights)


# In[98]:


loss_fcn = nn.CrossEntropyLoss(class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[99]:


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


# In[100]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)


# In[103]:


def training_loop(model, train_loader, val_loader, loss_fcn, optimizer, n_epochs=20, scheduler=None, early_stopping_patience = 5, save_path='best_model.pth'):
    
    model.to(device)
    
    best_val_loss = np.inf
    epochs_no_improve = 0
        
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print('---Starting Training---')

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_train_loss = 0.0
        running_train_acc = 0.0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{n_epochs} [Training]', leave=True)
        
        for batch_idx, (images, labels) in enumerate(train_progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            running_train_acc += accuracy(outputs, labels)
            
            train_progress_bar.set_postfix(loss=running_train_loss/(batch_idx+1),
                                           acc=running_train_acc/(batch_idx+1))
        
        train_loss = running_train_loss / len(train_loader)
        train_acc = running_train_acc / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_acc = 0.0
            val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{n_epochs} [Validating]', leave=True)
            
            for data_idx, (images, labels) in enumerate(val_progress_bar):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_fcn(outputs, labels)
                
                running_val_loss += loss.item()
                running_val_acc += accuracy(outputs, labels)
                
                val_progress_bar.set_postfix(loss=running_val_loss/(batch_idx+1),
                                             acc=running_val_acc/(batch_idx+1))
                
        val_loss = running_val_loss / len(val_loader)
        val_acc = running_val_acc / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, '
              f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')
            
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        
        else:
            epochs_no_improve += 1
            
        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
            
            
    print('---Training Finished---')
    
    return train_losses, train_accs, val_losses, val_accs
        


# In[104]:


train_losses, train_accs, val_losss, val_accs = training_loop(model, train_loader, val_loader, loss_fcn, optimizer, 30, scheduler, )


# In[150]:


def evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    
    report = classification_report(
        y_true, y_pred,
        target_names = ['healty', 'pituitary', 'glioma', 'meningioma']
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, report, cm


# In[151]:


test_acc, test_report, test_cm = evaluate(model, test_loader, device)

print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", test_report)
print("\nConfusion Matrix:\n", test_cm)


# In[152]:


plt.figure(figsize=(6,5))
sns.heatmap(
    test_cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['healthy','pituitary','glioma','meningioma'],
    yticklabels=['healthy','pituitary','glioma','meningioma']
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix – Test Set')
plt.show()


# In[153]:


torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': ['healthy','pituitary','glioma','meningioma'],
    'val_accuracy': 0.9488
}, 'brain_tumor_cnn.pth')


# In[ ]:




