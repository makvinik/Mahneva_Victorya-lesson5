"""
Задание 6: Дообучение предобученных моделей
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Добавляем путь к augmentations_basics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics')))
from datasets import CustomImageDataset

# ===== Конфигурация =====
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = '../data'
RESULTS_DIR = 'results/task6'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 1. Подготовка датасетов =====

# Нормализация для ImageNet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Трансформы для тренировки (без Resize – он будет внутри датасета)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Трансформы для валидации (без аугментаций)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Загружаем датасеты (передаём target_size=(224,224), убираем внешний Resize)
train_dataset = CustomImageDataset(
    os.path.join(DATA_ROOT, 'train'),
    transform=train_transform,
    target_size=(IMAGE_SIZE, IMAGE_SIZE)
)
val_dataset = CustomImageDataset(
    os.path.join(DATA_ROOT, 'val'),
    transform=val_transform,
    target_size=(IMAGE_SIZE, IMAGE_SIZE)
)

num_classes = len(train_dataset.get_class_names())
print(f"Количество классов: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ===== 2. Загрузка предобученной модели =====
model = models.resnet18(weights='IMAGENET1K_V1')
# Заменяем последний слой
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ===== 3. Оптимизатор и функция потерь =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ===== 4. Функция обучения и валидации =====

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc='Training'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Evaluating'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ===== 5. Цикл обучения =====
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0.0
best_model_path = os.path.join(RESULTS_DIR, 'best_model.pth')

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Сохраняем лучшую модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f" Новая лучшая модель сохранена (Val Acc: {best_val_acc:.4f})")

print(f"\n Лучшая точность на валидации: {best_val_acc:.4f}")

# ===== 6. Визуализация =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, EPOCHS+1), train_losses, label='Train Loss', marker='o')
axes[0].plot(range(1, EPOCHS+1), val_losses, label='Val Loss', marker='s')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, EPOCHS+1), train_accs, label='Train Acc', marker='o')
axes[1].plot(range(1, EPOCHS+1), val_accs, label='Val Acc', marker='s')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.show()

# ===== 7. Сохранение итогов =====
import csv
csv_path = os.path.join(RESULTS_DIR, 'training_log.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])
    for i in range(EPOCHS):
        writer.writerow([i+1, train_losses[i], train_accs[i], val_losses[i], val_accs[i]])

print(f"📁 Результаты сохранены в {RESULTS_DIR}/")