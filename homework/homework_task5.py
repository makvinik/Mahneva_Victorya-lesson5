"""
Задание 5: Эксперимент с размерами изображений
"""

import sys
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import random

# ===== Папка для сохранения =====
results_dir = 'results/task5'
os.makedirs(results_dir, exist_ok=True)

# ===== Настройки =====
SIZES = [64, 128, 224, 512]
NUM_IMAGES = 50
DATA_ROOT = '../data/train'

# ===== 1. Загрузка изображений =====
def load_images_from_folder(root, num_images):
    all_paths = []
    for class_name in os.listdir(root):
        class_path = os.path.join(root, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                all_paths.append(os.path.join(class_path, fname))
    random.shuffle(all_paths)
    selected = all_paths[:num_images]
    images = []
    for path in selected:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
    return images

print(f"Загрузка {NUM_IMAGES} изображений из {DATA_ROOT}...")
images = load_images_from_folder(DATA_ROOT, NUM_IMAGES)
print(f"Загружено {len(images)} изображений.")

# ===== 2. Пайплайны =====
# Без аугментаций (только ToTensor)
base_transform = transforms.ToTensor()

# С аугментациями (без ToTensor, он будет добавлен внутри)
aug_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor()
])

# ===== 3. Функция замера =====
def measure(size, images, base_transform, aug_pipeline, num_iter):
    resize = transforms.Resize((size, size))

    # Время только resize + to_tensor (без аугментаций)
    base_times = []
    for img in images[:num_iter]:
        start = time.perf_counter()
        img_resized = resize(img)
        _ = base_transform(img_resized)
        base_times.append(time.perf_counter() - start)

    # Время resize + aug + to_tensor
    aug_times = []
    for img in images[:num_iter]:
        start = time.perf_counter()
        img_resized = resize(img)
        _ = aug_pipeline(img_resized)
        aug_times.append(time.perf_counter() - start)

    avg_base = np.mean(base_times)
    avg_aug = np.mean(aug_times)
    avg_aug_only = avg_aug - avg_base  # время только аугментаций (без resize и to_tensor)

    # Память: размер тензора для данного размера (приблизительно)
    sample_tensor = base_transform(resize(images[0]))  # берем первое изображение
    mem = sample_tensor.element_size() * sample_tensor.nelement() / (1024 * 1024)

    return avg_base, avg_aug_only, mem

# ===== 4. Эксперимент =====
results = {}
for size in SIZES:
    print(f"Измерение для размера {size}x{size}...")
    base_time, aug_time, mem = measure(size, images, base_transform, aug_pipeline, NUM_IMAGES)
    results[size] = {
        'load_time': base_time,
        'aug_time': aug_time,
        'memory_mb': mem
    }
    print(f"  Load (resize+to_tensor): {base_time:.4f} с, Aug (only): {aug_time:.4f} с, Memory: {mem:.2f} МБ")

# ===== 5. Графики =====
sizes = list(results.keys())
load_times = [results[s]['load_time'] for s in sizes]
aug_times = [results[s]['aug_time'] for s in sizes]
memories = [results[s]['memory_mb'] for s in sizes]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(sizes, load_times, marker='o', color='blue')
axes[0].set_title('Время загрузки (resize + to_tensor)')
axes[0].set_xlabel('Размер (px)')
axes[0].set_ylabel('Время (с)')
axes[0].grid(True)

axes[1].plot(sizes, aug_times, marker='s', color='green')
axes[1].set_title('Время аугментаций (без resize и to_tensor)')
axes[1].set_xlabel('Размер (px)')
axes[1].set_ylabel('Время (с)')
axes[1].grid(True)

axes[2].plot(sizes, memories, marker='^', color='red')
axes[2].set_title('Потребление памяти (один тензор)')
axes[2].set_xlabel('Размер (px)')
axes[2].set_ylabel('Память (МБ)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'size_experiment.png'), dpi=150)
plt.show()

# ===== 6. CSV =====
import csv
csv_path = os.path.join(results_dir, 'results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Size', 'Load Time (s)', 'Aug Time (s)', 'Memory (MB)'])
    for s in sizes:
        writer.writerow([s, f"{results[s]['load_time']:.6f}", f"{results[s]['aug_time']:.6f}", f"{results[s]['memory_mb']:.2f}"])
print(f"Таблица сохранена: {csv_path}")
