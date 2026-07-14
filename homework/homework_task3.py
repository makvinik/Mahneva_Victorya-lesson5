"""
Задание 3: Анализ датасета
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter

# Добавляем путь к augmentations_basics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics')))
from datasets import CustomImageDataset

#  Параметры
data_root = '../data/train'   # анализируем только train
results_dir = 'results/task3'
os.makedirs(results_dir, exist_ok=True)

# 1. Сбор информации об изображениях
class_counts = Counter()
image_sizes = []  # список кортежей (width, height)

# Проходим по папкам классов
for class_name in os.listdir(data_root):
    class_path = os.path.join(data_root, class_name)
    if not os.path.isdir(class_path):
        continue
    # Считаем файлы изображений (по расширениям)
    count = 0
    for fname in os.listdir(class_path):
        if fname.lower().endswith(('.jpg', '.jpeg', 'png')):
            count += 1
            img_path = os.path.join(class_path, fname)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    image_sizes.append((w, h))
            except Exception as e:
                print(f" Не удалось открыть {img_path}: {e}")
    class_counts[class_name] = count

# ===== 2. Подсчёт статистики =====
total_images = sum(class_counts.values())
num_classes = len(class_counts)
widths = [s[0] for s in image_sizes]
heights = [s[1] for s in image_sizes]
min_w, max_w = min(widths), max(widths)
min_h, max_h = min(heights), max(heights)
avg_w = sum(widths) / len(widths)
avg_h = sum(heights) / len(heights)

print("\n Статистика датасета:")
print(f"  - Всего классов: {num_classes}")
print(f"  - Всего изображений: {total_images}")
print(f"  - Размеры (ширина x высота):")
print(f"      мин: {min_w} x {min_h}")
print(f"      макс: {max_w} x {max_h}")
print(f"      сред: {avg_w:.1f} x {avg_h:.1f}")

print("\n Количество изображений по классам:")
for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls}: {cnt}")

# 3. Визуализация
# 3.1 Гистограмма количества изображений по классам
plt.figure(figsize=(10, 5))
classes = list(class_counts.keys())
counts = list(class_counts.values())
plt.bar(classes, counts, color='skyblue')
plt.title('Количество изображений по классам')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'class_histogram.png'), dpi=150)
plt.show()

# 3.2 Распределение размеров (гистограмма ширины и высоты)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=30, color='lightgreen', edgecolor='black')
axes[0].set_title('Распределение ширины изображений')
axes[0].set_xlabel('Ширина (px)')
axes[0].set_ylabel('Частота')
axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
axes[1].set_title('Распределение высоты изображений')
axes[1].set_xlabel('Высота (px)')
axes[1].set_ylabel('Частота')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'size_distribution.png'), dpi=150)
plt.show()

