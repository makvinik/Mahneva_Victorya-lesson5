"""
Задание 1: Стандартные аугментации torchvision
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Добавляем путь к augmentations_basics, чтобы импортировать оттуда
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics')))

from datasets import CustomImageDataset
from torchvision import transforms

# ===== Папка для сохранения результатов =====
results_dir = 'results'

#  1. Загрузка датасета (без аугментаций)
root = '../data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Выбираем по одному изображению из первых 5 классов
num_classes = min(5, len(class_names))
sample_images = []
sample_labels = []
seen_classes = set()
for idx in range(len(dataset)):
    img, label = dataset[idx]
    if label not in seen_classes:
        seen_classes.add(label)
        sample_images.append(img)
        sample_labels.append(class_names[label])
        if len(sample_images) == num_classes:
            break

print(f"Выбрано {len(sample_images)} изображений из классов: {sample_labels}")

# 2. Определение пайплайнов аугментаций

# 2.1 Отдельные аугментации (каждая применяется отдельно)
single_augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]

# 2.2 Комбинированный пайплайн (все аугментации вместе)
combined_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(200, padding=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(degrees=20),
    transforms.RandomGrayscale(p=0.2)
])


# ===== Функция визуализации и сохранения =====

def show_and_save(original_img, aug_imgs, titles, class_name, suffix, subfolder=""):
    """
    Показывает и сохраняет график с оригиналом и аугментированными изображениями.
    """
    n = len(aug_imgs) + 1
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))

    # Оригинал
    axes[0].imshow(np.array(original_img))
    axes[0].set_title(f"Оригинал\n{class_name}")
    axes[0].axis('off')

    # Аугментации
    for i, (img, title) in enumerate(zip(aug_imgs, titles)):
        if hasattr(img, 'numpy'):
            img_np = img.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(img)
        axes[i + 1].imshow(img_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()

    # Сохраняем в папку results (подпапка создаётся, если её нет)
    save_folder = os.path.join(results_dir, subfolder)
    os.makedirs(save_folder, exist_ok=True)  # создаём подпапку, если её ещё нет
    clean_class = class_name.replace(' ', '_')
    filename = f"{clean_class}_{suffix}.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {save_path}")

    plt.show()
    plt.close(fig)

# 4. Применение аугментаций к каждому изображению
for img, class_name in zip(sample_images, sample_labels):
    print(f"\n=== Обработка класса: {class_name} ===")

    # 4.1 Отдельные аугментации
    single_results = []
    titles = []
    for name, aug in single_augs:
        aug_pipeline = transforms.Compose([aug, transforms.ToTensor()])
        aug_img = aug_pipeline(img)
        single_results.append(aug_img)
        titles.append(name)

    # Визуализируем отдельные аугментации (используем show_and_save)
    show_and_save(img, single_results, titles, class_name, suffix="single", subfolder="single")

    # 4.2 Комбинированный пайплайн (создаём 6 вариантов)
    combined_results = []
    for _ in range(6):
        combined_img = combined_aug(img)
        if not isinstance(combined_img, torch.Tensor):
            combined_img = transforms.ToTensor()(combined_img)
        combined_results.append(combined_img)

    # Визуализируем комбинированные аугментации
    titles_combined = [f"Комб. {i+1}" for i in range(len(combined_results))]
    show_and_save(img, combined_results, titles_combined, class_name, suffix="combined", subfolder="combined")