"""
Задание 2: Кастомные аугментации
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# Добавляем путь к augmentations_basics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics')))

from datasets import CustomImageDataset
from torchvision import transforms
from extra_augs import AddGaussianNoise, ElasticTransform, Solarize, Posterize, AutoContrast

# ===== Папка для сохранения =====
results_dir = 'results/task2'
os.makedirs(results_dir, exist_ok=True)


# 1. Реализация кастомных аугментаций

class RandomBlur:
    """Случайное размытие (GaussianBlur) со случайным радиусом."""

    def __init__(self, radius=(1, 5), p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            r = np.random.uniform(self.radius[0], self.radius[1])
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


class RandomPerspective:
    """Случайное перспективное искажение (аффинное) с заданными границами."""

    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            from torchvision.transforms import RandomPerspective as RP
            # Используем готовый трансформ из torchvision с фиксированными параметрами
            return RP(distortion_scale=self.distortion_scale, p=1.0)(img)
        return img


class RandomBrightnessContrast:
    """Случайное изменение яркости и контраста."""

    def __init__(self, brightness=(0.8, 1.2), contrast=(0.8, 1.2), p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            b = np.random.uniform(self.brightness[0], self.brightness[1])
            c = np.random.uniform(self.contrast[0], self.contrast[1])
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)
        return img


# 2. Загрузка данных
root = '../data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Берём по 2 изображения из первых 3 классов (всего 6)
sample_images = []
sample_labels = []
counts = {cls: 0 for cls in class_names[:3]}
for idx in range(len(dataset)):
    img, label = dataset[idx]
    cls = class_names[label]
    if cls in counts and counts[cls] < 2:
        sample_images.append(img)
        sample_labels.append(cls)
        counts[cls] += 1
        if all(v >= 2 for v in counts.values()):
            break

print(f"Выбрано {len(sample_images)} изображений из классов: {sample_labels}")

# 3. Определение пайплайнов

# Свои кастомные аугментации
custom_augs = [
    ("RandomBlur", RandomBlur(radius=(2, 6), p=1.0)),
    ("RandomPerspective", RandomPerspective(distortion_scale=0.3, p=1.0)),
    ("RandomBrightnessContrast", RandomBrightnessContrast(brightness=(0.5, 1.5), contrast=(0.5, 1.5), p=1.0))
]

# аугментации из extra_augs
extra_augs = [
    ("AddGaussianNoise", AddGaussianNoise(mean=0.0, std=0.1)),
    ("ElasticTransform", ElasticTransform(alpha=1, sigma=50, p=1.0)),
    ("Solarize", Solarize(threshold=128))
]


# 4. Функция визуализации и сохранения

def save_comparison(original, custom_results, extra_results, custom_names, extra_names, class_name, idx):
    """
    Сохраняет сравнение: оригинал, кастомные аугментации, готовые аугментации.
    """
    n_custom = len(custom_results)
    n_extra = len(extra_results)
    n_total = 1 + n_custom + n_extra
    fig, axes = plt.subplots(1, n_total, figsize=(n_total * 3, 3))

    # Оригинал
    axes[0].imshow(np.array(original))
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Кастомные
    for i, (img, title) in enumerate(zip(custom_results, custom_names)):
        if hasattr(img, 'numpy'):
            img_np = img.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(img)
        axes[1 + i].imshow(img_np)
        axes[1 + i].set_title(f"Каст.\n{title}")
        axes[1 + i].axis('off')

    # Готовые
    for i, (img, title) in enumerate(zip(extra_results, extra_names)):
        idx_extra = 1 + n_custom + i
        if hasattr(img, 'numpy'):
            img_np = img.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(img)
        axes[idx_extra].imshow(img_np)
        axes[idx_extra].set_title(f"Гот.\n{title}")
        axes[idx_extra].axis('off')

    plt.tight_layout()
    clean_class = class_name.replace(' ', '_')
    filename = f"{clean_class}_{idx}.png"
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {save_path}")
    plt.show()
    plt.close(fig)


# 5. Применение аугментаций

for i, (img, class_name) in enumerate(zip(sample_images, sample_labels)):
    print(f"\n=== Обработка класса: {class_name} (изображение {i + 1}) ===")

    # Кастомные
    custom_imgs = []
    custom_names = []
    for name, aug in custom_augs:
        aug_img = aug(img)
        custom_imgs.append(aug_img)
        custom_names.append(name)

    # Готовые
    extra_imgs = []
    extra_names = []
    for name, aug in extra_augs:
        # Применяем аугментацию (преобразуем в тензор, если требуется)
        if hasattr(aug, '__call__'):
            # Для простоты применяем к PIL Image (если аугментация работает с PIL)
            try:
                aug_img = aug(img)
            except:
                # Иначе через ToTensor
                to_tensor = transforms.ToTensor()
                aug_img = aug(to_tensor(img))
        else:
            aug_img = aug(img)
        extra_imgs.append(aug_img)
        extra_names.append(name)

    # Сохраняем сравнение
    save_comparison(img, custom_imgs, extra_imgs, custom_names, extra_names, class_name, i)
