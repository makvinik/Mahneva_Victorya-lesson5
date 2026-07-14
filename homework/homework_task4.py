"""
Задание 4: Pipeline аугментаций
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'augmentations_basics')))
from datasets import CustomImageDataset
from extra_augs import AddGaussianNoise, ElasticTransform, Solarize, Posterize, AutoContrast

# Папка для сохранения
results_dir = 'results/task4'
os.makedirs(results_dir, exist_ok=True)


# 1. Класс AugmentationPipeline

class AugmentationPipeline:
    """
    Класс для управления пайплайном аугментаций.
    """

    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        """Добавляет аугментацию по имени."""
        self.augmentations[name] = aug
        print(f" Добавлена аугментация: {name}")

    def remove_augmentation(self, name):
        """Удаляет аугментацию по имени."""
        if name in self.augmentations:
            del self.augmentations[name]
            print(f" Удалена аугментация: {name}")

    def get_augmentations(self):
        """Возвращает список имён всех аугментаций."""
        return list(self.augmentations.keys())

    def apply(self, image):
        """
        Применяет все аугментации последовательно к изображению.
        Возвращает список аугментированных изображений (по одному на каждую аугментацию).
        """
        results = []
        for name, aug in self.augmentations.items():
            try:
                # Пытаемся применить аугментацию напрямую к PIL Image
                aug_img = aug(image)
                results.append((name, aug_img))
            except:
                # Если не работает, пробуем через ToTensor (для тензорных аугментаций)
                try:
                    to_tensor = transforms.ToTensor()
                    tensor_img = to_tensor(image)
                    aug_tensor = aug(tensor_img)
                    # Преобразуем обратно в PIL для визуализации
                    to_pil = transforms.ToPILImage()
                    aug_img = to_pil(aug_tensor)
                    results.append((name, aug_img))
                except Exception as e:
                    results.append((name, image))  # возвращаем оригинал
        return results


# ===== 2. Создание конфигураций =====

def create_light_pipeline():
    """Лёгкая конфигурация: минимальные аугментации."""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation('SmallRotation', transforms.RandomRotation(degrees=10))
    return pipeline


def create_medium_pipeline():
    """Средняя конфигурация: умеренные аугментации."""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation('Rotation', transforms.RandomRotation(degrees=20))
    pipeline.add_augmentation('ColorJitter',
                              transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    pipeline.add_augmentation('RandomCrop', transforms.RandomCrop(200, padding=20))
    return pipeline


def create_heavy_pipeline():
    """Тяжёлая конфигурация: агрессивные аугментации."""
    pipeline = AugmentationPipeline()
    pipeline.add_augmentation('HorizontalFlip', transforms.RandomHorizontalFlip(p=0.5))
    pipeline.add_augmentation('Rotation', transforms.RandomRotation(degrees=45))
    pipeline.add_augmentation('ColorJitter',
                              transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
    pipeline.add_augmentation('RandomCrop', transforms.RandomCrop(180, padding=30))
    pipeline.add_augmentation('GaussianNoise', AddGaussianNoise(mean=0.0, std=0.1))
    pipeline.add_augmentation('ElasticTransform', ElasticTransform(alpha=1, sigma=40, p=0.5))
    return pipeline


# 3. Загрузка данных
root = '../data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
class_names = dataset.get_class_names()

# Берём по 2 изображения из первых 3 классов
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


# 4. Функция визуализации и сохранения

def save_pipeline_results(original, results, class_name, idx, pipeline_name):
    """
    Сохраняет график: оригинал + все аугментированные изображения из пайплайна.
    """
    n = 1 + len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))

    # Оригинал
    axes[0].imshow(np.array(original))
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментации
    for i, (name, aug_img) in enumerate(results):
        if hasattr(aug_img, 'numpy'):
            img_np = aug_img.numpy().transpose(1, 2, 0)
        else:
            img_np = np.array(aug_img)
        axes[i + 1].imshow(img_np)
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    plt.tight_layout()
    clean_class = class_name.replace(' ', '_')
    filename = f"{clean_class}_{idx}_{pipeline_name}.png"
    save_path = os.path.join(results_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# 5. Применение конфигураций

pipelines = {
    'light': create_light_pipeline(),
    'medium': create_medium_pipeline(),
    'heavy': create_heavy_pipeline()
}

for pipe_name, pipeline in pipelines.items():
    print(f"\n=== Применение {pipe_name} пайплайна ===")
    print(f"  Аугментации: {pipeline.get_augmentations()}")

    for i, (img, class_name) in enumerate(zip(sample_images, sample_labels)):
        print(f"  Класс: {class_name}, изображение {i + 1}")
        results = pipeline.apply(img)
        save_pipeline_results(img, results, class_name, i, pipe_name)
