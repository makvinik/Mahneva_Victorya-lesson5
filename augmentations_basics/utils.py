import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def show_single_augmentation(original, augmented, title):
    """
    Показывает оригинальное и аугментированное изображение.
    Поддерживает как PIL Image, так и тензоры.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Функция для преобразования в numpy
    def to_numpy(img):
        if hasattr(img, 'numpy'):  # тензор PyTorch
            return img.numpy().transpose(1, 2, 0)
        elif hasattr(img, 'shape'):  # уже numpy
            return img
        else:  # PIL Image
            return np.array(img)

    orig_np = to_numpy(original)
    aug_np = to_numpy(augmented)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    axes[1].imshow(aug_np)
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

def show_multiple_augmentations(original, augmented_list, titles):
    """
    Показывает оригинал и несколько аугментированных изображений.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def to_numpy(img):
        if hasattr(img, 'numpy'):
            return img.numpy().transpose(1, 2, 0)
        elif hasattr(img, 'shape'):
            return img
        else:
            return np.array(img)

    orig_np = to_numpy(original)
    n = len(augmented_list) + 1
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title('Оригинал')
    axes[0].axis('off')
    for i, (aug, title) in enumerate(zip(augmented_list, titles), start=1):
        aug_np = to_numpy(aug)
        axes[i].imshow(aug_np)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()