"""Erzeugt pro Bild mehrere augmentierte Varianten und speichert sie in einem Ausgabeordner."""

import torch
from torchvision.transforms import v2
from helpers import iter_images, save_image

def transform_image(img, transformation):
    """Wendet eine definierte Transformation auf ein einzelnes Bild an."""
    return transformation(img)

def main():
    """Lädt Bilder aus einem Ordner, augmentiert sie mehrfach und speichert die Ergebnisse."""
    folder = '../swissimage_annotator/static/data/st gallen/y/'
    output = 'aug_data_try'
    aug_per_img = 24

    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(6),
        v2.RandomPerspective(distortion_scale=0.6, p=1.0),
        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    for img_path, img in iter_images(folder):
        for i in range(aug_per_img):
            out = transform_image(img, transforms)
            # Eindeutiger Dateiname pro Augmentierung.
            aug_path = img_path.with_stem(f'{img_path.stem}_{i}')
            save_image(aug_path, out, output)
            print(f'Saved {aug_path.name}')
        break


main()
