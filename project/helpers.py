"""Hilfsfunktionen zum Laden, Iterieren, Plotten und Speichern von Bilddaten."""

import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_keypoints, draw_segmentation_masks


def save_image(img_path: Path, img: torch.Tensor, output_folder: str | Path):
    """Speichert ein Tensor-Bild als Datei im Zielordner."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    v2.ToPILImage()(img).save(output_folder / img_path.name)


def read_data(path: str):
    """Liest ein Bild als Tensor von einem Dateipfad."""
    return read_image(str(path))


def iter_images(folder: str | Path):
    """Iteriert über PNG-Bilder in einem Ordner und liefert Pfad plus Bildtensor."""
    for img_path in sorted(Path(folder).glob('*.png')):
        yield img_path, read_data(img_path)


def plot(imgs, row_title=None, bbox_width=3, **imshow_kwargs):
    """Zeigt Bilder optional mit Bounding Boxes, Masken oder Keypoints als Grid an."""
    if not isinstance(imgs[0], list):
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            points = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get('boxes')
                    masks = target.get('masks')
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                    if tv_tensors.is_rotated_bounding_format(boxes.format):
                        boxes = v2.ConvertBoundingBoxFormat('xyxyxyxy')(boxes)
                elif isinstance(target, tv_tensors.KeyPoints):
                    points = target
                else:
                    raise ValueError(f'Unexpected target type: {type(target)}')

            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Wertebereich nach Normalisierung wieder für Anzeige anpassen.
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors='yellow', width=bbox_width)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=['green'] * masks.shape[0], alpha=.65)
            if points is not None:
                img = draw_keypoints(img, points, colors='red', radius=10)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
