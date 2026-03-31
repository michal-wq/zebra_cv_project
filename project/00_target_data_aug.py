import torch
from torchvision.transforms import v2
from torchvision.io import decode_image, read_image
import matplotlib.pyplot as plt
from helpers import plot        # helper funktion von PyTorch Doku
from pathlib import Path

def read_data(path: str):
    return read_image(str(path))

def transform_image(img, transformation):
    return transformation(img)

def iter_images(folder: str | Path):
    for img_path in sorted(Path(folder).glob('*.png')):
        yield img_path, read_data(img_path)

def save_image(img_path: Path, img: torch.Tensor, output_folder: str | Path):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    v2.ToPILImage()(img).save(output_folder / img_path.name)

def main():

    FOLDER = '../swissimage_annotator/static/data/st gallen/y/'
    OUTPUT = 'aug_data_try'
    AUG_PER_IMG = 24

    transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(6),
    v2.RandomPerspective(distortion_scale=0.6, p=1.0),
    v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for img_path, img in iter_images(FOLDER):
        for i in range(AUG_PER_IMG):
            out = transform_image(img, transforms)
            aug_path = img_path.with_stem(f'{img_path.stem}_{i}')  # e.g. 2753025_1188375_0.png
            save_image(aug_path, out, OUTPUT)
            print(f'Saved {aug_path.name}')
        break

    
main()
