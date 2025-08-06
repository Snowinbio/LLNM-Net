import os
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from PIL import Image


class EdgeImageDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.image_paths = self._load_sorted_paths(root)

    def _load_sorted_paths(self, root):
        paths = []
        classes = self.classes  # ['benign', 'malignant']
        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root, class_name)
            image_files = [f for f in sorted(os.listdir(class_folder)) if f.endswith('.jpg')]
            image_paths = [os.path.join(class_folder, img_file) for img_file in image_files]
            paths.extend(image_paths)
        return paths

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


def get_edge_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
