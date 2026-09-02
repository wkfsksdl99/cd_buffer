import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KITTIDataset(Dataset):
    def __init__(self, root_dir, class_names, split='train', task='initial', severity=''):
        self.task = task
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, task, 'images')
        self.label_dir = os.path.join(root_dir, 'label_2')
        if task != 'initial':
            self.image_dir = os.path.join(root_dir, task, severity)

        split_path = os.path.join(root_dir, f'{split}.txt')
        if not os.path.exists(split_path):
            raise FileNotFoundError(f'Split file not found at: {split_path}')
        with open(split_path, 'r', encoding='utf-8') as split_file:
            self.image_files = [line.strip() for line in split_file]

        self.target_size = (352, 1216)
        self.transform = transforms.ToTensor()
        self.class_to_idx = {
            name: index + 1 for index, name in enumerate(class_names)
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(
            self.label_dir, image_name.replace('.png', '.txt')
        )

        with Image.open(image_path) as source_image:
            image = source_image.convert('RGB')
        current_width, current_height = image.size
        target_height, target_width = self.target_size
        image = image.resize((target_width, target_height), Image.BILINEAR)

        if self.task != 'initial':
            original_path = os.path.join(
                self.root_dir, 'initial', 'images', image_name
            )
            if os.path.exists(original_path):
                with Image.open(original_path) as original_image:
                    original_width, original_height = original_image.size
            else:
                original_width, original_height = 1224, 370
            scale_x = target_width / original_width
            scale_y = target_height / original_height
        else:
            scale_x = target_width / current_width
            scale_y = target_height / current_height

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    class_name = parts[0]
                    if class_name not in self.class_to_idx:
                        continue
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    boxes.append([
                        x1 * scale_x,
                        y1 * scale_y,
                        x2 * scale_x,
                        y2 * scale_y,
                    ])
                    labels.append(self.class_to_idx[class_name])

        target = {
            'boxes': (
                torch.as_tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.empty((0, 4), dtype=torch.float32)
            ),
            'labels': (
                torch.as_tensor(labels, dtype=torch.int64)
                if labels
                else torch.empty(0, dtype=torch.int64)
            ),
        }
        return self.transform(image), target
