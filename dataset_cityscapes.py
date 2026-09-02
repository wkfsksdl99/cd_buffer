import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, root_dir, class_names, split='train', task='initial', severity=''):
        self.task = task
        self.severity = severity
        self.root_dir = root_dir
        self.split = split

        if task == 'initial':
            self.image_dir = os.path.join(
                root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', split
            )
        elif task == 'foggy':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit_foggy', split)
        else:
            self.image_dir = os.path.join(root_dir, f'leftImg8bit_{task}', split)
        self.label_dir = os.path.join(
            root_dir, 'leftImg8bit_trainvaltest', 'annotations', split
        )

        self.image_files = []
        self.label_files = []
        if os.path.exists(self.image_dir):
            for location in os.listdir(self.image_dir):
                location_path = os.path.join(self.image_dir, location)
                if not os.path.isdir(location_path):
                    continue
                for image_name in os.listdir(location_path):
                    if not image_name.endswith('.png'):
                        continue
                    if severity and severity not in image_name:
                        continue
                    if task == 'initial':
                        label_name = image_name.replace(
                            '_leftImg8bit.png', '.json'
                        )
                    elif task == 'foggy':
                        label_name = image_name.replace(
                            f'_leftImg8bit_foggy_beta_{severity}.png', '.json'
                        )
                    else:
                        label_name = image_name.replace(
                            f'_leftImg8bit_{task}.png', '.json'
                        )
                    label_path = os.path.join(self.label_dir, label_name)
                    if os.path.exists(label_path):
                        self.image_files.append(
                            os.path.join(location_path, image_name)
                        )
                        self.label_files.append(label_path)

        self.target_size = (512, 1024)
        self.transform = transforms.ToTensor()
        self.class_to_idx = {
            name: index + 1 for index, name in enumerate(class_names)
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as source_image:
            image = source_image.convert('RGB')
        current_width, current_height = image.size
        target_height, target_width = self.target_size
        image = image.resize((target_width, target_height), Image.BILINEAR)
        scale_x = target_width / current_width
        scale_y = target_height / current_height

        boxes = []
        labels = []
        with open(self.label_files[index], 'r', encoding='utf-8') as label_file:
            label_data = json.load(label_file)
        for annotation in label_data['annotations']:
            class_name = annotation['class_name']
            if class_name not in self.class_to_idx:
                continue
            x1, y1, x2, y2 = annotation['bbox']
            x1 = max(0, min(x1 * scale_x, target_width))
            y1 = max(0, min(y1 * scale_y, target_height))
            x2 = max(0, min(x2 * scale_x, target_width))
            y2 = max(0, min(y2 * scale_y, target_height))
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([x1, y1, x2, y2])
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
