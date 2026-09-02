import json
import os
import warnings
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ACDCDataset(Dataset):
    def __init__(self, root_dir, class_names, condition='fog', use_train_val=True):
        self.condition = condition
        self.root_dir = root_dir
        self.use_train_val = use_train_val
        self.image_base_dir = os.path.join(root_dir, 'rgb_anon')
        self.label_base_dir = os.path.join(root_dir, 'gt_detection')
        self.image_files = []
        self.image_info_map = {}

        splits = ('train', 'val') if use_train_val else ('train',)
        for split in splits:
            if condition:
                label_path = os.path.join(
                    self.label_base_dir,
                    condition,
                    f'instancesonly_{condition}_{split}_gt_detection.json',
                )
            else:
                label_path = os.path.join(
                    self.label_base_dir,
                    f'instancesonly_{split}_gt_detection.json',
                )
            if not os.path.exists(label_path):
                warnings.warn(f'Label file not found: {label_path}')
                continue

            with open(label_path, 'r', encoding='utf-8') as label_file:
                label_data = json.load(label_file)
            images = {image['id']: image for image in label_data['images']}
            categories = {
                category['id']: category
                for category in label_data['categories']
            }
            annotations = defaultdict(list)
            for annotation in label_data['annotations']:
                annotations[annotation['image_id']].append(annotation)

            for image_id, image_info in images.items():
                path_parts = image_info['file_name'].split('/')
                if len(path_parts) >= 2:
                    path_parts[0] = condition
                    path_parts[1] = split
                image_path = os.path.join(
                    self.image_base_dir, '/'.join(path_parts)
                )
                if not os.path.exists(image_path):
                    continue
                self.image_info_map[len(self.image_files)] = {
                    'image_path': image_path,
                    'width': image_info['width'],
                    'height': image_info['height'],
                    'annotations': annotations[image_id],
                    'categories_dict': categories,
                }
                self.image_files.append(image_path)

        self.target_size = (512, 1024)
        self.transform = transforms.ToTensor()
        self.class_to_idx = {
            name.lower(): index + 1 for index, name in enumerate(class_names)
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        info = self.image_info_map[index]
        with Image.open(info['image_path']) as source_image:
            image = source_image.convert('RGB')
        current_width, current_height = image.size
        target_height, target_width = self.target_size
        image = image.resize((target_width, target_height), Image.BILINEAR)
        scale_x = target_width / current_width
        scale_y = target_height / current_height

        boxes = []
        labels = []
        categories = info['categories_dict']
        for annotation in info['annotations']:
            category = categories.get(annotation['category_id'])
            if category is None:
                continue
            category_name = category['name'].lower()
            if category_name not in self.class_to_idx:
                continue
            x, y, width, height = annotation['bbox']
            x1 = max(0, min(x * scale_x, target_width))
            y1 = max(0, min(y * scale_y, target_height))
            x2 = max(0, min((x + width) * scale_x, target_width))
            y2 = max(0, min((y + height) * scale_y, target_height))
            if x2 - x1 > 1 and y2 - y1 > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[category_name])

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
