import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from pycocotools.coco import COCO
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)

class VOCDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.image_ids = []
        with open(os.path.join(root, f'ImageSets/Main/{split}.txt')) as f:
            self.image_ids = [line.strip() for line in f]

        self.classes = [
            '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root, 'JPEGImages', f'{img_id}.jpg')
        anno_path = os.path.join(self.root, 'Annotations', f'{img_id}.xml')

        img = Image.open(img_path).convert('RGB')
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            boxes.append([
                float(bbox.find('xmin').text),
                float(bbox.find('ymin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymax').text)
            ])
            labels.append(self.class_to_idx[obj.find('name').text])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)
class COCODataset(Dataset):
    def __init__(self, root, split='train2017', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.cat_ids = self.coco.getCatIds()
        self.cat_to_label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        # 預加載圖片路徑和標註
        self.image_infos = [self.coco.loadImgs(img_id)[0] for img_id in self.image_ids]
        self.annotations = [self.coco.getAnnIds(imgIds=img_id) for img_id in self.image_ids]

    def __getitem__(self, idx):
        img_info = self.image_infos[idx]
        img_path = os.path.join(self.root, self.split, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.annotations[idx]
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_label[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([self.image_ids[idx]])
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    def __init__(self, root, split='train2017', transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        ann_file = os.path.join(root, 'annotations', f'instances_{split}.json')
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.cat_ids = self.coco.getCatIds()
        self.cat_to_label = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, self.split, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_label[ann['category_id']])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            logging.info(f"No valid annotations for image ID {img_id}")
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)

def get_dataset(dataset_type, root, split, transforms=None):
    if dataset_type.lower() == 'voc':
        return VOCDataset(root, split, transforms)
    elif dataset_type.lower() == 'coco':
        return COCODataset(root, split, transforms)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_dataloader(dataset_type, root, split, batch_size, transforms=None):
    dataset = get_dataset(dataset_type, root, split, transforms)
    shuffle = 'train' in split.lower()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    return dataloader