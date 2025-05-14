import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNModel(nn.Module):
    def __init__(self, num_classes=21):  # 20 類 + 背景
        super(FasterRCNNModel, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)