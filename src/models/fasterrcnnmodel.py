import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights

class FasterRCNNModel:
    def __init__(self, num_classes, backbone="resnet50", weights=True, **kwargs):
        """
        初始化 Faster R-CNN 模型
        Args:
            num_classes (int): 類別數量（包括背景）
            backbone (str): 主幹網絡 (e.g., 'resnet50')
            weights (bool): 是否使用預訓練權重
        """
        if backbone == "resnet50":
            # 加載 ResNet50 並使用 weights 參數
            model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if weights else None)
            # 移除全連接層，僅保留卷積層
            backbone = torch.nn.Sequential(*list(model.children())[:-2])
            backbone.out_channels = 2048  # ResNet50 的輸出通道數
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 定義 Anchor 生成器
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # 定義 ROI Pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )

        # 初始化 Faster R-CNN 模型
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            **kwargs
        )

    def get_model(self):
        """
        返回 Faster R-CNN 模型
        Returns:
            torch.nn.Module: 初始化好的 Faster R-CNN 模型
        """
        return self.model