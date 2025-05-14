from ultralytics import YOLO
import torch

class YOLOv8Model:
    def __init__(self, model_type='yolov8n', pretrained=True):
        """初始化 YOLOv8 模型。
        
        Args:
            model_type (str): YOLOv8 變體（例如 'yolov8n', 'yolov8s'）。
            pretrained (bool): 若為 True，加載預訓練權重。
        """
        self.model = YOLO(f"{model_type}.pt" if pretrained else f"{model_type}.yaml")
    
    def to(self, device):
        """將模型移動到指定設備。"""
        self.model.model.to(device)
        return self
    
    def __call__(self, imgs, targets=None):
        """前向傳播，支援訓練和推理。
        
        Args:
            imgs (torch.Tensor): 輸入圖像張量。
            targets (dict): 標註（包含 'boxes' 和 'labels'），訓練時使用。
        
        Returns:
            dict or list: 訓練時返回損失字典，推理時返回預測結果。
        """
        if self.model.model.training and targets is not None:
            # 訓練模式：返回損失
            outputs = self.model.model(imgs)  # YOLOv8 前向傳播
            # 簡化版損失計算（需根據 YOLOv8 實際輸出調整）
            loss_dict = {
                'box_loss': torch.tensor(0.0, device=imgs.device),  # 假設損失（需實現）
                'cls_loss': torch.tensor(0.0, device=imgs.device),
                'dfl_loss': torch.tensor(0.0, device=imgs.device)
            }
            return loss_dict
        else:
            # 推理模式：返回預測
            return self.model.predict(source=imgs, device=self.model.model.device)
    
    def train(self):
        """設置模型為訓練模式。"""
        self.model.model.train()
    
    def eval(self):
        """設置模型為評估模式。"""
        self.model.model.eval()
    
    def save(self, path):
        """保存模型檢查點。"""
        torch.save(self.model.model.state_dict(), path)