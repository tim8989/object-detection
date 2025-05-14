import torch
from PIL import Image
import torchvision.transforms as T
import os
import random

class Predictor:
    def __init__(self, model):
        """
        初始化預測器
        Args:
            model: Faster R-CNN 模型
        """
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        # 調整預測參數
        self.model.roi_heads.score_thresh = 0.7  # 提高置信度閾值
        self.model.roi_heads.nms_thresh = 0.3    # 降低 NMS IoU 閾值

    def predict(self, image_path):
        """
        預測單張圖片
        Args:
            image_path (str): 圖片路徑
        Returns:
            dict: 預測結果（boxes, labels, scores）
        """
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([T.Resize((600, 600)), T.ToTensor()])
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image)[0]
        
        return {
            'boxes': prediction['boxes'].cpu(),
            'labels': prediction['labels'].cpu(),
            'scores': prediction['scores'].cpu(),
            'image_path': image_path
        }

    def predict_random(self, image_dir, num_images):
        """
        隨機預測多張圖片
        Args:
            image_dir (str): 圖片目錄
            num_images (int): 預測圖片數量
        Returns:
            list: 預測結果列表
        """
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        if len(image_files) < num_images:
            raise ValueError(f"Not enough images in {image_dir}, found {len(image_files)}")

        selected_images = random.sample(image_files, num_images)
        predictions = []
        for image_path in selected_images:
            pred = self.predict(image_path)
            predictions.append(pred)
        return predictions