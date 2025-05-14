import torch
from PIL import Image
import torchvision.transforms as T
import os
import random
import torchvision.ops as ops

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

    def post_process(self, prediction, conf_threshold=0.7, nms_threshold=0.3):
        """
        後處理預測結果，應用置信度篩選和 NMS
        Args:
            prediction (dict): 模型原始預測（boxes, labels, scores）
            conf_threshold (float): 置信度閾值
            nms_threshold (float): NMS IoU 閾值
        Returns:
            dict: 過濾後的預測
        """
        # 篩選高置信度預測
        keep = prediction['scores'] > conf_threshold
        boxes = prediction['boxes'][keep]
        labels = prediction['labels'][keep]
        scores = prediction['scores'][keep]

        # 應用 NMS
        if boxes.numel() > 0:
            keep = ops.nms(boxes, scores, nms_threshold)
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }

    def predict(self, image_path):
        self.model.eval()
        # 載入原始圖片
        image_pil = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image_pil.size  # 原始尺寸
        # 轉換為 tensor 用於模型
        transform = T.Compose([T.Resize((600, 600)), T.ToTensor()])
        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)

        # 運行推理
        with torch.no_grad():
            prediction = self.model(image_tensor)[0]

        # 後處理預測
        prediction = self.post_process(prediction, conf_threshold=0.7, nms_threshold=0.3)

        # 縮放邊界框到原始圖片尺寸
        boxes = prediction['boxes']
        scale_x = orig_width / 600
        scale_y = orig_height / 600
        boxes[:, [0, 2]] *= scale_x  # 縮放 x 座標 (x_min, x_max)
        boxes[:, [1, 3]] *= scale_y  # 縮放 y 座標 (y_min, y_max)

        return {
            'boxes': boxes.cpu(),
            'labels': prediction['labels'].cpu(),
            'scores': prediction['scores'].cpu(),
            'image': image_pil,
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