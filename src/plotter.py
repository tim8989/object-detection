import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from datetime import datetime

class Plotter:
    def __init__(self, classes, output_dir='output'):
        """
        初始化視覺化工具
        Args:
            classes (list): 類別名稱列表（包括背景）
            output_dir (str): 輸出目錄
        """
        self.classes = classes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_predictions(self, predictions, filename=None):
        """
        繪製預測結果（邊框和類別標籤）
        Args:
            predictions (dict): 預測結果，包含 'boxes', 'labels', 'scores', 'image', 可選 'image_path'
            filename (str, optional): 儲存文件名，若為 None 則自動生成
        Returns:
            str: 儲存的圖片路徑
        """
        image = predictions['image']
        boxes = predictions['boxes']
        labels = predictions['labels']
        scores = predictions['scores']

        # 設置圖表
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # 繪製邊框和標籤
        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            # 繪製邊框
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # 添加標籤和分數
            label_text = f"{self.classes[label]}: {score:.2f}"
            ax.text(
                x_min, y_min - 5, label_text,
                fontsize=10, color='white',
                bbox=dict(facecolor='red', alpha=0.5)
            )

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if 'image_path' in predictions:
                # 從 image_path 提取圖片名稱（無副檔名）
                base_name = os.path.splitext(os.path.basename(predictions['image_path']))[0]
                filename = f"prediction_{base_name}_{timestamp}.png"
            else:
                filename = f"prediction_{timestamp}.png"

        # 儲存圖片
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.show()

        return output_path