import torch
import os
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)

class ModelLoader:
    def __init__(self, model, optimizer=None):
        """
        初始化 ModelLoader
        Args:
            model: Faster R-CNN 模型
            optimizer: 優化器（可選，用於繼續訓練）
        """
        self.model = model
        self.optimizer = optimizer

    def save_checkpoint(self, epoch, path='weights'):
        """
        儲存檢查點（模型權重、優化器狀態和 epoch）
        Args:
            epoch (int): 當前 epoch
            path (str): 儲存目錄
        """
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))
        logging.info(f"Saved checkpoint to {path}/checkpoint_epoch_{epoch}.pth")

    def load_checkpoint(self, checkpoint_path, strict=False):
        """
        加載檢查點（支援新舊格式，允許部分加載）
        Args:
            checkpoint_path (str): 檢查點文件路徑
            strict (bool): 是否嚴格匹配 state_dict（False 允許部分加載）
        Returns:
            int: 起始 epoch（加載的 epoch + 1）
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 新格式檢查點
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                if self.optimizer and checkpoint['optimizer_state_dict']:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
                return checkpoint['epoch'] + 1
            except RuntimeError as e:
                logging.warning(f"Partial loading due to mismatch: {e}")
                # 部分加載，忽略不匹配的參數（例如 roi_heads.box_predictor）
                state_dict = checkpoint['model_state_dict']
                model_dict = self.model.state_dict()
                # 僅加載形狀匹配的參數
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(state_dict)
                self.model.load_state_dict(model_dict, strict=False)
                logging.info(f"Partially loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
                return checkpoint['epoch'] + 1
        else:
            # 舊格式權重（僅 model.state_dict）
            try:
                self.model.load_state_dict(checkpoint, strict=strict)
                logging.info(f"Loaded legacy checkpoint from {checkpoint_path} (assuming epoch 30)")
                return 31  # 手動指定 epoch_30.pth 的起始 epoch
            except RuntimeError as e:
                logging.warning(f"Partial loading due to mismatch: {e}")
                model_dict = self.model.state_dict()
                state_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(state_dict)
                self.model.load_state_dict(model_dict, strict=False)
                logging.info(f"Partially loaded legacy checkpoint from {checkpoint_path} (assuming epoch 30)")
                return 31