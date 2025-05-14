import torch
from tqdm import tqdm
import logging
from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self, model, dataloader, optimizer, scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.scaler = GradScaler()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch}")):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            with autocast():
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses.item()
        
        avg_loss = total_loss / len(self.dataloader)
        logging.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, epochs, start_epoch=1):
        for epoch in range(start_epoch, epochs + 1):
            avg_loss = self.train_one_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step()