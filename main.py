import torch
from src.models.fasterrcnnmodel import FasterRCNNModel
from src.dataset import get_dataloader
from src.model_loader import ModelLoader
from src.trainer import Trainer
from src.predictor import Predictor
from src.plotter import Plotter
from src.utils import parse_args
import torchvision.transforms as T
import os
from pycocotools.coco import COCO
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)

def main():
    args = parse_args()

    # 動態設置類別數
    num_classes = 21 if args.dataset_type == 'voc' else 81  # VOC: 21 類, COCO: 81 類
    model = FasterRCNNModel(num_classes=num_classes).get_model()

    # 初始化優化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # 初始化 ModelLoader
    loader = ModelLoader(model, optimizer)

    # 如果指定 resume，則加載檢查點（訓練模式）
    start_epoch = 1
    if args.resume:
        start_epoch = loader.load_checkpoint(args.resume, strict=True)
        logging.info(f"Resuming training from epoch {start_epoch}")

    # 預測模式需要加載權重
    if args.mode == 'predict':
        if not args.weight_path:
            raise ValueError("Must specify --weight_path for predict mode")
        loader.load_checkpoint(args.weight_path, strict=False)
        logging.info(f"Loaded weights from {args.weight_path}")

    if args.mode == 'train':
        # 數據加載（添加縮放轉換）
        transforms = T.Compose([T.Resize((600, 600)), T.ToTensor()])
        split = 'train' if args.dataset_type == 'voc' else 'train2017'
        dataloader = get_dataloader(args.dataset_type, args.data_dir, split, args.batch_size, transforms)

        # 訓練（添加學習率調度）
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        trainer = Trainer(model, dataloader, optimizer, scheduler)
        trainer.train(args.epochs, start_epoch=start_epoch)

        # 儲存檢查點
        loader.save_checkpoint(args.epochs)

    elif args.mode == 'predict':
        # 初始化預測器和視覺化工具
        predictor = Predictor(model)

        # 動態加載類別名稱
        if args.dataset_type == 'voc':
            classes = [
                '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
        else:  # coco
            coco = COCO(os.path.join(args.data_dir, 'annotations', 'instances_val2017.json'))
            classes = ['__background__'] + [cat['name'] for cat in coco.loadCats(coco.getCatIds())]

        plotter = Plotter(classes)

        if args.num_random > 0:
            image_dir = os.path.join(args.data_dir, 'val' if args.dataset_type == 'voc' else 'val2017')
            predictions_list = predictor.predict_random(image_dir, args.num_random)
            for predictions in predictions_list:
                output_path = plotter.plot_predictions(predictions)
                logging.info(f"Prediction saved to: {output_path}")
        elif args.image_path:
            predictions = predictor.predict(args.image_path)
            output_path = plotter.plot_predictions(predictions)
            logging.info(f"Prediction saved to: {output_path}")
        else:
            raise ValueError("Must specify --image_path or --num_random for predict mode")

if __name__ == "__main__":
    main()