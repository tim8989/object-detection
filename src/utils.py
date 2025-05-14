# src/utils.py
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Training and Prediction")
    parser.add_argument("--dataset_type", choices=['voc', 'coco'], default='voc', help="Dataset type: 'voc' or 'coco'")
    parser.add_argument("--data_dir", default=None, help="Path to dataset (default depends on dataset_type)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (ignored for YOLOv8)")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes (default depends on dataset_type)")
    parser.add_argument("--mode", choices=['train', 'predict'], default='train', help="Mode: train or predict")
    parser.add_argument("--image_path", help="Path to single image for prediction")
    parser.add_argument("--num_random", type=int, default=0, help="Number of random images to predict (e.g., 10)")
    parser.add_argument("--resume", help="Path to checkpoint for resuming training")
    parser.add_argument("--weight_path", help="Path to model weights for prediction")

    args = parser.parse_args()

    # Set default values based on dataset_type
    if args.dataset_type == 'voc':
        args.data_dir = args.data_dir or 'data/VOCdevkit/VOC2012'
        args.num_classes = args.num_classes or 21  # 20 classes + background
    elif args.dataset_type == 'coco':
        args.data_dir = args.data_dir or 'data/coco'
        args.num_classes = args.num_classes or 81  # 80 classes + background

    # Validate arguments
    if args.mode == 'predict' and args.image_path and args.num_random > 0:
        parser.error("--image_path and --num_random cannot be used together")
    if args.resume and args.mode != 'train':
        parser.error("--resume can only be used with --mode train")
    if not os.path.exists(args.data_dir):
        parser.error(f"Dataset directory {args.data_dir} does not exist")

    return args