# Faster R-CNN 物件檢測專案

## 專案概述
本專案是一個基於 PyTorch 和 torchvision 的模組化、物件導向 Faster R-CNN 物件檢測框架，支援 PASCAL VOC 2012 和 MS COCO 2017 數據集。提供訓練、推理（單張或多張隨機圖片）和視覺化功能，支援檢查點繼續訓練、靈活的模型配置，以及監控訓練進度和視覺化預測結果的工具。專案位於 GitHub 倉庫：[https://github.com/tim8989/object-detection](https://github.com/tim8989/object-detection)，適合研究人員和開發者進行物件檢測任務。

## 專案結構
```
object_detection/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fasterrcnnmodel.py  # Faster R-CNN 模型定義
│   ├── dataset.py              # 數據加載與預處理（VOC 和 COCO）
│   ├── model_loader.py         # 檢查點加載與儲存
│   ├── trainer.py              # 訓練邏輯
│   ├── predictor.py            # 推理邏輯（單張或隨機多張）
│   ├── plotter.py              # 視覺化工具
│   ├── utils.py                # 通用工具函數
├── main.py                     # 專案入口
├── README.md                   # 專案說明
├── requirements.txt            # 依賴清單
├── .gitignore                  # Git 忽略文件
├── LICENSE                     # MIT 許可證
├── download_data.sh            # 數據集下載腳本
```

## 模組說明
- **src/models/fasterrcnnmodel.py**：定義 `FasterRCNNModel` 類，初始化 Faster R-CNN，採用 ResNet50 主幹網絡。
- **src/dataset.py**：實現 `VOCDataset` 和 `COCODataset`，支援 PASCAL VOC 和 MS COCO 數據集。
- **src/model_loader.py**：處理檢查點（模型權重、優化器狀態和 epoch）的儲存與加載。
- **src/trainer.py**：管理訓練流程，支援混合精度訓練和學習率調度。
- **src/predictor.py**：支援單張或隨機多張圖片的推理，可調整置信度和 NMS 閾值。
- **src/plotter.py**：視覺化預測結果，包括邊框、標籤和分數。
- **src/utils.py**：提供通用工具，如命令列參數解析。
- **main.py**：專案入口，支援訓練、繼續訓練和預測模式。

## 安裝

### 環境要求
- Python 3.8 或以上
- PyTorch 1.13.1（支援 CUDA 的 GPU 版本推薦）
- torchvision 0.14.1
- pycocotools（用於 COCO 數據集）
- matplotlib、tqdm、Pillow

### 安裝步驟
1. 克隆倉庫：
   ```bash
   git clone https://github.com/tim8989/object-detection.git
   cd object-detection
   ```
2. （可選）創建虛擬環境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

### GPU 支持
對於 GPU 加速，安裝與您的 CUDA 版本兼容的 PyTorch。例如，對於 CUDA 11.7：
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.com/whl/torch_stable.html
```
檢查 CUDA 可用性：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 注意事項
- 驗證 PyTorch 版本：
  ```bash
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```
- 安裝 pycocotools 需預裝編譯工具（Ubuntu/Debian）：
  ```bash
  sudo apt-get install python3-dev
  ```

## 數據準備
數據集（PASCAL VOC 2012 和 MS COCO 2017）需手動下載並放置在 `data/` 目錄。使用 `download_data.sh` 可自動下載部分數據集。

### 自動下載
運行腳本下載 VOC 2012 和 COCO 2017 驗證集：
```bash
bash download_data.sh
```
注意：腳本僅下載 VOC 完整數據集和 COCO 驗證集。COCO 訓練集需手動下載。

### PASCAL VOC 2012
1. 從 [官方網站](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 下載 `VOCtrainval_11-May-2012.tar`。
2. 解壓到 `data/VOCdevkit/VOC2012/`，確保結構如下：
   ```
   data/VOCdevkit/VOC2012/
   ├── Annotations/
   ├── JPEGImages/
   ├── ImageSets/
   │   ├── Main/
   │   │   ├── train.txt
   │   │   ├── val.txt
   ```
3. 驗證：
   ```bash
   ls -l data/VOCdevkit/VOC2012/JPEGImages/ | wc -l  # 約 17,125 張圖片
   cat data/VOCdevkit/VOC2012/ImageSets/Main/train.txt | wc -l  # 約 5,717 行
   ```

### MS COCO 2017
1. 從 [COCO 官網](http://cocodataset.org/#download) 下載：
   - 2017 Train images (`train2017.zip`)
   - 2017 Val images (`val2017.zip`)
   - 2017 Train/Val annotations (`annotations_trainval2017.zip`)
2. 解壓到 `data/coco/`，確保結構如下：
   ```
   data/coco/
   ├── annotations/
   │   ├── instances_train2017.json
   │   ├── instances_val2017.json
   ├── train2017/
   ├── val2017/
   ```
3. 驗證：
   ```bash
   ls -l data/coco/train2017/ | wc -l  # 約 118,287 張圖片
   ls -l data/coco/val2017/ | wc -l  # 約 5,000 張圖片
   ```

## 使用方法

### 訓練
- **VOC 數據集**：
  ```bash
  python main.py --dataset_type voc --epochs 10 --batch_size 4 --lr 0.001
  ```
  示例輸出：
  ```
  INFO:root:Loaded 5717 images for train split in data/VOCdevkit/VOC2012
  Epoch 1/10: 100%|██████████| 1429/1429 [10:00<00:00, 2.38it/s]
  ```

- **COCO 數據集**：
  ```bash
  python main.py --dataset_type coco --epochs 10 --batch_size 4 --lr 0.001
  ```
  示例輸出：
  ```
  INFO:root:Loaded 118287 images for train2017 split in data/coco
  Epoch 1/10: 100%|██████████| 29572/29572 [1:00:00<00:00, 8.21it/s]
  ```

- **繼續訓練**：
  ```bash
  python main.py --dataset_type voc --epochs 40 --batch_size 4 --lr 0.001 --resume weights/checkpoint_epoch_30.pth
  ```
  示例輸出：
  ```
  INFO:root:Resuming training from epoch 31
  INFO:root:Loaded 5717 images for train split in data/VOCdevkit/VOC2012
  ```

### 推理與視覺化
- **單張圖片（VOC）**：
  ```bash
  python main.py --dataset_type voc --mode predict --image_path data/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg --weight_path weights/checkpoint_epoch_30.pth
  ```
  示例輸出：
  ```
  INFO:root:Prediction saved to: output/prediction_2007_000027_20250514_135623.png
  ```

- **單張圖片（COCO）**：
  ```bash
  python main.py --dataset_type coco --mode predict --image_path data/coco/val2017/000000000139.jpg --weight_path weights/checkpoint_epoch_30.pth
  ```
  示例輸出：
  ```
  INFO:root:Prediction saved to: output/prediction_000000000139_20250514_135623.png
  ```

- **隨機多張圖片**：
  ```bash
  python main.py --dataset_type coco --mode predict --num_random 10 --weight_path weights/checkpoint_epoch_30.pth
  ```
  示例輸出：
  ```
  INFO:root:Selected 10 random images from data/coco/val2017
  INFO:root:Prediction saved to: output/prediction_000000000139_20250514_135623.png
  ```

### 數據集切換
使用 `--dataset_type` 切換數據集：
- `--dataset_type voc`：PASCAL VOC 2012（預設，21 類）
- `--dataset_type coco`：MS COCO 2017（81 類）

可選：指定 `--data_dir` 和 `--num_classes`：
```bash
python main.py --dataset_type coco --data_dir /path/to/coco --num_classes 81
```

## 預訓練模型
下載預訓練檢查點並放置在 `weights/` 目錄：
- VOC: [checkpoint_epoch_30.pth](https://drive.google.com/file/d/xxx/view?usp=sharing)（待更新）
- COCO: [checkpoint_epoch_30.pth](https://drive.google.com/file/d/yyy/view?usp=sharing)（待更新）

## 維護與更新
- **模組化設計**：數據集處理集中在 `dataset.py`，通過 `get_dataset` 切換數據集。
- **檢查點**：按 epoch 命名（如 `checkpoint_epoch_30.pth`）。
- **測試**：使用 `unittest` 或 `pytest` 測試 `VOCDataset` 和 `COCODataset`。
- **文檔**：詳細 docstring 支援自動生成 API 文檔。

### 未來改進
- 分離數據集模組（`voc_dataset.py`、`coco_dataset.py`）。
- 使用工廠模式優化 `get_dataset`。
- 加入 mAP 計算（VOC 使用官方腳本，COCO 使用 pycocotools）。
- 使用 `albumentations` 實現進階數據增強。
- 支援多 GPU 訓練（`torch.nn.DataParallel`）。

## 常見問題
- **錯誤：不支持的數據集類型**：
  確保 `--dataset_type` 為 `voc` 或 `coco`：
  ```bash
  python main.py --dataset_type voc
  ```
- **錯誤：數據集路徑無效**：
  驗證 `--data_dir`：
  ```bash
  ls -l data/VOCdevkit/VOC2012/  # 或 data/coco/
  ```
  如有需要，重新下載數據集。
- **檢查點不匹配**：
  VOC（21 類）和 COCO（81 類）檢查點不兼容，需針對目標數據集重新訓練。

## 聯繫方式
專案倉庫：[https://github.com/tim8989/object-detection](https://github.com/tim8989/object-detection)  
如有問題，請開啟 GitHub Issue 或聯繫：your_email@example.com