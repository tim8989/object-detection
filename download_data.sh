```
#!/bin/bash

# 下載 PASCAL VOC 2012
echo "Downloading PASCAL VOC 2012..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -P data/
tar -xf data/VOCtrainval_11-May-2012.tar -C data/
mv data/VOCdevkit/VOC2012 data/VOCdevkit/VOC2012
rm data/VOCtrainval_11-May-2012.tar

# 下載 MS COCO 2017 驗證集（訓練集需手動下載）
echo "Downloading MS COCO 2017 validation set..."
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
unzip data/coco/annotations_trainval2017.zip -d data/coco/
rm data/coco/val2017.zip data/coco/annotations_trainval2017.zip

# 驗證
echo "Verifying VOC..."
ls -l data/VOCdevkit/VOC2012/JPEGImages/ | wc -l
echo "Verifying COCO..."
ls -l data/coco/val2017/ | wc -l
```