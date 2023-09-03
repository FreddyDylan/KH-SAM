# KH-SAM

This project focuses on the problem of track defect detection. Based on SAM, we added knowledge heuristics to improve the performance and prediction speed of track defect detection. We tested our model on the datasets Type-I_RSDDs_dataset, Type-II_RSDDs_dataset, rsdds-dataset_link, rail400_2048x2000_nc3, and rail400_2048x2000_nc8, and achieved significant performance improvements.

# Contents

- Overview
- Dataset
- Dependencies
- Training
- Inference
- pre-trained models

# Overview

Track defect detection is a critical task for ensuring the safe operation and maintenance of rail transportation. Existing machine vision-based detection methods primarily focus on segmenting track images, which suffer from high model time complexity, serious background noise interference, and unsatisfactory segmentation results. To address these issues, we propose the Knowledge Heuristic SAM (KH-SAM) model, which is an image semantic segmentation model based on the improved Segment Anything Model (SAM) by introducing heuristic knowledge. This effectively enhances the efficiency and accuracy of defect segmentation.

# Dataset

This project utilized defect datasets including Type-I_RSDDs_dataset, Type-II_RSDDs_dataset, rsdds-dataset_link, rail400_2048x2000_nc3, and rail400_2048x2000_nc8, with image sizes of (160, 1000), (55, 1250), (535, 252), (2048, 2000), and (2048, 2000) respectively.

Dataset download link: https://github.com/FreddyDylan/Dataset

# Dependencies

```python
pip install -r requirements.txt
```

# Training

```python
python /KH-SAM/ultralytics/yolo/v8/detect/train.py
```

# Inference

```python
python /KH-SAM/ultralytics/yolo/v8/detect/predict.py
python /KH-SAM/segment-anythin-main/scripts/Inference.py
```

# pre-trained models

The pre-trained models on the five datasets are saved in the "KH-SAM/pre-trained models" folder.