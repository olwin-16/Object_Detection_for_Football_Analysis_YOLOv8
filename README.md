# Objectify: Real-Time Object Detection for Football Analysis

## Project Overview

The Objectify project is designed to develop an advanced deep learning model for analyzing football video clips. The goal is to identify and track players, the ball, and referees in real-time. Using state-of-the-art deep learning frameworks such as Faster R-CNN and YOLOv8, Objectify aims to efficiently process video frames to deliver accurate object detection, complete with bounding boxes and class labels. The project employs Python-based tools like PyTorch and Roboflow to facilitate model training and testing on custom-annotated datasets within Google Colab.

<img width="473" alt="FasterRCNN_Public_Results" src="https://github.com/user-attachments/assets/11f7e837-07d1-4a4f-badc-216c547cc612">

## Project Structure

The repository is organized into the following main directories:

### 1. Detectron2 (Aeroplane)

Description: Contains code and data for training object detection models using Detectron2 on the Aeroplane dataset.

Contents:
Dataset images and annotations
Configuration files
Python scripts for model training and evaluation

### 2. Balloon Dataset

Description: Contains code and data for training models on the Balloon dataset.

Contents:
Dataset images and annotations
Python scripts for training and evaluation
Configuration files

### 3. Football (YOLOv8)

Description: Contains code and data for training object detection models using YOLOv8 on the Football dataset.

Contents:
Dataset images and annotations
Python scripts for training and evaluation
Configuration files

### 4. Visuals, Architecture, Graphs

Description: Includes visualizations, architecture diagrams, and training graphs for better understanding and analysis.

Contents:
Architecture diagrams for model training
Training graphs and loss curves
Visual examples of object detection results

## Datasets Used

### COCO 2017 Dataset Class Names

Below is the list of class names:

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]


The `coco_classes.txt` file contains a list of 80 class names used for object detection in the COCO 2017 dataset. 

This file can be referenced when training or evaluating models in the `Detectron2` and `Balloon` modules.

To use these class names in your project, you can import the file into your code or copy and use the list directly.

### YAML Dataset

train: /content/drive/MyDrive/Colab Notebooks/ObjectDetection/Dataset/train/images
val: /content/drive/MyDrive/Colab Notebooks/ObjectDetection/Dataset/valid/images
#test: /content/drive/MyDrive/Colab Notebooks/ObjectDetection/Dataset/test/images  #optional

nc: 3
names: ['Ball', 'Player', 'Referee']

#roboflow:
 #url: https://universe.roboflow.com/nikhil-chapre-xgndf/detect-players-dgxz0/dataset/7

## Installation & Setup

Clone the repository:
`git clone https://github.com/olwin-16/Objectify.git`

Navigate into the directory:
`cd Objectify`

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments

Special thanks to the developers and contributors of Detectron2 and YOLOv8 for their innovative open-source frameworks.

The COCO dataset team for providing a benchmark dataset for object detection tasks.

## Contact

For any questions or contributions, please contact Olwin Christian at olwinchristian1626@gmail.com

