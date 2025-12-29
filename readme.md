# Paediatric Caries Detection
Published the [Caries Research Journal 2025](https://doi.org/10.1159/000550079).

## Setup Environment
Disclaimer: The codes have only been tested on Ubuntu 18.04, Python 3.8.18 and Pytorch 1.7.1 CUDA 11.0, but they should work on environments with similar major versions.
1. Install required libraries by running 
    ```bash
    conda create -n bdc python=3.8
    conda activate bdc
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```
2. Download pretrained model weights (optional)
    ```bash
    mkdir pretrained_weights
    cd pretrained_weights
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    ```
   
### Bring your own datasets
1. Convert your annotations to YOLO labelling format (one txt file per image).
   - Train annotations follows format: x1,y1,x2,y2,class_id
   - Test annotations (if available, else create an empty text file for each test image) follows format: x1,y1,x2,y2,class_id
2. Create a new yaml file and modify the settings in it following the example in `data` folder.

## How to run
- To train YOLOv7 or Faster-RCNN models:
    ```bash
    python train_yolov7.py --project <project name> --plots --weights ./pretrained_weights/yolov7_training.pt 
    python train_faster_rcnn.py --project <project name> --plots 
    ```

- To test YOLOv7 or Faster-RCNN models:
    ```bash
    python test_yolov7.py --data ./data/usp.yaml --coco-eval --no-trace --exist-ok --project <project name> --name <exp name> --weights <checkpoint path>
    python test_faster_rcnn.py --data ./data/usp.yaml --coco-eval --exist-ok --project <project name> --name <exp name> --weights <checkpoint path>
    ```

Outputs are stored in `<project>/<name>`.
All experiments are logged to Tensorboard and WandB automatically. They can be disabled in the training codes.

Please refer to [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (YOLOX-L), [CenterNet2](https://github.com/xingyizhou/CenterNet2) (CenterNet2_DLA-BiFPN-P3) and [DINO](https://github.com/IDEA-Research/DINO) (DINO-4scale) for other object detector models.
