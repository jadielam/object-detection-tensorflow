# Object Detection Tensorflow

This repository is mainly based on the work of [smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF) and [rbgirshick](https://github.com/rbgirshick/py-faster-rcnn). 

For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

### Acknowledgments: 

1. [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

2. [Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

3. [ROI pooling](https://github.com/zplizzi/tensorflow-fast-rcnn)

### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict` (recommend to install: [Anaconda](https://www.continuum.io/downloads))

### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  git clone https://github.com/CharlesShang/TFFRCNN.git
  ```

2. Build the Cython modules
    ```Shell
    cd TFFRCNN/lib
    make # compile cython and roi_pooling_op, you may need to modify make.sh for your platform
    ```

### Demo
*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $TFFRCNN
python ./faster_rcnn/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Download list

1. [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)

2. [VGG16 - TFFRCNN (0.689 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagX0JWRlR0cTZ5OGc/view?usp=sharing).

3. [VGG16 - TFFRCNN (0.748 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagVmt5VHlCU25vUEE/view?usp=sharing)

4. [Resnet50 trained on ImageNet](https://drive.google.com/file/d/0B_xFdh9onPagSWU1ZTAxUTZkZTQ/view?usp=sharing)

5. [Resnet50 - TFFRCNN (0.712 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagbXk1b0FIeDRJaU0/view?usp=sharing)

6. [PVANet trained on ImageNet, converted from caffemodel](https://drive.google.com/open?id=0B_xFdh9onPagQnJBdWl3VGQxam8)

### Training on Custom dataset

1. Place the data in VOC format

2. The data directory should have the following
structure
    ```
    VOC Format Data
        |-- Annotations
                |-- [000000-007480].xml
        |-- ImageSets
                |-- Main
                    |-- [train|val|trainval].txt
        |-- JPEGImages
                

    ```

3. Download pre-trained model [VGG16](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path `./data/pretrain_model/VGG_imagenet.npy`

4. TO TRAIN, DO:

    ```Shell
    cd $TFFRCNN
    python ./faster_rcnn/train_net.py --gpu 0 --weights ./data/pretrain_model/VGG_imagenet.npy --imdb hands_train --iters 70000 --cfg  ./experiments/cfgs/faster_rcnn_end2end_hands.yml --network VGGnet_train --set EXP_DIR exp_dir --restore 0
    ```

5. Run a profiling

    ```Shell
    cd $TFFRCNN
    # install a visualization tool
    sudo apt-get install graphviz  
    ./experiments/profiling/run_profiling.sh 
    # generate an image ./experiments/profiling/profile.png
    ```

6. TO DO DEMO TRAINED MODEL, DO:
    
    ```Shell
    cd $TFFRCNN
    python ./faster_rcnn/demo_hands.py --gpu 0 --net VGGnet_test --model /home/ubuntu/object-detection-tensorflow/output/faster_rcnn_voc_vgg/train/VGGnet_fast_rcnn_iter_70000.ckpt --cfg ~/object-detection-tensorflow/experiments/cfgs/faster_rcnn_end2end_hands.yml
    ```

### Training on KITTI detection dataset

1. Download the KITTI detection dataset

    ```
    http://www.cvlibs.net/datasets/kitti/eval_object.php
    ```

2. Extract all of these tar into `./TFFRCNN/data/` and the directory structure looks like this:
    
    ```
    KITTI
        |-- training
                |-- image_2
                    |-- [000000-007480].png
                |-- label_2
                    |-- [000000-007480].txt
        |-- testing
                |-- image_2
                    |-- [000000-007517].png
                |-- label_2
                    |-- [000000-007517].txt
    ```

3. Convert KITTI into Pascal VOC format
    
    ```Shell
    cd $TFFRCNN
    ./experiments/scripts/kitti2pascalvoc.py \
    --kitti $TFFRCNN/data/KITTI --out $TFFRCNN/data/KITTIVOC
    ```

4. The output directory looks like this:

    ```
    KITTIVOC
        |-- Annotations
                |-- [000000-007480].xml
        |-- ImageSets
                |-- Main
                    |-- [train|val|trainval].txt
        |-- JPEGImages
                |-- [000000-007480].jpg
    ```

5. Training on `KITTIVOC` is just like on Pascal VOC 2007

    ```Shell
    python ./faster_rcnn/train_net.py \
    --gpu 0 \
    --weights ./data/pretrain_model/VGG_imagenet.npy \
    --imdb kittivoc_train \
    --iters 160000 \
    --cfg ./experiments/cfgs/faster_rcnn_kitti.yml \
    --network VGGnet_train
    ```


