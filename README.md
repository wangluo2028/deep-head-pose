# HopelessNet: Improving Hopenet #

<div align="center">
  <img src="https://i.imgur.com/K7jhHOg.png" width="380"><br><br>
</div>

**Hopenet** is an accurate and easy to use head pose estimation network. It uses the 300W-LP dataset for training the models and have been tested on real data with good qualitative performance.

The original repository of Hopenet is [GitHub](https://github.com/natanielruiz/deep-head-pose)
For details about the method and quantitative results please check their CVPR Workshop [paper](https://arxiv.org/abs/1710.00925).

Here I am trying to revisit this method and improve its performance, specifically for testing on AFLW2000 dataset.  
I applied minor changes to the code so that I can work with in using **PyTorch version 2.6** and **Python3**.

<div align="center">
<img src="output-amir.gif"/><br><br>
</div>

## Better Training for Hopenet

The best reported results for AFLW2000 dataset, provided in the CVPRW paper (Table 1), are:  
Yaw: 6.470, Pitch: 6.559, Roll: 5.436, and **MAE: 6.155**

As reported in the paper, to achieve this result, they used below settings:
* Training Dataset: 300W-LP
* Alpha: 2
* Batch Size: 128
* Learning Rate: 1e-5

Using the provided code, I tried similar settings.  
Except for **batch size** for which I had to reduce to **64** due to the memory limitation of my GPU.  
What I found was after few epochs, the test error starts raising.  
To achieve a smoother error curve, I reduced the **learning rate** to **1e-6** and tried the training with different alpha values.

The best model I got so far was from **alpha = 1** which performs as below on AFLW2000:  
Yaw: 5.4517, Pitch: 6.3541, Roll: 5.3127, **MAE: 5.7062**  
The snapshot of this model can be downloaded from [models/hopenet_snapshot_a1.pkl](https://github.com/shahroudy/deep-head-pose/raw/master/models/hopenet_snapshot_a1.pkl).

## Improve the Efficiency of the Model with HopeLessNet :D

The original Hopenet method uses a ResNet50 convnet which is considered to be a heavy weight and inefficient model, specifically to be used on embedded or mobile platform.  
To mitigate this issue, we can think of replacing this module with a lighter network e.g. ResNet18, Squeezenet, or MobileNet.  
An argument is now added to the train_hopenet.py and test_hopenet.py modules called "arch" which can change the base network's architecture to:
* ResNet18
* ResNet34
* ResNet50
* ResNet101
* ResNet152
* Squeezenet_1_0
* Squeezenet_1_1
* MobileNetV2

The best performing model with **ResNet18** architecture [(snapshot)](https://github.com/shahroudy/deep-head-pose/raw/master/models/hopenet_resnet18.pkl) achieves:  
Yaw: 6.0897, Pitch: 6.9588, Roll: 6.0907, **MAE: 6.3797**

With **MobileNetV2** architechture [(snapshot)](https://github.com/shahroudy/deep-head-pose/raw/master/models/mobilenetv2.pkl) I could reach to:  
Yaw: 7.3247, Pitch: 6.9425, Roll: 6.2106, **MAE: 6.8259**

And with **Squeezenet_1_0** architechture [(snapshot)](https://github.com/shahroudy/deep-head-pose/raw/master/models/squeezenet_1_0.pkl) we can get:  
Yaw: 7.2015, Pitch: 7.9230, Roll: 6.8532, **MAE: 7.3259**

Lastly, **Squeezenet_1_1** architechture [(snapshot)](https://github.com/shahroudy/deep-head-pose/raw/master/models/squeezenet_1_1.pkl) could perform:  
Yaw: 8.8815, Pitch: 7.4020, Roll: 7.1891, **MAE: 7.8242**

It is good to mention about [HopeNet-Lite](https://github.com/OverEuro/deep-head-pose-lite), which also adopted a MobileNet like architecture for HopeNet.
**new** [GoT trailer example video](https://youtu.be/OZdOrSLBQmI)

**new** [Conan-Cruise-Car example video](https://youtu.be/Bz6eF4Nl1O8)


To use please install [PyTorch](http://pytorch.org/) and [OpenCV](https://opencv.org/) (for video) - I believe that's all you need apart from usual libraries such as numpy. You need a GPU to run Hopenet (for now).

To test on a video using dlib face detections (center of head will be jumpy):
```bash
python code/test_on_video_dlib.py --snapshot PATH_OF_SNAPSHOT --face_model PATH_OF_DLIB_MODEL --video PATH_OF_VIDEO --output_string STRING_TO_APPEND_TO_OUTPUT --n_frames N_OF_FRAMES_TO_PROCESS --fps FPS_OF_SOURCE_VIDEO


python code/test_on_video_dlib.py --snapshot checkpoints/hopenet_robust_alpha1.pkl --face_model models/mmod_human_face_detector.dat --video /path/to/video/file --output_string dhp --n_frames 400 --fps 30


```
To test on a video using your own face detections (we recommend using [dockerface](https://github.com/natanielruiz/dockerface), center of head will be smoother):
```bash
python code/test_on_video_dockerface.py --snapshot PATH_OF_SNAPSHOT --video PATH_OF_VIDEO --bboxes FACE_BOUNDING_BOX_ANNOTATIONS --output_string STRING_TO_APPEND_TO_OUTPUT --n_frames N_OF_FRAMES_TO_PROCESS --fps FPS_OF_SOURCE_VIDEO
```
Face bounding box annotations should be in Dockerface format (n_frame x_min y_min x_max y_max confidence).

Pre-trained models:

[300W-LP, alpha 1](https://drive.google.com/open?id=1EJPu2sOAwrfuamTitTkw2xJ2ipmMsmD3)

[300W-LP, alpha 2](https://drive.google.com/open?id=16OZdRULgUpceMKZV6U9PNFiigfjezsCY)

[300W-LP, alpha 1, robust to image quality](https://drive.google.com/open?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR)

For more information on what alpha stands for please read the paper. First two models are for validating paper results, if used on real data we suggest using the last model as it is more robust to image quality and blur and gives good results on video.

Please open an issue if you have an problem.

Some very cool implementations of this work on other platforms by some cool people:

[Gluon](https://github.com/Cjiangbpcs/gazenet_mxJiang)

[MXNet](https://github.com/haofanwang/mxnet-Head-Pose)

[TensorFlow with Keras](https://github.com/Oreobird/tf-keras-deep-head-pose)

A really cool lightweight version of HopeNet:

[Deep Head Pose Light](https://github.com/OverEuro/deep-head-pose-lite)


If you find Hopenet useful in your research please cite:

```
@InProceedings{Ruiz_2018_CVPR_Workshops,
author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
title = {Fine-Grained Head Pose Estimation Without Keypoints},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}
```

*Nataniel Ruiz*, *Eunji Chong*, *James M. Rehg*

Georgia Institute of Technology
