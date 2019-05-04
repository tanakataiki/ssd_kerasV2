# Keras Implemention of CustomNetwork-SSD

## This Work is going to support

### MobileNet-SSD

![MobileNet-SSD](https://github.com/tanakataiki/ssd_kerasV2/blob/master/example/MobileNet-SSD.gif)

### VGG16-SSD300

![VGG16-SSD](https://github.com/tanakataiki/ssd_kerasV2/blob/master/example/VGG16-SSD.gif)

### VGG16-SSD512

![VGG16-SSD512](https://github.com/tanakataiki/ssd_kerasV2/blob/master/example/VGG16-SSD512.gif)

### FeatureFused-SSD300

![FeatureFused-SSD300](https://github.com/tanakataiki/ssd_kerasV2/blob/master/example/FeatureFused-SSD.gif)

### Xception-SSDLite (Tanaka Original Ver)

![Xception-SSDLite](https://github.com/tanakataiki/ssd_kerasV2/blob/master/example/Xception-SSDLite.gif)

### MobileNetV2-SSDLite


Video is from free to use 
https://www.pexels.com/video/a-day-in-the-park-1466210/

I set threshold 0.9 to ignore wrong detection but usually thresh=0.6
So Please do not try this at home(this doesnt affect loss or map at all)


## Requirements
This code was tested with `Keras` v2.1.5, `Tensorflow` v1.6.0  GTX1080
Tensorflow・Keras・Numpy・Scipy・opencv-python・pillow・matplotlib・h5py


## My Weights Are Available From Here and WELCOME to upload your fine tuned weights
https://drive.google.com/drive/u/0/folders/1F8GjD3BFhf_hv9Ipez0twRptYc3P8YwP

Please write loss, acc and if possible mAp and your name if you want as your weight name
https://drive.google.com/drive/folders/1u-INV0pNjSjwNgbupXVpr1lwEsTMKW3F?usp=sharing

## Pull Request Is always welcome
As the truely perfect model doesn't exist forever there is still a way better.
(currently I don't have enought time to search very deep into details too...)



## Reference
SSD : https://github.com/rykov8/ssd_keras/blob/master/ssd.py

Caffe : https://github.com/weiliu89/caffe/tree/ssd

SSD : https://arxiv.org/abs/1512.02325

FSSD : https://arxiv.org/abs/1712.00960

FFSSD : https://arxiv.org/abs/1712.00960

DSSD : https://arxiv.org/abs/1701.06659

VGG : https://arxiv.org/abs/1409.1556

MoileNet : https://arxiv.org/abs/1704.04861

MobileNetV2 : https://arxiv.org/abs/1801.04381

Xception : https://arxiv.org/abs/1610.02357

MobileNetSSD : https://github.com/chuanqi305/MobileNet-SSD

MobileNetV2-SSDLite : https://github.com/chuanqi305/MobileNetv2-SSDLite

VGG16-SSD : https://qiita.com/tanakataiki/items/226c2460738361d2c4eb

MobileNet-SSD : https://qiita.com/tanakataiki/items/41509e1b0f4a9dcd01b1

FeatureFused-SSD : https://qiita.com/tanakataiki/items/36e71e7d2f5705bd98bb

Xception-SSDLite : https://qiita.com/tanakataiki/items/63fa46f529174d8e4c03

# Licence
The MIT License (MIT)

Copyright (c) 2018 Taiki Tanaka