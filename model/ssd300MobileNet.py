from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from keras.layers import Conv2D,SeparableConv2D
from keras.layers import Flatten
from keras.layers.merge import concatenate

from ssd_layers import PriorBox

import os
import warnings

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine import get_source_inputs
from keras.engine.base_layer import InputSpec
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):

    return imagenet_utils.preprocess_input(x, mode='tf')


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    
    channel_axis = 1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = Conv2D(filters, kernel,padding='valid',use_bias=False,strides=strides,name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),padding='valid',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),padding='same',use_bias=False,strides=(1, 1),name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def _depthwise_conv_block_f(inputs, depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1
    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d'  % block_id)(inputs)
    x = DepthwiseConv2D((3, 3),padding='valid',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)


def _conv_blockSSD_f(inputs, filters, alpha, kernel, strides,block_id=11):
    channel_axis = 1
    filters = int(filters * alpha)
    Conv = Conv2D(filters, kernel,padding='valid',use_bias=False,strides=strides,name='conv__%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn' % block_id)(Conv)
    return Activation(relu6, name='conv_%d_relu' % block_id)(x),Conv

def _conv_blockSSD(inputs, filters, alpha,block_id=11):
    channel_axis = 1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d_1' % block_id)(inputs)
    x = Conv2D(filters, (1,1),padding='valid',use_bias=False,strides=(1, 1),name='conv__%d_1'%block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn_1'% block_id)(x)
    x = Activation(relu6, name='conv_%d_relu_1'% block_id)(x)
    Conv = Conv2D(filters*2, (3,3), padding='valid', use_bias=False, strides=(2, 2), name='conv__%d_2' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_%d_bn_2' % block_id)(Conv)
    x = Activation(relu6, name='conv_%d_relu_2' % block_id)(x)
    return x,Conv

def SSD(input_shape, num_classes):

    img_size=(input_shape[1],input_shape[0])
    input_shape=(input_shape[1],input_shape[0],3)
    alpha = 1.0
    depth_multiplier = 1
    input0 = Input(input_shape)
    x = _conv_block(input0, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block_f(x, depth_multiplier,strides=(1, 1), block_id=11)
    x,conv11=_conv_blockSSD_f(x,512,depth_multiplier,kernel=(1, 1), strides=(1, 1),block_id=11)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,strides=(2, 2), block_id=12)
    x = _depthwise_conv_block_f(x, depth_multiplier,strides=(1, 1), block_id=13)
    x, conv13 = _conv_blockSSD_f(x, 512, alpha, kernel=(1, 1), strides=(1, 1), block_id=13)
    x, conv14_2 = _conv_blockSSD(x, 256, alpha, block_id=14)
    x, conv15_2 = _conv_blockSSD(x, 128, alpha, block_id=15)
    x, conv16_2 = _conv_blockSSD(x, 128, alpha, block_id=16)
    x, conv17_2 = _conv_blockSSD(x, 64, alpha, block_id=17)


    #Prediction from conv11
    num_priors = 3
    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv11_mbox_loc')(conv11)
    conv11_mbox_loc = x
    flatten = Flatten(name='conv11_mbox_loc_flat')
    conv11_mbox_loc_flat = flatten(conv11_mbox_loc)
    name = 'conv11_mbox_conf'  # type: str
    conv11_mbox_conf = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(conv11)
    flatten = Flatten(name='conv11_mbox_conf_flat')
    conv11_mbox_conf_flat = flatten(conv11_mbox_conf)
    priorbox = PriorBox(img_size,60,max_size=None, aspect_ratios=[2],variances=[0.1, 0.1, 0.2, 0.2],name='conv11_mbox_priorbox')
    conv11_mbox_priorbox = priorbox(conv11)

    num_priors = 6
    x = Conv2D(num_priors * 4, (1,1),padding='same',name='conv13_mbox_loc')(conv13)
    conv13_mbox_loc = x
    flatten = Flatten(name='conv13_mbox_loc_flat')
    conv13_mbox_loc_flat = flatten(conv13_mbox_loc)
    name = 'conv13_mbox_conf'
    conv13_mbox_conf = Conv2D(num_priors * num_classes, (1,1),padding='same',name=name)(conv13)
    flatten = Flatten(name='conv13_mbox_conf_flat')
    conv13_mbox_conf_flat = flatten(conv13_mbox_conf)
    priorbox = PriorBox(img_size, 105.0, max_size=150.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv13_mbox_priorbox')
    conv13_mbox_priorbox = priorbox(conv13)
    num_priors = 6

    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv14_2_mbox_loc')(conv14_2)
    conv14_2_mbox_loc = x
    flatten = Flatten(name='conv14_2_mbox_loc_flat')
    conv14_2_mbox_loc_flat = flatten(conv14_2_mbox_loc)
    name = 'conv14_2_mbox_conf'
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(conv14_2)
    conv14_2_mbox_conf = x
    flatten = Flatten(name='conv14_2_mbox_conf_flat')
    conv14_2_mbox_conf_flat = flatten(conv14_2_mbox_conf)
    priorbox = PriorBox(img_size, 150, max_size=195.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv14_2_mbox_priorbox')
    conv14_2_mbox_priorbox = priorbox(conv14_2)
    num_priors = 6

    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv15_2_mbox_loc')(conv15_2)
    conv15_2_mbox_loc = x
    flatten = Flatten(name='conv15_2_mbox_loc_flat')
    conv15_2_mbox_loc_flat = flatten(conv15_2_mbox_loc)
    name = 'conv15_2_mbox_conf'
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(conv15_2)
    conv15_2_mbox_conf = x
    flatten = Flatten(name='conv15_2_mbox_conf_flat')
    conv15_2_mbox_conf_flat = flatten(conv15_2_mbox_conf)
    priorbox = PriorBox(img_size, 195.0, max_size=240.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv15_2_mbox_priorbox')
    conv15_2_mbox_priorbox = priorbox(conv15_2)
    num_priors = 6

    x = Conv2D(num_priors * 4, (1,1), padding='same',name='conv16_2_mbox_loc')(conv16_2)
    conv16_2_mbox_loc = x
    flatten = Flatten(name='conv16_2_mbox_loc_flat')
    conv16_2_mbox_loc_flat = flatten(conv16_2_mbox_loc)
    name = 'conv16_2_mbox_conf'
    x = Conv2D(num_priors * num_classes, (1,1), padding='same',name=name)(conv16_2)
    conv16_2_mbox_conf = x
    flatten = Flatten(name='conv16_2_mbox_conf_flat')
    conv16_2_mbox_conf_flat = flatten(conv16_2_mbox_conf)
    priorbox = PriorBox(img_size, 240.0, max_size=285.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2],name='conv16_2_mbox_priorbox')
    conv16_2_mbox_priorbox = priorbox(conv16_2)

    num_priors = 6
    x = Conv2D(num_priors * 4,(1, 1), padding='same', name='conv17_2_mbox_loc')(conv17_2)
    conv17_2_mbox_loc = x
    flatten = Flatten(name='conv17_2_mbox_loc_flat')
    conv17_2_mbox_loc_flat = flatten(conv17_2_mbox_loc)
    name = 'conv17_2_mbox_conf'
    x = Conv2D(num_priors * num_classes, (1,1), padding='same', name=name)(conv17_2)
    conv17_2_mbox_conf = x
    flatten = Flatten(name='conv17_2_mbox_conf_flat')
    conv17_2_mbox_conf_flat = flatten(conv17_2_mbox_conf)
    priorbox = PriorBox(img_size, 285.0, max_size=300.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2],name='conv17_2_mbox_priorbox')
    conv17_2_mbox_priorbox = priorbox(conv17_2)

    mbox_loc = concatenate([conv11_mbox_loc_flat,conv13_mbox_loc_flat,conv14_2_mbox_loc_flat,conv15_2_mbox_loc_flat,conv16_2_mbox_loc_flat,conv17_2_mbox_loc_flat],axis=1, name='mbox_loc')
    mbox_conf = concatenate([conv11_mbox_conf_flat,conv13_mbox_conf_flat,conv14_2_mbox_conf_flat,conv15_2_mbox_conf_flat,conv16_2_mbox_conf_flat,conv17_2_mbox_conf_flat],axis=1, name='mbox_conf')
    mbox_priorbox = concatenate([conv11_mbox_priorbox,conv13_mbox_priorbox,conv14_2_mbox_priorbox,conv15_2_mbox_priorbox,conv16_2_mbox_priorbox,conv17_2_mbox_priorbox],axis=1,name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,mbox_conf,mbox_priorbox],axis=2,name='predictions')
    model = Model(inputs=input0, outputs=predictions)
    return model
