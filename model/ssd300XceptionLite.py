"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Conv2D,DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers import Reshape
from keras.models import Model
from ssd_layers import PriorBox
from keras.applications import Xception


def relu6(x):
    return K.relu(x, max_value=6)

def LiteConv(x,i,filter_num):
    x = Conv2D(filter_num//2, (1, 1), padding='same', use_bias=False, name=str(i)+'_pwconv1')(x)
    x = BatchNormalization(momentum=0.99,name=str(i)+'_pwconv1_bn')(x)
    x = Activation('relu', name=str(i)+'_pwconv1_act')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=2, activation=None,use_bias=False, padding='same', name=str(i)+'_dwconv2')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_sepconv2_bn')(x)
    x = Activation('relu', name=str(i) + '_sepconv2_act')(x)
    net = Conv2D(filter_num, (1, 1), padding='same', use_bias=False, name=str(i) + '_pwconv3')(x)
    x = BatchNormalization(momentum=0.99,name=str(i) + '_pwconv3_bn')(net)
    x = Activation('relu', name=str(i) + '_pwconv3_act')(x)
    return x,net

def Conv(x,filter_num):
    net = Conv2D(filter_num,kernel_size=1,use_bias=False,name='Conv_1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(net)
    x = Activation(relu6, name='Conv_1_relu')(x)
    return x,net



def prediction(x,i,num_priors,min_s,max_s,aspect,num_classes,img_size):
    a=Conv2D(num_priors*4,(3,3),padding='same',name=str(i)+'_mbox_loc')(x)
    mbox_loc_flat=Flatten(name=str(i)+'_mbox_loc_flat')(a)
    b=Conv2D(num_priors*num_classes,(3,3),padding='same',name=str(i)+'_mbox_conf')(x)
    mbox_conf_flat=Flatten(name=str(i)+'_mbox_conf_flat')(b)
    mbox_priorbox=PriorBox(img_size,min_size=min_s,max_size=max_s,aspect_ratios=aspect,variances=[0.1,0.1,0.2,0.2],name=str(i)+'_mbox_priorbox')(x)
    return mbox_loc_flat,mbox_conf_flat,mbox_priorbox


def SSD(input_shape, num_classes):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    img_size=(input_shape[1],input_shape[0])
    input_shape=(input_shape[1],input_shape[0],3)
    xception_input_shape = (299,299, 3)

    Input0 = Input(input_shape)
    xception=Xception(input_shape=xception_input_shape,include_top=False,weights='imagenet')
    FeatureExtractor=Model(inputs=xception.input, outputs=xception.get_layer('add_11').output)

    x= FeatureExtractor(Input0)
    x, pwconv3 = Conv(x, 1024)
    x, pwconv4 = LiteConv(x, 4, 1024)
    x, pwconv5 = LiteConv(x, 5, 512)
    x, pwconv6 = LiteConv(x, 6, 512)
    x, pwconv7 = LiteConv(x, 7, 256)
    x, pwconv8 = LiteConv(x, 8, 256)

    pwconv3_mbox_loc_flat, pwconv3_mbox_conf_flat, pwconv3_mbox_priorbox = prediction(pwconv3, 3, 3,60.0 ,None ,[2]   ,num_classes, img_size)
    pwconv4_mbox_loc_flat, pwconv4_mbox_conf_flat, pwconv4_mbox_priorbox = prediction(pwconv4, 4, 6,105.0,150.0,[2, 3],num_classes, img_size)
    pwconv5_mbox_loc_flat, pwconv5_mbox_conf_flat, pwconv5_mbox_priorbox = prediction(pwconv5, 5, 6,150.0,195.0,[2, 3],num_classes, img_size)
    pwconv6_mbox_loc_flat, pwconv6_mbox_conf_flat, pwconv6_mbox_priorbox = prediction(pwconv6, 6, 6,195.0,240.0,[2, 3],num_classes, img_size)
    pwconv7_mbox_loc_flat, pwconv7_mbox_conf_flat, pwconv7_mbox_priorbox = prediction(pwconv7, 7, 6,240.0,285.0,[2, 3],num_classes, img_size)
    pwconv8_mbox_loc_flat, pwconv8_mbox_conf_flat, pwconv8_mbox_priorbox = prediction(pwconv8, 8, 6,285.0,300.0,[2, 3],num_classes, img_size)


    # Gather all predictions
    mbox_loc = concatenate(
        [pwconv3_mbox_loc_flat, pwconv4_mbox_loc_flat, pwconv5_mbox_loc_flat, pwconv6_mbox_loc_flat,
         pwconv7_mbox_loc_flat, pwconv8_mbox_loc_flat], axis=1, name='mbox_loc')
    mbox_conf = concatenate(
        [pwconv3_mbox_conf_flat, pwconv4_mbox_conf_flat, pwconv5_mbox_conf_flat, pwconv6_mbox_conf_flat,
         pwconv7_mbox_conf_flat, pwconv8_mbox_conf_flat], axis=1, name='mbox_conf')
    mbox_priorbox = concatenate(
        [pwconv3_mbox_priorbox, pwconv4_mbox_priorbox, pwconv5_mbox_priorbox, pwconv6_mbox_priorbox,
         pwconv7_mbox_priorbox, pwconv8_mbox_priorbox], axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc,mbox_conf,mbox_priorbox],axis=2,name='predictions')
    model = Model(inputs=Input0,  outputs=predictions)
    return model