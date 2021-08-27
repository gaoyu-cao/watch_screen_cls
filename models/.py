'''
Descripttion: 
version: 
Author: 
Date: 2021-08-19 06:03:40
LastEditors: Please set LastEditors
LastEditTime: 2021-08-19 07:56:20
'''
"""MobileNet v3 models for Keras.

Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

from re import A
from typing_extensions import Concatenate
import tensorflow
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers.core import Dropout
import os

alpha = 1
def relu6(x):
    # relu函数
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    # 利用relu函数乘上x模拟sigmoid
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def return_activation(x, nl):
    # 用于判断使用哪个激活函数
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)

    return x

def conv_block(inputs, filters, kernel, strides, nl):
    # 一个卷积单元，也就是conv2d + batchnormalization + activation
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)

    return return_activation(x, nl)

def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(int(input_channels/16))(x)
    x = Activation(relu6)(x)
    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x

def bottleneck(inputs, filters, kernel, up_dim, stride, se, nl, block_id):
    prefix = 'block_{}_'.format(block_id)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    input_shape = K.int_shape(inputs)

    tchannel = int(up_dim)
    cchannel = int(alpha * filters)

    r = stride == 1 and input_shape[3] == filters
    # 1x1卷积调整通道数，通道数上升
    x = conv_block(inputs, tchannel, (1, 1), (1, 1), nl)
    # 进行3x3深度可分离卷积
    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = return_activation(x, nl)
    # 引入注意力机制
    if se:
        x = squeeze(x)
    # 下降通道数
    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name=prefix + 'BN')(x)


    if r:
        x = Add()([x, inputs])

    return x

def MobileNetv3_small(shape = (224,224,3), include_top=None, pooling=None, n_class=1000):
    inputs = Input(shape)
    # 224,224,3 -> 112,112,16
    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

    # 112,112,16 -> 56,56,16
    x = bottleneck(x, 16, (3, 3), up_dim=16, stride=2, se=True, nl='RE', block_id=0)

    # 56,56,16 -> 28,28,24
    x = bottleneck(x, 24, (3, 3), up_dim=72, stride=2, se=False, nl='RE', block_id=1)
    x = bottleneck(x, 24, (3, 3), up_dim=88, stride=1, se=False, nl='RE', block_id=2)
    
    # 28,28,24 -> 14,14,40
    x = bottleneck(x, 40, (5, 5), up_dim=96, stride=2, se=True, nl='HS', block_id=3)
    x = bottleneck(x, 40, (5, 5), up_dim=240, stride=1, se=True, nl='HS', block_id=4)
    x = bottleneck(x, 40, (5, 5), up_dim=240, stride=1, se=True, nl='HS', block_id=5)
    # 14,14,40 -> 14,14,48
    x = bottleneck(x, 48, (5, 5), up_dim=120, stride=1, se=True, nl='HS', block_id=6)
    x = bottleneck(x, 48, (5, 5), up_dim=144, stride=1, se=True, nl='HS', block_id=7)

    # 14,14,48 -> 7,7,96
    x = bottleneck(x, 96, (5, 5), up_dim=288, stride=2, se=True, nl='HS', block_id=8)
    x = bottleneck(x, 96, (5, 5), up_dim=576, stride=1, se=True, nl='HS', block_id=9)
    x = bottleneck(x, 96, (5, 5), up_dim=576, stride=1, se=True, nl='HS', block_id=10)

    x = conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 576))(x)

    # x = Conv2D(1024, (1, 1), padding='same')(x)
    # x = return_activation(x, 'HS')

    if include_top:
        x = Conv2D(n_class, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((n_class,))(x)


    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    model = Model(inputs, x)

    return model

def HTCNetv3(
            crop_h,
            crop_w, 
            screen_img_h, 
            screen_img_w,
            head_img_h,
            head_img_w,
            dropout=False, 
            dropout_ratdio=0.5,
):  

    base_model_1 = MobileNetv3_small(shape=((crop_h, crop_w, 3)))
    base_model_2 = MobileNetv3_small(shape=(screen_img_h,screen_img_w, 3))
    base_model_3 = MobileNetv3_small(shape=(head_img_h,head_img_w,3))

    for layer in base_model_1.layers:
        layer._name = layer.name + '_1'
    for layer in base_model_2.layers:
        layer._name = layer.name + '_2'
    for layer in base_model_3.layers:
        layer._name = layer.name + '_3'
        
    # feature map of crop imgs
    x_1 = base_model_1.output
    x_1 = GlobalAveragePooling2D()(x_1)
    x_1 = Dense(1024, activation='relu', name='x_1_dense_1')(x_1)

    # feature map of screen
    x_2 = base_model_2.get_layer('block_10_BN_2').output
    x_2 = GlobalAveragePooling2D()(x_2)

    # feature map of head
    x_3 = base_model_3.get_layer('block_10_BN_3').output
    x_3 = GlobalAveragePooling2D()(x_3)

    # concate the feature map of screen and head
    x_t = Concatenate()([x_2, x_3])
    x_t = Dense(256, activation='relu', name='cat_dense_1')(x_t)

    out = Concatenate()([x_t, x_1])
    out = Dense(512, activation='relu', name='cat_dense_2')(out)
    if dropout:
        out = Dropout(dropout_ratdio)(out)

    out = Dense(256, activation='relu',name='cat_dense_3')(out)
    out = Dense(2)(out)
    out = Activation('softmax')(out)
    
    model = Model([base_model_1.input, base_model_2.input, base_model_3.input], out)

    return model


if __name__ == "__main__":
    # model = MobileNetv3_small(shape = (224, 224, 3))
    # print(model.get_layer('feature').output)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    model = HTCNetv3(
            crop_h=224,
            crop_w=224, 
            screen_img_h=224, 
            screen_img_w=128,
            head_img_h=128,
            head_img_w=128,
            dropout=True, 
            dropout_ratdio=0.5,
    )
    # print(model.output)
    model.summary()

# 3,110,055
