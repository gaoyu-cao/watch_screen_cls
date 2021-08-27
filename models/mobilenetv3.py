'''
Descripttion: 
version: 
Author: 
Date: 2021-08-20 01:44:03
LastEditors: Please set LastEditors
LastEditTime: 2021-08-23 06:07:34
'''

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D,Input
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

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

    x = Conv2D(filters, kernel, padding='same', strides=strides, kernel_initializer='he_normal')(inputs)
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
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    prefix = 'block_{}_'.format(block_id)
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
    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', name=prefix+'conv')(x)
    x = BatchNormalization(axis=channel_axis, name=prefix+'conv_BN')(x)

    if r:
        x = Add()([x, inputs])

    return x

def MobileNetv3_small(input_shape=(224,224,3), n_class = 1000, include_top=None):
    inputs = Input(input_shape)
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
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1024, (1, 1), padding='same')(x)
        x = return_activation(x, 'HS')

        x = Conv2D(n_class, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((n_class,))(x)

    model = Model(inputs, x)

    return model

def MobileNetv3_large(input_shape=(224,224,3), n_class = 1000, include_top=None):
    inputs = Input(input_shape)
    # 224,224,3 -> 112,112,16
    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

    # 112,112,16 -> 56,56,16
    x = bottleneck(x, 16, (3, 3), up_dim=16, stride=1, se=False, nl='RE', block_id=0)
    x = bottleneck(x, 24, (3, 3), up_dim=64, stride=2, se=False, nl='RE', block_id=1)
    x = bottleneck(x, 24, (3, 3), up_dim=72, stride=1, se=False, nl='RE', block_id=2)
    x = bottleneck(x, 40, (5, 5), up_dim=72, stride=2, se=False, nl='RE', block_id=3)
    x = bottleneck(x, 40, (5, 5), up_dim=120, stride=1, se=False, nl='RE', block_id=4)
    x = bottleneck(x, 40, (5, 5), up_dim=120, stride=1, se=False, nl='RE', block_id=5)
    x = bottleneck(x, 80, (3, 3), up_dim=240, stride=2, se=False, nl='HS', block_id=6)
    x = bottleneck(x, 80, (3, 3), up_dim=200, stride=1, se=False, nl='HS', block_id=7)
    x = bottleneck(x, 80, (3, 3), up_dim=184, stride=1, se=False, nl='HS', block_id=8)
    x = bottleneck(x, 80, (3, 3), up_dim=184, stride=1, se=False, nl='HS', block_id=9)
    x = bottleneck(x, 112, (3, 3), up_dim=480, stride=1, se=True, nl='HS', block_id=10)
    x = bottleneck(x, 112, (3, 3), up_dim=672, stride=1, se=True, nl='HS', block_id=11)
    x = bottleneck(x, 160, (5, 5), up_dim=672, stride=2, se=True, nl='HS', block_id=12)
    x = bottleneck(x, 160, (5, 5), up_dim=960, stride=1, se=True, nl='HS', block_id=13)
    x = bottleneck(x, 160, (5, 5), up_dim=960, stride=1, se=True, nl='HS', block_id=14)

    x = conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = return_activation(x, 'HS')

        x = Conv2D(n_class, (1, 1), padding='same', activation='softmax')(x)
        x = Reshape((n_class,))(x)

    model = Model(inputs, x)
    return model

def HTCMobileNetv3_small(
            crop_h,
            crop_w, 
            screen_img_h, 
            screen_img_w,
            head_img_h,
            head_img_w,
            dropout=False, 
            dropout_ratdio=0.5):
    base_model_1 = MobileNetv3_small(input_shape=(crop_h, crop_w, 3), include_top=False)
    base_model_2 = MobileNetv3_small(input_shape=(screen_img_h, screen_img_w, 3), include_top=False)
    base_model_3 = MobileNetv3_small(input_shape=(head_img_h, head_img_w, 3), include_top=False)
    
    # crop_block_target_layer=target_layer + '_2'
    # screen_block_target_layer=target_layer + '_2'
    # head_block_target_layer=target_layer + '_3'

    # model_1  process(crop img) 
    for layer in base_model_1.layers:
        layer._name = layer.name + '_1'

    # model_2 process(screen pics)
    for layer in base_model_2.layers:
        layer._name = layer.name + '_2'
    
    # model_2 process(screen pics)
    for layer in base_model_3.layers:
        layer._name = layer.name + '_3'

    ## the feature map of crop imgs
    x_1 = base_model_1.output
    x_1 = tf.keras.layers.GlobalAveragePooling2D()(x_1) 
    x_1 = tf.keras.layers.Dense(1024, activation='relu',name='x_1_dense_1')(x_1)

    x_2 = base_model_2.output
    x_2 = tf.keras.layers.GlobalAveragePooling2D()(x_2) 

    x_3 = base_model_3.output
    x_3 = tf.keras.layers.GlobalAveragePooling2D()(x_3) 

    ## concate the feature map of screen and head
    x_t = tf.keras.layers.Concatenate()([x_2, x_3])
    x_t = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_1')(x_t)

    out = tf.keras.layers.Concatenate()([x_t, x_1])
    out = tf.keras.layers.Dense(512, activation='relu',name='cat_dense_2')(out)
    if dropout:
        out = tf.keras.layers.Dropout(dropout_ratdio)(out)
    
    out = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_3')(out)
    out = tf.keras.layers.Dense(2)(out)
    out = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model([base_model_1.input, base_model_2.input, base_model_3.input], out)
 
    return model

def HTCMobileNetv3_large(
            crop_h,
            crop_w, 
            screen_img_h, 
            screen_img_w,
            head_img_h,
            head_img_w,
            dropout=False, 
            dropout_ratdio=0.5):
    base_model_1 = MobileNetv3_large(input_shape=(crop_h, crop_w, 3), include_top=False)
    base_model_2 = MobileNetv3_large(input_shape=(screen_img_h,screen_img_w, 3), include_top=False)
    base_model_3 = MobileNetv3_large(input_shape=(head_img_h,head_img_w,3), include_top=False)
    
    # model_1  process(crop img) 
    for layer in base_model_1.layers:
        layer._name = layer.name + '_1'

    # model_2 process(screen pics)
    for layer in base_model_2.layers:
        layer._name = layer.name + '_2'
    
    # model_2 process(screen pics)
    for layer in base_model_3.layers:
        layer._name = layer.name + '_3'


    ## the feature map of crop imgs
    x_1 = base_model_1.output
    x_1 = tf.keras.layers.GlobalAveragePooling2D()(x_1) 
    x_1 = tf.keras.layers.Dense(1024, activation='relu',name='x_1_dense_1')(x_1)

    x_2 = base_model_2.output
    x_2 = tf.keras.layers.GlobalAveragePooling2D()(x_2) 

    x_3 = base_model_3.output
    x_3 = tf.keras.layers.GlobalAveragePooling2D()(x_3) 

    ## concate the feature map of screen and head
    x_t = tf.keras.layers.Concatenate()([x_2, x_3])
    x_t = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_1')(x_t)

    out = tf.keras.layers.Concatenate()([x_t, x_1])
    out = tf.keras.layers.Dense(512, activation='relu',name='cat_dense_2')(out)
    if dropout:
        out = tf.keras.layers.Dropout(dropout_ratdio)(out)
    
    out = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_3')(out)
    out = tf.keras.layers.Dense(2)(out)
    out = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model([base_model_1.input, base_model_2.input, base_model_3.input], out)
 
    return model


#-------------------------------------------------------------#
#   MobileNet的网络部分
#-------------------------------------------------------------#
import warnings
import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K


def MobileNet(input_shape=[224,224,3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):


    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x)
    # model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    # model_name = 'mobilenet_1_0_224_tf.h5'
    # model.load_weights(model_name)

    return model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)


if __name__ == "__main__":
    model = HTCMobileNetv3_small(
            crop_h=224,
            crop_w=224, 
            screen_img_h=224, 
            screen_img_w=128,
            head_img_h=128,
            head_img_w=128,
            dropout=True, 
            dropout_ratdio=0.5,
    )
    # model = MobileNet()
    model.summary()
