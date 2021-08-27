'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:05:59
LastEditors: Please set LastEditors
LastEditTime: 2021-08-12 07:51:26
'''
'''
Descripttion: 
version: 
Author: 
Date: 2021-06-08 07:01:17
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 02:28:47
'''
import tensorflow as tf
import numpy as np

from models.model import HTCNet
import os
from opts import parser


def generator_keras_model(model_weights_path,  keras_model_root):
    
    # ============== using modelv2 import HTCNet ===========
    # block_3_depthwise_relu, block_5_depthwise_relu（目前训练使用）, block_2_depthwise_relu
    block_target_layer = 'block_2_depthwise_relu'
    model = HTCNet(
            crop_h=224,
            crop_w=224, 
            screen_img_h=224, 
            screen_img_w=128,
            head_img_h=128,
            head_img_w=128,
            dropout=False, 
            dropout_ratdio=0.5,
            target_layer=block_target_layer)

    
    model.summary()

    model.load_weights(model_weights_path)
    print('model load weights from {} successful '.format(model_weights_path))

    keras_model_name = 'keras_watch_screen_model.h5'
    model_full_path = os.path.join(keras_model_root, keras_model_name)
    model.save(model_full_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_weights_path = '/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/data_list_20210319/train_model_2021-08-11-07-52-56/MobilenetV2Screen_025.h5'
    keras_model_root = '/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/data_list_20210319/train_model_2021-08-11-07-52-56/'
    generator_keras_model(model_weights_path, keras_model_root)
