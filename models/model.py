'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:10:56
LastEditors: Please set LastEditors
LastEditTime: 2021-08-19 07:56:39
'''

import tensorflow as tf
import os

# block_3_depthwise_relu
def HTCNet(
            crop_h,
            crop_w, 
            screen_img_h, 
            screen_img_w,
            head_img_h,
            head_img_w,
            dropout=False, 
            dropout_ratdio=0.5,
            target_layer='block_3_depthwise_relu'):

    base_model_1 = tf.keras.applications.MobileNetV2(input_shape=(crop_h, crop_w, 3),weights='imagenet', include_top=False)  
    base_model_2 = tf.keras.applications.MobileNetV2(input_shape=(screen_img_h,screen_img_w, 3), weights='imagenet', include_top=False)
    base_model_3 = tf.keras.applications.MobileNetV2(input_shape=(head_img_h,head_img_w,3), weights='imagenet', include_top=False)
    # block_5_depthwise_relu_2, block_3_depthwise_relu

    # setting the output layer of mobilenetv2(branch of screen and head)
    screen_block_target_layer=target_layer + '_2'
    head_block_target_layer=target_layer + '_3'

    # model_1  process(crop img) 
    for layer in base_model_1.layers:
        layer._name = layer.name + '_1'

    ## the feature map of crop imgs
    x_1 = base_model_1.output
    # x_1 = base_model_1.get_layer('block_16_depthwise_relu_1').output
    x_1 = tf.keras.layers.GlobalAveragePooling2D()(x_1) 
    # model_v18 1024
    x_1 = tf.keras.layers.Dense(1024, activation='relu',name='x_1_dense_1')(x_1)

    # model_2 process(screen pics)
    for layer in base_model_2.layers:
        layer._name = layer.name + '_2'
    
    x_2 = base_model_2.get_layer(screen_block_target_layer).output
    # x_2 = base_model_2.output
    x_2 = tf.keras.layers.GlobalAveragePooling2D()(x_2) 

    # model_2 process(screen pics)
    for layer in base_model_3.layers:
        layer._name = layer.name + '_3'
    
    x_3 = base_model_3.get_layer(head_block_target_layer).output
    # x_3 = base_model_3.output
    x_3 = tf.keras.layers.GlobalAveragePooling2D()(x_3) 
    
    ## concate the feature map of screen and head

    x_t = tf.keras.layers.Concatenate()([x_2, x_3])
    # modelv12 dense 512
    # model_v13/14/15 dense 256
    # model_v18 1024
    # model_v19 256
    x_t = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_1')(x_t)

    # out = tf.keras.layers.Add()([x_t, x_1])
    out = tf.keras.layers.Concatenate()([x_t, x_1])
     # modelv12 dense 1024
     # model_v13/14/15 dense 512
    out = tf.keras.layers.Dense(512, activation='relu',name='cat_dense_2')(out)
    if dropout:
        out = tf.keras.layers.Dropout(dropout_ratdio)(out)
    # modelv12 dense 512
    # model_v13/14/15/19 dense 256
    out = tf.keras.layers.Dense(256, activation='relu',name='cat_dense_3')(out)
    out = tf.keras.layers.Dense(2)(out)
    out = tf.keras.layers.Activation('softmax')(out)

    model = tf.keras.models.Model([base_model_1.input, base_model_2.input, base_model_3.input], out)
 
    return model
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    model =HTCNet(
            crop_h=224,
            crop_w=224, 
            screen_img_h=224, 
            screen_img_w=128,
            head_img_h=128,
            head_img_w=128,
            dropout=True, 
            dropout_ratdio=0.5,
            target_layer='block_5_depthwise_relu')

    model.summary()
# Total params: 4,561,474
    
    

    