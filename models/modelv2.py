'''
Descripttion: 
version: 
Author: 
Date: 2021-06-08 06:02:57
LastEditors: Please set LastEditors
LastEditTime: 2021-08-06 07:55:06
'''
from re import X
import tensorflow as tf
import os
from tensorflow.keras.layers import BatchNormalization, Flatten, ReLU, Dot, Dense, Conv2D

def h_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6
    
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
            bilinear=True,
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
    x_1_ori = base_model_1.output
    print("Crop feature dim:", x_1_ori.shape)

    x_1 = tf.keras.layers.GlobalAveragePooling2D()(x_1_ori)
    x_1 = tf.keras.layers.Dense(x_1_ori.shape[-1]//16)(x_1)
    x_1 = ReLU()(x_1)
    x_1 = tf.keras.layers.Dense(x_1_ori.shape[-1])(x_1)
    x_1 = h_sigmoid(x_1)
    x_1 = tf.expand_dims(x_1, axis=1)
    x_1 = tf.expand_dims(x_1, axis=1)
    x_1_ori = x_1 * x_1_ori

    # model_2 process(screen pics)
    for layer in base_model_2.layers:
        layer._name = layer.name + '_2'
    
    x_2 = base_model_2.get_layer(screen_block_target_layer).output
    x_2 = tf.keras.layers.GlobalAveragePooling2D()(x_2)

    # model_2 process(screen pics)
    for layer in base_model_3.layers:
        layer._name = layer.name + '_3'
    
    x_3 = base_model_3.get_layer(head_block_target_layer).output
    x_3 = tf.keras.layers.GlobalAveragePooling2D()(x_3)

    if bilinear:
        x_1 = Conv2D(filters=4, kernel_size=(3,3), name='x_1_conv_1')(x_1_ori)
        x_1 = BatchNormalization(name='bn_x_1_conv_1')(x_1)
        x_1 = ReLU()(x_1)
        x_1 = Flatten()(x_1)
        print("Conv feature dims:", x_1.shape)
        #x_1 = tf.keras.layers.Dense(128, name='x_1_dense_1')(x_1)
        #x_1 = BatchNormalization(name='bn_x_1_dense_1')(x_1)
        #x_1 = ReLU()(x_1)
        x_2 = tf.keras.layers.Dense(64, name='x_2_dense_1')(x_2)
        x_2 = BatchNormalization(name='bn_x_2_dense_1')(x_2)
        x_2 = ReLU()(x_2)
        x_3 = tf.keras.layers.Dense(64, name='x_3_dense_1')(x_3)
        x_3 = BatchNormalization(name='bn_x_3_dense_1')(x_3)
        x_3 = ReLU()(x_3)

        '''x_2 = tf.keras.layers.Reshape(target_shape=(x_2.shape[-1], 1))(x_2)
        x_3 = tf.keras.layers.Reshape(target_shape=(x_3.shape[-1], 1))(x_3)
        x_t = tf.keras.layers.Dot(axes=2)([x_2, x_3])
        x_t = tf.keras.layers.Flatten()(x_t)
        x_t = tf.keras.layers.Dense(128, name='dot_dense_1')(x_t)
        x_t = BatchNormalization(name='bn_dot_dense_1')(x_t)
        x_t = ReLU()(x_t)'''
        x_t = tf.keras.layers.Concatenate()([x_2, x_3])

        x_t = tf.keras.layers.Reshape(target_shape=(x_t.shape[-1], 1))(x_t)
        x_1 = tf.keras.layers.Reshape(target_shape=(x_1.shape[-1], 1))(x_1)
        out = tf.keras.layers.Dot(axes=2)([x_1, x_t])
        out = tf.keras.layers.Flatten()(out)
        out = tf.keras.layers.Dense(256, name='dot_dense_2')(out)
        out = BatchNormalization(name='bn_dot_dense_2')(out)
        out = ReLU()(out)

    else:
        # x_1 = base_model_1.get_layer('block_16_depthwise_relu_1').output
        x_1 = tf.keras.layers.GlobalAveragePooling2D()(x_1) 
        # model_v18 1024
        #x_1 = tf.keras.layers.Dense(1024, activation='relu',name='x_1_dense_1')(x_1)
        x_1 = tf.keras.layers.Dense(256, activation='relu',name='x_1_dense_1')(x_1)

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
    model =HTCNet(
            crop_h=224,
            crop_w=224, 
            screen_img_h=224, 
            screen_img_w=128,
            head_img_h=128,
            head_img_w=128,
            dropout=True, 
            dropout_ratdio=0.5,
            target_layer='block_2_depthwise_relu')

    model.summary()
    
    
    

    