'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:08:21
LastEditors: Please set LastEditors
LastEditTime: 2021-08-25 03:08:58
'''

import argparse
from tokenize import String

parser = argparse.ArgumentParser(description='Keras implementation of video action classification')

# ================================data config===============================

parser.add_argument('--train_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/train_list_new.txt')
parser.add_argument('--val_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/val_list_new.txt')
parser.add_argument('--test_list', type=str, default='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/test_list_new.txt')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--data_augmentation', action='store_true')

# ================================img config==============================
parser.add_argument('--crop_h', type=int, default=224)
parser.add_argument('--crop_w', type=int, default=224)
parser.add_argument('--head_img_h', type=int, default=128)
parser.add_argument('--head_img_w', type=int, default=128)
parser.add_argument('--screen_img_h', type=int, default=224)
parser.add_argument('--screen_img_w', type=int, default=128)

# ================================model config==============================
parser.add_argument('--use_summary', action='store_true')
parser.add_argument('--basemodel_name', type=str, default='mobilenetv2')
parser.add_argument('--dataloader_mask', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--screen_block_target_layer', type=str, default='block_5_depthwise_relu_2')
parser.add_argument('--head_block_target_layer', type=str, default='block_5_depthwise_relu_3')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dropout_ratdio', type=float, default=0.5)

#================================lr config=================================
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--use_lr_decay', action='store_true')
parser.add_argument('--lr_decay_epoch_period', type=int, default=10)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)

#===============================gpu config=================================
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--gpu_memory_fraction', type=float, default=0.9)

#===============================saving checkpoing and eventfile============
parser.add_argument('--save_model_root', type=str, default='/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/')
parser.add_argument('--model_folder', type=str, default='train_model_{}')
parser.add_argument('--eventfiles', type=str, default='/mnt/sda1/cgy/Keras_Watch_Screen/eventfiles/')
parser.add_argument('--log_path', type=str,default='/mnt/sda1/cgy/Keras_Watch_Screen/log/')
parser.add_argument('--save_model_name', type=str, default='MobilenetV2Screen_{epoch:03d}.h5')

#===============================eval.py or test.py config============
parser.add_argument('--test_model_path', type=str, default='/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/Watch_Screen_model/Q2/data_list3/train_model_V4/keras_watch_model.h5')
parser.add_argument('--saved_model_path', type=str, default='/mnt/sda1/cgy/Keras_Watch_Screen/convert_tensorlite/2021_05_20_KD/saved_model')
parser.add_argument('--version_num', type=str, default='1')

parser.add_argument('--pre_weights', type=str, default='/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/data_list_20210319/train_model_2021-08-23-06-10-48/MobilenetV2Screen_030.h5')