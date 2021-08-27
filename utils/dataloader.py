'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:11:48
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:11:49
'''
import os
import cv2
import numpy as np
import random

import tensorflow as tf
import sys


class TxtRecord(object):
    '''
    # 解析文本文件
    '''
    def __init__(self, row):
        self._data = row

    @property
    def head_images_path(self):
        return self._data[0]
    
    @property
    def full_image_path(self):
        return self._data[1]

    @property
    def label(self):
        return int(self._data[2])

    @property
    def sc(self):
        
        return [int(i) for i in self._data[3:7]]
    
    @property
    def hc(self):
        return [int(i) for i in self._data[7:]]


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    '''
    @description: augment img hsv
    @param {*}
    @return {*}
    '''    
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, data_list, num_class, crop_h, crop_w,
                    head_img_h, head_img_w, screen_img_h, screen_img_w, 
                    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                    model='train', shuffle=True):

        self.batch_size = batch_size
        self.data_list = data_list
        self.num_class = num_class
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.head_img_h = head_img_h
        self.head_img_w = head_img_w
        self.screen_img_h = screen_img_h
        self.screen_img_w = screen_img_w
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.model = model
        self.shuffle = shuffle
        
        self._parse_list()
        self.on_epoch_end()
        self.max = len(self.video_list) // self.batch_size
        

    def _load_image(self, full_image_path):
        
        fullimage = cv2.imread(full_image_path)
        fullimage = cv2.resize(fullimage, (720, 1280))
      
        # fullimage = random_noise(fullimage)
        # fullimage = fullimage[:,:,(2,1,0)]
        if self.model == 'train':
            augment_hsv(fullimage, self.hsv_h, self.hsv_s, self.hsv_v)
            fullimage = fullimage.astype('uint8')
            fullimage = self._data_augmentation(fullimage)
        fullimage = fullimage.astype('float32')
        fullimage = fullimage[:,:,(2,1,0)]
        fullimage = tf.keras.applications.mobilenet_v2.preprocess_input(fullimage)
        
        return fullimage
    

    def _data_augmentation(self, img):
        '''
        args:
            img: 3D numpy tensor 
        return:
            augmentation data
        '''
        img = tf.image.random_jpeg_quality(img, 30, 70)
        return img.numpy()
    

    def __getitem__(self,index):
        left_index = index * self.batch_size
        right_index = (index + 1) * self.batch_size
        
        indexes = self.indexes[left_index:right_index]
        # print(indexes)
        record_list = [self.video_list[k] for k in indexes]
        # self.index_data += 1
        return self.data_generate(record_list)

    def data_generate(self, record_list):
        '''
        @description: 
        @param {*}
        @return {*}
        '''
        batch_head_image = np.empty((self.batch_size, self.head_img_h, self.head_img_w, 3), dtype=np.float32)
        batch_screen_images = np.empty((self.batch_size, self.screen_img_h, self.screen_img_w, 3),dtype=np.float32)
        batch_crop_images = np.empty((self.batch_size, self.crop_h, self.crop_w, 3),dtype=np.float32)
        batch_anno = np.empty((self.batch_size), dtype=int)
        batch_path = []

        for label_id, record in enumerate(record_list):
            new_crop, screen, head = self.get(record)
            batch_anno[label_id] = record.label
            batch_crop_images[label_id,] = new_crop 
            batch_head_image[label_id,] = head
            batch_screen_images[label_id,] = screen
            batch_path.append(record.head_images_path)
        if self.model == 'train' or self.model == 'test':
            return [batch_crop_images, batch_screen_images, batch_head_image], tf.keras.utils.to_categorical(batch_anno, num_classes=self.num_class)
        else:
            return batch_path, [batch_crop_images, batch_screen_images, batch_head_image], batch_anno.tolist()
    
    def get(self, record):
        fullimage = self._load_image(record.full_image_path)
        # label =
        # print(record.full_image_path,fullimage.shape)
        sc_xmin = record.sc[0]
        sc_ymin = record.sc[1]
        sc_xmax = record.sc[2]
        sc_ymax = record.sc[3]
        
        hc_xmin = record.hc[0]
        hc_ymin = record.hc[1]
        hc_xmax = record.hc[2]
        hc_ymax = record.hc[3]
        
        new_xmin = min(sc_xmin, hc_xmin)
        new_ymin = min(sc_ymin, hc_ymin)
        new_xmax = max(sc_xmax, hc_xmax)
        new_ymax = max(sc_ymax, hc_ymax)

        screen = fullimage[sc_ymin:sc_ymax, sc_xmin:]
        head = fullimage[hc_ymin:hc_ymax, hc_xmin:hc_xmax]
        new_crop = fullimage[new_ymin:new_ymax, new_xmin:new_xmax]

        new_crop = cv2.resize(new_crop, (self.crop_w, self.crop_h))
        screen = cv2.resize(screen, (self.screen_img_w, self.screen_img_h))
        head = cv2.resize(head, (self.head_img_w, self.head_img_h))
        
        return  new_crop,  screen,  head

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.video_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _parse_list(self):
        self.video_list = [TxtRecord(x.strip().split(',')) for x in open(self.data_list)]
    
    def __len__(self):
        return len(self.video_list) // self.batch_size


if __name__ == '__main__':
    data_loader = Dataloader(batch_size=32, data_list='/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list_20210319/test_list.txt', num_class=2, crop_h=224, crop_w=224,
                    head_img_h=96, head_img_w=96, screen_img_h=96, screen_img_w=96,shuffle=False, model='train')
    
    for i, (data, label) in enumerate(data_loader):
      
        print(i, data[0].shape, data[1].shape, data[1].shape, label.shape)
        # except:
        #     print(path)
            
    # tf_data = tf.data.Dataset.from_generator(data_loader)
