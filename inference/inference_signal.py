'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:09:43
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:10:37
'''

from logging import NOTSET
import tensorflow as tf
import cv2
import numpy as np
import os
import time 
import pandas as pd
# '/mnt/sda2/cj/TF2_keras_watch_screen/checkpoints/Watch_Screen_model/Q2/data_list3/train_model_V4/keras_watch_model.h5'
MODEL_PATH = '/mnt/sda2/cj/TF2_keras_watch_screen/convert_tensorlite/2021_02_25/saved_model/5'
# ORIGION_PATH_LIST = '/mnt/sda2/Public_Data/Data_lookscreen/train_dataset/data_list3'
# TEST_LIST = '/mnt/sda2/cj/TF2_keras_watch_screen/test_data/test_list_sample.txt'
TEST_LIST = '/mnt/sda2/cj/TF2_keras_watch_screen/test_data/000E9A000A32_detect_res2.txt'
SAVE_ROOT = '/mnt/sda2/cj/TF2_keras_watch_screen/inference'
# JPG_QUALITY_LIST = [10, 15, 20, 25, 30, 35, 40, 45, 50]

def detect(model,full_image_path, screen_bboxes, head_bboxes, draw=False, save_root=None):
    fullimage = cv2.imread(full_image_path)
    fullimage = cv2.resize(fullimage, (576, 704))
    fullimage_ = fullimage[:,:,(2,1,0)]
    fullimage_ = tf.keras.applications.mobilenet_v2.preprocess_input(fullimage_)
    
    sc_xmin = screen_bboxes[0]
    sc_ymin = screen_bboxes[1]
    sc_xmax = screen_bboxes[2]
    sc_ymax = screen_bboxes[3]
    
    hc_xmin = head_bboxes[0]
    hc_ymin = head_bboxes[1]
    hc_xmax = head_bboxes[2]
    hc_ymax = head_bboxes[3]
    
    
    new_xmin = min(sc_xmin, hc_xmin)
    new_ymin = min(sc_ymin, hc_ymin)
    new_xmax = max(sc_xmax, hc_xmax)
    new_ymax = max(sc_ymax, hc_ymax)


    # =========== draw the rectangle ===========
    
    screen = fullimage_[sc_ymin:sc_ymax, sc_xmin:sc_xmax]
    head = fullimage_[hc_ymin:hc_ymax, hc_xmin:hc_xmax]
    new_crop = fullimage_[new_ymin:new_ymax, new_xmin:new_xmax]

    new_crop = cv2.resize(new_crop, (224, 224))
    screen = cv2.resize(screen, (128, 224))
    head = cv2.resize(head, (128, 128))

    new_crop = np.expand_dims(new_crop, 0)
    screen = np.expand_dims(screen, 0)
    head = np.expand_dims(head, 0)
    # result = model.predict([new_crop, screen, head])
    start_time = time.time()
    result = model([new_crop, screen, head]).numpy()
    result = np.squeeze(result, 0)
    end_time = time.time() - start_time
    if draw:
        cv2.rectangle(fullimage, (sc_xmin, sc_ymin), (sc_xmax, sc_ymax), (0, 255, 0), 4)
        cv2.rectangle(fullimage, (hc_xmin, hc_ymin), (hc_xmax, hc_ymax), (255, 0, 0), 4)
        
        cv2.putText(fullimage, "YES:{:.4f}".format(result[1]), (hc_xmin-60, hc_ymin-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(fullimage, "No:{:.4f}".format(result[0]), (hc_xmin-60, hc_ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        # jpg_quality = full_image_path.split('/')[-2]
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        new_name = full_image_path.split('/')[-1][:-4]+'_'+str(time_str)+"_{:.4f}.jpg".format(end_time)
        # new_name = full_image_path.split('/')[-1][:-4]+"_draw.jpg"
        save_jpg_full_path = os.path.join(save_root, new_name)
        cv2.imwrite(save_jpg_full_path, fullimage)

    return full_image_path, result


def inference(test_list_path, model_path, save_folder_path, quality_list=None):
    if  model_path.split('/')[-1].startswith('keras'):
        model = tf.keras.models.load_model(model_path)
        print("load keras model from '{}' successful".format(model_path))
    else:
        model = tf.saved_model.load(model_path)
        print('load saved model from {} successful'.format(model_path))
    
    # ===== load img path =====
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
    result_dict = dict()
    
    for line in lines:
        anno = line.split(',')
        label = anno[2]
        
        head_bboxes = [int(i) for i in anno[7:]]
        screen_bboxes = [int(i) for i in anno[3:7]]
        full_image_path = anno[1]
        # for quality in quality_list:
        #     new_list = []
        #     temp_path = anno[1].split('/')
        #     new_list.extend(temp_path[:-2])
        #     new_list.append(str(quality))
        #     new_list.append(temp_path[-1])
            # full_image_path = '/'.join(new_list)
            # print(full_image_path)
            # jpg_quality = str(quality)
            # full_image_path = anno[1]
        test_folder = full_image_path.split('/')[-2]
        save_full_path = os.path.join(save_folder_path, test_folder)
        os.makedirs(save_full_path, exist_ok=True)
        
        full_image_path, result = detect(model, full_image_path, 
                    screen_bboxes, head_bboxes, save_root=save_full_path)
        result = result.tolist()
        if full_image_path not in result_dict.keys():
            result_dict[full_image_path] = result[1]
        else:
            result_dict[full_image_path] = max(result_dict[full_image_path], result[1])
        print('image_path:{}, predict:{}'.format(full_image_path, result))
    print('----save csv -----')
    all_result_list = []
    # score_list = []
    for img_path in result_dict.keys():
        result_list = []
        result_list.append(img_path)
        result_list.append(result_dict[img_path])
        all_result_list.append(result_list)
    # save_list = [img_path_list, score_list]
    name = ['image_path', 'positive_scorce']
    
    test=pd.DataFrame(columns=name,data=all_result_list)
    test.to_csv('/mnt/sda2/cj/TF2_keras_watch_screen/inference/watch_screen_model.scv', encoding='gbk')    
if __name__ == "__main__":
    # inference(TEST_LIST, MODEL_PATH, SAVE_ROOT, JPG_QUALITY_LIST)
    inference(TEST_LIST, MODEL_PATH, SAVE_ROOT)
                

