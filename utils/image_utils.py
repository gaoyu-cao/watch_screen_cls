'''
Descripttion: 
version: 
Author: 
Date: 2021-08-17 01:23:43
LastEditors: Please set LastEditors
LastEditTime: 2021-08-17 02:04:36
'''
import numpy as np
import cv2


def resize_image(img, target_sizes, keep_ratio=True):
    # if keep_ratio is True, letterbox using padding
    if not isinstance(target_sizes, (list, set, tuple)):
        target_sizes = [target_sizes, target_sizes]
    target_h, target_w = target_sizes                       

    h, w, _ = img.shape                                     
    scale = min(target_h / h, target_w / w)                 
    temp_h, temp_w = int(scale * h), int(scale * w)         
    image_resize = cv2.resize(img, (temp_w, temp_h))        

    if keep_ratio:
        image_new = np.full(shape=(target_h, target_w, 3), fill_value=128.0)                    
        delta_h, delta_w = (target_h - temp_h) // 2, (target_w - temp_w) // 2                   
        image_new[delta_h: delta_h + temp_h, delta_w: delta_w + temp_w, :] = image_resize       
        
        return image_new


if __name__ == '__main__':
    img = cv2.imread("/mnt/sda1/cgy/new_crop.jpg")
    img = resize_image(img,(96, 224))
    cv2.imwrite('resize_image.jpg', img)