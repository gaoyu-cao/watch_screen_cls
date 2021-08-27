'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:07:14
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:07:15
'''
'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 01:56:05
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 02:40:56
'''
import h5py
from models.model import HTCNet
from opts import parser
import numpy as np



def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))  

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                print("      {}: {}".format(name. d.value))
    finally:
        f.close()


if __name__ == '__main__':
    # weight_file_path = '/mnt/sda2/cgy/TF2_keras_watch_screen/checkpoints/Watch_Screen_model/train_model_2021/model_v1/MobilenetV2Screen_013.h5'
    weight_file_path = '/mnt/sda2/cgy/Keras_Watch_Screen-master/checkpoints/data_list_20210319/train_model_2021-06-21-02-37-30/MobilenetV2Screen_020.h5'
    # print_keras_wegiths(weight_file_path)
    global args
    args = parser.parse_args()
    model = HTCNet(crop_h=args.crop_h, crop_w=args.crop_w,
                        screen_img_h=args.screen_img_h,screen_img_w=args.screen_img_w,
                        head_img_h=args.head_img_h,head_img_w=args.head_img_w,
                        dropout=args.dropout,dropout_ratdio=args.dropout_ratdio) 
    model.load_weights(weight_file_path)
    print("load keras weights from '{}' successful".format(weight_file_path))
    for layer in model.layers:
    #     if np.isnan(layer.get_weights()).any():
    #         print(layer.name)
    #     else:
    # print('no nan')
        if layer.name=='Conv1_3':
            
            print("layer.name {}\nlayer's weights {}".format(layer.name, layer.get_weights()))