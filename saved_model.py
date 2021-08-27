'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:08:34
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:08:34
'''
'''
Descripttion: 
version: 
Author: 
Date: 2021-01-27 02:21:18
LastEditors: Please set LastEditors
LastEditTime: 2021-05-20 05:49:24
'''
import tensorflow as tf 
import numpy as np
from utils.dataloader import Dataloader
# from model_new import HTCNet
# from modelv2 import HTCNet
import os
from opts import parser



def save_model(args):
    model_path = args.test_model_path
    # model_path = '/mnt/sda2/cj/Keras_Knowledge_Distillation/checkpoints/train_model_TEMPERATURE_30_Teacher_train_model_2021-07-14-08-00-04/train_model_2021-07-15-07-40-04/keras_watch_screen_model_epoch_030_acc_0.8342_loss_0.0304.h5'
    # model_path = '/mnt/sda2/cj/TF2_keras_watch_screen/convert_tensorlite/2021_05_20_KD/keras_model/watch_screen_model_epoch_21.h5'
    # ================ from modelv2 import HTCNet ==================
    # model = HTCNet(
    #         crop_h=args.crop_h,
    #         crop_w=args.crop_w, 
    #         screen_img_h=args.screen_img_h, 
    #         screen_img_w=args.screen_img_w,
    #         head_img_h=args.head_img_h,
    #         head_img_w=args.head_img_w,
    #         dropout=args.dropout, 
    #         dropout_ratdio=args.dropout_ratdio,
    #         target_layer='block_5_depthwise_relu') 
    # ================ from model_new import HTCNet ================
    # model = HTCNet(base_model=args.basemodel_name,crop_h=args.crop_h, crop_w=args.crop_w,
    #                 screen_img_h=args.screen_img_h,screen_img_w=args.screen_img_w,
    #                 head_img_h=args.head_img_h,head_img_w=args.head_img_w,
    #                 dropout=args.dropout,dropout_ratdio=args.dropout_ratdio)
    # model.load_weights(model_path)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print(' ======================= ')
    print("load keras weights from '{}' successful".format(model_path)) 

    print("model input list {}\nmodel output list {}".format(model.input, model.output))
    if not isinstance(model.input, list):
        input_names = [model.input.name]
    else:
        input_tensors = model.input
        input_names = [tensor.name for tensor in input_tensors]

    if not isinstance(model.output, list):
        output_names = [model.output.name]
    else:
        output_tensors = model.output
        output_names = [tensor.name for tensor in output_tensors]

    # ========saved_model.save======
    print('input_name :{}\noutput_names:{}'.format(input_names, output_names)) 
    model_save_path= os.path.join(args.saved_model_path,args.version_num)
    os.makedirs(model_save_path,exist_ok=True)
    tf.saved_model.save(model, model_save_path) 




if __name__ == "__main__":
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # ==== tf2.x set_memory_growth
    # gpus = tf.config.list_physical_devices(device_type='GPU')
    # print(gpus)

    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.compat.v1.Session(config=config)
    print(("""
        GPU Congfig
        CUDA_VISIBLE_DEVICES :              {}
        PER_PROCESS_GPU_MEMORY_FRACTION     {}
    """).format(args.gpu_id, args.gpu_memory_fraction))
   
    save_model(args)
