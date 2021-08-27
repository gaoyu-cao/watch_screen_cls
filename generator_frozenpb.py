'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:06:31
LastEditors: Please set LastEditors
LastEditTime: 2021-08-11 02:53:37
'''
'''
Descripttion: 
version: 
Author: 
Date: 2021-01-20 02:46:53
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 02:28:57
'''
import tensorflow as tf


from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
# from model_new import HTCNet
from models.model import HTCNet
import os
from opts import parser


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
    # model_path = '/mnt/sda2/cgy/Keras_Knowledge_Distillation/checkpoints/train_model_TEMPERATURE_30_Teacher_train_model_2021-07-14-08-00-04/train_model_2021-07-15-07-40-04/keras_watch_screen_model_epoch_030_acc_0.8342_loss_0.0304.h5'
    model_path = '/mnt/sda1/cgy/Keras_Watch_Screen/checkpoints/data_list_20210319/train_model_2021-08-05-03-35-09/keras_watch_screen_model.h5'
    # model_path = '/mnt/sda2/cj/TF2_keras_watch_screen/checkpoints/Watch_Screen_model/KD/epoch_022_acc_0.892489_loss_0.042395/watch_screen_model_epoch_21.h5'
    # ================ from models import HTCNet ==================
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
    model = tf.keras.models.load_model(model_path)
    # model.load_weights(model_path)
    model.summary()
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
    full_model = tf.function(lambda x:model(x))
    full_model = full_model.get_concrete_function(x=(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype), 
                                                tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype),
                                                 tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)))
    frozen_func = convert_variables_to_constants_v2(full_model, lower_control_flow=False)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    frozen_pb_pull_path = "/mnt/sda1/cgy/Keras_Watch_Screen/convert_tensorlite/2021_07_15_KD/frozen_pb"
    os.makedirs(frozen_pb_pull_path, exist_ok=True)
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                        logdir=frozen_pb_pull_path,
                        name="complex_frozen_graph.pb",
                        as_text=False)
    
    # ========saved_model.save======
    # print('input_name :{}\noutput_names:{}'.format(input_names, output_names)) 
    # convert_keras_model_to_pb(model)
