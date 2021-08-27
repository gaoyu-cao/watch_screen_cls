'''
Author: cheng jie 
Date: 2021-01-13 14:28:20
LastEditTime: 2021-08-12 03:29:21
LastEditors: Please set LastEditors
Description: 加载模型，验证模型(loss,acc)
FilePath: /cgy/Keras_Video_Recognization/eval.py
'''

import tensorflow.keras.layers as layers
# from tensorflow_backend import set_session
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD

from utils.dataloader_mask import Dataloader

from opts import parser
from models.model import HTCNet
import os


def eval(args):
    
    # if args.dataset.lower() == 'ucf101':
    #     num_class = 101
    #     image_tmpl = '{}_{:0>5d}.jpg'
    # elif args.dataset.lower() == 'hmdb51':
    #     num_class = 51
    #     image_tmpl = '{}_{:0>5d}.jpg'
    # elif args.dataset.lower() == 'electric_bike':
    #     num_class = 2
    #     image_tmpl = '{}_{}.png'
    # else:
    #     raise ValueError('Unknow data ', args.dataset)
    batch_size = args.batch_size

    eval_loader = Dataloader(batch_size=batch_size, data_list=args.train_list, num_class=args.num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w)
   
    checkpoints =  args.test_model_path
    if checkpoints.split('/')[-1].startswith('keras'):
        model = load_model(checkpoints)
        print("load keras model from '{}' successful".format(checkpoints))
    else:
        model = HTCNet(crop_h=args.crop_h, crop_w=args.crop_w,
                screen_img_h=args.screen_img_h,screen_img_w=args.screen_img_w,
                head_img_h=args.head_img_h,head_img_w=args.head_img_w,
                screen_block_target_layer=args.screen_block_target_layer,
                head_block_target_layer=args.head_block_target_layer, 
                dropout=args.dropout,dropout_ratdio=args.dropout_ratdio)
        # block_4_project_BN  block_5_expand
        print(("""
        Initializing  model： 
        watch screen model Configurations:
                batch_size:                 {}
                epoch:                      {}
                crop_h:                     {}
                crop_w:                     {}
                head_img_h:                 {}
                head_img_w:                 {}
                screen_img_h:               {}
                screen_img_w:               {}
                num_classes:                {}
                screen_block_target_layer:  '{}'
                head_block_target_layer:    '{}'
                use dropout:                {}
                dropout_ratdio:             {}
                base_lr:                    {}
        """).format(args.batch_size, args.max_epochs, args.crop_h, args.crop_w,args.head_img_h,args.head_img_w,
                args.screen_img_h,args.screen_img_w, args.num_classes,args.screen_block_target_layer, args.head_block_target_layer, 
                args.dropout,args.dropout_ratdio, args.base_lr))
        # optimizer
        sgd = SGD(lr=args.base_lr, momentum=0.9, decay=0)

        # compilte model
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        weights_path = args.checkpoints
        model.load_weights(weights_path)
        
        print('load weights successful from {}'.format(weights_path))


    if  args.use_summary:
        model.summary()
   

    print('---------------evaluate----------------')
    scores = model.evaluate_generator(eval_loader, verbose=1, workers=8)
    print('%s: %.2f' % (model.metrics_names[0], scores[0])) # Loss
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100)) # metrics1

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    print(("""
        Eval MODEL ........
        GPU Congfig
        CUDA_VISIBLE_DEVICES :              {}
        PER_PROCESS_GPU_MEMORY_FRACTION     {}
    """).format(args.gpu_id, args.gpu_memory_fraction))
    use_gpu_id = os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_memory_fraction)
    # set_session(tf.Session(config=config))
    sess = tf.compat.v1.Session(config=config)
    eval(args)
    