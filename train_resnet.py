'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:08:57
LastEditors: Please set LastEditors
LastEditTime: 2021-08-24 01:29:15
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf 
import numpy as np
from models.resnet import HTCResNet 
from opts import parser
import time
from tensorflow.keras.optimizers import SGD
                                    
def lr_decay(model, lr_decay_epoch_period, lr_decay_rate):
    def scheduler(epoch):
        if (epoch % lr_decay_epoch_period == 0) and (epoch != 0):
            lr = tf.keras.backend.get_value(model.optimizer.lr)
            tf.keras.backend.set_value(model.optimizer.lr, lr * lr_decay_rate)
            print('Lr reduced to {}'.format(lr * lr_decay_rate))

        return tf.keras.backend.get_value(model.optimizer.lr)

    return scheduler

def lr_schedule(epoch):
    lr = args.base_lr
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def train(args):
    batch_size = args.batch_size
    epochs = args.max_epochs
    num_classes = args.num_classes
                                        
    if args.dataloader_mask:
        print('dataloader type is mask')
        from utils.dataloader_mask import Dataloader

    else:
        from utils.dataloader import Dataloader
    train_loader = Dataloader(batch_size=batch_size, data_list=args.train_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w)
    val_loader = Dataloader(batch_size=batch_size, data_list=args.val_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w,shuffle=False, model='test')
    test_loader = Dataloader(batch_size=batch_size, data_list=args.test_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w,shuffle=False, model='test')
    
    # ============== using modelv2 import HTCNet ===========
    # block_3_depthwise_relu, block_5_depthwise_relu, block_2_depthwise_relu
    block_target_layer = 'block_2_depthwise_relu'
    model = HTCResNet(
            crop_h=args.crop_h,
            crop_w=args.crop_w, 
            screen_img_h=args.screen_img_h, 
            screen_img_w=args.screen_img_w,
            head_img_h=args.head_img_h,
            head_img_w=args.head_img_w,
            dropout=args.dropout, 
            dropout_ratdio=args.dropout_ratdio,
            # bilinear=True,
            # target_layer=block_target_layer
            )

    if args.use_summary:
        model.summary()

    if args.resume:
        model.load_weights(args.resume_path)
        print('load weights successful')

    if args.pre_weights:
        model.load_weights(args.pre_weights)
        print('load weights successful')
        
    # callback 
    callbacks = []
    data_folder_path = args.train_list.split('/')[-2]       # data_folder_path = data_list_20210319

    save_model_root = args.save_model_root

    # 根据训练集不用再次划分一下模型
    save_model_data_root = os.path.join(save_model_root, data_folder_path)
    # model_folder = args.model_folder
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # print(time_str)
    model_folder = args.model_folder.format(time_str)
    save_checkpoints_path = os.path.join(save_model_data_root, model_folder)
    os.makedirs(save_checkpoints_path, exist_ok=True)
    # print(save_checkpoints_path)

    # 1.checkpoints
    save_model_name = args.save_model_name
    save_model_path = os.path.join(save_checkpoints_path, save_model_name)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_accuracy', verbose=1, save_weights_only=True, save_best_only=False)
    callbacks.append(checkpoints)
    
    # # 2.tensorboard
    # eventfile_dir = os.path.join(args.eventfiles, args.model_folder)
    # tensorboard =  tf.keras.callbacks.TensorBoard(log_dir=eventfile_dir, histogram_freq=0, write_graph=True, batch_size=args.batch_size, write_images=False)
    # callbacks.append(tensorboard)

    # 3.early stop
    # early_stopping =  tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=2)
    # callbacks.append(early_stopping)
    
    # 4.LearningRateScheduler
    if args.use_lr_decay :
        lr_decay_epoch_period = args.lr_decay_epoch_period
        lr_decay_rate = args.lr_decay_rate
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay(model, lr_decay_epoch_period, lr_decay_rate))
        callbacks.append(lr_scheduler)

    print(("""
        Data path:
            train_data_path:               '{}'
            val_data_path:                 '{}'
            test_data_path:                '{}'
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
                block_target_layer:        '{}'
                use dropout:                {}
                dropout_ratdio:             {}
                base_lr:                    {}
    """).format(args.train_list, args.val_list, args.test_list,
            args.batch_size, args.max_epochs, args.crop_h, args.crop_w,
            args.head_img_h,args.head_img_w,args.screen_img_h,args.screen_img_w, 
            num_classes,block_target_layer, 
            args.dropout,args.dropout_ratdio, args.base_lr))

    optimizer=tf.keras.optimizers.SGD(lr=args.base_lr, momentum=0.9, decay=1e-5)
    # loss binary_crossentropy/categorical_crossentropy
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(generator=train_loader,   epochs=epochs, verbose=1, callbacks=callbacks, workers=4,class_weight=None, validation_data=val_loader)
    model.fit(train_loader, epochs=epochs, callbacks=callbacks, validation_data=val_loader,workers=8)
   
    print('---------------evaluate----------------')
    scores = model.evaluate_generator(test_loader, verbose=1, workers=8)
    print('%s: %.2f' % (model.metrics_names[0], scores[0])) # Loss
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100)) # metrics1
    print(model.metrics_names)
    print(scores)

    # ==================== save finall keras h5 =============
    # model_h5 = 'keras_watch_screen.hdf5'
    # keras_model_full_path = os.path.join(save_checkpoints_path, model_h5)
    # model.save(keras_model_full_path)

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.compat.v1.Session(config=config)
    print(("""
        GPU Congfig
        CUDA_VISIBLE_DEVICES :              {}
        PER_PROCESS_GPU_MEMORY_FRACTION     {}
    """).format(args.gpu_id, args.gpu_memory_fraction))
   
    train(args)
