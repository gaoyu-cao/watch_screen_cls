'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:07:55
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:07:55
'''
import tensorflow as tf 
import numpy as np
from tensorflow.python.eager.execute import args_to_matching_eager
from tensorflow.python.keras.backend import dropout 
from tensorflow.python.keras import backend as K 
from utils.dataloader import Dataloader
from model_new import HTCNet
import os
from opts import parser


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

    train_loader = Dataloader(batch_size=batch_size, data_list=args.train_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w)
    val_loader = Dataloader(batch_size=batch_size, data_list=args.val_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w)
    test_loader = Dataloader(batch_size=batch_size, data_list=args.test_list, num_class=num_classes, crop_h=args.crop_h, crop_w=args.crop_w,
                    head_img_h=args.head_img_h, head_img_w=args.head_img_w, screen_img_h=args.screen_img_h, screen_img_w=args.screen_img_w)
    
    # model = HTCNet(crop_h=args.crop_h, crop_w=args.crop_w,
    #                 screen_img_h=args.screen_img_h,screen_img_w=args.screen_img_w,
    #                 head_img_h=args.head_img_h,head_img_w=args.head_img_w,
    #                 screen_block_target_layer=args.screen_block_target_layer,
    #                 head_block_target_layer=args.head_block_target_layer, 
    #                 dropout=args.dropout,dropout_ratdio=args.dropout_ratdio)
    model = HTCNet(base_model=args.basemodel_name,crop_h=args.crop_h, crop_w=args.crop_w,
                    screen_img_h=args.screen_img_h,screen_img_w=args.screen_img_w,
                    head_img_h=args.head_img_h,head_img_w=args.head_img_w,
                    dropout=args.dropout,dropout_ratdio=args.dropout_ratdio)               
    if args.use_summary:
        model.summary()


    return model
    for layer in model.layers:
        layer.trainable = True

    # callback 
    callbacks = []
    save_model_root = args.save_model_root
    model_folder = args.model_folder
    save_checkpoints_path = os.path.join(save_model_root, model_folder)
    os.makedirs(save_checkpoints_path, exist_ok=True)

    # 1.checkpoints
    save_model_name = args.save_model_name
    save_model_path = os.path.join(save_checkpoints_path, save_model_name)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_accuracy', 
                        verbose=1)
    #checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_accuracy', verbose=1,save_weights_only=True, save_best_only=True)
    callbacks.append(checkpoints)
    
    # 2.tensorboard
    # eventfile_dir = os.path.join(args.eventfiles, args.model_folder)
    # tensorboard =  tf.keras.callbacks.TensorBoard(log_dir=eventfile_dir, histogram_freq=0, write_graph=True, batch_size=args.batch_size, write_images=True)
    # callbacks.append(tensorboard)

    # 3.early stop
    # early_stopping =  tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=2)
    # callbacks.append(early_stopping)

    # 4.LearningRateScheduler
    if args.use_lr_decay :
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        # lr_decay_epoch_period = args.lr_decay_epoch_period
        # lr_decay_rate = args.lr_decay_rate
        # reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_decay(model, lr_decay_epoch_period, lr_decay_rate))
        callbacks.append(lr_scheduler)

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
     args.screen_img_h,args.screen_img_w, num_classes,args.screen_block_target_layer, args.head_block_target_layer, 
     args.dropout,args.dropout_ratdio, args.base_lr))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=args.base_lr, momentum=0.9, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(generator=train_loader,   epochs=epochs, verbose=1, callbacks=callbacks, workers=4,class_weight=None, validation_data=val_loader)
    model.fit(train_loader, epochs=epochs, callbacks=callbacks,validation_data=val_loader,workers=4 )
    # ========save model
    # print('---------------save keras model----------------')
    # keras_model_name = 'Keras_Watch_Screen_model.h5'
    # keras_model_path = os.path.join(save_checkpoints_path, keras_model_name)
    # model.save(keras_model_path)
    print('---------------evaluate----------------')
    scores = model.evaluate_generator(test_loader, verbose=1, workers=8)
    print('%s: %.2f' % (model.metrics_names[0], scores[0])) # Loss
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100)) # metrics1


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        #freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        print(tf.compat.v1.global_variables())
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        print("====================: fz name: ", freeze_var_names)
        output_names = output_names or []
        #output_names += [v.op.name for v in tf.global_variables()]
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)
        return frozen_graph

def export_frozen_pb(model, frozen_model_path, frozen_model_name):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full_model = tf.function(lambda x:model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.input[0].shape, model.input[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("=================================================================================> :", layers)
    from tensorflow.python.framework import graph_io
    K.set_learning_phase(0)
    print(type(model.input))
    print(type(model.output))
    print('input is :', model.input)
    print ('output is:', model.output.name)
    print ('output op name is:', model.output.op.name)

    frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
    if not os.path.isdir(frozen_model_path):
        os.mkdir(frozen_model_path)
    graph_io.write_graph(frozen_graph, frozen_model_path, frozen_model_name, as_text=False)#True


def convert_saved_model_to_pb(output_node_names, input_saved_model_dir, output_graph_dir):
    from tensorflow.python.tools import freeze_graph

    output_node_names = ','.join(output_node_names)

    freeze_graph.freeze_graph(input_graph=None, input_saver=None,
            input_binary=None,
            input_checkpoint=None,
            output_node_names=output_node_names,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=output_graph_dir,
            clear_devices=None,
            initializer_nodes=None,
            input_saved_model_dir=input_saved_model_dir)


def save_output_tensor_to_pb():
    output_names = ['dense/Softmax']
    save_pb_model_path = './freeze_graph.pb'
    model_dir = 't'
    convert_saved_model_to_pb(output_names, model_dir, save_pb_model_path)

def convert(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype), tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype), tf.TensorSpec(model.inputs[2].shape, model.inputs[2].dtype)))

    frozen_func = convert_variables_to_constants_v2(full_model)
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

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
            logdir="./frozen_models",
            name="frozen_graph.pb",
            as_text=False)


if __name__ == "__main__":

    from  tensorflow.python.tools import freeze_graph
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
   
    model = train(args)
    model.load_weights("./convert_tensorlite/2021_02_25/checkpoints/MobilenetV2Screen_013.h5")
    print("================================")
    model.summary()
    frozen_model_path = "./"
    frozen_model_name = "fg.pb"
    #export_frozen_pb(model, frozen_model_path, frozen_model_name)
    #model.save('t')
    #convert(model)
    model.save_mlir()
