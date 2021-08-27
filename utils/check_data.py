'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:11:19
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:11:20
'''
'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 02:08:16
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 02:37:37
'''
import os
from tqdm import tqdm 
import random

def show_data_distribuion(data_path):
    positive_num, negative_num = 0, 0
    with open(data_path, 'r') as f:
        data_lists = f.readlines()

    for data in data_lists:
        if data.split(',')[2] == '1':

            positive_num += 1
        else:
            negative_num += 1
    print("""
    'data distribution：{}'
    all:{}\npositive_num :{}\nnegative_num :{}
    """.format(data_path, len(data_lists),positive_num, negative_num ))


def split_data(data_list, train_ratdio=0.8, val_ratdio=0.1):
    '''

    '''
    num_video = len(data_list)
    random.shuffle(data_list)
    train_val_num = int(num_video * train_ratdio)
    train_val_lists = data_list[:train_val_num]
    test_lists = data_list[train_val_num:]

    train_num = int(len(train_val_lists) * (1-val_ratdio))
    train_lists = train_val_lists[:train_num]
    val_lists = train_val_lists[train_num:]
    return train_lists, val_lists, test_lists


def write_txt(data_list, save_path):
    random.shuffle(data_list)
    with open(save_path, 'a', encoding='utf-8') as f:
        for info in data_list:
            info = info
            f.write(info)


# def split_data(data_path):
data_path = '/mnt/sda2/cgy/headscreen_cls/data_list_20210319/train_list.txt'  
with open(data_path, 'r') as f:
    data_lists = f.readlines()

positive_sample = list()
negative_sample = list()

for data in data_lists:
    if data.split(',')[2] == '1':
        positive_sample.append(data)
    else: 
        negative_sample.append(data)

print('all sample number is: {}\npositive sample number is:{}\nnegative sample number is :{}'.format(len(data_lists), len(positive_sample), len(negative_sample)))


random.shuffle(negative_sample)

negative_sample_select = negative_sample[:int(len(positive_sample)* 2.5)]
negative_sample_NO_select = negative_sample[int(len(positive_sample)* 2.5):]
print('negative sample select is :{}\other number is {}'.format(len(negative_sample_select), len(negative_sample_NO_select)))
positive_sample.extend(negative_sample_select)
print('new data:', len(positive_sample))
write_txt(positive_sample, '/mnt/sda2/cgy/TF2_keras_watch_screen/data_list/train_new3.txt')
# data_Path_list = ['/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt',
#                   '/mnt/sda2/cj/headscreen_cls/data_list_20210319/val_list.txt',
#                 '/mnt/sda2/cj/headscreen_cls/data_list_20210319/test_list.txt',
#          ]
# '''
#     'data distribution：/mnt/sda2/cj/headscreen_cls/data_list_20210319/train_list.txt'
#     all:68300
# positive_num :18358
# negative_num :49942
    

#     'data distribution：/mnt/sda2/cj/headscreen_cls/data_list_20210319/val_list.txt'
#     all:17075
# positive_num :4492
# negative_num :12583
    

#     'data distribution：/mnt/sda2/cj/headscreen_cls/data_list_20210319/test_list.txt'
#     all:11694
# positive_num :1645
# negative_num :10049

# '''
# for data_path in data_Path_list:
#     show_data_distribuion(data_path)