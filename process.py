from __future__ import division
from __future__ import print_function
import os, time, scipy.io, shutil
import numpy as np
import cv2 as cv
import glob
import argparse
import re

'''
搜索丢失的数据并使用临近数据填充：
1. 文件，即整个时间点的通道的数据丢失；
2. 区域丢失，某时间点的三个数据通道文件都存在，但是数据本身存在不正常（大面积为0）。

处理方法：
直接用前一时间点的数据对本时间点进行填充
'''
def filling(process_dir, HEADS):
    for head in HEADS:
        '''
        找到数据集小时ID的最大值和最小值，供之后的遍历搜索
        '''
        band1name = head + '_Hour_*_Band_08.npy'
        band1files = glob.glob(os.path.join(process_dir, band1name))
        tids = []
        for i in range(len(band1files)):
            tids.append(int(re.findall(r'Hour_(\d+)_Band', band1files[i])[0]))
        
        first_tid = min(tids)
        last_tid = max(tids)

        for cid in range(first_tid, last_tid + 1):
            # check files miss
            name1 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_08.npy')
            name2 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_09.npy')
            name3 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_10.npy')

            if not (os.path.isfile(name1) and os.path.isfile(name2) and os.path.isfile(name3)):
                # 存在文件丢失，检验文件是否能正常加载，如无法正常加载，加载上一时间点的对应通道的文件，并保存为该时间点的文件
                print("missing:", name1, os.path.isfile(name1), name2, os.path.isfile(name2), name3, os.path.isfile(name3))
                try:
                    data1 = np.load(name1)
                except:
                    print(name1)
                    data1 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_08.npy'))
                    np.save(os.path.join(name1), data1)

                try:
                    data2 = np.load(name2)
                except:
                    print(name2)
                    data2 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_09.npy'))
                    np.save(os.path.join(name2), data2)

                try:
                    data3 = np.load(name3)
                except:
                    print(name3)
                    data3 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_10.npy'))
                    np.save(os.path.join(name3), data3)

            # check data loss and 
            data1 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_08.npy')) / 16383.0
            data2 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_09.npy')) / 2047.0
            data3 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_10.npy')) / 2047.0
            
            
            if sum(sum(data1 >= 1)) > 2048 or sum(sum(data1 <= 0)) > 2048:
                print(name1)
                data1 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_08.npy'))
                np.save(os.path.join(name1), data1)

            if sum(sum(data2 >= 1)) > 2048 or sum(sum(data2 <= 0)) > 2048:
                print(name2)
                data2 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_09.npy'))
                np.save(os.path.join(name2), data2)

            if sum(sum(data3 >= 1)) > 2048 or sum(sum(data3 <= 0)) > 2048:
                print(name3)
                data3 = np.load(os.path.join(process_dir, head + '_Hour_' + str(cid - 1) + '_Band_10.npy'))
                np.save(os.path.join(name3), data3)


def save2img(process_dir, output_dir, HEADS, ps):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for head in HEADS:
        band1name = head + '_Hour_*_Band_08.npy'
        band1files = glob.glob(os.path.join(process_dir, band1name))
        tids = []
        for i in range(len(band1files)):
            tids.append(int(re.findall(r'Hour_(\d+)_Band', band1files[i])[0]))
        
        first_tid = min(tids)
        last_tid = max(tids)

        for cid in range(first_tid, last_tid + 1):
            name1 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_08.npy')
            name2 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_09.npy')
            name3 = os.path.join(process_dir, head + '_Hour_' + str(cid) + '_Band_10.npy')

            data1 = np.expand_dims(np.load(name1), axis=2) / 16383.0
            data2 = np.expand_dims(np.load(name2), axis=2) / 2047.0
            data3 = np.expand_dims(np.load(name3), axis=2) / 2047.0
            data = np.clip(np.concatenate((data1, data2, data3), axis=2), 0, 1)

            img = cv.resize(data, (ps, ps), interpolation=cv.INTER_CUBIC)
            scipy.misc.toimage(img*255, high=255, low=0, cmin=0, cmax=255).save(os.path.join(output_dir, head + '_Hour_' + str(cid) + '.png'))


if __name__ == '__main__':
    process_dir = '/content/typhoon-baseline/data/train/'
    output_dir = '/content/typhoon-baseline/img_data/train/'
    HEADS = ['A']#, 'B', 'C']

    print('Searching for missing data of train dataset...')
    filling(process_dir, HEADS)

    print('Transfering...')
    save2img(process_dir, output_dir, HEADS, 500)

    process_dir = '/content/typhoon-baseline/data/test/'
    output_dir = '/content/typhoon-baseline/img_data/test/'
    HEADS = ['U']#, 'V', 'W']#, 'X', 'Y', 'Z']

    print('Searching for missing data of test dataset...')
    filling(process_dir, HEADS)

    print('Transfering...')
    save2img(process_dir, output_dir, HEADS, 500)
