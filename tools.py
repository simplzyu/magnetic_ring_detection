# encoding: utf-8
"""
@author: guozhenyu
@contact: guozhenyu@pku.edu.cn

@version: 1.0
@file: run.py
@time: 2023/9/10 4:59 PM
"""

import os
from collections import defaultdict
from pathlib import Path
import cv2


def load_all_pic(root):
    nam2pig2num = defaultdict(dict)
    for fnam in os.listdir(root):
        # print(fnam)
        parts = fnam.split('.')[0].split('_')
        # print(parts)
        # print(fnam)
        if '.DS_Store' == fnam: continue
        if parts[1] not in nam2pig2num[parts[0]]:
            nam2pig2num[parts[0]][parts[1]] = []
        nam2pig2num[parts[0]][parts[1]].append(parts[2])

    plist, plist2nums = [], {}
    for vnam, detail in sorted(nam2pig2num.items(), key=lambda x:int(x[0][1:])):
        # print('==',vnam, detail)
        for pnam, nums in sorted(detail.items(), key=lambda x:int(x[0])):
            plist.append((vnam, pnam))
            plist2nums[(vnam, pnam)] = sorted(nums, key=lambda x:int(x))

    return nam2pig2num, plist, plist2nums

def load_pic(root):
    plist = []
    for dir in sorted(os.listdir(root), key=lambda x:int(x[1:])):
        if '.DS_Store' == dir: continue
        fnames = os.listdir(os.path.join(root, dir))
        fnames = sorted(fnames, key=lambda x:int(x.split('.')[0]))
        for fnam in fnames:
            if '.DS_Store' == fnam: continue
            plist.append(dir + '_' + fnam.split('.')[0])
    return plist

def reshape_image(img, shape=(20, 40)):
    return cv2.resize(img,shape)

def image_spilt(sub_img, x_p=0, y_p=0):
    x_len, y_len = len(sub_img), len(sub_img[0])
    dig1 = reshape_image(sub_img[int(0.2621*x_len)+x_p:int(0.9677*x_len)+x_p, int(0.1701*y_len)+y_p:int(0.3403*y_len)]+y_p)
    dig2 = reshape_image(sub_img[int(0.2621*x_len)+x_p:int(0.9677*x_len)+x_p, int(0.3403*y_len)+y_p:int(0.5198*y_len)]+y_p)
    dig3 = reshape_image(sub_img[int(0.2621*x_len)+x_p:int(0.9677*x_len)+x_p, int(0.5198*y_len)+y_p:int(0.6900*y_len)]+y_p)
    dig4 = reshape_image(sub_img[int(0.2621*x_len)+x_p:int(0.9677*x_len)+x_p, int(0.6994*y_len)+y_p:int(0.8696*y_len)]+y_p)
    return [dig1, dig2, dig3, dig4]


def valid_pic(img):
    sub_img = img[int(0.2456*len(img)):, int(0.0820*img.shape[1]):int(0.8525*img.shape[1])]
    return sub_img