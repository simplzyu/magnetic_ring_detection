# encoding: utf-8
"""
@author: guozhenyu 
@contact: guozhenyu@pku.edu.cn

@version: 1.0
@file: test.py.py
@time: 2023/9/10 4:38 PM
"""

import os
from collections import defaultdict
IMAGE_FOLDER = os.path.join('static', 'images')


def load_all_pic():
    nam2pig2num = defaultdict(dict)
    for fnam in os.listdir(IMAGE_FOLDER):
        print(fnam)
        parts = fnam.split('.')[0].split('_')
        print(parts)
        if parts[1] not in nam2pig2num[parts[0]]:
            nam2pig2num[parts[0]][parts[1]] = []
        nam2pig2num[parts[0]][parts[1]].append(parts[2])

    plist = []
    for vnam, detail in sorted(nam2pig2num.items(), key=lambda x:x[0]):
        print('==',vnam, detail)
        for pnam, info in sorted(detail.items(), key=lambda x:int(x[0])):
            plist.append((vnam, pnam))

    pig2next, pig2pre = {}, {}
    for i, pig in enumerate(plist):
        if i+1 == len(plist):
            pig2next[pig] = plist[0]
        else:
            pig2next[pig] = plist[i+1]
        pig2pre[pig] = plist[i-1]
    return nam2pig2num, pig2next, pig2pre

nam2pig2num, pig2next, pig2pre = load_all_pic()
print(pig2next)
print(pig2pre)