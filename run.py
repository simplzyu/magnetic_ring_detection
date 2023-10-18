# encoding: utf-8
"""
@author: guozhenyu
@contact: guozhenyu@pku.edu.cn

@version: 1.0
@file: run.py
@time: 2023/9/10 4:59 PM
"""

import os
import imageio
import pandas as pd
from tools import *
from collections import defaultdict
from config import conn
from flask import Flask, render_template, request, url_for, redirect
from cnocr import CnOcr
from datetime import datetime


# 图片文件夹路径
app = Flask(__name__)
# 图片文件夹路径
IMAGE_FOLDER = os.path.join('static', 'images')
ocr = CnOcr(rec_model_name='en_number_mobile_v2.0')

def load_local_labels():
    nam2label = {}
    sheet = ['s3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
    for sn in sheet:
        label_df = pd.read_excel('/Users/simple-hz/Documents/gzy/视频处理/resized_video/labels.xlsx',
                                 sheet_name=sn).dropna()
        for idx, row in label_df.iterrows():
            nam = sn + '_' + str(int(row['x']))
            parts = str(row['y']).split('.')
            zs = parts[0]
            zs = 'x' * (3 - len(zs)) + zs
            xs = parts[1]
            new_num = f'{zs}.{xs}'
            nam2label[nam] = new_num
    return nam2label


def load_db_label_now(pic_nam, tb_nam):
    all_labels = pd.read_sql(f"select label from {tb_nam} where pic_nam = '{pic_nam}' order by date desc limit 1",
                             con=conn)
    if len(all_labels) == 0:
        return ''
    else:
        return all_labels.head(1).values[0][0]


def load_db_label(tb_nam):
    all_labels = pd.read_sql(f"select * from {tb_nam}", con=conn)
    nam2dbbabel = {}
    all_labels = all_labels.groupby(by='pic_nam').apply(lambda x: x.sort_values(by='date', ascending=False).iloc[0])
    for idx, row in all_labels.iterrows():
        if len(row['label']) > 5:
            print('error', row['pic_nam'], row['label'])
        else:
            nam2dbbabel[row['pic_nam']] = row['label']
    return nam2dbbabel


def to_db(nam, label, tb_nam):
    df = pd.DataFrame([[nam, label, 0]], columns=['pic_nam', 'label', 'is_pred'])
    df.to_sql(tb_nam, con=conn, index=False, if_exists='append')


@app.route('/gallery', methods=['GET', 'POST'])
def gallery():
    nam2pig2num, plist, plist2nums = load_all_pic(IMAGE_FOLDER)

    # print(plist)
    if request.method == 'POST':
        pic_nam = request.form['pic_nam']
        label = request.form['label']
        if len(label) > 0:
            to_db(pic_nam, label, 'label')
        current_image = request.form['current_image']
        if request.form['submit'] == 'Next':
            current_image = (int(current_image) + 1) % len(plist)
        elif request.form['submit'] == 'Previous':
            current_image = (int(current_image) - 1) % len(plist)
    else:
        pic_nam = request.args.get('pic_nam')
        try:
            vnam, pnam = pic_nam.split('_')
            current_image = plist.index((vnam, pnam))
        except:
            current_image = 0
        if current_image is None:
            current_image = 0
        else:
            current_image = int(current_image)

    pic = '_'.join(plist[current_image])
    label1 = load_db_label_now(pic, 'label')
    label2 = nam2locallabel.get(pic)
    label3 = nam2dblabel.get(pic)
    if label1 is not None:
        label = label1
    elif label3 is not None:
        label = label3
    elif label2 is not None:
        label = label2
    else:
        label = ''

    cnocr_labels = []
    images = [f'{pic}_{num}.jpg' for num in plist2nums[plist[current_image]]]
    for img in images:
        out = ocr.ocr(os.path.join('./static/images', img))
        print(out)
        if len(out) != 0:
            cnocr_labels.append(out[0]['text'])
        else:
            cnocr_labels.append('x')
    cnocr_label = ''.join(cnocr_labels)
    return render_template('gallery2.html'
                           , num_nam=pic
                           , images=images
                           , label=label
                           , cnocr_label=cnocr_label
                           , current_image=current_image)


@app.route('/gallery_new', methods=['GET', 'POST'])
def gallery_new():
    plist = app.config.get('plist')
    video_nam = app.config.get('video_nam')
    # print(plist)
    if request.method == 'POST':
        pic_nam = request.form['pic_nam']
        label = request.form['label']
        if len(label) > 0:
            to_db(video_nam + '_' + pic_nam, label, 'label')
        current_image = request.form['current_image']
        if request.form['submit'] == 'Next':
            current_image = (int(current_image) + 1) % len(plist)
        elif request.form['submit'] == 'Previous':
            current_image = (int(current_image) - 1) % len(plist)
    else:
        pic_nam = request.args.get('pic_nam')
        try:
            current_image = plist.index(pic_nam)
        except:
            current_image = 0
        if current_image is None:
            current_image = 0
        else:
            current_image = int(current_image)
    pic_nam = plist[current_image]
    label = load_db_label_now(video_nam + '_' + pic_nam, 'label')

    img = valid_pic(cv2.imread(os.path.join('static/images/', video_nam, 'pic', pic_nam + '.jpg')))
    cnocr_label = ''
    out = ocr.ocr(img)
    if len(out) != 0:
        cnocr_label = out[0]['text']
    if label == '':
        label = cnocr_label

    return render_template('gallery_new.html'
                           , video_nam=app.config.get('video_nam')
                           , pic_nam=pic_nam
                           , images=[pic_nam + '_' + str(i) for i in range(4)]
                           , label=label
                           , cnocr_label=cnocr_label
                           , current_image=current_image)


@app.route('/gallery_all', methods=['GET', 'POST'])
def gallery_all():
    plist = load_pic('static/images3')
    pre_label = ''
    # print(plist)
    if request.method == 'POST':
        pic_nam = request.form['pic_nam']
        pre_label = label = request.form['label']
        if len(label) > 0:
            to_db(pic_nam, label, 'label2')
        current_image = request.form['current_image']
        if request.form['submit'] == 'Next':
            current_image = (int(current_image) + 1) % len(plist)
        elif request.form['submit'] == 'Previous':
            current_image = (int(current_image) - 1) % len(plist)
    else:
        pic_nam = request.args.get('pic_nam')
        try:
            current_image = plist.index(pic_nam)
        except:
            current_image = 0
        if current_image is None:
            current_image = 0
        else:
            current_image = int(current_image)

    pic_nam = plist[current_image]
    # label = nam2dblabel.get(pic_nam, '')
    label = load_db_label_now(pic_nam, 'label2')
    if label == '':
        label = pre_label
    dir, pic_nam = pic_nam.split('_')
    return render_template('gallery3.html', dir=dir, pic_nam=pic_nam, label=label, current_image=current_image)


@app.route('/upload', methods=['POST'])
def upload():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file = request.files['file']
    file_nam = file.filename.split('.')[0]
    file_nam = f'video_{timestamp}_{file_nam}'
    upload_path = os.path.join(os.path.dirname(__file__), 'static/uploads')
    file.save(os.path.join(upload_path, f'video_{timestamp}_{file.filename}'))
    # 读取视频
    video = cv2.VideoCapture(os.path.join(upload_path, f'video_{timestamp}_{file.filename}'))
    # 读取第一帧
    ret, frame = video.read()
    height, width = frame.shape[0], frame.shape[1]
    cv2.imwrite(os.path.join(upload_path, f'{file_nam}.jpg')
                , frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    print(width, height)
    return render_template('web.html',
                           pic_nam=file_nam,
                           width=width,
                           height=height)


@app.route('/web', methods=['GET'])
def web():
    return render_template('web.html')


@app.route('/loc_submit', methods=['POST'])
def submit():
    video_nam = request.form.get('video_nam')
    sav_dir = f"static/images/{video_nam}"
    sav_pic_dir = os.path.join(sav_dir, 'pic')
    sav_num_dir = os.path.join(sav_dir, 'num')
    os.mkdir(sav_dir)
    os.mkdir(sav_pic_dir)
    os.mkdir(sav_num_dir)
    start_x, start_y = eval(request.form.get('start'))
    end_x, end_y = eval(request.form.get('end'))

    cap = cv2.VideoCapture(f"static/uploads/{video_nam}.mp4")
    flag = cap.isOpened()
    if not flag:
        print("\033[31mLine 65 error\033[31m: open" + video_nam + "error!")

    frame_count = 0  # 给每一帧标号
    fnames = []
    while True:
        frame_count += 1
        if frame_count % 10 != 0: continue
        flag, frame = cap.read()
        if not flag:  # 如果已经读取到最后一帧则退出
            break
        imgs = image_spilt(frame[start_y:end_y, start_x:end_x])
        fnames.append(str(frame_count))
        cv2.imwrite(os.path.join(sav_pic_dir, f"{str(frame_count)}.jpg"), frame[start_y:end_y, start_x:end_x])
        for j, img in enumerate(imgs):
            # fnames.append()
            cv2.imwrite(os.path.join(sav_num_dir, f"{str(frame_count)}_{j}.jpg"), img)
    cap.release()
    img = valid_pic(cv2.imread(os.path.join(sav_pic_dir, fnames[0] + '.jpg')))

    cnocr_label = ''
    out = ocr.ocr(img)
    if len(out) != 0:
        cnocr_label = out[0]['text']
    app.config['plist'] = fnames
    app.config['video_nam'] = video_nam
    return render_template('gallery_new.html'
                           , video_nam=video_nam
                           , pic_nam=fnames[0]
                           , images=[fnames[0] + '_' + str(i) for i in range(4)]
                           , label=''
                           , cnocr_label=cnocr_label
                           , current_image=0)


@app.route('/')
def index():
    images = os.listdir('static/uploads')
    all_vidoes = set()
    for image in images:
        if image == '.DS_Store': continue
        image = image.split('.')[0]
        all_vidoes.add(image)
    all_vidoes = sorted(all_vidoes, key=lambda x: int(x.split('_')[1]))

    return render_template('index.html', images=all_vidoes)


@app.route('/show_video')
def show_video():
    print(request.method)
    video_nam = request.args.get('video_nam')
    plist = sorted([fnam.split('.')[0] for fnam in os.listdir(f'static/images/{video_nam}/pic')], key=lambda x:int(x))
    app.config['plist'] = plist
    app.config['video_nam'] = video_nam
    return redirect(url_for('gallery_new', pic_nam=plist[0]))


if __name__ == '__main__':
    nam2locallabel = load_local_labels()
    nam2dblabel = load_db_label('label')
    # 整个数字作为label
    # nam2dblabel2 = load_db_label('label2')
    app.run(debug=True)
