import os
from PIL import Image
from collections import defaultdict
from flask import Flask, render_template, request

app = Flask(__name__)

# 图片文件夹路径
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

# 设置路由
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gallery')
def gallery():

    return render_template('gallery.html', image_names=image_names)

@app.route('/gallery/<image_name>')
def image(image_name):
    return render_template('image.html', image_name=image_name)

# 加载图像
@app.before_request
def before_request():
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            with Image.open(os.path.join(IMAGE_FOLDER, filename)) as img:
                img.load()

if __name__ == '__main__':
    # app.run(debug=True)
    print(load_all_pic())

