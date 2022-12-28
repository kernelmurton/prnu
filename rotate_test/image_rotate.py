import os
from glob import glob
from multiprocessing import cpu_count, Pool
import datetime
from statistics import quantiles

import numpy as np
from PIL import Image,ImageFilter

def rotateImage(img, orientation):
    """
    画像ファイルをOrientationの値に応じて回転させる
    """
    rotate_flag = 1
    #orientationの値に応じて画像を回転させる
    if orientation == 1:
        img_rotate = img
        rotate_flag = 0
    elif orientation == 2:
        #左右反転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        #180度回転
        img_rotate = img.transpose(Image.ROTATE_180)
    elif orientation == 4:
        #上下反転
        img_rotate = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        #左右反転して90度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    elif orientation == 6:
        #270度回転
        img_rotate = img.transpose(Image.ROTATE_270)
    elif orientation == 7:
        #左右反転して270度回転
        img_rotate = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
    elif orientation == 8:
        #90度回転
        img_rotate = img.transpose(Image.ROTATE_90)
    else:
        rotate_flag =0
    return img_rotate,rotate_flag

def main():
    dt_now = datetime.datetime.now()
    im = Image.open('test.jpg')
    exifinfo=im._getexif()
    orientation=exifinfo.get(0x112,1)
    if orientation == 6:
        print('read scusessed')
    else:
        print('read false')
    im_rotate=im.rotate(-270,expand=True)
    path = 'rotateImage'+str(dt_now.isoformat)+'.jpg'
    im_rotate.save(path,exif = im_rotate.info['exif'],quality=95)

if __name__ == '__main__':
    main()
