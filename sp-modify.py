# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
Politecnico di Milano 2018
"""

import os
from glob import glob
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image
import prnu

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
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.t
    Check the detection performance obtained with cross-correlation and PCE
    :return:23.0258880561172	22.2813862555823	33.2573042222027	22.7495359261404

    """
    # PRNUが検出しやすい平面を撮影したデータ
    ff_dirlist = np.array(sorted(glob('data-modify/ff-jpg/*.jpg')))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    # 風景を撮影したデータ
    nat_dirlist = np.array(sorted(glob('data-modify/nat-jpg/*.jpg')))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])

    print('Computing fingerprints')
    # 今回検出するデバイスを策定
    fingerprint_device = sorted(np.unique(ff_device))#ユニークなデバイス
    k = []
    for device in fingerprint_device:
        print('The unique device is {}'.format(device))
        imgs = []
        for img_path in ff_dirlist[ff_device == device]:
            im = Image.open(img_path)
            exifinfo = im._getexif()
            orientation=exifinfo.get(0x112,1)
            if orientation != 1:
                print('{} is not true position'.format(img_path))
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue
            #データのトリミングを行う
            im_cut = prnu.cut_ctr(im_arr, (512, 512, 3))
            imgs += [im_cut] # +=で配列追加
        k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]#prnuを抽出して配列に追加する
    k = np.stack(k, 0)

    #風景写真について扱う
    imgs= []
    for img_path in nat_dirlist:
        im=Image.open(img_path)
        exifinfo=im._getexif()
        orientation=exifinfo.get(0x112,1)
        im_rotate=rotateImage(im,orientation)[0]
        im_rotate.save(img_path,exif = im_rotate.info['exif'],quality=95)
        imgs+= [prnu.cut_ctr(np.asarray(im), (512, 512, 3))]
    #Python で関数の実行を並列化する
    pool = Pool(cpu_count())
    w = pool.map(prnu.extract_single, imgs)
    pool.close()
    w = np.stack(w, 0)
    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']

    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    print('Computing PCE')
    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))
    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
            print('PCE value:{:.3f}'.format(prnu.pce(cc2d)['pce']))
    print('Computing statistics on PCE')
    stats_pce = prnu.stats(pce_rot, gt)
    print('AUC on CC {:.2f}'.format(stats_cc['auc']))
    print('AUC on PCE {:.2f}'.format(stats_pce['auc']))

if __name__ == '__main__':
    main()
