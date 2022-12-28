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
import csv

import numpy as np
from PIL import Image

import prnu


def main():
    """
    Main example script. Load a subset of flatfield and natural images from Dresden.
    For each device compute the fingerprint from all the flatfield images.
    For each natural image compute the noise residual.t
    Check the detection performance obtained with cross-correlation and PCE
    :return:
    """
    # PRNUが検出しやすい平面を撮影したデータ
    dirlist = np.array(sorted(glob('airisuzuki_official_uf/*.jpg')))
    numlist = np.array([os.path.split(i)[1].rsplit('_', 1)[1] for i in dirlist])
    
    print('Computing fingerprints')
    # 今回検出するデバイスを策定
    k = []
    for device in numlist:
        # print('The number of pictures is {}'.format(device))
        imgs = []
        for img_path in dirlist[numlist == device]:
            im = Image.open(img_path)
            im_arr = np.asarray(im)
            if im_arr.dtype != np.uint8:
                print('Error while reading image: {}'.format(img_path))
                continue
            if im_arr.ndim != 3:
                print('Image is not RGB: {}'.format(img_path))
                continue
            #データのトリミングを行う
            im_cut = prnu.cut_ctr(im_arr, (300, 300, 3))
            imgs += [im_cut] # +=で配列追加
        k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]#prnuを抽出して配列に追加する
    k = np.stack(k, 0)
    # print(k)
    print('Computing residuals')
    #風景写真について扱う
    imgs = []
    for img_path in dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (300, 300, 3))]
    #Python で関数の実行を並列化する
    pool = Pool(cpu_count())
    print(cpu_count())
    w = pool.map(prnu.extract_single, imgs)
    pool.close()
    w = np.stack(w, 0)
    # Computing Ground Truth
    # gt = prnu.gt(fingerprint_device, nat_device)

    print('Computing cross correlation')
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']

    # print('Computing statistics cross correlation')
    # stats_cc = prnu.stats(cc_aligned_rot, gt)

    print('Computing PCE')
    pce_rot = np.zeros((len(numlist), len(numlist)))

    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            if fingerprint_idx < natural_idx:
                cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
                pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
                # print('PCE value:{:.3f}'.format(prnu.pce(cc2d)['pce']))

    #ここでCSVファイルに出力する
    f = open('pcelist_airisuzuki.csv','w',newline='')
    writer=csv.writer(f)
    writer.writerows(pce_rot)
    f.close()



if __name__ == '__main__':
    main()
