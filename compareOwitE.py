# -*- coding: UTF-8 -*-

import os
from glob import glob
from multiprocessing import cpu_count, Pool
from unittest import result
import numpy as np
import pandas as pd
from PIL import Image
import prnu

def main():
    device_name = 'apple_iPhone8_1'
    method = 'GaussianBlur(5,2.0)'
    ffpath= 'e-Data/effected/'+device_name+'/'+method+'/*.jpg'
    natpath = 'allData/*.jpg'
    # PRNUが検出しやすい平面を撮影したデータ
    ff_dirlist = np.array(sorted(glob(ffpath)))
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    # 風景を撮影したデータ
    nat_dirlist = np.array(sorted(glob(natpath)))
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])
    # 今回検出するデバイスを策定
    fingerprint_device = sorted(np.unique(ff_device))#ユニークなデバイス
    k = []
    for device in fingerprint_device:
        imgs = []
        for img_path in ff_dirlist[ff_device == device]:
            im = Image.open(img_path)
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
        imgs+= [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512,3))]
    #Python で関数の実行を並列化する
    pool = Pool(cpu_count())
    w = pool.map(prnu.extract_single, imgs)
    pool.close()

    w = np.stack(w, 0)
    gt = prnu.gt(fingerprint_device, nat_device)
    
    cc_aligned_rot = prnu.aligned_cc(k, w)['cc']
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    pce_rot = np.zeros((len(fingerprint_device), len(nat_device)))
    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
    stats_pce = prnu.stats(pce_rot, gt)
    result_path = device_name+'-'+method+'.csv'
    if os.path.exists(result_path) == False:
        f = open(result_path,'w')
        f.close
    DF = pd.DataFrame(pce_rot)
    DF.to_csv(result_path)
if __name__ == '__main__':
    main()
