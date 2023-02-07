# -*- coding: UTF-8 -*-

import os
import csv
from glob import glob
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from PIL import Image
import prnu


def main(device_name: str, method: str) -> list:
    print(device_name)
    print(method)
    ffpath = 'e-Data/Categorized/' + device_name + '/*.jpg'
    natpath = 'e-Data/Effected/'+device_name+'/'+method+'/*.jpg'
    ff_dirlist = np.array(sorted(glob(ffpath)))
    ff_imgs = np.array([os.path.split(i)[1] for i in ff_dirlist])
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0]
                         for i in ff_dirlist])
    nat_dirlist = np.array(sorted(glob(natpath)))
    nat_imgs = np.array([os.path.split(i)[1] for i in nat_dirlist])
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0]
                          for i in nat_dirlist])
    fingerprint_device = sorted(np.unique(ff_device))  # ユニークなデバイス

    imgs = []
    for img_path in ff_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]
    pool = Pool(cpu_count())
    k = pool.map(prnu.extract_single, imgs)
    pool.close()
    k = np.stack(k, 0)

    imgs = []
    for img_path in nat_dirlist:
        imgs += [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512, 3))]
    pool = Pool(cpu_count())
    w = pool.map(prnu.nat_extract_single, imgs)
    pool.close()

    w = np.stack(w, 0)
    gt_img = prnu.gt(ff_imgs, nat_imgs)
    gt_device = prnu.gt(ff_device, nat_device)
    print(gt_img)
    pce_rot = np.zeros((len(ff_imgs), len(nat_imgs)))
    print(pce_rot)
    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
    stats_pce = prnu.stats(pce_rot, gt_img)
    result_path = 'csv_img/'+device_name+'-'+method+'.csv'
    if os.path.exists(result_path) == False:
        f = open(result_path, 'w')
        f.close()
        DF = pd.DataFrame(pce_rot)
        DF.to_csv(result_path)
    outList = [method,stats_pce['auc'], stats_pce['eer'], stats_pce['th_eer']]
    with open(result_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(outList)
    # print('AUC,EER,TH_EER on PCE {:.2f}{:.2f}{:.2f}'.format(
    #     stats_pce['auc'], stats_pce['eer'], stats_pce['th_eer']))
    return outList
    


if __name__ == '__main__':
    main(device_name='google_pixel6a_0',method='GaussianBlur(3,1.0)')
