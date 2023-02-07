# -*- coding: UTF-8 -*-

import os
import csv
from glob import glob
from multiprocessing import cpu_count, Pool
from unittest import result
import numpy as np
import pandas as pd
from PIL import Image
import prnu

def main():
    device_name = 'all'   
    method = 'standard'    
    ffpath= 'allData/*.jpg'    
    natpath= 'allData/*.jpg'
    # PRNUが検出しやすい平面を撮影したデータ
    ff_dirlist  = np.array(sorted(glob(ffpath)))
    ff_imgs     = np.array([os.path.split(i)[1] for i in ff_dirlist])
    ff_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in ff_dirlist])
    nat_dirlist = np.array(sorted(glob(natpath)))
    nat_imgs    = np.array([os.path.split(i)[1] for i in nat_dirlist])
    nat_device = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in nat_dirlist])
      
    imgs= []
    for img_path in ff_dirlist:
        imgs+= [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512,3))]
    pool = Pool(cpu_count())
    k = pool.map(prnu.extract_single, imgs)
    pool.close()
    k = np.stack(k, 0)
    
    imgs= []
    for img_path in nat_dirlist:
        imgs+= [prnu.cut_ctr(np.asarray(Image.open(img_path)), (512, 512,3))]
    pool = Pool(cpu_count())
    w = pool.map(prnu.nat_extract_single, imgs)
    pool.close()

    w = np.stack(w, 0)
    gt_img = prnu.gt(ff_imgs, nat_imgs)
    gt_device = prnu.gt(ff_device, nat_device)
    
    # cc_aligned_rot = prnu.aligned_cc(k, w)['cc']
    # print(cc_aligned_rot)
    # # stats_cc = prnu.stats(cc_aligned_rot, gt)

    pce_rot = np.zeros((len(ff_imgs), len(nat_imgs)))
    for fingerprint_idx, fingerprint_k in enumerate(k):
        for natural_idx, natural_w in enumerate(w):
            cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
            pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
    stats_pce = prnu.stats(pce_rot, gt_device)
    result_path = 'img_'+device_name+'-'+method+'.csv'
    if os.path.exists(result_path) == False:
        f = open(result_path,'w') 
        f.close()
    DF = pd.DataFrame(pce_rot)
    DF.to_csv(result_path)
    outList = [stats_pce['auc'],stats_pce['eer'],stats_pce['th_eer']]
    with open(result_path,'a') as f :
         writer =csv.writer(f)
         writer.writerow(outList)
    df = pd.DataFrame(ff_imgs)
    df.to_csv('sortImgList.csv')
    print('AUC,EER,TH_EER on PCE {:.2f}{:.2f}{:.2f}'.format(stats_pce['auc'],stats_pce['eer'],stats_pce['th_eer']))
    
if __name__ == '__main__':
    main()
