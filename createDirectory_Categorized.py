# -*- coding: UTF-8 -*-

import os
from glob import glob
import numpy as np
import shutil


def main():
    dirlist= np.array(sorted(glob('allData/*.jpg')))
    devicelist_all = np.array([os.path.split(i)[1].rsplit('_', 1)[0] for i in dirlist])
    devicelist = sorted(np.unique(devicelist_all)) 
    
    for device_name in devicelist:
        savedir = 'e-Data/Categorized/' + device_name
        if os.path.exists(savedir) == False:
            os.mkdir(savedir)
        for img_path in dirlist[devicelist_all == device_name]:
            shutil.copy2(img_path,savedir)

if __name__ == '__main__':
    main()