# -*- coding: UTF-8 -*-

import os
import cv2
from glob import glob
from multiprocessing import cpu_count, Pool
import numpy as np
from PIL import Image
import functions



def main():
    device = 'apple_iPhone8_0'
    folderpath = '../effectedData/'+ device + '/*.jpg'
    dirlist = np.array(sorted(glob(folderpath)))
    print(dirlist)
if __name__ == '__main__':
    main()
