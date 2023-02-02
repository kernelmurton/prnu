# -*- coding: UTF-8 -*-

import os
import cv2
from glob import glob
from multiprocessing import cpu_count, Pool
import numpy as np
from PIL import Image

class ArgumentError(Exception):
    pass

def GammaCorrection (im:np.ndarray,gamma:float) -> np.ndarray:
    #明度の変化をガンマ補正で行う
    imax = im.max()
    gamma_img = imax * (im/imax)**(1/gamma)
    return gamma_img



