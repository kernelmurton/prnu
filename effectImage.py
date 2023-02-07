# -*- coding: UTF-8 -*-

from itertools import count
import os
import cv2
from glob import glob
from multiprocessing import cpu_count, Pool
import numpy as np
from PIL import Image
import prnu
import functions


def main(device: str):
    # 変数設定
    sizeNums = [3, 5]
    sigmax = float(1.0)
    gamma = float(0.25)
    # set Path
    getPath = 'e-Data/Categorized/' + device + '/*.jpg'
    effectedDirpath = 'e-Data/Effected/' + device
    if os.path.exists(effectedDirpath) == False:
        os.mkdir(effectedDirpath)
    # 配列設定
    effectName_path = []
    dict_index = []
    saveImpath_array = []
    dict = {}
    # 画像加工の手法
    e_methods = ['blur', 'GaussianBlur', 'medianBlur', 'GammaCorrection','morphologyOP','morphologyCL']
    # 画像加工の手法ごとにフォルダを作成
    for method in e_methods:
        if method == 'GaussianBlur':
            for sizeNum in sizeNums:
                for i in range(3):
                    path = effectedDirpath + '/' + method + '(' + str(sizeNum) + ','+str(sigmax*2**i)+')'
                    effectName_path.append(path)
                    dict_index.append(
                        method+'(' + str(sizeNum) + ','+str(sigmax*2**i)+')')
                    if os.path.exists(path) == False:
                        os.mkdir(path)
        elif method == 'GammaCorrection':
            for i in range(3):
                path = effectedDirpath + '/' + method + str(gamma*2**i)
                effectName_path.append(path)
                dict_index.append(method+str(gamma*2**i))
                if os.path.exists(path) == False:
                    os.mkdir(path)
        else:
            for sizeNum in sizeNums:
                path = effectedDirpath + '/' + method + str(sizeNum)
                effectName_path.append(path)
                dict_index.append(method+str(sizeNum))
                if os.path.exists(path) == False:
                    os.mkdir(path)

    for path in effectName_path:
        tmp = path + '/' + device + '_'
        saveImpath_array.append(tmp)
    
    # 画像加工
    dirlist = np.array(sorted(glob(getPath)))
    count = 0
    for img_path in dirlist:
        im = cv2.imread(img_path)
        gray = prnu.rgb2gray(im)
        gray_3ch = np.stack((gray,)*3, -1)
        # make dictionary
        # when you add method, add elif
        for method in e_methods:
            if method == 'GaussianBlur':
                for sizeNum in sizeNums:
                    for i in range(3):
                        k_size = (sizeNum, sizeNum)
                        dst_key = method + \
                            '(' + str(sizeNum) + ','+str(sigmax*2**i)+')'
                        dst_value = cv2.GaussianBlur(
                            gray_3ch, ksize=k_size, sigmaX=sigmax*2**i)
                        dict[dst_key] = dst_value
            elif method == 'GammaCorrection':
                for i in range(3):
                    dst_key = method+str(gamma*2**i)
                    dst_value = functions.GammaCorrection(gray_3ch, gamma*2**i)
                    dict[dst_key] = dst_value
            elif method == 'blur':
                for sizeNum in sizeNums:
                    k_size = (sizeNum, sizeNum)
                    dst_key = method+str(sizeNum)
                    dst_value = cv2.blur(gray_3ch, ksize=k_size)
                    dict[dst_key] = dst_value
            elif method == 'medianBlur':
                for sizeNum in sizeNums:
                    dst_key = method+str(sizeNum)
                    dst_value = cv2.medianBlur(gray_3ch, ksize=sizeNum)
                    dict[dst_key] = dst_value
            elif method == 'morphologyOP':
                for sizeNum in sizeNums:
                    k_size = (sizeNum, sizeNum)
                    dst_key = method+str(sizeNum)
                    dst_value = cv2.morphologyEx(gray_3ch, cv2.MORPH_OPEN, kernel=k_size)
                    dict[dst_key] = dst_value
            elif method == 'morphologyCL':
                for sizeNum in sizeNums:
                    k_size = (sizeNum, sizeNum)
                    dst_key = method+str(sizeNum)
                    dst_value = cv2.morphologyEx(gray_3ch, cv2.MORPH_CLOSE, kernel=k_size)
                    dict[dst_key] = dst_value

        for i in range(len(saveImpath_array)):
            saveImpath = saveImpath_array[i] + str(count)+'.jpg'
            if os.path.exists(saveImpath) == False:
                cv2.imwrite(saveImpath, dict[dict_index[i]])
        count = count + 1


if __name__ == '__main__':
    editpath = 'e-Data/Categorized'
    files = sorted(os.listdir(editpath))
    device_list = [f for f in files if os.path.isdir(
        os.path.join(editpath, f))]
    for device in device_list:
        main(device=device)
