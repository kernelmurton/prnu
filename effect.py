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



def main():
    #変数設定
    device = 'apple_iPhone8_1'
    sizeNum = 3
    sigmax = float(1.3)
    gamma = float(0.5)
    k_size = (sizeNum,sizeNum)
    #特定のデバイスのフォルダを作成
    dirpath = 'e-Data/'+ device + '/*.jpg'
    effectedDirpath = 'e-Data/effected/'+ device 
    if os.path.exists(effectedDirpath) == False:
        os.mkdir(effectedDirpath)
    #画像加工の手法
    e_methods = ['blur','GaussianBlur','medianBlur','GammaCorrection']
    #配列設定
    patharray = []
    dst_index = []
    savepath_array =[]
    #画像加工の手法ごとにフォルダを作成
    for method in e_methods:
        if method == 'GaussianBlur':            
            path = effectedDirpath + '/'+ method +'('+ str(sizeNum) +','+str(sigmax)+')'
            patharray.append(path)        
        elif method == 'GammaCorrection':
            path = effectedDirpath + '/'+ method + str(gamma)  
            patharray.append(path)
        else:
            path = effectedDirpath + '/'+ method + str(sizeNum)  
            patharray.append(path)
        dst_index.append(str('dst_'+method))
        if os.path.exists(path) == False:os.mkdir(path)
    for path in patharray:
        tmp = path + '/' + device + '_'
        savepath_array.append(tmp)
    #画像加工
    dirlist = np.array(sorted(glob(dirpath)))
    count = 0
    for img_path in dirlist :
        im = cv2.imread(img_path)
        gray = prnu.rgb2gray(im)
        gray_3ch = np.stack((gray,)*3, -1)
        dict = {
            #ぼかし
            'dst_blur' : cv2.blur(gray_3ch,ksize=k_size),#平均化フィルタ
            'dst_GaussianBlur' : cv2.GaussianBlur(gray_3ch, ksize=k_size, sigmaX = sigmax),#ガウシアンフィルタ
            'dst_medianBlur' : cv2.medianBlur(gray_3ch,ksize=sizeNum),#メディアンフィルタ
            #明度変換
            'dst_GammaCorrection' : functions.GammaCorrection(gray_3ch,gamma)
            #RGB変換
            #モルフォロジー変換
        }
        for i in range(len(savepath_array)):
            savepath = savepath_array[i] +str(count)+'.jpg'
            if os.path.exists(savepath) == False:
                cv2.imwrite(savepath,dict[dst_index[i]])
        count = count + 1

        
if __name__ == '__main__':
    main()
