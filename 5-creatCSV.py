import os
import csv
from glob import glob
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
from PIL import Image


def main():
    editpath = 'csv/*.csv'
    dirlist = np.array(sorted(glob(editpath)))
    devicelist = np.array([os.path.split(i)[1] for i in dirlist])
    resultpath = 'graph.csv'

    for device in devicelist:
        dir = dirlist[devicelist == device][0]
        df = pd.read_csv(dir)
        original = df.iloc[35, 3]
        thresholdlist = [device]
        for i in range(0, 36):
            value = df.iloc[i, 4] / original * 100
            thresholdlist.append(value)
        with open(resultpath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(thresholdlist)


if __name__ == '__main__':
    main()
