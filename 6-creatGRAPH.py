from multiprocessing import Pool, cpu_count
import datetime
import numpy as np
import pandas as pd
import pywt
from numpy.fft import fft2, ifft2
from scipy.ndimage import filters
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    resultpath = 'graph.csv'
    df = pd.read_csv(resultpath)
    methodlist = list(df.columns)
    averagelist = []
    for i in range(1, 36):
        value = df.iloc[49, i]
        averagelist.append(value)
    tmp = np.array(methodlist)
    methodlist = np.delete(tmp, 0)
    averagelist = np.array(averagelist)

    # print(methodlist)
    print(averagelist)
    print(len(methodlist))
    print(len(averagelist))
    x = methodlist.tolist()
    y = averagelist.tolist()
    values = x
    print(x, y)
    plt.plot(x, y, marker="o")
    plt.xlabel("Method Name")
    plt.ylabel("%")
    plt.xticks(x, values, rotation=90)
    plt.show()
    # x = [0.01, 0.1, 1, 10, 100]
    # y = [2, 1, 6, 4, 8]
    # values = ['A', 'B', 'C', 'D', 'E', 'F']
    # plt.plot(x, y, marker="o")
    # plt.xlabel("X-Axis")
    # plt.ylabel("Y-Axis")
    # plt.title("Set X labels in Matplotlib Plot")
    # plt.xticks(x, values)
    # plt.show()


if __name__ == '__main__':
    main()
