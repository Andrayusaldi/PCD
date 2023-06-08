import cv2
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
from scipy.stats import skew, kurtosis

#ganti Apple dengan nama folder citra
images = os.listdir('./foto/data_testing/')
num_images = len(images)

data = []

for i in range(num_images):
    img = cv2.imread('./foto/data_testing/' + images[i])

    R = (img[:, :, 2])
    G = (img[:, :, 1])
    B = (img[:, :, 0])

    meanR = np.mean(R)
    meanG = np.mean(G)
    meanB = np.mean(B)

    stdR = np.std(R)
    stdG = np.std(G)
    stdB = np.std(B)

    # convert matrix to array
    Rflat = R.flatten()
    Gflat = G.flatten()
    Bflat = B.flatten()

    skewR = skew(Rflat)
    skewG = skew(Gflat)
    skewB = skew(Bflat)

    kurtR = kurtosis(Rflat)
    kurtG = kurtosis(Gflat)
    kurtB = kurtosis(Bflat)
    
    kelas = '?'

    data.append([meanR, meanG, meanB, stdR, stdG, stdB,
                 skewR, skewG, skewB, kurtR, kurtG, kurtB, kelas])

df = pd.DataFrame(data, columns=['meanR', 'meanG', 'meanB', 'stdR', 'stdG', 'stdB',
                                 'skewR', 'skewG', 'skewB', 'kurtR', 'kurtG', 'kurtB', 'kelas'])
print(df)
df.to_csv('data_testing.csv', index=False)

