import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import ml_metrics as metrics
import scipy.spatial.distance as dist


num_query = 10
num_db = 287

with open('./qsd1_w1/gt_corresps.pkl', 'rb') as f:
    x = pickle.load(f)
    print(x)

def norm_histo(img):
    """Devuelve histograma de cada canal para range (0,256)"""
    histo_1 = cv2.calcHist(img , [0] ,None, [48] , [0, 256])
    #cv2.normalize(histo_1,histo_1,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    histo_2 = cv2.calcHist(img, [1], None, [48], [0, 256])
    #cv2.normalize(histo_2, histo_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    histo_3 = cv2.calcHist(img, [2], None, [48], [0, 256])
    #cv2.normalize(histo_3, histo_3, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    #histo = histo_1+histo_2+histo_3
    return histo_1,histo_2,histo_3

def get_k_values(extreme:str,k:int,similarity_val:list):
    """Coge los ultimos o primeros k valores de la lista
    extreme: 'top' or 'bottom' para que coja los primero valores o los ultimos"""
    sort_list =[]
    for j in sorted(similarity_val):
        sort_list.append(j)
    index = []
    if extreme=='top':
        for i in range(k):
            index.append(similarity_val.index(sort_list[-i]))

    if extreme == 'bottom':
        for i in range(k):
            index.append(similarity_val.index(sort_list[i]))

    return index

#lista de indices de la database predecidos
positions = []

for i in range(num_query):

    if i < 10:
        zero_0 = '00'
    else:
        zero_0 = '0'
    query_file = './qsd1_w1/00' + zero_0 + str(i) + '.jpg'

    query_img = cv2.imread(query_file)

    #query_hsv = cv2.cvtColor(query_img,cv2.COLOR_RGB2HSV)
    #query_cielab = cv2.cvtColor(query_img,cv2.COLOR_RGB2Lab)

    #histo_queryhsv = norm_histo(query_hsv)
    #histo_querylab = norm_histo(query_cielab)
    histo_queryrgb= norm_histo(query_img)

    #lista en que guarda distancias de query con cada imagen de la database
    rgb = []

    for j in range(num_db):
        if j <= 10:
            zero_1 = '00'
        if 10<=j and j<100:
            zero_1 = '0'
        if j>=100:
            zero_1 = ''

        db_file = './BBDD/bbdd_00' + zero_1 + str(j) + '.jpg'

        db_img = cv2.imread(db_file)

        #db_hsv = cv2.cvtColor(db_img, cv2.COLOR_RGB2HSV)
        #db_cielab = cv2.cvtColor(db_img, cv2.COLOR_RGB2Lab)

        #histo_dbhsv = norm_histo(db_hsv)
        #histo_dblab = norm_histo(db_cielab)
        histo_dbrgb = norm_histo(db_img)

        #metric_hsv = cv2.compareHist(histo_queryhsv, histo_dbhsv, cv2.HISTCMP_CHISQR)
        #metric_lab = cv2.compareHist(histo_querylab, histo_dblab, cv2.HISTCMP_CHISQR)
        distance = 0
        for j,k in zip(histo_queryrgb,histo_dbrgb):
            distance = dist.euclidean(j,k)
            distance += distance
        #metric_rgb = cv2.compareHist(histo_queryrgb, histo_dbrgb, cv2.HISTCMP_CHISQR)
        #mean_i = min(metric_lab,metric_hsv,metric_rgb)
        rgb.append(distance/3)
        #hsv.append(metric_hsv)
        #lab.append(metric_lab)
        #mean.append(mean_i)

    position_1 = get_k_values('bottom',5,rgb)
    #position_2 = get_k_values('bottom',5,hsv)
    #position_3 = get_k_values('bottom',5,lab)
    #position_4 = get_k_values('bottom', 5, mean)

    positions.append(position_1)
    print(positions)

mapk = metrics.mapk(x,positions,5)
print(mapk)






