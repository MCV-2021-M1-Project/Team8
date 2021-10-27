from tqdm import tqdm
from typing import List

import cv2
import numpy as np

"""
Divides image into symetrically blocks (n_rows x n_cols blocks).
Returns a List with all the blocks.
"""
def image_to_windows(image: np.ndarray, n_rows: int, n_cols: int) -> List[np.ndarray]:
    # Estimate block size
    x, y = image.shape[0]/n_rows, image.shape[1]/n_cols
    x, y = int(x), int(y)

    # Split image into blocks
    windows = []
    for i in range(n_rows):
       for j in range(n_cols):
           # Split
           windows.append(image[i*x:(i+1)*x, j*y:(j+1)*y])
    
    return windows


"""
Calculates Normalized 3D Histogram with n_bins for the given image.
"""
def get_3d_norm_histogram(image: np.ndarray, n_bins: int) -> np.ndarray:
    hist_1 = cv2.calcHist([image], [0, 1, 2], None, [n_bins, n_bins, n_bins], None)
    hist_norm_1 = cv2.normalize(hist_1, hist_1, 0, 1, cv2.NORM_MINMAX)
    return hist_norm_1


"""
Calculates 3D Normalized Block-Based Histogram for each image of the given dataset.
Returns a Feature Matrix.
"""
def calculate_histograms(data: np.ndarray,n_bins: int, n_cols: int, n_rows: int,desc: str) -> np.ndarray:
    histograms = []
    for image in tqdm(data, desc = desc):
        # Generate Blocks and their histogram
        windows = image_to_windows(image,n_rows,n_cols)
        sub_histograms = [get_3d_norm_histogram(im,n_bins) for im in windows]
        sub_histograms = np.array(sub_histograms).flatten()
        histograms.append(sub_histograms)
            
    return np.array(histograms)
"""
Masks the detected text over an image with a rectangle mask of 0s. N_imgs is the expected number of text boxes to be removed 1 or 2 
"""
def text_removal_image(image:np.ndarray,num_images:int):
    
    #Transform to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Defining the kernels to use
    filterSize =(12, 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)
    
    #Applying tophat and blackhat morph operators
    tophat_img = cv2.morphologyEx(img, 
                              cv2.MORPH_TOPHAT,
                              kernel)
    blackhat_img = cv2.morphologyEx(img, 
                              cv2.MORPH_BLACKHAT,
                              kernel)
    #Defining the kernels to use
    filterSize =(3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)
    #Adding the results of tophat and blackhat morph operators and thresholding 
    #to get the letters that have the most intensity
    mean = tophat_img/2 +blackhat_img/2
    hat = np.uint8(mean>70)
    
    #Opening filter to remove little artifacts
    opening = cv2.morphologyEx(hat, cv2.MORPH_OPEN, kernel,iterations = 1)

    #Defining the kernels to use
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80,18 ))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(opening, rect_kernel, iterations = 1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
    
    bounded = np.zeros((img.shape))
    
    #Loop to get the contour with the greatest area
    if num_images==1:
        area = 0
        for cnt in contours:
            area_1 = cv2.contourArea(cnt)
            if area_1 >= area:
                area = area_1
                x, y, w, h = cv2.boundingRect(cnt)
        
        rect = cv2.rectangle(image, (x,y),(x+w,y+h) , (0,0,0), -1)
        
        return image
    if num_images==2:
        area_1 = 0
        area_2 = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= area_1:
                area_1 = area
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
            if area >= area_2:
                area_2 = area
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
        
        rect = cv2.rectangle(image, (x1,y1),(x1+w1,y1+h1) , (0,0,0), -1)
        rect = cv2.rectangle(image, (x2,y2),(x2+w2,y2+h2) , (0,0,0), -1)
        return image
    
    
def text_removal(data: np.ndarray,num_images,desc: str) -> np.ndarray:
    text_removed_image =[]
    for image in tqdm(data, desc = desc):
        # Generate Blocks and their histogram
        masked_images = text_removal_image(image,num_images)
        text_removed_image.append(masked_images)

    return np.array(text_removed_image)
    