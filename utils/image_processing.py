from tqdm import tqdm
from skimage.feature import hog, ORB, local_binary_pattern, multiblock_lbp
from skimage.color import rgb2gray
from skimage.transform import resize, integral_image
from typing import List
from joblib import Parallel, delayed
import pytesseract
import cv2
import numpy as np
import re


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
and returns the x,y,w,h for the rectangle used to mask
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
    
    #Loop to get the contour with the greatest area depending if one box or two have to be detected
    if num_images==1:
        area = 0
        x1=0
        y1=0
        w1=0
        h1=0
        for cnt in contours:
            area_1 = cv2.contourArea(cnt)
            if area_1 >= area:
                area = area_1
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
        
        rect = cv2.rectangle(image, (x1,y1),(x1+w1,y1+h1) , (0,0,0), -1)
        
        return image, [x1,y1,w1,h1]
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
        return image, [[x1,y1,w1,h1],[x2,y2,w2,h2]]    
""""
Applies text removal to a set
"""
def text_removal(data: np.ndarray,num_images,desc: str) -> np.ndarray:
    text_removed_image =[]
    contours =[]
    for image in tqdm(data, desc = desc):
        # Generate masked images and the x,y,w,h
        masked_images,coords = text_removal_image(image,num_images)
        text_removed_image.append(masked_images)
        contours.extend(coords)

    return np.array(text_removed_image),contours
#Detects text and returns the string of the title
"""
Returns the detected text of an image as a string with nonspecial characters
"""
def title_reading(image:np.ndarray,num_images:int) -> list:
    im2 = image.copy()
    _, [x_0,y_0,width,height] = text_removal_image(image,num_images)
    # Cropping the text block for giving input to OCR
    cropped = im2[y_0:y_0 + height, x_0:x_0 + width]
    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
    return re.sub("[^A-Za-z0-9- ]","",text)

#Generate list of titles    
def text_reading(data:np.ndarray,num_images,desc:str) -> list:
    detected_titles =[]
    for image in tqdm(data, desc=desc):
        title = title_reading(image,num_images)
        detected_titles.append(title)
    return(detected_titles)


def hog_image(image: np.ndarray) -> np.ndarray:
    image = resize(image = image, output_shape=(300,300))
    h_im, im = hog(image,orientations=9, pixels_per_cell=(8, 16),
                    cells_per_block=(2, 2), visualize=True)
    return im.astype(np.float32)

def calculate_hog(data: np.ndarray, desc: str) -> np.ndarray:
    return np.array(Parallel(n_jobs=-1)(delayed(hog_image)(file) for file in tqdm(data, desc = desc))).astype(np.float32)


def lbp_image(image: np.ndarray) -> np.ndarray:
    image = resize(image = image, output_shape=(300,300))
    bw_image = rgb2gray(image)
    return local_binary_pattern(image = bw_image, P = 5, R = 10).astype(np.float32)

def lbp_block_image(image: np.ndarray) -> np.ndarray:
    image = resize(image = image, output_shape=(300,300))
    bw_image = (rgb2gray(image)*255).astype(np.uint8)
    bw_image = integral_image(bw_image)
    return np.array([multiblock_lbp(int_image = bw_image, r = 3, c = 3, width = int(bw_image.shape[0]/9), height = int(bw_image.shape[0]/9))])


def calculate_lbp(data: np.ndarray, block: bool, desc: str) -> np.ndarray:
    if not block:
        return np.array(Parallel(n_jobs=-1)(delayed(lbp_image)(file) for file in tqdm(data, desc = desc))).astype(np.float32)
    
    return np.array(Parallel(n_jobs=-1)(delayed(lbp_block_image)(file) for file in tqdm(data, desc = desc))).astype(np.float32)
