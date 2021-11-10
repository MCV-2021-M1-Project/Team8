from tqdm import tqdm
from skimage.feature import hog, ORB, local_binary_pattern, multiblock_lbp
from skimage.color import rgb2gray
from skimage.transform import resize, integral_image
from typing import List, Tuple
from joblib import Parallel, delayed
import pytesseract
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt


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

def find_greatest_contour(image:np.ndarray,num_image:int) -> list:
    contours, hierarchy = cv2.findContours(np.uint8(image), cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
    x1=0
    y1=0
    w1=0
    h1=0    
    x2=0
    y2=0
    w2=0
    h2=0
    x3=0
    y3=0
    w3=0
    h3=0
    area = []
 
    if num_image==1:
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        sorted_list = sorted(area,reverse=True)
        top_n = sorted_list[0:num_image]
        
        x1, y1, w1, h1 = cv2.boundingRect(contours[area.index(top_n[0])])
    
        return [x1,y1,w1,h1]
    
    if num_image==2:
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        sorted_list = sorted(area,reverse=True)
        top_n = sorted_list[0:num_image]
        
        x1, y1, w1, h1 = cv2.boundingRect(contours[area.index(top_n[0])])
        if len(top_n)==2:
            x2, y2, w2, h2 = cv2.boundingRect(contours[area.index(top_n[1])])

        return [[x1,y1,w1,h1],[x2,y2,w2,h2]]  
    
    if num_image==3:
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        sorted_list=sorted(area,reverse=True)
        top_n = sorted_list[0:num_image]
        x1, y1, w1, h1 = cv2.boundingRect(contours[area.index(top_n[0])])
        if len(top_n)==2:
            x2, y2, w2, h2 = cv2.boundingRect(contours[area.index(top_n[1])])
        elif len(top_n)==3:
            x2, y2, w2, h2 = cv2.boundingRect(contours[area.index(top_n[1])])
            x3, y3, w3, h3 = cv2.boundingRect(contours[area.index(top_n[2])])
        
        return [[x1,y1,w1,h1],[x2,y2,w2,h2],[x3,y3,w3,h3]]  


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
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80,18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(opening, rect_kernel, iterations = 1)

   
   
    im_copy = image.copy()
    #Loop to get the contour with the greatest area depending if one box or two have to be detected
    if num_images==1:
        coords = find_greatest_contour(dilation,num_images) 
        
        rect = cv2.rectangle(im_copy, (coords[0],coords[1]),(coords[0]+coords[2],coords[1]+coords[3]) , (0,0,0), -1)
        
        return im_copy, coords
    if num_images==2:
        coords = find_greatest_contour(dilation,num_images)       
        
        rect = cv2.rectangle(im_copy, (coords[0][0],coords[0][1]),(coords[0][0]+coords[0][2],coords[0][1]+coords[0][3]) , (0,0,0), -1)
        rect = cv2.rectangle(im_copy, (coords[1][0],coords[1][1]),(coords[1][0]+coords[1][2],coords[1][1]+coords[1][3]) , (0,0,0), -1)
        
        return im_copy, coords
"""
Applies a morphological filter to the image
"""
def morph_filter(mask, kernel, filter):
    return cv2.morphologyEx(mask, filter, np.ones(kernel, np.uint8))

"""Masks with a  rectangle of 0s the detected text in the image. Return x,y,w,h"""
def better_text_removal_image(image:np.ndarray,num_images:int):
                             
    value_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,2]
    #Converting blacks to white and white to whites to whites
    abs_v = np.absolute(value_hsv - np.amax(value_hsv) / 2)
   
    #Applying  blackhat morph operator and thresholding
    blackhat = morph_filter(abs_v,(3,3),cv2.MORPH_BLACKHAT)
    blackhat = blackhat / np.max(blackhat)
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[blackhat > 0.4] = 1
    
    mask = morph_filter(mask, (2, 10), cv2.MORPH_CLOSE)  ##Fill letter
    mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)  # Delete vetical kernel = (1,3) and horizontal lines kernel = (4,1). Total kernel = (4,3)
    mask = morph_filter(mask, (1, 29), cv2.MORPH_CLOSE)  # Join letters
    
    
    im_copy = image.copy()
    #Detects greatest contour 
    x1,y1,w1,h1 = find_greatest_contour(mask, 1)
    
    top = y1-int(h1*3/2)
    bottom = y1+int(3*h1/2)
    if top<0:
        top = 0
    if bottom>abs_v.shape[0]:
        bottom = int(abs_v.shape[0])
    
    box = np.zeros((image.shape[0], image.shape[1]))
    mask = np.zeros((box.shape[0], box.shape[1]))
    
    box[top:bottom, 30:-30] = blackhat[top:bottom,30:-30]
    box = box / np.amax(box)
    mask[box > 0.46] = 1
    
    #Applies morph filter to a smaller region to finally detect bounding box                          
    mask = morph_filter(mask, (5, 14), cv2.MORPH_CLOSE)  # Fill letter
    mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)
    mask = morph_filter(mask, (1, 91), cv2.MORPH_CLOSE)  # Join letters
    mask = morph_filter(mask, (1, 2), cv2.MORPH_OPEN)  # Delete remaining vertical lines
    
    coords = find_greatest_contour(mask, num_images)
    if num_images==1:
        rect = cv2.rectangle(im_copy, (coords[0],coords[1]),(coords[0]+coords[2],coords[1]+coords[3]) , (0,0,0), -1)
    if num_images==2:
        rect = cv2.rectangle(im_copy, (coords[0][0],coords[0][1]),(coords[0][0]+coords[0][2],coords[0][1]+coords[0][3]) , (0,0,0), -1)
        rect = cv2.rectangle(im_copy, (coords[1][0],coords[1][1]),(coords[1][0]+coords[1][2],coords[1][1]+coords[1][3]) , (0,0,0), -1)
     
    return im_copy, coords
""" Method for semitransparent bounding boxes, it discards boxes that are too big"""
def transparent_text_removal(image:np.ndarray, num_images:int)->np.ndarray:
    
    value_hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)[:,:,2]
    
    dst = cv2.cornerHarris(np.uint8(value_hsv),7,5,0.004)
    dst = cv2.dilate(dst,None)
    mask_corners = np.zeros((image.shape))
    #result is dilated for marking the corners, not important
    mask_corners = dst>0.1*dst.max()
    mask = morph_filter(np.uint8(mask_corners), (2, 10), cv2.MORPH_CLOSE)  ##Fill letter
    mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)  # Delete vetical kernel = (1,3) and horizontal lines kernel = (4,1). Total kernel = (4,3)
    mask = morph_filter(mask, (40, 80), cv2.MORPH_CLOSE)  # Join letters
    #Detects 3 greatest contours
    tops = []
    bottoms = []
    lefts = []
    rights = []
    contours = find_greatest_contour(mask,num_images)
    
    for i in range(0,num_images):
        tops.append(contours[i][1]-int(contours[i][3]*1/2))
        bottoms.append(contours[i][1]+int(contours[i][3]*3/2))
    for i in range(0,len(tops)):
        if tops[i]<0:
            tops[i] = 0
    for i in range(0,len(bottoms)):
        if bottoms[i]>image.shape[0]:
            bottoms[i] = int(image.shape[0])

    dim = 0
    im_copy = image.copy()
    for m,n in zip(tops,bottoms):
        if m ==0 and n == 0:
            continue
        
        box = np.zeros((image.shape[0], image.shape[1]))
        box[m:n,:] = cv2.cornerHarris(np.uint8(value_hsv[m:n,:]),7,5,0.004)
        dst = cv2.dilate(box,None)
        mask_corners = np.zeros((image.shape))
        #result is dilated for marking the corners, not important
        mask_corners = dst>0.05*dst.max()
        mask = morph_filter(np.uint8(mask_corners), (10, 10), cv2.MORPH_CLOSE)  # Fill letter
        mask = morph_filter(mask, (4, 4),
                        cv2.MORPH_OPEN)
        mask = morph_filter(mask, (40, 80), cv2.MORPH_CLOSE)  # Join letters
        coords = find_greatest_contour(mask,1)
        if coords[3]>image.shape[0]/2:
            continue
        rect = cv2.rectangle(im_copy, (coords[0],coords[1]),(coords[0]+coords[2],coords[1]+coords[3]) , (0,0,0), -1)
    return im_copy, contours
        
""""
Applies text removal to a set
"""
def text_removal(data: np.ndarray,num_images: int,method: str, desc: str) -> np.ndarray:
    text_removed_image =[]
    contours =[]
    if method=='first':
        for image in tqdm(data, desc = desc):
            # Generate masked images and the x,y,w,h
            masked_images,coords = text_removal_image(image,num_images)
            text_removed_image.append(masked_images)
            contours.append(coords)
    if method=='better':
        for image in tqdm(data, desc = desc):
            # Generate masked images and the x,y,w,h
            masked_images,coords = better_text_removal_image(image,num_images)
            text_removed_image.append(masked_images)
            contours.append(coords)
    if method=='transparent':
        for image in tqdm(data, desc = desc):
            # Generate masked images and the x,y,w,h
            masked_images,coords = transparent_text_removal(image,num_images)
            text_removed_image.append(masked_images)
            contours.append(coords)

    return np.array(text_removed_image),contours

"""
Returns the detected text of an image as a string with nonspecial characters
"""
def title_reading(image:np.ndarray,num_images:int) -> list:
    im2 = image.copy()
    if num_images==1:
        _, [x_0,y_0,width,height] = text_removal_image(image,num_images)
        # Cropping the text block for giving input to OCR
        if x_0==0 and y_0==0 and width==0 and height ==0:
            cropped = im2
        else:
            cropped = im2[y_0:y_0 + height, x_0:x_0 + width]
        # Apply OCR on the cropped image
  
        
        text = pytesseract.image_to_string(cropped)
        return re.sub("[^A-Za-z0-9- ]","",text)
        
    if num_images==2:
        _, coords = text_removal_image(image,num_images)
        # Cropping the text block for giving input to OCR
        if coords[0][0]==0 and coords[0][1]==0 and coords[0][2]==0 and coords[0][3]==0:
            cropped_1 = im2
        else:
            cropped_1 = im2[coords[0][1]:coords[0][1]+coords[0][3],coords[0][0]:coords[0][0]+coords[0][2]]
        
        if coords[1][0]==0 and coords[1][1]==0 and coords[1][2]==0 and coords[1][3]==0:
            cropped_2 = im2
        else:
            cropped_2 = im2[coords[1][1]:coords[1][1]+coords[1][3],coords[1][0]:coords[1][0]+coords[1][2]]
        
        text_1 = pytesseract.image_to_string(cropped_1)
        text_2 = pytesseract.image_to_string(cropped_2)
        
        return (re.sub("[^A-Za-z0-9- ]","",text_1),re.sub("[^A-Za-z0-9- ]","",text_2))
    if num_images==3:
        _, coords = transparent_text_removal(image,num_images)
        # Cropping the text block for giving input to OCR
        if coords[0][0]==0 and coords[0][1]==0 and coords[0][2]==0 and coords[0][3]==0:
            text_1 = ''
        else:
            cropped_1 = im2[coords[0][1]:coords[0][1]+coords[0][3],coords[0][0]:coords[0][0]+coords[0][2]]
            text_1 = pytesseract.image_to_string(cropped_1)
        if coords[1][0]==0 and coords[1][1]==0 and coords[1][2]==0 and coords[1][3]==0:
            text_2 = ''
        else:
            cropped_2 = im2[coords[1][1]:coords[1][1]+coords[1][3],coords[1][0]:coords[1][0]+coords[1][2]]
            text_2 = pytesseract.image_to_string(cropped_2)
        if coords[2][0]==0 and coords[2][1]==0 and coords[2][2]==0 and coords[2][3]==0:
            text_3 = ''
        else:
            cropped_3 = im2[coords[2][1]:coords[2][1]+coords[2][3],coords[2][0]:coords[2][0]+coords[2][2]]
            text_3 = pytesseract.image_to_string(cropped_3)
            
        
        return (re.sub("[^A-Za-z0-9- ]","",text_1),re.sub("[^A-Za-z0-9- ]","",text_2),re.sub("[^A-Za-z0-9- ]","",text_3))

#Generate list of titles    
def text_reading(data:np.ndarray,num_images,desc:str) -> list:
    detected_titles =[]
    for image in tqdm(data, desc=desc):
        title = title_reading(image,num_images)
        detected_titles.append(title)
    return(detected_titles)

def hog_image(image: np.ndarray) -> np.ndarray:
    image = resize(image = image, output_shape=(300,300))
    h_im, im = hog(image,orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
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


def calculate_orb_image(image: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=50)
    image = resize(image = image, output_shape=(300,300))
    bw_image = rgb2gray(image)*255
    bw_image = bw_image.astype(np.uint8)
    keypoints, descriptors = orb.detectAndCompute(bw_image,None)
    return keypoints, descriptors

def calculate_orb(data: np.ndarray, desc: str) -> Tuple[List, List]:
    features = [calculate_orb_image(file) for file in tqdm(data, desc = desc)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)


def calculate_sift_image(image: np.ndarray) -> np.ndarray:
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)
    image = resize(image = image, output_shape=(300,300))
    bw_image = rgb2gray(image)*255
    bw_image = bw_image.astype(np.uint8)
    keypoints, descriptors = sift.detectAndCompute(bw_image,None)
    return keypoints, descriptors


def calculate_sift(data: np.ndarray, desc: str) -> Tuple[List, List]:
    features = [calculate_sift_image(file) for file in tqdm(data, desc = desc)]
    keypoints, descriptors = list(zip(*features))
    return keypoints, np.array(descriptors)
