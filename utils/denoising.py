import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import math
from tqdm import tqdm


def denoise_images (images, method='nlmeans', ksize=(3,3)):
    
    ksize = ksize
    
    (lpfw,lpfh) = ksize
    (gfw,gfh) = ksize
    
    new_images = []
    for img in tqdm(images):
        
        if method == 'low_pass':
            lowPassFilter = np.ones((lpfw,lpfh))*1/(lpfw*lpfh)
            denoised_img = cv2.filter2D(img,-1,lowPassFilter)
        elif method == 'gauss':
            #gaussianFilter = gaussFilter((gfw,gfh),1)
            #gaussian_image = cv2.filter2D(img,-1,gaussianFilter)
            denoised_img = cv2.GaussianBlur(img,ksize,0)

        elif method == 'average':
            denoised_img = cv2.blur(img,ksize)
        
        elif method == 'nlmeans':
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 13, 21)
            
        elif method == 'median':
            denoised_img = cv2.medianBlur(img,ksize[0])
        else:
            return None 
        
        new_images.append(denoised_img)
        
    return new_images