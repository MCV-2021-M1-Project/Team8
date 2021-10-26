from tqdm import tqdm
import numpy as np
import cv2

def image_to_windows(image: np.ndarray, n_rows: int, n_cols: int):
    x, y = image.shape[0]/n_rows, image.shape[1]/n_cols
    x, y = int(x), int(y)
    windows = []
    for i in range(n_rows):
       for j in range(n_cols):
           windows.append(image[i*x:(i+1)*x, j*y:(j+1)*y])
    
    return windows

def get_3d_norm_histogram(image: np.ndarray, n_bins: int):
    hist_1 = cv2.calcHist([image], [0, 1, 2], None, [n_bins, n_bins, n_bins], None)
    #hist_norm_1 = cv2.normalize(hist_1, hist_1, 0, 1, cv2.NORM_MINMAX)
    return hist_1

def calculate_histograms(data: np.ndarray,n_bins: int, n_cols: int, n_rows: int,desc: str):
    histograms = []
    for image in tqdm(data, desc = desc):
        windows = image_to_windows(image,n_rows,n_cols)
        sub_histograms = [get_3d_norm_histogram(im,n_bins) for im in windows]
        sub_histograms = np.array(sub_histograms).flatten()
        #sub_histograms = ((sub_histograms - np.min(sub_histograms)) / (np.max(sub_histograms) - np.min(sub_histograms))).astype(np.float32)
        histograms.append(sub_histograms)
            
    return np.array(histograms)