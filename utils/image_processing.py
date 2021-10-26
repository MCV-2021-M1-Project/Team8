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