from skimage.io import imread
from typing import Tuple, List
from tqdm import tqdm
from joblib import Parallel,delayed
import os
import numpy as np

def read_image(path):
   return imread(path), path

def load_data(folder: str, extension: str, desc: str):
   files = [folder+image for image in os.listdir(folder) if extension in image]
   data = Parallel(n_jobs=2)(delayed(read_image)(file) for file in tqdm(files, desc = desc))
   print('{} read: {} images'.format(folder,len(data)))
   images, images_names = list(zip(*data))
   return images, images_names


def image_to_windows(image: np.ndarray, n_rows: int, n_cols: int):
    x, y = image.shape[0]/n_rows, image.shape[1]/n_cols
    x, y = int(x), int(y)
    windows = []
    for i in range(n_rows):
       for j in range(n_cols):
           windows.append(image[i*x:(i+1)*x, j*y:(j+1)*y])
    
    return windows