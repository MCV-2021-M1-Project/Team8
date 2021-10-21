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