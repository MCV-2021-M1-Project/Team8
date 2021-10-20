from skimage.io.collection import ImageCollection
from typing import Tuple, List
from tqdm import tqdm

def load_data(folder: str) -> ImageCollection:
   # We use ImageCollection to load all images inside folders
   d = ImageCollection(folder)
   print('{} read: {} images'.format(folder,len(d)))
    
   return d