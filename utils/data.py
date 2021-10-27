

from skimage.io import imread
from typing import Tuple, List
from tqdm import tqdm
from joblib import Parallel,delayed
from sklearn.decomposition import PCA

import os
import pickle
import multiprocessing
import numpy as np
import platform 


"""
DataManager Class: Functions related to data and datasets
such as I/O operations or data cardinality.
"""
class DataManager(object):

   def __init__(self) -> None:
      # Empty PCA
      self.pca = None
      # Select number of process
      self.n_process = multiprocessing.cpu_count()
      if self.n_process > 1 and self.n_process < 4:
         self.n_process = 2
      elif self.n_process >= 4:
         self.n_process = 4
      else:
         self.n_process = 1


   """
   Reads image at path.
   Returns both the image and path.
   """    
   def read_image(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
      return path,imread(path)


   """
   Reads all images at folder with chosen extension.
   Returns images and paths.
   """
   def load_data(self, folder: str, extension: str, desc: str) -> Tuple[np.ndarray, List[str]]:
      # List files and read data
      files = [folder+image for image in os.listdir(folder) if extension in image]
      data = Parallel(n_jobs=self.n_process)(delayed(self.read_image)(file) for file in tqdm(files, desc = desc))
      data = sorted(data) # Sort by path (0000.png, 0001.png, ...)
      print('{} read: {} images'.format(folder,len(data)))

      # Split Images and Paths
      images_names, images  = list(zip(*data))

      return images, images_names


   """
   Reduces data cardinality applying PCA. We keep 95% variance needed to explain data.
   Returns new data with less variables.
   """
   def reduce_cardinality(self, data: np.ndarray) -> np.ndarray:
      # First time needs Fit
      if not self.pca:
         self.pca = PCA(n_components=0.95)
         return self.pca.fit_transform(data)

      # Rest of the times we just transform
      else:
         return self.pca.transform(data)
   

   """
   Image path to ID
   """
   def get_image_id(self, image: str) -> str:
      # Extract BBBD_XXX.jpg from relative path
      file = image.split("/")[3]
      # Extract XXX id from BBBD_XXX.jpg 
      id = file.replace(".jpg","").split("_")[1]
      return int(id)


   """
   Save Results properly formated for QS with 1 image
   """
   def save_results_1(self, results: List[List[str]], path: str, save: bool) -> List[List[int]]:
      # Vectorize function to apply to each element in results
      get_ids = np.vectorize(self.get_image_id)
      results = get_ids(results)

      # Creates Save Folder 
      if not os.path.exists(path):
         os.makedirs(path)

      if save:
      # Saves data
         pickle.dump(obj = results,file = open(path+"/result.pkl","wb"))
         print("Results Saved!")

      return results


   """
   Save Results properly formated for QS with multi-image
   """
   def save_results_2(self, results: List[List[str]], path: str, save: bool) -> List[List[int]]:
      result = [] # Final Result
      for element in results: # [[[top_k_image1.1],[top_k_image1.2]],[[top_k_image2]], ...]
         tmp = []
         for image in element: # Iterate through images from original image
               tmp.append(self.get_image_id(image))
         
         result.append(tmp)

      # Creates Save Folder 
      if not os.path.exists(path):
         os.makedirs(path)

      if save:
      # Saves data
         pickle.dump(obj = result,file = open(path+"/result.pkl","wb"))
         print("Results Saved!")

      return result
      