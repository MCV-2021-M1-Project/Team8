from joblib import Parallel,delayed
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import cv2
import multiprocessing


class Similarity(object):
    
    def __init__(self):
        # Choosing Number of Cores to use for multiprocessing
        n_process = multiprocessing.cpu_count()

        if n_process > 1 and n_process < 4:
            n_process = 2
        elif n_process >= 4:
            n_process = 4
        else:
            n_process = 1
    
    """
    Computes cos similarity between feature vector and all BBDD feature vectors
    """    
    def cos_similarity(self, vector: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        # We take profit of numpy broadcasting to calculate cos similarity between the provided vector and all BBDD features
        return db_feature_matrix.dot(vector)/ (np.linalg.norm(db_feature_matrix, axis=1) * np.linalg.norm(vector))


    """
    Computes euclidean similarity between feature vector and all BBDD feature vectors
    """   
    def euclidean_similarity(self, vector: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        # We take profit of numpy broadcasting to calculate euclidean similarity between the vector and all BBDD features
        dist = (db_feature_matrix - vector)**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        dist = np.max(dist) - dist
        return dist


    """
    Computes histogram intersection for each channel between 2 feature vectors
    """
    def compute_histogram_intersect_vector(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        # Recover each channel and Calculate intersection
        n_bins = int(len(vector1)/4)
        r = np.sum(np.minimum(vector1[:n_bins], vector2[:n_bins]))
        g = np.sum(np.minimum(vector1[n_bins:2*n_bins], vector2[n_bins:2*n_bins]))
        b = np.sum(np.minimum(vector1[2*n_bins:3*n_bins], vector2[2*n_bins:3*n_bins]))
        gray = np.sum(np.minimum(vector1[3*n_bins:], vector2[3*n_bins:]))
        # Retrieve mean
        return np.mean([r,g,b,gray])


    """
    Computes histogram intersection for each channel between 2 feature vectors
    """
    def histogram_similarity(self, vector1: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        return np.array([self.compute_histogram_intersect_vector(vector1,vector2) for vector2 in db_feature_matrix])


    def compute_histogram_correlation_vector(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        n_bins = int(len(vector1)/4)
        r = cv2.compareHist(vector1[:n_bins], vector2[:n_bins],cv2.HISTCMP_CORREL)
        g = cv2.compareHist(vector1[n_bins:2*n_bins], vector2[n_bins:2*n_bins],cv2.HISTCMP_CORREL)
        b = cv2.compareHist(vector1[2*n_bins:3*n_bins], vector2[2*n_bins:3*n_bins],cv2.HISTCMP_CORREL)
        gray = cv2.compareHist(vector1[3*n_bins:], vector2[3*n_bins:],cv2.HISTCMP_CORREL)
        return np.mean([r,g,b,gray])


    """
    Computes histogram correlation bewteen vector and BBDD feature vectors
    """
    def correlation_similarity(self, vector1: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        return np.array([self.compute_histogram_correlation_vector(vector1,vector2) for vector2 in db_feature_matrix])


    
    def hellinger_similarity_vector(self, vector1: np.ndarray, vector2: np.ndarray):
        return cv2.compareHist(vector1, vector2, cv2.HISTCMP_BHATTACHARYYA)
    
    
    def hellinger_similarity(self, vector1: np.ndarray, db_feature_matrix: np.ndarray):
        return np.array([self.hellinger_similarity_vector(vector1,vector2) for vector2 in db_feature_matrix])
    
    
    """
    Computes similairty for an entire QuerySet
    """    
    def compute_similarities(self, qs: np.ndarray, db_feature_matrix: np.ndarray, desc: str, similarity: str = 'cos') -> np.ndarray:
        # Perform similarity for each vector in the QuerySet
        if similarity == "cos":
            return np.array([self.cos_similarity(vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])
        
        elif similarity == "intersection":
            return np.array([self.histogram_similarity(vector1=vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])
        
        elif similarity == "euclidean":
            return np.array([self.euclidean_similarity(vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])
        
        elif similarity == "correlation":
            return np.array([self.correlation_similarity(vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])
        
        elif similarity == "hellinger":
            return np.array([self.hellinger_similarity(vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])
        
        elif similarity == "mixed":
            return np.array([self.correlation_similarity(vector,db_feature_matrix=db_feature_matrix)+self.cos_similarity(vector,db_feature_matrix=db_feature_matrix) for vector in tqdm(qs,desc=desc)])