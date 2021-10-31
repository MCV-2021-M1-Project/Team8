from joblib import Parallel,delayed
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import cv2
import multiprocessing
import textdistance


"""
Similarity Class: Functions related to measure similarity between images
based on their feature vectors. 
"""
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
    Computes histogram intersection for each channel between feature vector and all BBDD feature vectors
    """
    def histogram_similarity(self, vector1: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        return np.array([self.compute_histogram_intersect_vector(vector1,vector2) for vector2 in db_feature_matrix])


    """
    Computes histogram correlation for each channel between 2 feature vectors
    """
    def compute_histogram_correlation_vector(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        n_bins = int(len(vector1)/4)
        r = cv2.compareHist(vector1[:n_bins], vector2[:n_bins],cv2.HISTCMP_CORREL)
        g = cv2.compareHist(vector1[n_bins:2*n_bins], vector2[n_bins:2*n_bins],cv2.HISTCMP_CORREL)
        b = cv2.compareHist(vector1[2*n_bins:3*n_bins], vector2[2*n_bins:3*n_bins],cv2.HISTCMP_CORREL)
        gray = cv2.compareHist(vector1[3*n_bins:], vector2[3*n_bins:],cv2.HISTCMP_CORREL)
        return np.mean([r,g,b,gray])


    """
    Computes histogram correlation between vector and BBDD feature vectors
    """
    def correlation_similarity(self, vector1: np.ndarray, db_feature_matrix: np.ndarray) -> np.ndarray:
        return np.array([self.compute_histogram_correlation_vector(vector1,vector2) for vector2 in db_feature_matrix])


    """
    Computes hellinger distance between 2 feature vectors
    """
    def hellinger_similarity_vector(self, vector1: np.ndarray, vector2: np.ndarray):
        # 0 is max similarity, 1 is min similarity
        dist = cv2.compareHist(vector1, vector2, cv2.HISTCMP_BHATTACHARYYA)
        # We turn 0 to big number and 1 to 0.
        return (1-dist) # 1.e-5 avoids division by zero
    

    """
    Computes hellinger distance between vector and BBDD feature vectors
    """
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
    """
   Returns levensthein similarity of a string with the titles in the database
    """
    def levenshtein_text_similarity(self,qs:str,db_string_list:list) ->np.ndarray: 
        return np.array([textdistance.levenshtein.normalized_similarity(qs,expected) for expected in db_string_list])
    """
   Returns jaccard similarity of a string with the titles in the database
    """    
    def jaccard_text_similarity(self,qs:str,db_string_list:list) ->np.ndarray: 
        return np.array([textdistance.jaccard.normalized_similarity(qs,expected) for expected in db_string_list])

    """
    Computes similairty for an entire QuerySet
    """    
    def compute_string_similarities(self, qs: list, db_string_list: list, desc: str, similarity: str = 'levenshtein') -> np.ndarray:
    # Perform similarity for each vector in the QuerySet
        if similarity == "levenshtein":
            return np.array([self.levenshtein_text_similarity(qs=detected,db_string_list=db_string_list) for detected in tqdm(qs,desc=desc)])
        if similarity == "jaccard":
            return np.array([self.jaccard_text_similarity(qs=detected,db_string_list=db_string_list) for detected in tqdm(qs,desc=desc)])
    """
    
    """
    def get_index_positions(self,list_of_elems, element):
        ''' Returns the indexes of all occurrences of give element in
        the list- listOfElements '''
        index_pos_list = []
        index_pos = 0
        while True:
            try:
                # Search for item in list from indexPos to the end of list
                index_pos = list_of_elems.index(element, index_pos)
                # Add the index position in list
                index_pos_list.append(index_pos)
                index_pos += 1
            except ValueError as e:
                break
        return index_pos_list

    """
   Retrieves the top k similar images for a vector.
    """    
    def get_top_k_vector(self, similarity_vector: np.ndarray, db_files: List[str], k: int) -> List[str]:
        # We get top K index of the vector (unordered)
        idx = np.argpartition(similarity_vector, -k)[-k:]
        # Then we order index in order to get the ordered top k values
        top_k = list(similarity_vector[idx])
        sorted_top = list(sorted(top_k,reverse=True))
        sorted_top = list(dict.fromkeys(sorted_top))
        sorted_index=[]
        for j in sorted_top:  
            sorted_index.extend(self.get_index_positions(top_k,j))
        idx = [idx[m] for m in sorted_index]
        # ImageCollection also saves in .files so we can easily retrieve them
        return [db_files[i] for i in idx]

    """
    Retrieves the top k similar images for a QuerySet
    """    
    def get_top_k(self, similarity_matrix: np.ndarray, db_files: List[str], k: int, desc: str) -> List[List[str]]:
        # Estimate top k values for all the Queryet
        return [self.get_top_k_vector(similarity_vector = vector, db_files = db_files, k = k) for vector in tqdm(similarity_matrix, desc = desc)]
    
