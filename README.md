# Team8
M1. Introduction to Human and Computer Vision

## Team Members
* José Manuel López (Josep)
* Alex Martin
* Marcos V. Conde


### Week 1 : Image Retrieval and Background Subtraction (Color Based)

* Image Retrieval consists on finding the most similar images (in a Database) to a given query image based on images' features.
* Background subtraction is any technique which allows an image's foreground to be extracted for further processing (object recognition etc.). 
* Feature Vectors &#8594; ```1D Vector Based on Concatenated Normalized Individual Channels' Histograms (R, G, B, Gray) with 16 Bins.```
* Metrics &#8594; ```Cosinus, Histogram Intersection, Histogran Correlation, Euclidean Distance.```
* Substraction Method:
    * Otsu's Binarization
    * Pixel Diff

* Results
    |                        | Dataset | MAP@1 | MAP@5 | Comment                                                 |
    |------------------------|---------|-------|-------|---------------------------------------------------------|
    | Method 1 - Correlation | QSD1    | 0.366 | 0.428 | Baseline Retrieval Method for QSD1.                     |
    | Method 1 - Correlation | QSD2    | 0.1   | 0.177 | Performance decay due to "wild" images with background. |
    | Method 2 - Otsu        | QSD2    | 0.433 | 0.461 | After Cropping the images performance improves notably. |
    | Method 3 - Pixel Diff  | QSD2    | 0.3   | 0.386 | Overall worse performance method than Otsu.             |
    
