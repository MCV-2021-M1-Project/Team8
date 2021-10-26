from skimage.io import imread
import matplotlib.pyplot as plt


"""
   Plot 1st Query Results
"""   
def plot_image_and_similar(qs,top_qs) -> None:
    
    for i in range(len(qs)):
        f, axarr = plt.subplots(1,2, figsize=(10,10))
        base = qs[i]
        query = imread(top_qs[i][0])
        axarr[0].imshow(base)
        axarr[0].title.set_text("Base")
        axarr[1].imshow(query)
        axarr[1].title.set_text("Query")
        plt.show()