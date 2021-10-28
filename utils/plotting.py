from skimage.io import imread
import matplotlib.pyplot as plt


"""
   Plot 1st Query Results
"""   
def plot_image_and_similar(qs,top_qs, topk=1) -> None:
    
    for i in range(len(qs)):
        
        query = qs[i]
        f, axarr = plt.subplots(1,1+topk, figsize=(10,10))
        axarr[0].imshow(query)
        axarr[0].title.set_text("Query")
        axarr[0].axis('off')
        
        for k in range(topk):
            reference = imread(top_qs[i][k])
            axarr[k+1].imshow(reference)
            axarr[k+1].title.set_text(f"Top {k+1} BD")
            axarr[k+1].axis('off')
        
        plt.show()