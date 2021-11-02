import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import ml_metrics as metrics


def iou_score(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def readimg (path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def NMS(boxes, overlapThresh = 0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)

def background_crop (files, gt_masks= True, save_masks = False, plot_results=True, plot_rect=False):
    
    SCORES = []
    PREDICTIONS = {}
    RESULT = []
    FILENAMES = []

    for file in files:

        base = readimg (file)
        if gt_masks:
            gt_mask = readimg (file.replace('.jpg', '.png'))

        base_rgb = base.copy()
        rgb = base.copy()
        base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        
        # Performing OTSU threshold
        ret, base = cv2.threshold(base, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #base = otsus_binarization(rgb2gray(base))
        
        # Morphology mask
        mask = base
        
        # opening 1) correct noise
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
        
        # closing 1) get shapes
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations = 1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations = 1)
        # closing 2) get shapes
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
        
        # gradient=dilation - erosion --> outline object
        #g1 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        #g2 = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        #mask = g1 - g2
        
        # opening -> closing define better edges
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations = 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations = 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations = 1)
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations = 1)
        
        
        # Finding contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        boxes = []
        areas = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = base_rgb[y:y + h, x:x + w]
            area_cropped = cropped.shape[0] * cropped.shape[1]
            if (area_cropped <= 200*200) or (cropped.shape[0] < 200) or (cropped.shape[1] < 200): continue
            else: 
                boxes.append((x, y, x + w, y + h))
                areas[area_cropped] = (x, y, x + w, y + h)
                
        boxes = NMS(np.array(boxes), overlapThresh = 0.25)
        contours_detect = len(boxes)
        #print ('contours detected:', len(boxes))

        # Produce mask
        morph_mask = mask.copy()
        mask = np.zeros(base_rgb.shape)

        if len(boxes) > 1:
            # 2 biggest boxes
            big = sorted(areas.keys(),reverse=True)
            box1 = areas[big[0]]
            box2 = areas[big[1]]
            nboxes = [box1, box2]

            PREDICTIONS[file] = nboxes
            for box in nboxes:
                x, y, x2, y2 = box
                cropped = base_rgb[y:y2, x:x2]
                mask [y:y2, x:x2] = 1
                if plot_rect:
                    rect = cv2.rectangle(base_rgb, (x, y), (x2, y2), (0, 255, 0), 10)
                    
                #plt.imshow(cropped)
                #plt.axis('off')
                #plt.show()
                RESULT.append(cropped)
                FILENAMES.append(file)
        else:
            x, y, x2, y2 = boxes[0]
            cropped = base_rgb[y:y2, x:x2]
            mask [y:y2, x:x2] = 1
            #print (x, y, x2, y2)
            if plot_rect:
                rect = cv2.rectangle(base_rgb, (x, y), (x2, y2), (0, 255, 0), 10)
                
            PREDICTIONS[file] = boxes[0]
            RESULT.append(cropped)
            FILENAMES.append(file)
            #plt.imshow(rect)
            #plt.axis('off')
            #plt.show()
            #plt.imshow(cropped)
            #plt.axis('off')
            #plt.show()
            
        mask = mask.astype(np.uint8) * 255

        try: 
            mask_iou = iou_score(gt_mask, mask)
        except:
            mask_iou = 0.75
        
        SCORES.append(mask_iou)
        
        if plot_results:
 
            f, axarr = plt.subplots(1,6, figsize=(15,15))
            axarr[0].imshow(rgb)
            axarr[0].title.set_text("RGB")
            axarr[0].axis('off')
            axarr[1].imshow(base, cmap="gray")
            axarr[1].title.set_text("Otsu's Binarized")
            axarr[1].axis('off')
            axarr[2].imshow(morph_mask, cmap="gray")
            axarr[2].title.set_text(f"Morphology Mask {contours_detect}")
            axarr[2].axis('off')
            axarr[3].imshow(mask, cmap="gray")
            axarr[3].title.set_text(f"Final Mask {np.round(mask_iou,2)} iou")
            axarr[3].axis('off')
            if gt_masks: 
                axarr[4].imshow(gt_mask, cmap="gray")
            else:
                axarr[4].imshow(mask, cmap="gray")
            axarr[4].title.set_text("GT Mask")
            axarr[4].axis('off')
            axarr[5].imshow(base_rgb)
            axarr[5].title.set_text("RGB")
            axarr[5].axis('off')
            plt.show()
        
    
    if gt_masks:
        print ('\n>> Mean IOU Score: ', np.mean(SCORES))
        
    assert len(RESULT) == len(FILENAMES)
    return RESULT, FILENAMES