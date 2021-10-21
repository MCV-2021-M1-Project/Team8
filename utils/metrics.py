import numpy as np


def iou_score(target: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def prec_recall (true_labels: np.ndarray, pred_labels: np.ndarray):
    # Labels as Boolean
    true_labels = np.asarray(true_labels).astype(np.bool)
    pred_labels = np.asarray(pred_labels).astype(np.bool)
    
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))

    precision = TP / (TP+FP)
    recall    = TP / (TP+FN)
    return precision, recall
    

def f1_dice(im1: np.ndarray, im2: np.ndarray):

    # Images as Boolean
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())