import torch
from collections import deque
from pylib.imaging.lesion_tools import get_clusters, combine_clusters_by_distance, get_max_diameter_of_cluster
import numpy as np
from typing import List, Tuple, Type, Union

def f1_from_tp_fp_fn(
    true_positive: int,
    false_positive: int,
    false_negative: int
) -> float:
    """
    Calculate the F1 score from true positive, false positive, and false negative counts.
    
    Args:
        true_positive (int): True positive count.
        false_positive (int): False positive count.
        false_negative (int): False negative count.
        
    Returns:
        float: F1 score.
    """

    if true_positive == 0:
        return 0.0

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score







def lesion_wise_tp_fp_fn(
    y_true: Union[torch.Tensor, List[torch.Tensor]], 
    y_pred: Union[torch.Tensor, List[torch.Tensor]],
    channel: int = 0,
    threshold: float = 0.5,
    min_inter_lesion_distance: int = 0,
    min_lesion_diameter: int = 0,
    return_counts: bool = False
) -> float:
    """
    Calculate the lesion-wise F1 score.
    
    Args:
        y_true (torch.Tensor): Ground truth mask.
        y_pred (torch.Tensor): Predicted mask.
        threshold (float): Threshold for binary classification.
        
    Returns:
        tuple: (true_positive, false_positive, false_negative)
    """

    true_positive = 0
    false_positive = 0
    false_negative = 0

    if isinstance(y_true, torch.Tensor):
        y_true = [y_true]
    if isinstance(y_pred, torch.Tensor):
        y_pred = [y_pred]
    

    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length!"
    assert len(y_true) > 0, "y_true and y_pred must not be empty!"
    n_pred_lesions = 0
    n_true_lesions = 0

    for i in range(len(y_true)):

        assert y_true[i].shape == y_pred[i].shape, f"y_true and y_pred must have the same shape! {y_true[i].shape} != {y_pred[i].shape}"

        if y_true[i].dim() == 3:
            truth = y_true[i][channel]
            pred = y_pred[i][channel]
        else:
            truth = y_true[i]
            pred = y_pred[i]
        
        assert truth.dim() == 2, f"y_true and y_pred must be 2D tensors! {pred.dim()} != 2"


        # Get list of clusters
        y_true_clusters = get_clusters(truth)
        y_pred_clusters = get_clusters(pred)

        if min_inter_lesion_distance > 0:
            y_pred_clusters = combine_clusters_by_distance(y_pred_clusters, min_inter_lesion_distance)

        if min_lesion_diameter > 0:
            # Filter clusters based on minimum lesion diameter
            y_true_clusters = [cluster for cluster in y_true_clusters if get_max_diameter_of_cluster(cluster) >= min_lesion_diameter]
            y_pred_clusters = [cluster for cluster in y_pred_clusters if get_max_diameter_of_cluster(cluster) >= min_lesion_diameter]

        n_pred_lesions += len(y_pred_clusters)
        n_true_lesions += len(y_true_clusters)

        for pred_cluster in y_pred_clusters:

            # Find the best matching cluster in y_true
            best_iou = 0
            best_intersection = 0
            for true_cluster in y_true_clusters:

                intersection = np.intersect1d(true_cluster, pred_cluster).size
                union = np.union1d(true_cluster, pred_cluster).size

                if union == 0:
                    continue

                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                if intersection > best_intersection:
                    best_intersection = intersection

            if threshold is not None and best_iou > threshold:
                true_positive += 1
            elif best_intersection > 0:
                true_positive += 1
            else:
                false_positive += 1
        
        for true_cluster in y_true_clusters:
            best_iou = 0
            best_intersection = 0
            for pred_cluster in y_pred_clusters:

                intersection = np.intersect1d(true_cluster, pred_cluster).size
                union = np.union1d(true_cluster, pred_cluster).size

                if union == 0:
                    continue

                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                if intersection > best_intersection:
                    best_intersection = intersection


            if threshold is not None and best_iou <= threshold:
                false_negative += 1
            elif best_intersection == 0:
                false_negative += 1

                
    if return_counts:
        return true_positive, false_positive, false_negative, n_pred_lesions, n_true_lesions
    return true_positive, false_positive, false_negative

    




