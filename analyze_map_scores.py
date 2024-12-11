import os
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def load_predictions(experiment_folder, method):
    """Load predictions from a pickle file"""
    pred_file = os.path.join(experiment_folder, f"{method}.pkl")
    try:
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
            if isinstance(predictions, list):
                # Convert list of (predictions, image_name, size) tuples to dictionary
                result = {}
                for pred_list, img_name, _ in predictions:
                    # Convert predictions to the expected format
                    formatted_preds = []
                    for pred in pred_list:
                        if len(pred) == 3:  # box, score, class format
                            formatted_preds.append(pred)
                        else:  # other format, try to adapt
                            print(f"Unexpected prediction format in {method} for {img_name}: {pred}")
                            continue
                    result[img_name] = formatted_preds
                return result
            return predictions
    except Exception as e:
        print(f"Error loading predictions for {method}: {e}")
        return {}

def calculate_precision_recall(gt_dets, pred_dets, iou_threshold=0.5):
    """Calculate precision and recall for a single image"""
    if not gt_dets or not pred_dets:
        return [], [], []
    
    matches = []
    scores = []
    matched_gt = set()
    
    # Sort predictions by confidence score
    pred_dets_sorted = sorted(pred_dets, key=lambda x: x[1], reverse=True)
    
    for pred_idx, pred_det in enumerate(pred_dets_sorted):
        best_iou = 0
        best_gt_idx = -1
        
        # Handle different prediction formats
        try:
            pred_box = pred_det[0]
            pred_score = pred_det[1]
            pred_class = pred_det[2]
        except (IndexError, TypeError) as e:
            print(f"Error processing prediction: {pred_det}, Error: {e}")
            continue
        
        for gt_idx, gt_det in enumerate(gt_dets):
            try:
                gt_box = gt_det[0]
                gt_class = gt_det[2]
            except (IndexError, TypeError) as e:
                print(f"Error processing ground truth: {gt_det}, Error: {e}")
                continue
                
            if gt_idx in matched_gt or gt_class != pred_class:
                continue
                
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        matches.append(1 if best_iou >= iou_threshold else 0)
        scores.append(pred_score)
        
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
    
    # Calculate cumulative precision and recall
    matches = np.array(matches)
    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)
    
    n_gt = len(gt_dets)
    recall = tp / n_gt if n_gt > 0 else np.zeros_like(tp)
    precision = tp / (tp + fp)
    
    return precision, recall, scores

def calculate_ap(precision, recall):
    """Calculate Average Precision using 11-point interpolation"""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    # 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap

def calculate_map(experiment_folder, methods, reference_method='lossless_data'):
    """Calculate mAP for each method compared to the reference method"""
    results = {}
    reference_preds = load_predictions(experiment_folder, reference_method)
    
    if not reference_preds:
        print(f"Error: Could not load reference predictions from {reference_method}")
        return results
    
    for method in methods:
        if method == reference_method:
            continue
            
        print(f"\nAnalyzing {method} vs {reference_method}")
        method_preds = load_predictions(experiment_folder, method)
        
        if not method_preds:
            print(f"Error: Could not load predictions for {method}")
            continue
        
        # Calculate AP for each class
        class_aps = defaultdict(list)
        common_images = set(reference_preds.keys()) & set(method_preds.keys())
        
        if not common_images:
            print(f"Error: No common images found between {method} and {reference_method}")
            continue
            
        print(f"Analyzing {len(common_images)} common images")
        
        for img_name in common_images:
            gt_dets = reference_preds[img_name]
            pred_dets = method_preds[img_name]
            
            # Group detections by class
            gt_by_class = defaultdict(list)
            pred_by_class = defaultdict(list)
            
            for det in gt_dets:
                gt_by_class[det[2]].append(det)
            for det in pred_dets:
                pred_by_class[det[2]].append(det)
            
            # Calculate AP for each class
            all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
            for class_id in all_classes:
                precision, recall, _ = calculate_precision_recall(
                    gt_by_class[class_id],
                    pred_by_class[class_id]
                )
                ap = calculate_ap(precision, recall)
                class_aps[class_id].append(ap)
        
        # Calculate mAP
        mean_aps = {class_id: np.mean(aps) for class_id, aps in class_aps.items()}
        map_score = np.mean(list(mean_aps.values()))
        
        results[method] = {
            'mAP': map_score,
            'class_APs': mean_aps,
            'num_images': len(common_images)
        }
        
        print(f"\nResults for {method}:")
        print(f"mAP: {map_score:.4f}")
        print("AP by class:")
        for class_id, ap in mean_aps.items():
            print(f"  Class {class_id}: {ap:.4f}")
    
    return results

def main():
    # Update methods to match actual file names
    methods = [
        'lossless_data',  # reference method
        'lossy_data',
        'lossy_data_clevel_0',
        'lossy_data_clevel_4',
        'lossy_data_clevel_8'
    ]
    
    print("Starting MAP analysis...")
    print(f"Methods to analyze: {methods}")
    
    results = calculate_map(experiment_folder=".", methods=methods, reference_method='lossless_data')
    
    if not results:
        print("No results were generated. Check if the prediction files exist and are in the correct format.")
        return
        
    # Save results to file
    output_file = "map_analysis_results.txt"
    with open(output_file, 'w') as f:
        f.write("MAP Analysis Results\n")
        f.write("===================\n\n")
        for method, metrics in results.items():
            f.write(f"\n{method} vs lossless_data:\n")
            f.write(f"mAP: {metrics['mAP']:.4f}\n")
            f.write("AP by class:\n")
            for class_id, ap in metrics['class_APs'].items():
                f.write(f"  Class {class_id}: {ap:.4f}\n")
            f.write(f"Number of images analyzed: {metrics['num_images']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nResults have been saved to {output_file}")
    
    # Print summary to console
    print("\nSummary of Results:")
    for method, metrics in results.items():
        print(f"\n{method} vs lossless_data:")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"Number of images analyzed: {metrics['num_images']}")

if __name__ == "__main__":
    main() 