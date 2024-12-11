import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import sys
from collections import defaultdict

def load_predictions(experiment_folder, experiment_name):
    """Load predictions.pkl file for a given experiment"""
    pred_file = os.path.join(experiment_folder, experiment_name, "predictions.pkl")
    print(f"Loading predictions from: {pred_file}")
    
    with open(pred_file, 'rb') as f:
        preds = pickle.load(f)
        print(f"Number of predictions loaded for {experiment_name}: {len(preds)}")
        if len(preds) > 0:
            print(f"Sample prediction format: {preds[0]}")
    return preds

def calculate_bbox_metrics(box1, box2):
    """Calculate IoU and distance between two bounding boxes"""
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    # Calculate center point distance
    c1_x = (b1_x1 + b1_x2) / 2
    c1_y = (b1_y1 + b1_y2) / 2
    c2_x = (b2_x1 + b2_x2) / 2
    c2_y = (b2_y1 + b2_y2) / 2
    
    center_dist = np.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
    
    return iou, center_dist

def calculate_precision_recall(gt_dets, pred_dets, iou_threshold=0.5):
    """Calculate precision and recall values"""
    if not gt_dets or not pred_dets:
        return [], [], []
    
    # Sort predictions by confidence score
    pred_dets = sorted(pred_dets, key=lambda x: x[1], reverse=True)
    
    matches = []
    scores = []
    matched_gt = set()
    
    for pred in pred_dets:
        pred_box, pred_score, pred_class = pred
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_dets):
            if gt_idx in matched_gt:
                continue
                
            gt_box, _, gt_class = gt
            if gt_class != pred_class:
                continue
                
            iou = calculate_bbox_metrics(pred_box, gt_box)[0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        matches.append(1 if best_iou >= iou_threshold else 0)
        scores.append(pred_score)
        
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
    
    # Calculate cumulative precision and recall
    tp = np.cumsum(matches)
    fp = np.cumsum([1 - m for m in matches])
    recall = tp / len(gt_dets) if len(gt_dets) > 0 else np.zeros_like(tp)
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

def compare_predictions(pred1, pred2):
    """Compare two sets of predictions"""
    metrics = {
        'confidence_diff': [],
        'iou': [],
        'center_dist': [],
        'class_matches': 0,
        'total_detections': 0,
        'ap_by_class': defaultdict(list),
        'map': 0.0,
        'mismatches': []
    }
    
    # Create dictionaries with image names as keys
    pred1_dict = {img: det for det, img in pred1}
    pred2_dict = {img: det for det, img in pred2}
    
    # Find common images
    common_images = set(pred1_dict.keys()) & set(pred2_dict.keys())
    
    print(f"Total images in first set: {len(pred1_dict)}")
    print(f"Total images in second set: {len(pred2_dict)}")
    print(f"Common images: {len(common_images)}")
    
    if len(common_images) == 0:
        print("Warning: No common images found between the two prediction sets!")
        return metrics
    
    # Sort common images to ensure consistent order
    common_images = sorted(common_images)
    
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'ap': 0.0})
    
    for img in common_images:
        det1 = pred1_dict[img]
        det2 = pred2_dict[img]
        
        # Calculate precision-recall and AP for each class
        all_classes = set(d[2] for d in det1) | set(d[2] for d in det2)
        
        for class_id in all_classes:
            # Filter detections by class
            gt_dets = [d for d in det1 if d[2] == class_id]
            pred_dets = [d for d in det2 if d[2] == class_id]
            
            precision, recall, scores = calculate_precision_recall(gt_dets, pred_dets)
            if len(precision) > 0:
                ap = calculate_ap(precision, recall)
                metrics['ap_by_class'][class_id].append(ap)
        
        # Original comparison logic
        for d1 in det1:
            metrics['total_detections'] += 1
            best_match = None
            best_iou = 0
            
            for d2 in det2:
                if d1[2] == d2[2]:  # Same class
                    iou, dist = calculate_bbox_metrics(d1[0], d2[0])
                    if iou > best_iou:
                        best_match = d2
                        best_iou = iou
            
            if best_match:
                conf_diff = abs(d1[1] - best_match[1])
                metrics['confidence_diff'].append(conf_diff)
                metrics['iou'].append(best_iou)
                metrics['center_dist'].append(calculate_bbox_metrics(d1[0], best_match[0])[1])
                metrics['class_matches'] += 1
                
                if conf_diff > 0.01 or best_iou < 0.99:
                    metrics['mismatches'].append({
                        'image': img,
                        'class': d1[2],
                        'ground_truth': d1,
                        'prediction': best_match,
                        'iou': best_iou,
                        'conf_diff': conf_diff
                    })
    
    # Calculate final MAP
    for class_id in metrics['ap_by_class']:
        class_aps = metrics['ap_by_class'][class_id]
        class_metrics[class_id]['ap'] = np.mean(class_aps)
    
    metrics['map'] = np.mean([metrics['ap'] for metrics in class_metrics.values()])
    metrics['class_metrics'] = class_metrics
    
    return metrics

def analyze_lossless_methods(experiment_folder):
    """Compare lossless compression methods"""
    methods = ['lossless_blosc2_clevel_4', 'lossless_blosc2_clevel_4', 'lossless_blosc2_clevel_4']
    predictions = {}
    
    for method in methods:
        predictions[method] = load_predictions(experiment_folder, method)
    
    results = {}
    for m1 in methods:
        for m2 in methods:
            if m1 < m2:
                metrics = compare_predictions(predictions[m1], predictions[m2])
                results[f"{m1}_vs_{m2}"] = metrics
    
    return results

def calculate_map_metrics(pred1, pred2):
    """Calculate MAP and per-class AP for comparison"""
    metrics = {
        'map': 0.0,
        'ap_by_class': defaultdict(float),
        'precision_by_class': defaultdict(float),
        'recall_by_class': defaultdict(float)
    }
    
    # Create dictionaries with image names as keys
    pred1_dict = {img: det for det, img in pred1}
    pred2_dict = {img: det for det, img in pred2}
    
    common_images = sorted(set(pred1_dict.keys()) & set(pred2_dict.keys()))
    if not common_images:
        return metrics
    
    class_aps = defaultdict(list)
    class_precisions = defaultdict(list)
    class_recalls = defaultdict(list)
    
    for img in common_images:
        det1 = pred1_dict[img]
        det2 = pred2_dict[img]
        
        # Get all unique classes
        all_classes = set(d[2] for d in det1) | set(d[2] for d in det2)
        
        for class_id in all_classes:
            # Filter detections by class
            gt_dets = [d for d in det1 if d[2] == class_id]
            pred_dets = [d for d in det2 if d[2] == class_id]
            
            # Calculate precision-recall for this class
            matches = []
            scores = []
            matched_gt = set()
            
            for pred in sorted(pred_dets, key=lambda x: x[1], reverse=True):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_dets):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_bbox_metrics(pred[0], gt[0])[0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                matches.append(1 if best_iou >= 0.5 else 0)
                scores.append(pred[1])
                
                if best_gt_idx >= 0 and best_iou >= 0.5:
                    matched_gt.add(best_gt_idx)
            
            if matches:
                tp = np.cumsum(matches)
                fp = np.cumsum([1 - m for m in matches])
                recall = tp / len(gt_dets) if len(gt_dets) > 0 else np.zeros_like(tp)
                precision = tp / (tp + fp)
                
                ap = calculate_ap(precision, recall)
                class_aps[class_id].append(ap)
                class_precisions[class_id].append(np.mean(precision))
                class_recalls[class_id].append(np.mean(recall))
    
    # Calculate final metrics
    for class_id in class_aps:
        metrics['ap_by_class'][class_id] = np.mean(class_aps[class_id])
        metrics['precision_by_class'][class_id] = np.mean(class_precisions[class_id])
        metrics['recall_by_class'][class_id] = np.mean(class_recalls[class_id])
    
    metrics['map'] = np.mean([ap for ap in metrics['ap_by_class'].values()])
    return metrics

def plot_map_comparison(map_results):
    """Plot MAP comparison across experiments"""
    plt.figure(figsize=(12, 6))
    methods = list(map_results.keys())
    map_scores = [results['map'] for results in map_results.values()]
    
    bars = plt.bar(methods, map_scores)
    plt.title('MAP Comparison Across Compression Methods')
    plt.xlabel('Compression Method')
    plt.ylabel('Mean Average Precision (MAP)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('map_comparison.png')
    plt.close()

def analyze_lossy_pairs(experiment_folder):
    """Compare lossy compression methods with different parameters"""
    pairs = [
        ('custom_quantization_bits_4', 'custom_quantization_bits_8'),
        ('custom_quantization_bits_2', 'custom_quantization_bits_8'),
        ('custom_quantization_bits_10', 'custom_quantization_bits_8'),
        ('lossless_blosc2_clevel_4', 'custom_quantization_bits_4'),
        ('lossless_blosc2_clevel_4', 'custom_quantization_bits_2')
    ]
    
    results = {}
    map_results = {}  # Store MAP results for each method
    
    for method1, method2 in pairs:
        print(f"\nAnalyzing pair: {method1} vs {method2}")
        try:
            pred1 = load_predictions(experiment_folder, method1)
            pred2 = load_predictions(experiment_folder, method2)
            
            if not pred1 or not pred2:
                print(f"Warning: No predictions found for {method1} or {method2}")
                continue
            
            # Original metrics
            metrics = compare_predictions(pred1, pred2)
            results[f"{method1}_vs_{method2}"] = metrics
            
            # Calculate MAP metrics
            map_metrics = calculate_map_metrics(pred1, pred2)
            map_results[method1] = map_metrics
            
            # Print detailed results
            print(f"\nResults for {method1} vs {method2}:")
            print(f"Mean Average Precision (MAP): {map_metrics['map']:.4f}")
            print("\nPer-class metrics:")
            for class_id in map_metrics['ap_by_class']:
                print(f"\nClass {class_id}:")
                print(f"  AP: {map_metrics['ap_by_class'][class_id]:.4f}")
                print(f"  Precision: {map_metrics['precision_by_class'][class_id]:.4f}")
                print(f"  Recall: {map_metrics['recall_by_class'][class_id]:.4f}")
            
        except Exception as e:
            print(f"Error analyzing pair {method1} vs {method2}: {str(e)}")
            continue
    
    return results, map_results

def main():
    # Specify your experiment folder
    experiment_folder = "compression_experiments_20241121_200536"  # Update with your timestamp
    
    # Create output directory for analysis
    analysis_output = os.path.join(experiment_folder, "analysis")
    os.makedirs(analysis_output, exist_ok=True)
    
    # Analyze lossless methods
    print("Analyzing lossless compression methods...")
    lossless_results = analyze_lossless_methods(experiment_folder)
    
    # Analyze lossy pairs
    print("Analyzing lossy compression pairs...")
    lossy_results, map_results = analyze_lossy_pairs(experiment_folder)
    
    # Plot MAP comparison
    plot_map_comparison(map_results)
    
    # Save detailed results to file
    output_file = os.path.join(analysis_output, "detailed_analysis.txt")
    with open(output_file, 'w') as f:
        f.write("Compression Analysis Results\n")
        f.write("==========================\n\n")
        
        # Write lossless results
        f.write("Lossless Compression Results:\n")
        f.write("-" * 30 + "\n")
        for pair, metrics in lossless_results.items():
            f.write(f"\n{pair}:\n")
            write_metrics(f, metrics)
        
        # Write lossy results
        f.write("\nLossy Compression Results:\n")
        f.write("-" * 30 + "\n")
        for pair, metrics in lossy_results.items():
            f.write(f"\n{pair}:\n")
            write_metrics(f, metrics)
        
        # Write MAP comparison results
        f.write("\nMAP Comparison Results:\n")
        f.write("-" * 30 + "\n")
        for method, metrics in map_results.items():
            f.write(f"\n{method}:\n")
            f.write(f"MAP: {metrics['map']:.4f}\n")
            f.write("\nPer-class metrics:\n")
            for class_id in metrics['ap_by_class']:
                f.write(f"Class {class_id}:\n")
                f.write(f"  AP: {metrics['ap_by_class'][class_id]:.4f}\n")
                f.write(f"  Precision: {metrics['precision_by_class'][class_id]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_by_class'][class_id]:.4f}\n")
    
    print(f"\nDetailed results have been saved to {output_file}")
    print("MAP comparison plot saved as map_comparison.png")

def write_metrics(f, metrics):
    """Helper function to write metrics to file"""
    f.write(f"Mean Average Precision (mAP): {metrics['map']:.4f}\n")
    
    f.write("\nPer-class Average Precision:\n")
    for class_id, aps in metrics['ap_by_class'].items():
        f.write(f"  Class {class_id}: {np.mean(aps):.4f}\n")
    
    if metrics['confidence_diff']:
        conf_diffs = np.array(metrics['confidence_diff'])
        f.write(f"\nAverage confidence difference: {np.mean(conf_diffs):.4f}\n")
    
    if metrics['iou']:
        ious = np.array(metrics['iou'])
        f.write(f"Average IoU: {np.mean(ious):.4f}\n")
    
    if metrics['total_detections'] > 0:
        match_rate = 100 * metrics['class_matches'] / metrics['total_detections']
        f.write(f"Detection match rate: {match_rate:.1f}%\n")
    
    if metrics['mismatches']:
        f.write("\nMismatch Summary:\n")
        f.write(f"Total mismatches: {len(metrics['mismatches'])}\n")
        
        # Group mismatches by class
        class_mismatches = defaultdict(int)
        for mismatch in metrics['mismatches']:
            class_mismatches[mismatch['class']] += 1
        
        f.write("\nMismatches by class:\n")
        for class_id, count in class_mismatches.items():
            f.write(f"  Class {class_id}: {count} mismatches\n")
    
    f.write("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main() 