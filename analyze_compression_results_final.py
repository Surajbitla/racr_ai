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
import argparse

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
            
            iou, _ = calculate_bbox_metrics(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        matches.append(1 if best_iou >= iou_threshold else 0)
        scores.append(pred_score)
        
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
    
    tp = np.cumsum(matches)
    fp = np.cumsum([1 - m for m in matches])
    recall = tp / len(gt_dets) if len(gt_dets) > 0 else np.zeros_like(tp)
    precision = tp / (tp + fp)
    
    return precision, recall, scores

def calculate_ap(precision, recall):
    """Calculate Average Precision using 11-point interpolation"""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap

def analyze_experiment_pair(pred1, pred2, name1, name2):
    """Analyze and compare two experiments in detail"""
    metrics = {
        'confidence_diffs': [],
        'iou_values': [],
        'center_distances': [],
        'class_matches': 0,
        'total_detections': 0,
        'ap_by_class': defaultdict(list),
        'map': 0.0,
        'mismatches': [],
        'class_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0})
    }
    
    # Create dictionaries with image names as keys
    pred1_dict = {img: det for det, img in pred1}
    pred2_dict = {img: det for det, img in pred2}
    
    common_images = sorted(set(pred1_dict.keys()) & set(pred2_dict.keys()))
    print(f"\nAnalyzing {len(common_images)} common images between {name2} and {name1}")
    
    for img in common_images:
        det1 = pred1_dict[img]
        det2 = pred2_dict[img]
        
        # Analyze per class
        all_classes = set(d[2] for d in det1) | set(d[2] for d in det2)
        
        for class_id in all_classes:
            # Filter detections by class
            gt_dets = [d for d in det1 if d[2] == class_id]
            pred_dets = [d for d in det2 if d[2] == class_id]
            
            precision, recall, scores = calculate_precision_recall(gt_dets, pred_dets)
            if len(precision) > 0:
                ap = calculate_ap(precision, recall)
                metrics['ap_by_class'][class_id].append(ap)
                
                # Update class metrics
                metrics['class_metrics'][class_id]['precision'] = np.mean(precision)
                metrics['class_metrics'][class_id]['recall'] = np.mean(recall)
            
            # Detailed matching analysis
            for gt in gt_dets:
                best_match = None
                best_iou = 0
                best_dist = float('inf')
                
                for pred in pred_dets:
                    if gt[2] == pred[2]:
                        iou, dist = calculate_bbox_metrics(gt[0], pred[0])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred
                            best_dist = dist
                
                if best_match and best_iou >= 0.5:
                    metrics['class_metrics'][class_id]['tp'] += 1
                    metrics['iou_values'].append(best_iou)
                    metrics['confidence_diffs'].append(abs(gt[1] - best_match[1]))
                    metrics['center_distances'].append(best_dist)
                    metrics['class_matches'] += 1
                    
                    # Record significant differences
                    if abs(gt[1] - best_match[1]) > 0.1 or best_iou < 0.8:
                        metrics['mismatches'].append({
                            'image': img,
                            'class': class_id,
                            'ground_truth': gt,
                            'prediction': best_match,
                            'iou': best_iou,
                            'confidence_diff': abs(gt[1] - best_match[1]),
                            'center_distance': best_dist
                        })
                else:
                    metrics['class_metrics'][class_id]['fn'] += 1
            
            # Count false positives
            matched_preds = sum(1 for pred in pred_dets if any(
                calculate_bbox_metrics(gt[0], pred[0])[0] >= 0.5 and gt[2] == pred[2]
                for gt in gt_dets
            ))
            metrics['class_metrics'][class_id]['fp'] += len(pred_dets) - matched_preds
        
        metrics['total_detections'] += len(det1)
    
    # Calculate final MAP
    for class_id in metrics['ap_by_class']:
        class_aps = metrics['ap_by_class'][class_id]
        metrics['ap_by_class'][class_id] = np.mean(class_aps)
    
    metrics['map'] = np.mean(list(metrics['ap_by_class'].values()))
    
    return metrics

def plot_comparison_metrics(results_dict, output_dir):
    """Generate visualization plots for the analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot MAP comparison
    plt.figure(figsize=(12, 6))
    methods = list(results_dict.keys())
    maps = [results['map'] for results in results_dict.values()]
    
    plt.bar(methods, maps)
    plt.title('Mean Average Precision (MAP) Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MAP')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_comparison.png'))
    plt.close()
    
    # Plot IoU distribution
    plt.figure(figsize=(12, 6))
    for method, results in results_dict.items():
        if results['iou_values']:
            sns.kdeplot(data=results['iou_values'], label=method)
    plt.title('IoU Distribution Comparison')
    plt.xlabel('IoU')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_distribution.png'))
    plt.close()
    
    # Plot confidence differences
    plt.figure(figsize=(12, 6))
    for method, results in results_dict.items():
        if results['confidence_diffs']:
            sns.kdeplot(data=results['confidence_diffs'], label=method)
    plt.title('Confidence Score Differences Distribution')
    plt.xlabel('Absolute Confidence Difference')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_diffs.png'))
    plt.close()
    
    # Plot center distances
    plt.figure(figsize=(12, 6))
    for method, results in results_dict.items():
        if results['center_distances']:
            sns.kdeplot(data=results['center_distances'], label=method)
    plt.title('Bounding Box Center Distance Distribution')
    plt.xlabel('Center Distance (pixels)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'center_distances.png'))
    plt.close()

def write_detailed_analysis(results, baseline, output_dir):
    """Write detailed analysis to file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "detailed_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write("Compression Analysis Results\n")
        f.write("==========================\n\n")
        
        # Write summary table
        f.write("Summary Table:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Method':<30} {'MAP':>10} {'Avg IoU':>10} {'Avg Conf Diff':>15} {'Match Rate':>12}\n")
        f.write("-" * 100 + "\n")
        
        for method, result in results.items():
            map_score = result['map']
            avg_iou = np.mean(result['iou_values']) if result['iou_values'] else 0
            avg_conf = np.mean(result['confidence_diffs']) if result['confidence_diffs'] else 0
            match_rate = 100 * result['class_matches'] / result['total_detections'] if result['total_detections'] > 0 else 0
            
            f.write(f"{method:<30} {map_score:>10.4f} {avg_iou:>10.4f} {avg_conf:>15.4f} {match_rate:>11.1f}%\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Write detailed analysis for each method
        for method, result in results.items():
            f.write(f"\nDetailed Analysis: {method} vs {baseline}\n")
            f.write("=" * 50 + "\n")
            
            # Overall metrics
            f.write(f"Mean Average Precision (MAP): {result['map']:.4f}\n")
            f.write(f"Total Detections: {result['total_detections']}\n")
            f.write(f"Matched Detections: {result['class_matches']}\n")
            f.write(f"Match Rate: {100 * result['class_matches'] / result['total_detections']:.1f}%\n\n")
            
            # Per-class analysis
            f.write("Per-class Analysis:\n")
            f.write("-" * 30 + "\n")
            for class_id, metrics in result['class_metrics'].items():
                f.write(f"\nClass {class_id}:\n")
                f.write(f"  Average Precision: {result['ap_by_class'][class_id]:.4f}\n")
                f.write(f"  True Positives: {metrics['tp']}\n")
                f.write(f"  False Positives: {metrics['fp']}\n")
                f.write(f"  False Negatives: {metrics['fn']}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
            
            # Statistical analysis
            f.write("\nStatistical Analysis:\n")
            f.write("-" * 30 + "\n")
            
            if result['iou_values']:
                ious = np.array(result['iou_values'])
                f.write("\nIoU Statistics:\n")
                f.write(f"  Mean: {np.mean(ious):.4f}\n")
                f.write(f"  Median: {np.median(ious):.4f}\n")
                f.write(f"  Std Dev: {np.std(ious):.4f}\n")
                f.write(f"  Min: {np.min(ious):.4f}\n")
                f.write(f"  Max: {np.max(ious):.4f}\n")
            
            if result['confidence_diffs']:
                conf_diffs = np.array(result['confidence_diffs'])
                f.write("\nConfidence Difference Statistics:\n")
                f.write(f"  Mean: {np.mean(conf_diffs):.4f}\n")
                f.write(f"  Median: {np.median(conf_diffs):.4f}\n")
                f.write(f"  Std Dev: {np.std(conf_diffs):.4f}\n")
                f.write(f"  Max: {np.max(conf_diffs):.4f}\n")
            
            if result['center_distances']:
                distances = np.array(result['center_distances'])
                f.write("\nBounding Box Center Distance Statistics:\n")
                f.write(f"  Mean: {np.mean(distances):.4f}\n")
                f.write(f"  Median: {np.median(distances):.4f}\n")
                f.write(f"  Std Dev: {np.std(distances):.4f}\n")
                f.write(f"  Max: {np.max(distances):.4f}\n")
            
            # Mismatch analysis
            if result['mismatches']:
                f.write("\nMismatch Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total mismatches: {len(result['mismatches'])}\n")
                
                # Group mismatches by class
                class_mismatches = defaultdict(int)
                for mismatch in result['mismatches']:
                    class_mismatches[mismatch['class']] += 1
                
                f.write("\nMismatches by class:\n")
                for class_id, count in class_mismatches.items():
                    f.write(f"  Class {class_id}: {count} mismatches\n")
                
                # Show worst mismatches
                f.write("\nTop 5 Worst Mismatches:\n")
                sorted_mismatches = sorted(result['mismatches'], 
                                        key=lambda x: (x['iou'], -x['confidence_diff']))[:5]
                for i, mismatch in enumerate(sorted_mismatches, 1):
                    f.write(f"\nMismatch {i}:\n")
                    f.write(f"  Image: {mismatch['image']}\n")
                    f.write(f"  Class: {mismatch['class']}\n")
                    f.write(f"  IoU: {mismatch['iou']:.4f}\n")
                    f.write(f"  Confidence Difference: {mismatch['confidence_diff']:.4f}\n")
                    f.write(f"  Center Distance: {mismatch['center_distance']:.4f}\n")
            
            f.write("\n" + "=" * 100 + "\n")

def write_detailed_mismatch_analysis(results, baseline, output_dir):
    """Write detailed analysis focusing on experiments with mismatches"""
    output_file = os.path.join(output_dir, "detailed_mismatch_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write("Detailed Mismatch Analysis Report\n")
        f.write("================================\n\n")
        f.write(f"Baseline: {baseline}\n")
        f.write("This report focuses on experiments that don't have 100% match rate\n")
        f.write("=" * 80 + "\n\n")
        
        # Find experiments with mismatches
        for method, result in results.items():
            match_rate = 100 * result['class_matches'] / result['total_detections']
            if match_rate < 100:
                f.write(f"\nDetailed Analysis for: {method}\n")
                f.write("-" * 50 + "\n")
                
                # Overall statistics
                f.write("\nOverall Performance:\n")
                f.write(f"Match Rate: {match_rate:.2f}%\n")
                f.write(f"Total Detections: {result['total_detections']}\n")
                f.write(f"Matched Detections: {result['class_matches']}\n")
                f.write(f"Mismatched Detections: {result['total_detections'] - result['class_matches']}\n")
                f.write(f"MAP Score: {result['map']:.4f}\n")
                
                # Per-class analysis where mismatches occur
                f.write("\nPer-class Analysis:\n")
                for class_id, metrics in result['class_metrics'].items():
                    if metrics['fn'] > 0 or metrics['fp'] > 0:
                        f.write(f"\nClass {class_id}:\n")
                        f.write(f"  Average Precision: {result['ap_by_class'][class_id]:.4f}\n")
                        f.write(f"  True Positives: {metrics['tp']}\n")
                        f.write(f"  False Positives: {metrics['fp']}\n")
                        f.write(f"  False Negatives: {metrics['fn']}\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall: {metrics['recall']:.4f}\n")
                
                # Confidence score analysis
                if result['confidence_diffs']:
                    conf_diffs = np.array(result['confidence_diffs'])
                    f.write("\nConfidence Score Analysis:\n")
                    f.write(f"  Mean Difference: {np.mean(conf_diffs):.4f}\n")
                    f.write(f"  Median Difference: {np.median(conf_diffs):.4f}\n")
                    f.write(f"  Max Difference: {np.max(conf_diffs):.4f}\n")
                    f.write(f"  Std Dev: {np.std(conf_diffs):.4f}\n")
                    
                    # Distribution of confidence differences
                    f.write("\nConfidence Difference Distribution:\n")
                    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, float('inf')]
                    hist, _ = np.histogram(conf_diffs, bins=bins)
                    for i in range(len(bins)-1):
                        f.write(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} instances\n")
                
                # IoU analysis
                if result['iou_values']:
                    ious = np.array(result['iou_values'])
                    f.write("\nIoU Analysis:\n")
                    f.write(f"  Mean IoU: {np.mean(ious):.4f}\n")
                    f.write(f"  Median IoU: {np.median(ious):.4f}\n")
                    f.write(f"  Min IoU: {np.min(ious):.4f}\n")
                    f.write(f"  Max IoU: {np.max(ious):.4f}\n")
                    
                    # Distribution of IoU values
                    f.write("\nIoU Distribution:\n")
                    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    hist, _ = np.histogram(ious, bins=bins)
                    for i in range(len(bins)-1):
                        f.write(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} instances\n")
                
                # Detailed mismatch analysis
                if result['mismatches']:
                    f.write("\nDetailed Mismatch Analysis:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Group mismatches by class
                    class_mismatches = defaultdict(list)
                    for mismatch in result['mismatches']:
                        class_mismatches[mismatch['class']].append(mismatch)
                    
                    for class_id, mismatches in class_mismatches.items():
                        f.write(f"\nClass {class_id} Mismatches:\n")
                        f.write(f"Total mismatches: {len(mismatches)}\n")
                        
                        # Calculate average metrics for this class
                        avg_iou = np.mean([m['iou'] for m in mismatches])
                        avg_conf_diff = np.mean([m['confidence_diff'] for m in mismatches])
                        
                        f.write(f"Average IoU: {avg_iou:.4f}\n")
                        f.write(f"Average confidence difference: {avg_conf_diff:.4f}\n")
                        
                        # List worst mismatches for this class
                        f.write("\nWorst 5 mismatches for this class:\n")
                        sorted_mismatches = sorted(mismatches, key=lambda x: (x['iou'], -x['confidence_diff']))[:5]
                        for i, mismatch in enumerate(sorted_mismatches, 1):
                            f.write(f"\nMismatch {i}:\n")
                            f.write(f"  Image: {mismatch['image']}\n")
                            f.write(f"  IoU: {mismatch['iou']:.4f}\n")
                            f.write(f"  Confidence Difference: {mismatch['confidence_diff']:.4f}\n")
                            if 'center_distance' in mismatch:
                                f.write(f"  Center Distance: {mismatch['center_distance']:.4f}\n")
                            f.write(f"  Ground Truth: conf={mismatch['ground_truth'][1]:.4f}\n")
                            f.write(f"  Prediction: conf={mismatch['prediction'][1]:.4f}\n")
                    
                    # Analysis of mismatch patterns
                    f.write("\nMismatch Pattern Analysis:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Analyze if mismatches are more common in certain confidence ranges
                    gt_confs = [m['ground_truth'][1] for m in result['mismatches']]
                    pred_confs = [m['prediction'][1] for m in result['mismatches']]
                    
                    f.write("\nGround Truth Confidence Distribution in Mismatches:\n")
                    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    hist, _ = np.histogram(gt_confs, bins=bins)
                    for i in range(len(bins)-1):
                        f.write(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} instances\n")
                    
                    f.write("\nPrediction Confidence Distribution in Mismatches:\n")
                    hist, _ = np.histogram(pred_confs, bins=bins)
                    for i in range(len(bins)-1):
                        f.write(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} instances\n")
                    
                    # Summary of findings
                    f.write("\nSummary of Mismatch Patterns:\n")
                    f.write("- Most common confidence range for mismatches: ")
                    gt_hist, _ = np.histogram(gt_confs, bins=bins)
                    most_common_range_idx = np.argmax(gt_hist)
                    f.write(f"{bins[most_common_range_idx]:.1f}-{bins[most_common_range_idx+1]:.1f}\n")
                    
                    # Recommendations
                    f.write("\nRecommendations:\n")
                    if np.mean(conf_diffs) > 0.3:
                        f.write("- High confidence differences suggest potential calibration issues\n")
                    if np.mean(ious) < 0.7:
                        f.write("- Low IoU values indicate localization problems\n")
                    if len(class_mismatches) == 1:
                        f.write("- Mismatches concentrated in one class suggest class-specific issues\n")
                    
                f.write("\n" + "=" * 80 + "\n")

def main():
    # Hardcoded configuration
    experiment_folder = "compression_experiments_20241127_130804"  # Your experiment folder
    output_dir = os.path.join(experiment_folder, "analysis")
    baseline = "lossless_blosc2_clevel_4"  # Your baseline experiment
    
    print(f"\nAnalyzing compression experiments in: {experiment_folder}")
    print(f"Using baseline: {baseline}")
    print(f"Results will be saved to: {output_dir}")
    
    # Validate experiment folder
    if not os.path.isdir(experiment_folder):
        print(f"Error: Experiment folder '{experiment_folder}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available experiments
    experiments = [d for d in os.listdir(experiment_folder) 
                  if os.path.isdir(os.path.join(experiment_folder, d))]
    
    if not experiments:
        print(f"Error: No experiment directories found in {experiment_folder}")
        sys.exit(1)
    
    if baseline not in experiments:
        print(f"Error: Baseline experiment '{baseline}' not found")
        print("Available experiments:", experiments)
        sys.exit(1)
    
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = {}
    
    # Load baseline first
    try:
        predictions[baseline] = load_predictions(experiment_folder, baseline)
        print(f"Loaded baseline: {baseline}")
    except Exception as e:
        print(f"Error loading baseline {baseline}: {e}")
        sys.exit(1)
    
    # Load other experiments
    for exp in experiments:
        if exp == baseline:
            continue
        try:
            predictions[exp] = load_predictions(experiment_folder, exp)
            print(f"Loaded experiment: {exp}")
        except Exception as e:
            print(f"Error loading {exp}: {e}")
            continue
    
    # Analyze experiments
    results = {}
    for exp in experiments:
        if exp == baseline or exp not in predictions:
            continue
        
        print(f"\nAnalyzing {exp} vs {baseline}...")
        results[exp] = analyze_experiment_pair(
            predictions[baseline],
            predictions[exp],
            baseline,
            exp
        )
    
    # Generate visualizations
    print("\nGenerating visualization plots...")
    plot_comparison_metrics(results, output_dir)
    
    # Write detailed analysis
    print("\nWriting detailed analysis...")
    write_detailed_analysis(results, baseline, output_dir)
    
    # Add the detailed mismatch analysis
    print("\nGenerating detailed mismatch analysis...")
    write_detailed_mismatch_analysis(results, baseline, output_dir)
    
    # Save configuration
    config = {
        'experiment_folder': os.path.abspath(experiment_folder),
        'output_directory': os.path.abspath(output_dir),
        'baseline': baseline,
        'experiments_analyzed': list(results.keys()),
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, "analysis_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved in {output_dir}/")
    
    # Print summary
    print("\nQuick Summary:")
    print("-" * 80)
    print(f"{'Method':<30} {'MAP':>10} {'Avg IoU':>10} {'Match Rate':>12}")
    print("-" * 80)
    
    for method, result in results.items():
        map_score = result['map']
        avg_iou = np.mean(result['iou_values']) if result['iou_values'] else 0
        match_rate = 100 * result['class_matches'] / result['total_detections'] if result['total_detections'] > 0 else 0
        print(f"{method:<30} {map_score:>10.4f} {avg_iou:>10.4f} {match_rate:>11.1f}%")
    
    # Print recommendations
    print("\nAnalysis Recommendations:")
    print("-" * 80)
    
    # Find best method based on different metrics
    best_map = max(results.items(), key=lambda x: x[1]['map'])
    best_iou = max(results.items(), key=lambda x: np.mean(x[1]['iou_values']) if x[1]['iou_values'] else 0)
    best_match_rate = max(results.items(), key=lambda x: x[1]['class_matches'] / x[1]['total_detections'])
    
    print(f"Best MAP Score: {best_map[0]} ({best_map[1]['map']:.4f})")
    print(f"Best IoU: {best_iou[0]} (Average IoU: {np.mean(best_iou[1]['iou_values']):.4f})")
    print(f"Best Detection Match Rate: {best_match_rate[0]} ({100 * best_match_rate[1]['class_matches'] / best_match_rate[1]['total_detections']:.1f}%)")
    
    # Overall recommendation
    print("\nOverall Recommendation:")
    methods_scores = {}
    for method, result in results.items():
        # Combine multiple metrics for overall score
        map_score = result['map']
        iou_score = np.mean(result['iou_values']) if result['iou_values'] else 0
        match_rate = result['class_matches'] / result['total_detections']
        
        # Weight the metrics (you can adjust these weights)
        overall_score = 0.5 * map_score + 0.3 * iou_score + 0.2 * match_rate
        methods_scores[method] = overall_score
    
    best_method = max(methods_scores.items(), key=lambda x: x[1])
    print(f"Best overall method: {best_method[0]} (Combined score: {best_method[1]:.4f})")
    print("This recommendation considers MAP score (50%), IoU (30%), and detection match rate (20%)")

if __name__ == "__main__":
    main() 