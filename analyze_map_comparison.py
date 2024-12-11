import os
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
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
                
            iou = calculate_iou(pred_box, gt_box)
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

def calculate_map_for_experiment(predictions, reference_predictions):
    """Calculate MAP for a single experiment compared to reference"""
    metrics = {
        'map': 0.0,
        'ap_by_class': defaultdict(float),
        'precision_by_class': defaultdict(float),
        'recall_by_class': defaultdict(float),
        'mismatches': []
    }
    
    # Create dictionaries for easier matching
    pred_dict = {img: dets for dets, img, _ in predictions}
    ref_dict = {img: dets for dets, img, _ in reference_predictions}
    
    common_images = sorted(set(pred_dict.keys()) & set(ref_dict.keys()))
    if not common_images:
        return metrics
    
    # Calculate metrics per class
    class_aps = defaultdict(list)
    class_precisions = defaultdict(list)
    class_recalls = defaultdict(list)
    
    for img in common_images:
        pred_dets = pred_dict[img]
        ref_dets = ref_dict[img]
        
        # Get all unique classes
        all_classes = set(d[2] for d in pred_dets) | set(d[2] for d in ref_dets)
        
        for class_id in all_classes:
            # Filter detections by class
            class_preds = [d for d in pred_dets if d[2] == class_id]
            class_refs = [d for d in ref_dets if d[2] == class_id]
            
            precision, recall, _ = calculate_precision_recall(class_refs, class_preds)
            if len(precision) > 0:
                ap = calculate_ap(precision, recall)
                class_aps[class_id].append(ap)
                class_precisions[class_id].append(np.mean(precision))
                class_recalls[class_id].append(np.mean(recall))
                
                # Record significant mismatches
                for ref_det in class_refs:
                    best_match = None
                    best_iou = 0
                    for pred_det in class_preds:
                        iou = calculate_iou(ref_det[0], pred_det[0])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred_det
                    
                    if best_iou < 0.5:  # Significant mismatch
                        metrics['mismatches'].append({
                            'image': img,
                            'class': class_id,
                            'reference': ref_det,
                            'prediction': best_match,
                            'iou': best_iou
                        })
    
    # Calculate final metrics
    for class_id in class_aps:
        metrics['ap_by_class'][class_id] = np.mean(class_aps[class_id])
        metrics['precision_by_class'][class_id] = np.mean(class_precisions[class_id])
        metrics['recall_by_class'][class_id] = np.mean(class_recalls[class_id])
    
    # Calculate MAP
    metrics['map'] = np.mean([ap for ap in metrics['ap_by_class'].values()])
    
    return metrics

def plot_map_comparison(experiment_results):
    """Plot MAP comparison across experiments"""
    methods = list(experiment_results.keys())
    map_scores = [results['map'] for results in experiment_results.values()]
    
    plt.figure(figsize=(12, 6))
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

def main():
    # Specify your experiment folder
    experiment_folder = "compression_experiments_20241121_200536"  # Update with your timestamp
    
    # Define experiments to analyze
    experiments = [
        'custom_quantization_bits_2',
        'custom_quantization_bits_4',
        'custom_quantization_bits_8',
        'custom_quantization_bits_10',
        'lossless_blosc2_clevel_4',
        'lossless_zlib_level_9'
    ]
    
    # Load reference predictions (using bits_8 as reference)
    reference_method = 'custom_quantization_bits_8'
    try:
        reference_file = os.path.join(experiment_folder, reference_method, "predictions.pkl")
        with open(reference_file, 'rb') as f:
            reference_predictions = pickle.load(f)
    except Exception as e:
        print(f"Error loading reference predictions: {e}")
        return
    
    # Calculate MAP for each experiment
    results = {}
    for experiment in experiments:
        if experiment == reference_method:
            continue
            
        print(f"\nAnalyzing {experiment}...")
        try:
            pred_file = os.path.join(experiment_folder, experiment, "predictions.pkl")
            with open(pred_file, 'rb') as f:
                predictions = pickle.load(f)
            
            metrics = calculate_map_for_experiment(predictions, reference_predictions)
            results[experiment] = metrics
            
            # Print results
            print(f"\nResults for {experiment}:")
            print(f"MAP: {metrics['map']:.4f}")
            print("\nPer-class metrics:")
            for class_id in metrics['ap_by_class']:
                print(f"\nClass {class_id}:")
                print(f"  AP: {metrics['ap_by_class'][class_id]:.4f}")
                print(f"  Precision: {metrics['precision_by_class'][class_id]:.4f}")
                print(f"  Recall: {metrics['recall_by_class'][class_id]:.4f}")
            
            if metrics['mismatches']:
                print(f"\nTotal mismatches: {len(metrics['mismatches'])}")
                print("Top 3 worst mismatches:")
                sorted_mismatches = sorted(metrics['mismatches'], key=lambda x: x['iou'])[:3]
                for i, mismatch in enumerate(sorted_mismatches):
                    print(f"\nMismatch {i+1}:")
                    print(f"  Image: {mismatch['image']}")
                    print(f"  Class: {mismatch['class']}")
                    print(f"  IoU: {mismatch['iou']:.4f}")
        
        except Exception as e:
            print(f"Error analyzing {experiment}: {e}")
            continue
    
    # Plot MAP comparison
    plot_map_comparison(results)
    
    # Save detailed results to file
    output_file = os.path.join(experiment_folder, "map_analysis_results.txt")
    with open(output_file, 'w') as f:
        f.write("MAP Analysis Results\n")
        f.write("==================\n\n")
        f.write(f"Reference method: {reference_method}\n\n")
        
        for experiment, metrics in results.items():
            f.write(f"\n{experiment}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"MAP: {metrics['map']:.4f}\n")
            
            f.write("\nPer-class metrics:\n")
            for class_id in metrics['ap_by_class']:
                f.write(f"\nClass {class_id}:\n")
                f.write(f"  AP: {metrics['ap_by_class'][class_id]:.4f}\n")
                f.write(f"  Precision: {metrics['precision_by_class'][class_id]:.4f}\n")
                f.write(f"  Recall: {metrics['recall_by_class'][class_id]:.4f}\n")
            
            if metrics['mismatches']:
                f.write(f"\nTotal mismatches: {len(metrics['mismatches'])}\n")
                f.write("Mismatches by class:\n")
                class_mismatches = defaultdict(int)
                for mismatch in metrics['mismatches']:
                    class_mismatches[mismatch['class']] += 1
                for class_id, count in class_mismatches.items():
                    f.write(f"  Class {class_id}: {count} mismatches\n")
            
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"\nResults saved to {output_file}")
    print("MAP comparison plot saved as map_comparison.png")

if __name__ == "__main__":
    main() 