import os
import pickle
import numpy as np
from collections import defaultdict

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

def analyze_predictions(method1_data, method2_data, method1_name, method2_name):
    """Analyze predictions and calculate metrics including mAP"""
    results = {
        'confidence_diff': [],
        'iou_values': [],
        'class_matches': 0,
        'total_detections': 0,
        'mismatches': [],
        'ap_by_class': defaultdict(list),
        'map': 0.0
    }
    
    # Create dictionaries for easier matching
    method1_dict = {img: dets for dets, img, _ in method1_data}
    method2_dict = {img: dets for dets, img, _ in method2_data}
    
    common_images = sorted(set(method1_dict.keys()) & set(method2_dict.keys()))
    print(f"\nAnalyzing {len(common_images)} common images")
    
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'ap': 0.0})
    
    for img in common_images:
        dets1 = method1_dict[img]
        dets2 = method2_dict[img]
        
        # Calculate precision-recall and AP for each class
        all_classes = set(d[2] for d in dets1) | set(d[2] for d in dets2)
        
        for class_id in all_classes:
            # Filter detections by class
            gt_dets = [d for d in dets1 if d[2] == class_id]
            pred_dets = [d for d in dets2 if d[2] == class_id]
            
            precision, recall, scores = calculate_precision_recall(gt_dets, pred_dets)
            if len(precision) > 0:
                ap = calculate_ap(precision, recall)
                results['ap_by_class'][class_id].append(ap)
            
            # Count matches and mismatches
            matched_gt = set()
            matched_pred = set()
            
            for i, gt in enumerate(gt_dets):
                best_iou = 0
                best_pred = None
                best_pred_idx = -1
                
                for j, pred in enumerate(pred_dets):
                    if j in matched_pred:
                        continue
                        
                    iou = calculate_iou(gt[0], pred[0])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred
                        best_pred_idx = j
                
                if best_iou >= 0.5:
                    matched_gt.add(i)
                    matched_pred.add(best_pred_idx)
                    class_metrics[class_id]['tp'] += 1
                    
                    # Record metrics for matched detections
                    results['iou_values'].append(best_iou)
                    results['confidence_diff'].append(abs(gt[1] - best_pred[1]))
                    results['class_matches'] += 1
                else:
                    # Record mismatch details
                    results['mismatches'].append({
                        'image': img,
                        'class': class_id,
                        'ground_truth': gt,
                        'best_prediction': best_pred,
                        'iou': best_iou
                    })
                    class_metrics[class_id]['fn'] += 1
            
            # Count false positives
            class_metrics[class_id]['fp'] += len(pred_dets) - len(matched_pred)
        
        results['total_detections'] += len(dets1)
    
    # Calculate final metrics
    for class_id in results['ap_by_class']:
        class_aps = results['ap_by_class'][class_id]
        class_metrics[class_id]['ap'] = np.mean(class_aps)
    
    results['map'] = np.mean([metrics['ap'] for metrics in class_metrics.values()])
    
    # Print detailed results
    print(f"\nComparison Results: {method2_name} vs {method1_name}")
    print("-" * 50)
    print(f"Mean Average Precision (mAP): {results['map']:.4f}")
    print("\nPer-class Metrics:")
    
    for class_id, metrics in class_metrics.items():
        print(f"\nClass {class_id}:")
        print(f"  Average Precision: {metrics['ap']:.4f}")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
        recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
    
    if results['mismatches']:
        print("\nDetailed Mismatch Analysis:")
        print(f"Total mismatches: {len(results['mismatches'])}")
        print("\nTop 5 worst mismatches:")
        sorted_mismatches = sorted(results['mismatches'], key=lambda x: x['iou'])
        for i, mismatch in enumerate(sorted_mismatches[:5]):
            print(f"\nMismatch {i+1}:")
            print(f"  Image: {mismatch['image']}")
            print(f"  Class: {mismatch['class']}")
            print(f"  IoU: {mismatch['iou']:.4f}")
            print(f"  Ground Truth: {mismatch['ground_truth']}")
            print(f"  Best Prediction: {mismatch['best_prediction']}")
    
    return results

def main():
    # Load prediction files
    methods = [
        ('custom_quantization_bits_8', 'custom_quantization_bits_8.pkl'),  # reference
        ('custom_quantization_bits_4', 'custom_quantization_bits_4.pkl'),
        ('custom_quantization_bits_2', 'custom_quantization_bits_2.pkl'),
        ('custom_quantization_bits_10', 'custom_quantization_bits_10.pkl'),
        ('lossless_blosc2_clevel_4', 'lossless_blosc2_clevel_4.pkl')
    ]
    
    print("Loading prediction files...")
    data = {}
    for method_name, filename in methods:
        try:
            with open(filename, 'rb') as f:
                data[method_name] = pickle.load(f)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    # Compare each method against the reference
    reference_method = 'custom_quantization_bits_8'
    if reference_method not in data:
        print(f"Error: Reference method {reference_method} not found")
        return
    
    results = {}
    for method_name in data:
        if method_name == reference_method:
            continue
        
        print(f"\nAnalyzing {method_name} vs {reference_method}...")
        results[method_name] = analyze_predictions(
            data[reference_method],
            data[method_name],
            reference_method,
            method_name
        )
    
    # Save detailed results to file
    output_file = "detailed_map_analysis.txt"
    with open(output_file, 'w') as f:
        f.write("Detailed MAP Analysis Results\n")
        f.write("===========================\n\n")
        
        for method_name, result in results.items():
            f.write(f"\n{method_name} vs {reference_method}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean Average Precision (mAP): {result['map']:.4f}\n")
            
            f.write("\nPer-class Average Precision:\n")
            for class_id, aps in result['ap_by_class'].items():
                f.write(f"  Class {class_id}: {np.mean(aps):.4f}\n")
            
            if result['confidence_diff']:
                f.write("\nConfidence Score Differences:\n")
                conf_diffs = np.array(result['confidence_diff'])
                f.write(f"  Mean: {np.mean(conf_diffs):.6f}\n")
                f.write(f"  Std: {np.std(conf_diffs):.6f}\n")
                f.write(f"  Max: {np.max(conf_diffs):.6f}\n")
            
            if result['iou_values']:
                f.write("\nIoU Statistics:\n")
                ious = np.array(result['iou_values'])
                f.write(f"  Mean: {np.mean(ious):.4f}\n")
                f.write(f"  Min: {np.min(ious):.4f}\n")
                f.write(f"  Max: {np.max(ious):.4f}\n")
            
            f.write(f"\nTotal detections: {result['total_detections']}\n")
            f.write(f"Matched detections: {result['class_matches']}\n")
            f.write(f"Match rate: {100 * result['class_matches'] / result['total_detections']:.2f}%\n")
            
            if result['mismatches']:
                f.write("\nMismatch Summary:\n")
                f.write(f"Total mismatches: {len(result['mismatches'])}\n")
                
                # Group mismatches by class
                class_mismatches = defaultdict(int)
                for mismatch in result['mismatches']:
                    class_mismatches[mismatch['class']] += 1
                
                f.write("\nMismatches by class:\n")
                for class_id, count in class_mismatches.items():
                    f.write(f"  Class {class_id}: {count} mismatches\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nDetailed results have been saved to {output_file}")

if __name__ == "__main__":
    main() 