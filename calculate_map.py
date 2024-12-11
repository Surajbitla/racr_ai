import os
import pickle
import numpy as np
from pathlib import Path
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_predictions(experiment_folder, experiment_name):
    """Load predictions.pkl file for a given experiment"""
    pred_file = os.path.join(experiment_folder, experiment_name, "predictions.pkl")
    print(f"Loading predictions from: {pred_file}")
    
    with open(pred_file, 'rb') as f:
        preds = pickle.load(f)
        print(f"Number of predictions loaded for {experiment_name}: {len(preds)}")
        if len(preds) > 0:
            print(f"Sample prediction format:")
            print(f"Prediction type: {type(preds[0])}")
            print(f"First prediction: {preds[0]}")
            if isinstance(preds[0], tuple):
                print(f"First prediction boxes: {preds[0][0]}")
                print(f"First prediction image name: {preds[0][1]}")
    return preds

def load_ground_truth(gt_folder):
    """Load ground truth annotations in YOLO format from the same directory as images"""
    # Look for txt files in the same directory
    gt_files = glob.glob(os.path.join(gt_folder, "*.txt"))
    ground_truths = {}
    
    print(f"Looking for ground truth files in: {gt_folder}")
    
    for gt_file in gt_files:
        # Get corresponding image name (same name, different extension)
        base_name = os.path.basename(gt_file)
        image_name = os.path.splitext(base_name)[0] + '.jpg'
        boxes = []
        
        with open(gt_file, 'r') as f:
            for line in f:
                try:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert YOLO format to [x, y, w, h]
                    x = x_center - width/2
                    y = y_center - height/2
                    boxes.append(([x, y, width, height], 1.0, int(class_id)))  # confidence is 1.0 for GT
                except ValueError as e:
                    print(f"Warning: Skipping malformed line in {gt_file}: {line.strip()}")
                    continue
                
        ground_truths[image_name] = boxes
    
    print(f"Loaded {len(ground_truths)} ground truth annotations")
    if len(ground_truths) == 0:
        print("Warning: No ground truth annotations found!")
        print(f"Checked for .txt files in: {gt_folder}")
    return ground_truths

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in [x, y, w, h] format"""
    # Convert to [x1, y1, x2, y2] format
    b1_x1, b1_y1 = float(box1[0]), float(box1[1])
    b1_x2, b1_y2 = float(box1[0] + box1[2]), float(box1[1] + box1[3])
    
    b2_x1, b2_y1 = float(box2[0]), float(box2[1])
    b2_x2, b2_y2 = float(box2[0] + box2[2]), float(box2[1] + box2[3])
    
    # Calculate intersection coordinates
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # Check if there is valid intersection
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    # Calculate areas
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    # Avoid division by zero
    if union_area <= 0:
        return 0.0
        
    iou = inter_area / union_area
    
    # Debug output
    print(f"\nIoU Calculation Debug:")
    print(f"Box1: x1={b1_x1:.3f}, y1={b1_y1:.3f}, x2={b1_x2:.3f}, y2={b1_y2:.3f}")
    print(f"Box2: x1={b2_x1:.3f}, y1={b2_y1:.3f}, x2={b2_x2:.3f}, y2={b2_y2:.3f}")
    print(f"Intersection: x1={inter_x1:.3f}, y1={inter_y1:.3f}, x2={inter_x2:.3f}, y2={inter_y2:.3f}")
    print(f"Areas: box1={b1_area:.3f}, box2={b2_area:.3f}, inter={inter_area:.3f}, union={union_area:.3f}")
    print(f"IoU: {iou:.3f}")
    
    return float(iou)

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.3):
    """Calculate precision and recall values"""
    if not gt_boxes or not pred_boxes:
        return np.array([]), np.array([]), []
    
    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
    
    print(f"\nPrecision-Recall Calculation:")
    print(f"Number of GT boxes: {len(gt_boxes)}")
    print(f"Number of Pred boxes: {len(pred_boxes)}")
    print(f"IoU threshold: {iou_threshold}")
    
    matches = []
    scores = []
    matched_gt = set()
    
    for pred_idx, pred in enumerate(pred_boxes):
        pred_box, pred_score, pred_class = pred
        best_iou = 0.0
        best_gt_idx = -1
        
        print(f"\nProcessing prediction {pred_idx + 1} with confidence {pred_score:.3f}")
        
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
                
            gt_box, _, gt_class = gt
            if gt_class != pred_class:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            print(f"IoU with GT box {gt_idx + 1}: {iou:.3f}")
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        is_match = best_iou >= iou_threshold
        matches.append(float(is_match))
        scores.append(float(pred_score))
        
        print(f"Best IoU: {best_iou:.3f} with GT box {best_gt_idx + 1 if best_gt_idx >= 0 else 'none'}")
        print(f"Match: {'Yes' if is_match else 'No'}")
        
        if is_match:
            matched_gt.add(best_gt_idx)
    
    # Convert to numpy arrays with explicit float type
    matches = np.array(matches, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    
    # Calculate cumulative metrics
    tp = np.cumsum(matches)
    fp = np.cumsum(1.0 - matches)
    
    # Calculate precision and recall with explicit float division
    recall = tp / float(len(gt_boxes)) if len(gt_boxes) > 0 else np.zeros_like(tp)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    
    print("\nFinal Precision-Recall values:")
    print(f"Precision values: {precision}")
    print(f"Recall values: {recall}")
    
    return precision, recall, scores

def calculate_ap(precision, recall):
    """Calculate Average Precision using 11-point interpolation"""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0
    
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0.0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return float(ap)

def normalize_predictions(predictions, image_size=(512, 512)):
    """Normalize prediction coordinates from pixel values to [0-1] range"""
    normalized_predictions = []
    for pred_tuple in predictions:
        pred, image_name = pred_tuple
        normalized_boxes = []
        for box, score, class_id in pred:
            # Convert from [x, y, w, h] in pixels to normalized coordinates
            x = box[0] / image_size[0]
            y = box[1] / image_size[1]
            w = box[2] / image_size[0]
            h = box[3] / image_size[1]
            normalized_boxes.append(([x, y, w, h], score, class_id))
        normalized_predictions.append((normalized_boxes, image_name))
    return normalized_predictions

def evaluate_experiment(predictions, ground_truths):
    """Evaluate predictions against ground truth"""
    # Normalize predictions to match ground truth coordinate system
    normalized_predictions = normalize_predictions(predictions)
    
    results = {
        'ap_by_class': defaultdict(list),
        'map': 0.0,
        'class_metrics': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0, 'recall': 0})
    }
    
    print("\nEvaluation Debug Info:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of ground truths: {len(ground_truths)}")
    
    # Process each image
    for pred_tuple in normalized_predictions:
        pred, image_name = pred_tuple
        print(f"\nProcessing image: {image_name}")
        
        if image_name not in ground_truths:
            print(f"Warning: No ground truth found for {image_name}")
            continue
        
        gt_boxes = ground_truths[image_name]
        print(f"Ground truth boxes: {len(gt_boxes)}")
        print(f"Prediction boxes: {len(pred)}")
        
        # Calculate metrics per class
        all_classes = set(d[2] for d in gt_boxes) | set(d[2] for d in pred)
        print(f"Classes in this image: {all_classes}")
        
        for class_id in all_classes:
            # Filter boxes by class
            gt_class_boxes = [d for d in gt_boxes if d[2] == class_id]
            pred_class_boxes = [d for d in pred if d[2] == class_id]
            
            print(f"\nClass {class_id}:")
            print(f"Ground truth boxes for class: {len(gt_class_boxes)}")
            print(f"Prediction boxes for class: {len(pred_class_boxes)}")
            
            if len(gt_class_boxes) > 0:
                print(f"Sample GT box: {gt_class_boxes[0]}")
            if len(pred_class_boxes) > 0:
                print(f"Sample Pred box: {pred_class_boxes[0]}")
            
            precision, recall, scores = calculate_precision_recall(gt_class_boxes, pred_class_boxes)
            
            print(f"Precision values: {precision}")
            print(f"Recall values: {recall}")
            
            if len(precision) > 0:
                ap = calculate_ap(precision, recall)
                results['ap_by_class'][class_id].append(ap)
                print(f"AP for class {class_id}: {ap}")
                
                # Update class metrics
                results['class_metrics'][class_id]['precision'] = np.mean(precision)
                results['class_metrics'][class_id]['recall'] = np.mean(recall)
            
            # Count TP, FP, FN
            matched_gt = set()
            matched_pred = set()
            
            for pred_idx, pred_box in enumerate(pred_class_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_class_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred_box[0], gt_box[0])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5:
                    results['class_metrics'][class_id]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                else:
                    results['class_metrics'][class_id]['fp'] += 1
            
            results['class_metrics'][class_id]['fn'] += len(gt_class_boxes) - len(matched_gt)
            
            print(f"Class {class_id} metrics:")
            print(f"TP: {results['class_metrics'][class_id]['tp']}")
            print(f"FP: {results['class_metrics'][class_id]['fp']}")
            print(f"FN: {results['class_metrics'][class_id]['fn']}")
    
    # Calculate final AP for each class and mAP
    for class_id in results['ap_by_class']:
        results['ap_by_class'][class_id] = np.mean(results['ap_by_class'][class_id])
    
    results['map'] = np.mean(list(results['ap_by_class'].values()))
    
    return results

def plot_results(results_dict, output_dir):
    """Generate visualization plots for the analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot MAP comparison
    plt.figure(figsize=(12, 6))
    experiments = list(results_dict.keys())
    maps = [results['map'] for results in results_dict.values()]
    
    plt.bar(experiments, maps)
    plt.title('Mean Average Precision (MAP) Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('MAP')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_comparison.png'))
    plt.close()
    
    # Plot per-class AP for each experiment
    plt.figure(figsize=(12, 6))
    for exp_name, results in results_dict.items():
        class_aps = results['ap_by_class']
        classes = list(class_aps.keys())
        aps = [class_aps[c] for c in classes]
        plt.plot(classes, aps, marker='o', label=exp_name)
    
    plt.title('Per-class Average Precision')
    plt.xlabel('Class ID')
    plt.ylabel('AP')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_ap.png'))
    plt.close()

def write_detailed_analysis(results_dict, output_dir):
    """Write detailed analysis to file"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "map_analysis.txt")
    
    with open(output_file, 'w') as f:
        f.write("MAP Analysis Results\n")
        f.write("===================\n\n")
        
        # Summary table
        f.write("Summary Table:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Experiment':<30} {'MAP':>10} {'Classes':>10}\n")
        f.write("-" * 80 + "\n")
        
        for exp_name, results in results_dict.items():
            map_score = results['map']
            num_classes = len(results['ap_by_class'])
            f.write(f"{exp_name:<30} {map_score:>10.4f} {num_classes:>10d}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed analysis for each experiment
        for exp_name, results in results_dict.items():
            f.write(f"\nDetailed Analysis: {exp_name}\n")
            f.write("=" * 50 + "\n")
            
            # Overall metrics
            f.write(f"Mean Average Precision (MAP): {results['map']:.4f}\n")
            f.write(f"Number of Classes: {len(results['ap_by_class'])}\n\n")
            
            # Per-class analysis
            f.write("Per-class Analysis:\n")
            f.write("-" * 30 + "\n")
            for class_id, ap in results['ap_by_class'].items():
                metrics = results['class_metrics'][class_id]
                f.write(f"\nClass {class_id}:\n")
                f.write(f"  Average Precision: {ap:.4f}\n")
                f.write(f"  True Positives: {metrics['tp']}\n")
                f.write(f"  False Positives: {metrics['fp']}\n")
                f.write(f"  False Negatives: {metrics['fn']}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    # Also save as CSV for easier analysis
    csv_file = os.path.join(output_dir, "map_results.csv")
    data = []
    for exp_name, results in results_dict.items():
        for class_id, ap in results['ap_by_class'].items():
            metrics = results['class_metrics'][class_id]
            data.append({
                'Experiment': exp_name,
                'Class': class_id,
                'AP': ap,
                'TP': metrics['tp'],
                'FP': metrics['fp'],
                'FN': metrics['fn'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {output_file} and {csv_file}")

def main():
    # Configuration
    experiment_folder = "compression_experiments_20241127_130804"  # Your experiment folder
    gt_folder = "onion/testing"  # Path to ground truth labels and images
    output_dir = os.path.join(experiment_folder, "map_analysis")
    
    print(f"\nAnalyzing MAP scores for experiments in: {experiment_folder}")
    print(f"Using ground truth from: {gt_folder}")
    print(f"Results will be saved to: {output_dir}")
    
    # Load ground truth
    ground_truths = load_ground_truth(gt_folder)
    
    if len(ground_truths) == 0:
        print("Error: No ground truth annotations found. Please check the path and file format.")
        return
    
    # Get available experiments
    experiments = [d for d in os.listdir(experiment_folder) 
                  if os.path.isdir(os.path.join(experiment_folder, d))]
    
    if not experiments:
        print(f"Error: No experiment directories found in {experiment_folder}")
        return
    
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp}")
    
    # Analyze each experiment
    results_dict = {}
    for exp in experiments:
        try:
            print(f"\nAnalyzing {exp}...")
            predictions = load_predictions(experiment_folder, exp)
            results = evaluate_experiment(predictions, ground_truths)
            results_dict[exp] = results
            
            print(f"MAP Score: {results['map']:.4f}")
            print("Per-class AP:")
            for class_id, ap in results['ap_by_class'].items():
                print(f"  Class {class_id}: {ap:.4f}")
                
        except Exception as e:
            print(f"Error analyzing {exp}: {e}")
            continue
    
    if not results_dict:
        print("Error: No experiments could be analyzed successfully.")
        return
    
    # Generate visualizations and reports
    print("\nGenerating visualizations and reports...")
    plot_results(results_dict, output_dir)
    write_detailed_analysis(results_dict, output_dir)
    
    print(f"\nAnalysis complete! Results saved in {output_dir}/")
    
    # Print summary
    print("\nQuick Summary:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'MAP':>10} {'Classes':>10}")
    print("-" * 80)
    
    for exp_name, results in results_dict.items():
        map_score = results['map']
        num_classes = len(results['ap_by_class'])
        print(f"{exp_name:<30} {map_score:>10.4f} {num_classes:>10d}")

if __name__ == "__main__":
    main() 