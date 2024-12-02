import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, average_precision_score
from collections import defaultdict
import os

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def draw_detections(image_path, detections, output_path, class_names=["with_weeds", "without_weeds"]):
    """Draw detections on image and save"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, class_id in detections:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        color = 'red' if class_id == 0 else 'blue'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[class_id]}: {score:.2f}"
        
        text_size = draw.textbbox((0, 0), label, font)
        text_width = text_size[2]
        text_height = text_size[3]
        
        label_x = x1
        label_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + h + 2
        
        if label_x + text_width > image.width:
            label_x = image.width - text_width
        
        draw.rectangle([label_x, label_y - text_height - 2, label_x + text_width, label_y], fill=color)
        draw.text((label_x, label_y - text_height - 2), label, fill='white', font=font)

    image.save(output_path)

def evaluate_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    """Calculate precision, recall, F1, and mAP"""
    tp, fp, scores = defaultdict(list), defaultdict(list), defaultdict(list)
    
    for pred_box, pred_score, pred_class in zip(pred_boxes, pred_scores, pred_classes):
        matched = False
        for gt_box, gt_class in zip(gt_boxes, gt_classes):
            if pred_class == gt_class and calculate_iou(pred_box, gt_box) >= iou_threshold:
                tp[pred_class].append(1)
                fp[pred_class].append(0)
                matched = True
                break
        if not matched:
            tp[pred_class].append(0)
            fp[pred_class].append(1)
        scores[pred_class].append(pred_score)

    metrics = {}
    for cls in scores:
        if len(tp[cls]) > 0:
            precision, recall, _ = precision_recall_curve(tp[cls], scores[cls])
            f1 = f1_score(tp[cls], [1 if s >= 0.5 else 0 for s in scores[cls]])
            ap = average_precision_score(tp[cls], scores[cls])
            metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap
            }

    return metrics

def analyze_pkl_files(compression_levels=[1, 4, 8], include_lossless=True, 
                     save_images=True, calculate_metrics=True):
    results = {}
    base_path = Path('.')
    
    # Load lossless results if requested
    if include_lossless:
        lossless_path = base_path / 'lossless_data.pkl'
        if lossless_path.exists():
            results['lossless'] = load_pkl_file(lossless_path)
            print("\nLossless compression statistics:")
            analyze_single_compression(results['lossless'], 'lossless')
    
    # Load lossy results
    for level in compression_levels:
        file_path = base_path / f'lossy_data_compression_{level}.pkl'
        if file_path.exists():
            results[f'lossy_{level}'] = load_pkl_file(file_path)
            print(f"\nLossy compression level {level} statistics:")
            analyze_single_compression(results[f'lossy_{level}'], f'lossy_{level}')
    
    # Compare results
    all_versions = ['lossless'] if include_lossless else []
    all_versions.extend([f'lossy_{level}' for level in compression_levels])
    
    comparisons = compare_all_versions(results, all_versions)
    
    # Save visualization images if requested
    if save_images:
        os.makedirs('output_images', exist_ok=True)
        for version, data in results.items():
            for pred, img_name, _ in data:
                img_path = os.path.join('onion/testing', img_name)
                out_path = os.path.join('output_images', f'{version}_{img_name}')
                draw_detections(img_path, pred, out_path)
    
    # Calculate metrics if requested
    if calculate_metrics:
        plot_metrics(results, all_versions)
    
    # Create visualization plots
    create_comparison_plots(comparisons, results, all_versions)

def analyze_single_compression(data, version):
    """Print statistics for a single compression version"""
    total_detections = sum(len(pred) for pred, _, _ in data)
    total_size = sum(size for _, _, size in data)
    
    print(f"Total images: {len(data)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(data):.2f}")
    print(f"Total compressed size: {total_size:,} bytes")
    print(f"Average size per image: {total_size/len(data):,.2f} bytes")

def compare_all_versions(results, versions):
    """Compare all versions of compression"""
    comparisons = {}
    for i, ver1 in enumerate(versions):
        for ver2 in versions[i+1:]:
            key = f'{ver1}vs{ver2}'
            comparisons[key] = compare_two_versions(results[ver1], results[ver2])
            print(f"\nComparing {ver1} vs {ver2}:")
            print_comparison_results(comparisons[key])
    return comparisons

def compare_two_versions(data1, data2):
    """Compare two versions of compression"""
    comparison = {
        'box_diffs': [],
        'score_diffs': [],
        'class_mismatches': 0,
        'detection_count_diffs': [],
        'size_ratios': [],
        'confidence_diffs': []
    }
    
    for (pred1, img1, size1), (pred2, img2, size2) in zip(data1, data2):
        if img1 != img2:
            continue
            
        comparison['detection_count_diffs'].append(len(pred1) - len(pred2))
        comparison['size_ratios'].append(size1/size2 if size2 != 0 else float('inf'))
        
        # Compare individual detections
        for d1 in pred1:
            best_iou = 0
            best_match = None
            
            for d2 in pred2:
                iou = calculate_iou(d1[0], d2[0])
                if iou > best_iou:
                    best_iou = iou
                    best_match = d2
            
            if best_match and best_iou > 0.5:
                comparison['box_diffs'].append(best_iou)
                comparison['score_diffs'].append(abs(d1[1] - best_match[1]))
                if d1[2] != best_match[2]:
                    comparison['class_mismatches'] += 1
                comparison['confidence_diffs'].append(d1[1] - best_match[1])
    
    return comparison

def print_comparison_results(comparison):
    """Print detailed comparison results"""
    print(f"Average IoU: {np.mean(comparison['box_diffs']):.4f}")
    print(f"Average confidence score difference: {np.mean(comparison['score_diffs']):.4f}")
    print(f"Class mismatches: {comparison['class_mismatches']}")
    print(f"Average detection count difference: {np.mean(comparison['detection_count_diffs']):.2f}")
    print(f"Average size ratio: {np.mean(comparison['size_ratios']):.2f}x")
    print(f"Average confidence difference: {np.mean(comparison['confidence_diffs']):.4f}")

def plot_metrics(results, versions):
    """Create plots for precision-recall and other metrics"""
    plt.figure(figsize=(15, 10))
    
    # Precision-Recall curve
    plt.subplot(2, 2, 1)
    for version in versions:
        data = results[version]
        pred_boxes = [box for pred, _, _ in data for box, _, _ in pred]
        pred_scores = [score for pred, _, _ in data for _, score, _ in pred]
        pred_classes = [cls for pred, _, _ in data for _, _, cls in pred]
        metrics = evaluate_predictions(pred_boxes, pred_scores, pred_classes, [], [])
        
        for cls in metrics:
            plt.plot(metrics[cls]['recall'], metrics[cls]['precision'], 
                    label=f'{version} (AP={metrics[cls]["ap"]:.2f})')
    
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    
    # Add other metric plots as needed
    plt.savefig('metrics_comparison.png')
    plt.close()

def create_comparison_plots(comparisons, results, versions):
    """Create detailed comparison plots"""
    plt.figure(figsize=(20, 15))
    
    # Confidence score differences
    plt.subplot(3, 2, 1)
    for key, data in comparisons.items():
        plt.hist(data['confidence_diffs'], alpha=0.5, label=key, bins=20)
    plt.title('Confidence Score Differences')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Size comparison
    plt.subplot(3, 2, 2)
    sizes = {ver: [size for _, _, size in results[ver]] for ver in versions}
    plt.boxplot([sizes[ver] for ver in versions], labels=versions)
    plt.title('Compression Sizes')
    plt.ylabel('Bytes')
    plt.xticks(rotation=45)
    
    # Add more plots as needed
    
    plt.tight_layout()
    plt.savefig('compression_comparison_detailed.png')
    plt.close()

if __name__ == "__main__":
    analyze_pkl_files(
        compression_levels=[1, 4, 8],
        include_lossless=True,
        save_images=True,
        calculate_metrics=True
    ) 