import os
import pickle
import numpy as np
from collections import defaultdict

def load_predictions(filename):
    """Load predictions from a pickle file"""
    try:
        with open(filename, 'rb') as f:
            predictions = pickle.load(f)
            return predictions
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def analyze_differences(reference_data, comparison_data):
    """Analyze detailed differences between two sets of predictions"""
    stats = {
        'confidence_diffs': [],
        'bbox_diffs': [],
        'total_detections': 0,
        'matched_detections': 0,
        'class_matches': defaultdict(int),
        'class_totals': defaultdict(int)
    }
    
    # Create dictionaries for easier matching
    ref_dict = {img_name: dets for dets, img_name, _ in reference_data}
    comp_dict = {img_name: dets for dets, img_name, _ in comparison_data}
    
    common_images = set(ref_dict.keys()) & set(comp_dict.keys())
    
    for img_name in common_images:
        ref_dets = ref_dict[img_name]
        comp_dets = comp_dict[img_name]
        
        for ref_det in ref_dets:
            stats['total_detections'] += 1
            ref_box, ref_conf, ref_class = ref_det
            stats['class_totals'][ref_class] += 1
            
            # Find matching detection in comparison data
            best_match = None
            best_iou = 0
            
            for comp_det in comp_dets:
                comp_box, comp_conf, comp_class = comp_det
                if comp_class == ref_class:
                    # Calculate IoU
                    iou = calculate_iou(ref_box, comp_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = comp_det
            
            if best_match and best_iou > 0.5:
                stats['matched_detections'] += 1
                stats['class_matches'][ref_class] += 1
                
                # Calculate confidence difference
                conf_diff = abs(ref_conf - best_match[1])
                stats['confidence_diffs'].append(conf_diff)
                
                # Calculate bounding box coordinate differences
                bbox_diff = np.abs(np.array(ref_box) - np.array(best_match[0]))
                stats['bbox_diffs'].append(bbox_diff)
    
    return stats

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

def print_statistics(stats, method_name):
    """Print detailed statistics"""
    print(f"\nDetailed Analysis for {method_name}:")
    print("-" * 50)
    
    # Detection statistics
    print("\nDetection Statistics:")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Matched detections: {stats['matched_detections']}")
    print(f"Match rate: {100 * stats['matched_detections'] / stats['total_detections']:.2f}%")
    
    # Class-wise statistics
    print("\nClass-wise Statistics:")
    for class_id in stats['class_totals'].keys():
        total = stats['class_totals'][class_id]
        matches = stats['class_matches'][class_id]
        print(f"Class {class_id}:")
        print(f"  Total: {total}")
        print(f"  Matches: {matches}")
        print(f"  Match rate: {100 * matches / total:.2f}%")
    
    # Confidence differences
    if stats['confidence_diffs']:
        conf_diffs = np.array(stats['confidence_diffs'])
        print("\nConfidence Score Differences:")
        print(f"  Mean: {np.mean(conf_diffs):.6f}")
        print(f"  Std: {np.std(conf_diffs):.6f}")
        print(f"  Max: {np.max(conf_diffs):.6f}")
    
    # Bounding box differences
    if stats['bbox_diffs']:
        bbox_diffs = np.array(stats['bbox_diffs'])
        print("\nBounding Box Coordinate Differences:")
        print("  Mean absolute differences:")
        print(f"    X: {np.mean(bbox_diffs[:, 0]):.6f}")
        print(f"    Y: {np.mean(bbox_diffs[:, 1]):.6f}")
        print(f"    Width: {np.mean(bbox_diffs[:, 2]):.6f}")
        print(f"    Height: {np.mean(bbox_diffs[:, 3]):.6f}")
        print("  Max absolute differences:")
        print(f"    X: {np.max(bbox_diffs[:, 0]):.6f}")
        print(f"    Y: {np.max(bbox_diffs[:, 1]):.6f}")
        print(f"    Width: {np.max(bbox_diffs[:, 2]):.6f}")
        print(f"    Height: {np.max(bbox_diffs[:, 3]):.6f}")

def main():
    # Load reference data
    reference_data = load_predictions("lossless_data.pkl")
    if reference_data is None:
        return
    
    # Methods to compare
    methods = [
        'lossy_data',
        'lossy_data_clevel_0',
        'lossy_data_clevel_4',
        'lossy_data_clevel_8'
    ]
    
    # Analyze each method
    results = {}
    for method in methods:
        print(f"\nAnalyzing {method}...")
        comparison_data = load_predictions(f"{method}.pkl")
        if comparison_data is None:
            continue
            
        stats = analyze_differences(reference_data, comparison_data)
        results[method] = stats
        print_statistics(stats, method)
    
    # Save detailed results to file
    with open("detailed_analysis_results.txt", "w") as f:
        for method, stats in results.items():
            f.write(f"\nDetailed Analysis for {method}:\n")
            f.write("-" * 50 + "\n")
            
            # Write detection statistics
            f.write("\nDetection Statistics:\n")
            f.write(f"Total detections: {stats['total_detections']}\n")
            f.write(f"Matched detections: {stats['matched_detections']}\n")
            f.write(f"Match rate: {100 * stats['matched_detections'] / stats['total_detections']:.2f}%\n")
            
            # Write class-wise statistics
            f.write("\nClass-wise Statistics:\n")
            for class_id in stats['class_totals'].keys():
                total = stats['class_totals'][class_id]
                matches = stats['class_matches'][class_id]
                f.write(f"Class {class_id}:\n")
                f.write(f"  Total: {total}\n")
                f.write(f"  Matches: {matches}\n")
                f.write(f"  Match rate: {100 * matches / total:.2f}%\n")
            
            # Write confidence differences
            if stats['confidence_diffs']:
                conf_diffs = np.array(stats['confidence_diffs'])
                f.write("\nConfidence Score Differences:\n")
                f.write(f"  Mean: {np.mean(conf_diffs):.6f}\n")
                f.write(f"  Std: {np.std(conf_diffs):.6f}\n")
                f.write(f"  Max: {np.max(conf_diffs):.6f}\n")
            
            # Write bounding box differences
            if stats['bbox_diffs']:
                bbox_diffs = np.array(stats['bbox_diffs'])
                f.write("\nBounding Box Coordinate Differences:\n")
                f.write("  Mean absolute differences:\n")
                f.write(f"    X: {np.mean(bbox_diffs[:, 0]):.6f}\n")
                f.write(f"    Y: {np.mean(bbox_diffs[:, 1]):.6f}\n")
                f.write(f"    Width: {np.mean(bbox_diffs[:, 2]):.6f}\n")
                f.write(f"    Height: {np.mean(bbox_diffs[:, 3]):.6f}\n")
                f.write("  Max absolute differences:\n")
                f.write(f"    X: {np.max(bbox_diffs[:, 0]):.6f}\n")
                f.write(f"    Y: {np.max(bbox_diffs[:, 1]):.6f}\n")
                f.write(f"    Width: {np.max(bbox_diffs[:, 2]):.6f}\n")
                f.write(f"    Height: {np.max(bbox_diffs[:, 3]):.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nDetailed results have been saved to detailed_analysis_results.txt")

if __name__ == "__main__":
    main() 