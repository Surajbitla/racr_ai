import os

import pickle

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd
 
from pathlib import Path

import json

import sys



def load_predictions(experiment_folder, experiment_name):

    """Load predictions.pkl file for a given experiment"""

    pred_file = os.path.join(experiment_folder, experiment_name, "predictions.pkl")

    print(f"Loading predictions from: {pred_file}")  # Add debug print

    with open(pred_file, 'rb') as f:

        preds = pickle.load(f)

        # Add debug prints

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



def compare_predictions(pred1, pred2):

    """Compare two sets of predictions"""

    metrics = {

        'confidence_diff': [],

        'iou': [],

        'center_dist': [],

        'class_matches': 0,

        'total_detections': 0

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

    

    for img in common_images:

        det1 = pred1_dict[img]

        det2 = pred2_dict[img]

        

        # Add debug prints

        print(f"\nImage: {img}")

        print(f"Number of detections in first set: {len(det1)}")

        print(f"Number of detections in second set: {len(det2)}")

        

        for d1 in det1:

            best_match = None

            best_iou = 0

            

            for d2 in det2:

                if d1[2] == d2[2]:  # Same class

                    iou, dist = calculate_bbox_metrics(d1[0], d2[0])

                    if iou > best_iou:

                        best_match = d2

                        best_iou = iou

            

            if best_match:

                conf_diff = float(abs(d1[1] - best_match[1]))

                metrics['confidence_diff'].append(conf_diff)

                metrics['iou'].append(float(best_iou))

                metrics['center_dist'].append(float(calculate_bbox_metrics(d1[0], best_match[0])[1]))

                metrics['class_matches'] += 1

                

                # Add debug print for significant differences

                if conf_diff > 0.01 or best_iou < 0.99:

                    print(f"Significant difference found:")

                    print(f"Original detection: {d1}")

                    print(f"Matched detection: {best_match}")

                    print(f"Confidence diff: {conf_diff}, IoU: {best_iou}")

            

            metrics['total_detections'] += 1

    

    # Add summary statistics

    metrics['num_common_images'] = len(common_images)

    metrics['total_images_set1'] = len(pred1_dict)

    metrics['total_images_set2'] = len(pred2_dict)

    

    return metrics



def analyze_lossless_methods(experiment_folder):

    """Compare lossless compression methods"""

    methods = ['no_compression', 'lossless_zlib_level_9', 'lossless_blosc2_clevel_9']

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



def analyze_lossy_pairs(experiment_folder):

    """Compare lossy compression methods with different parameters"""

    pairs = [

        ('lossy_zfp_precision_precision_4', 'lossy_zfp_precision_precision_8'),

        ('lossy_zfp_accuracy_accuracy_0.1', 'lossy_zfp_accuracy_accuracy_0.01'),

        ('lossy_zfp_rate_rate_4', 'lossy_zfp_rate_rate_8'),

        ('custom_quantization_bits_4', 'custom_quantization_bits_8')

    ]

    

    results = {}

    for method1, method2 in pairs:

        print(f"\nAnalyzing pair: {method1} vs {method2}")

        try:

            pred1 = load_predictions(experiment_folder, method1)

            pred2 = load_predictions(experiment_folder, method2)

            

            if not pred1 or not pred2:

                print(f"Warning: No predictions found for {method1} or {method2}")

                continue

            

            # Verify predictions are different

            print(f"Checking if predictions are identical...")

            are_identical = True

            pred1_dict = {img: det for det, img in pred1}

            pred2_dict = {img: det for det, img in pred2}

            common_images = set(pred1_dict.keys()) & set(pred2_dict.keys())

            

            for img in common_images:

                det1 = pred1_dict[img]

                det2 = pred2_dict[img]

                if len(det1) != len(det2) or any(not np.array_equal(d1[0], d2[0]) or abs(d1[1] - d2[1]) > 1e-6 

                      for d1, d2 in zip(det1, det2)):

                    are_identical = False

                    break

            

            print(f"Predictions are {'identical' if are_identical else 'different'}")

            

            metrics = compare_predictions(pred1, pred2)

            results[f"{method1}_vs_{method2}"] = metrics

            

        except Exception as e:

            print(f"Error analyzing pair {method1} vs {method2}: {str(e)}")

            continue

    

    return results



def plot_comparison_results(results, title, output_path):

    """Generate plots for comparison results"""

    plt.figure(figsize=(15, 10))

    

    # Plot confidence differences

    plt.subplot(2, 2, 1)

    data = []

    labels = []

    for method, metrics in results.items():

        data.append(metrics['confidence_diff'])

        labels.extend([method] * len(metrics['confidence_diff']))

    

    sns.boxplot(data=data)

    plt.xticks(range(len(results)), list(results.keys()), rotation=45)

    plt.title('Confidence Score Differences')

    plt.ylabel('Absolute Difference')

    

    # Plot IoU

    plt.subplot(2, 2, 2)

    data = []

    for method, metrics in results.items():

        data.append(metrics['iou'])

    

    sns.boxplot(data=data)

    plt.xticks(range(len(results)), list(results.keys()), rotation=45)

    plt.title('IoU Distribution')

    plt.ylabel('IoU')

    

    # Plot detection matches

    plt.subplot(2, 2, 3)

    methods = list(results.keys())

    matches = [metrics['class_matches'] for metrics in results.values()]

    totals = [metrics['total_detections'] for metrics in results.values()]

    

    match_rates = [m/t*100 for m, t in zip(matches, totals)]

    

    plt.bar(methods, match_rates)

    plt.xticks(rotation=45)

    plt.title('Detection Match Rate')

    plt.ylabel('Match Rate (%)')

    

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()



def analyze_summary_files(experiment_folder):

    """Analyze and compare summary.txt files"""

    summaries = {}

    for exp_dir in os.listdir(experiment_folder):

        summary_file = os.path.join(experiment_folder, exp_dir, "summary.txt")

        if os.path.exists(summary_file):

            with open(summary_file, 'r') as f:

                content = f.read()

                # Parse relevant metrics from summary file

                # This is a simple example - extend based on your needs

                metrics = {

                    'compression_ratio': float(content.split('Compression ratio:')[1].split('\n')[0]),

                    'total_time': float(content.split('Total time:')[1].split('s')[0])

                }

                summaries[exp_dir] = metrics

    

    return summaries



# Add this new function to handle NumPy types

def convert_to_serializable(obj):

    """Convert numpy types to Python native types for JSON serialization"""

    if isinstance(obj, np.ndarray):

        return obj.tolist()

    elif isinstance(obj, np.float32):

        return float(obj)

    elif isinstance(obj, np.int64):

        return int(obj)

    elif isinstance(obj, dict):

        return {k: convert_to_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):

        return [convert_to_serializable(item) for item in obj]

    return obj



def verify_prediction_files(experiment_folder):
    """Verify prediction files in detail and save results to a file"""
    output_file = os.path.join(experiment_folder, "verification_results.txt")
    
    # Redirect print statements to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Save original stdout and redirect to Logger
    original_stdout = sys.stdout
    sys.stdout = Logger(output_file)
    
    try:
        methods = [
            'lossy_zfp_accuracy_accuracy_0.1',
            'lossy_zfp_accuracy_accuracy_0.01',
            'lossy_zfp_rate_rate_4',
            'lossy_zfp_rate_rate_8',
            'lossy_zfp_precision_precision_4',
            'lossy_zfp_precision_precision_8'
        ]
        
        print("=== Prediction File Analysis ===\n")
        
        # First, print file sizes
        print("File Sizes:")
        for method in methods:
            pred_file = os.path.join(experiment_folder, method, "predictions.pkl")
            if os.path.exists(pred_file):
                size = os.path.getsize(pred_file)
                print(f"{method}: {size/1024:.2f} KB")
        print("\n")
        
        predictions = {}
        for method in methods:
            pred_file = os.path.join(experiment_folder, method, "predictions.pkl")
            try:
                with open(pred_file, 'rb') as f:
                    preds = pickle.load(f)
                    predictions[method] = preds
                    print(f"\nMethod: {method}")
                    print(f"Number of predictions: {len(preds)}")
                    if len(preds) > 0:
                        first_det, first_img = preds[0]
                        print(f"First image: {first_img}")
                        if len(first_det) > 0:
                            print("First detection details:")
                            print(f"  Bounding box: {first_det[0][0]}")
                            print(f"  Confidence: {first_det[0][1]}")
                            print(f"  Class: {first_det[0][2]}")
                            bbox_array = np.array(first_det[0][0])
                            print(f"  Raw bbox values: {np.array2string(bbox_array, precision=8)}")
            except Exception as e:
                print(f"Error loading {method}: {str(e)}")
        
        print("\n=== Detailed Comparisons ===")
        pairs = [
            ('lossy_zfp_accuracy_accuracy_0.1', 'lossy_zfp_accuracy_accuracy_0.01'),
            ('lossy_zfp_rate_rate_4', 'lossy_zfp_rate_rate_8')
        ]
        
        for method1, method2 in pairs:
            print(f"\nComparing {method1} vs {method2}")
            if method1 in predictions and method2 in predictions:
                pred1 = predictions[method1]
                pred2 = predictions[method2]
                
                print(f"\nAnalyzing first 3 images:")
                for i, ((det1, img1), (det2, img2)) in enumerate(zip(pred1, pred2)):
                    if i >= 3:
                        break
                        
                    print(f"\nImage {i+1}: {img1}")
                    print(f"Number of detections: {len(det1)} vs {len(det2)}")
                    
                    for j, (d1, d2) in enumerate(zip(det1, det2)):
                        if j >= 3:
                            break
                        
                        bbox1 = np.array(d1[0])
                        bbox2 = np.array(d2[0])
                        bbox_diff = np.abs(bbox1 - bbox2).max()
                        conf_diff = abs(d1[1] - d2[1])
                        
                        print(f"\nDetection {j+1}:")
                        print(f"  Bbox difference: {bbox_diff}")
                        print(f"  Confidence difference: {conf_diff}")
                        if bbox_diff > 0 or conf_diff > 0:
                            print("  Original bbox:", d1[0])
                            print("  Compressed bbox:", d2[0])
                            print("  Original confidence:", d1[1])
                            print("  Compressed confidence:", d2[1])
                            print("  Class:", d1[2])
                
                # Add statistical analysis
                print("\nStatistical Analysis:")
                all_bbox_diffs = []
                all_conf_diffs = []
                
                for (det1, _), (det2, _) in zip(pred1, pred2):
                    for d1, d2 in zip(det1, det2):
                        bbox1 = np.array(d1[0])
                        bbox2 = np.array(d2[0])
                        all_bbox_diffs.append(np.abs(bbox1 - bbox2).max())
                        all_conf_diffs.append(abs(d1[1] - d2[1]))
                
                if all_bbox_diffs:
                    print(f"  Average bbox difference: {np.mean(all_bbox_diffs):.8f}")
                    print(f"  Max bbox difference: {np.max(all_bbox_diffs):.8f}")
                    print(f"  Average confidence difference: {np.mean(all_conf_diffs):.8f}")
                    print(f"  Max confidence difference: {np.max(all_conf_diffs):.8f}")
    
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Verification results saved to {output_file}")



def main():

    # Add this at the start of main()

    experiment_folder = "compression_experiments_20241121_200536"

    print("Verifying prediction files...")

    verify_prediction_files(experiment_folder)

    

    # Specify your experiment folder

    experiment_folder = "compression_experiments_20241121_200536"  # Update with your timestamp

    

    # Create output directory for analysis

    analysis_output = os.path.join(experiment_folder, "analysis")

    os.makedirs(analysis_output, exist_ok=True)

    

    # Analyze lossless methods

    print("Analyzing lossless compression methods...")

    lossless_results = analyze_lossless_methods(experiment_folder)

    plot_comparison_results(

        lossless_results,

        "Lossless Compression Comparison",

        os.path.join(analysis_output, "lossless_comparison.png")

    )

    

    # Analyze lossy pairs

    print("Analyzing lossy compression pairs...")

    lossy_results = analyze_lossy_pairs(experiment_folder)

    plot_comparison_results(

        lossy_results,

        "Lossy Compression Comparison",

        os.path.join(analysis_output, "lossy_comparison.png")

    )

    

    # Analyze summary files

    print("Analyzing summary files...")

    summary_results = analyze_summary_files(experiment_folder)

    

    # Plot summary comparisons

    plt.figure(figsize=(12, 6))

    methods = list(summary_results.keys())

    ratios = [m['compression_ratio'] for m in summary_results.values()]

    times = [m['total_time'] for m in summary_results.values()]

    

    plt.subplot(1, 2, 1)

    plt.bar(methods, ratios)

    plt.xticks(rotation=45)

    plt.title('Compression Ratios')

    plt.ylabel('Ratio')

    

    plt.subplot(1, 2, 2)

    plt.bar(methods, times)

    plt.xticks(rotation=45)

    plt.title('Total Processing Time')

    plt.ylabel('Time (s)')

    

    plt.tight_layout()

    plt.savefig(os.path.join(analysis_output, "summary_comparison.png"))

    plt.close()

    

    # Save all results to a JSON file

    all_results = {

        'lossless_comparison': convert_to_serializable(lossless_results),

        'lossy_comparison': convert_to_serializable(lossy_results),

        'summary_comparison': convert_to_serializable(summary_results)

    }

    

    with open(os.path.join(analysis_output, "analysis_results.json"), 'w') as f:

        json.dump(all_results, f, indent=4)

    

    # Also save as CSV for easier analysis

    csv_data = []

    for method, metrics in summary_results.items():

        row = {

            'method': method,

            'compression_ratio': metrics['compression_ratio'],

            'total_time': metrics['total_time']

        }

        csv_data.append(row)

    

    df = pd.DataFrame(csv_data)

    df.to_csv(os.path.join(analysis_output, "summary_results.csv"), index=False)

    

    print(f"Analysis complete. Results saved to {analysis_output}")

    

    # Print some key findings

    print("\nKey Findings:")

    print("-------------")

    print("Lossless Compression Comparison:")

    for method_pair, metrics in lossless_results.items():

        print(f"\n{method_pair}:")

        print(f"Average confidence difference: {np.mean(metrics['confidence_diff']):.4f}")

        print(f"Average IoU: {np.mean(metrics['iou']):.4f}")

        print(f"Detection match rate: {(metrics['class_matches']/metrics['total_detections']*100):.1f}%")

    

    print("\nLossy Compression Comparison:")

    for method_pair, metrics in lossy_results.items():

        print(f"\n{method_pair}:")

        print(f"Average confidence difference: {np.mean(metrics['confidence_diff']):.4f}")

        print(f"Average IoU: {np.mean(metrics['iou']):.4f}")

        print(f"Detection match rate: {(metrics['class_matches']/metrics['total_detections']*100):.1f}%")



if __name__ == "__main__":

    main() 
