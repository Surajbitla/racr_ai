import os
import numpy as np

# Function to read a YOLO format .txt file and return bounding boxes with their class ID and confidence scores
def read_yolo_txt(file_path):
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height, confidence = map(float, data[1:])
            detections.append((class_id, x_center, y_center, width, height, confidence))
    return detections

# Function to calculate Intersection over Union (IoU) for two bounding boxes in normalized YOLO format
def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2
    
    inter_width = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_height = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_area = inter_width * inter_height
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

# Function to compare regular and split model results
def compare_results(regular_txt, split_txt, iou_threshold=0.5):
    regular_detections = read_yolo_txt(regular_txt)
    split_detections = read_yolo_txt(split_txt)
    
    iou_list = []
    confidence_diff = []
    unmatched_regular = len(regular_detections)
    unmatched_split = len(split_detections)
    
    matched = 0
    
    for reg_det in regular_detections:
        reg_class, reg_x, reg_y, reg_w, reg_h, reg_conf = reg_det
        best_iou = 0
        best_split_det = None
        
        for split_det in split_detections:
            split_class, split_x, split_y, split_w, split_h, split_conf = split_det
            
            if reg_class == split_class:
                iou = calculate_iou((reg_x, reg_y, reg_w, reg_h), (split_x, split_y, split_w, split_h))
                if iou > best_iou:
                    best_iou = iou
                    best_split_det = split_det
        
        if best_iou >= iou_threshold:
            matched += 1
            iou_list.append(best_iou)
            confidence_diff.append(reg_conf - best_split_det[5])  # Difference in confidence scores
            unmatched_regular -= 1
            unmatched_split -= 1
    
    return {
        "iou_list": iou_list,
        "confidence_diff": confidence_diff,
        "unmatched_regular": unmatched_regular,
        "unmatched_split": unmatched_split,
        "total_regular_detections": len(regular_detections),
        "total_split_detections": len(split_detections),
        "matched_detections": matched
    }

# Function to summarize comparison results across all images
def summarize_overall(results):
    total_regular = sum([res['total_regular_detections'] for res in results])
    total_split = sum([res['total_split_detections'] for res in results])
    total_matched = sum([res['matched_detections'] for res in results])
    unmatched_regular = sum([res['unmatched_regular'] for res in results])
    unmatched_split = sum([res['unmatched_split'] for res in results])

    all_iou = [iou for res in results for iou in res['iou_list']]
    all_conf_diff = [diff for res in results for diff in res['confidence_diff']]

    avg_iou = np.mean(all_iou) if all_iou else 0
    avg_conf_diff = np.mean(all_conf_diff) if all_conf_diff else 0

    print("Overall Summary:")
    print(f"Total Regular Detections: {total_regular}")
    print(f"Total Split Detections: {total_split}")
    print(f"Matched Detections: {total_matched}")
    print(f"Unmatched Regular Detections: {unmatched_regular}")
    print(f"Unmatched Split Detections: {unmatched_split}")
    print(f"Average IoU for matched detections: {avg_iou:.4f}")
    print(f"Average confidence score difference: {avg_conf_diff:.4f}")

# Compare results for all images and summarize
regular_labels_dir = r"C:\Users\GenCyber\Documents\Yolov8\runs\detect\val42\labels"
split_labels_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New\output_labels"

overall_results = []
for txt_file in os.listdir(regular_labels_dir):
    regular_txt = os.path.join(regular_labels_dir, txt_file)
    split_txt = os.path.join(split_labels_dir, txt_file)
    
    if os.path.exists(split_txt):
        result = compare_results(regular_txt, split_txt)
        overall_results.append(result)

# Summarize the overall performance
summarize_overall(overall_results)
