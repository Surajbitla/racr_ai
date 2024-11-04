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
    # box format: (x_center, y_center, width, height)
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
            
            if reg_class == split_class:  # Only compare boxes with the same class
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

# Function to summarize the comparison results
def summarize_comparison(result):
    print(f"Total Regular Detections: {result['total_regular_detections']}")
    print(f"Total Split Detections: {result['total_split_detections']}")
    print(f"Matched Detections: {result['matched_detections']}")
    print(f"Unmatched Regular Detections: {result['unmatched_regular']}")
    print(f"Unmatched Split Detections: {result['unmatched_split']}")
    
    if result['iou_list']:
        avg_iou = np.mean(result['iou_list'])
        avg_conf_diff = np.mean(result['confidence_diff'])
        print(f"Average IoU for matched detections: {avg_iou:.4f}")
        print(f"Average confidence score difference: {avg_conf_diff:.4f}")
    else:
        print("No detections were matched.")

# Compare results for all images
regular_labels_dir = r"C:\Users\GenCyber\Documents\Yolov8\runs\detect\val42\labels"
split_labels_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New\output_labels"

for txt_file in os.listdir(regular_labels_dir):
    regular_txt = os.path.join(regular_labels_dir, txt_file)
    split_txt = os.path.join(split_labels_dir, txt_file)
    
    if os.path.exists(split_txt):
        result = compare_results(regular_txt, split_txt)
        print(f"Results for {txt_file}:")
        summarize_comparison(result)
        print("="*50)
