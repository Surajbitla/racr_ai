import os
import numpy as np

# Function to read YOLO format .txt files with optional confidence handling
def read_yolo_txt(file_path, has_confidence=True):
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, width, height = map(float, data[1:5])
            if has_confidence and len(data) == 6:
                confidence = float(data[5])
                detections.append((class_id, x_center, y_center, width, height, confidence))
            else:
                detections.append((class_id, x_center, y_center, width, height, None))  # No confidence for ground truth
    return detections

# Function to calculate Intersection over Union (IoU) between two bounding boxes
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

# Function to compare detections from two sets (ground truth vs regular/split)
def compare_detections(detections1, detections2, iou_threshold=0.45, has_confidence=True):
    iou_list = []
    confidence_diff = []
    unmatched_1 = len(detections1)
    unmatched_2 = len(detections2)

    matched = 0

    for det1 in detections1:
        class1, *box1 = det1[:5] if has_confidence else det1[:4]  # Handle class and bbox only
        best_iou = 0
        best_det2 = None

        for det2 in detections2:
            class2, *box2 = det2[:5] if has_confidence else det2[:4]  # Handle class and bbox only

            if class1 == class2:
                iou = calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_det2 = det2

        if best_iou >= iou_threshold:
            matched += 1
            iou_list.append(best_iou)
            if has_confidence and det1[-1] is not None and best_det2[-1] is not None:
                confidence_diff.append(det1[-1] - best_det2[-1])  # Difference in confidence
            unmatched_1 -= 1
            unmatched_2 -= 1

    return {
        "iou_list": iou_list,
        "confidence_diff": confidence_diff,
        "unmatched_1": unmatched_1,
        "unmatched_2": unmatched_2,
        "total_1": len(detections1),
        "total_2": len(detections2),
        "matched_detections": matched
    }

# Summarize results for a comparison across all images
def summarize_comparison_results(results):
    total_1 = sum([res['total_1'] for res in results])
    total_2 = sum([res['total_2'] for res in results])
    total_matched = sum([res['matched_detections'] for res in results])
    unmatched_1 = sum([res['unmatched_1'] for res in results])
    unmatched_2 = sum([res['unmatched_2'] for res in results])

    all_iou = [iou for res in results for iou in res['iou_list']]
    all_conf_diff = [diff for res in results for diff in res['confidence_diff']]

    avg_iou = np.mean(all_iou) if all_iou else 0
    avg_conf_diff = np.mean(all_conf_diff) if all_conf_diff else 0

    return {
        "total_1": total_1,
        "total_2": total_2,
        "total_matched": total_matched,
        "unmatched_1": unmatched_1,
        "unmatched_2": unmatched_2,
        "avg_iou": avg_iou,
        "avg_conf_diff": avg_conf_diff
    }

# Compare all three sets of labels (ground truth, regular, split)
def compare_all(regular_dir, split_dir, ground_dir):
    all_results = []

    for txt_file in os.listdir(ground_dir):
        ground_txt = os.path.join(ground_dir, txt_file)
        regular_txt = os.path.join(regular_dir, txt_file)
        split_txt = os.path.join(split_dir, txt_file)

        if os.path.exists(regular_txt) and os.path.exists(split_txt):
            ground_vs_regular = compare_detections(read_yolo_txt(ground_txt, has_confidence=False), read_yolo_txt(regular_txt))
            ground_vs_split = compare_detections(read_yolo_txt(ground_txt, has_confidence=False), read_yolo_txt(split_txt))
            regular_vs_split = compare_detections(read_yolo_txt(regular_txt), read_yolo_txt(split_txt))

            all_results.append({
                "ground_vs_regular": summarize_comparison_results([ground_vs_regular]),
                "ground_vs_split": summarize_comparison_results([ground_vs_split]),
                "regular_vs_split": summarize_comparison_results([regular_vs_split])
            })

    return all_results

# Summarize overall results across all images
def summarize_overall_comparison(all_results):
    ground_vs_regular = [res['ground_vs_regular'] for res in all_results]
    ground_vs_split = [res['ground_vs_split'] for res in all_results]
    regular_vs_split = [res['regular_vs_split'] for res in all_results]

    def summarize_all(results):
        total_1 = sum([res['total_1'] for res in results])
        total_2 = sum([res['total_2'] for res in results])
        total_matched = sum([res['total_matched'] for res in results])
        unmatched_1 = sum([res['unmatched_1'] for res in results])
        unmatched_2 = sum([res['unmatched_2'] for res in results])

        avg_iou = np.mean([res['avg_iou'] for res in results])
        avg_conf_diff = np.mean([res['avg_conf_diff'] for res in results])

        return {
            "total_1": total_1,
            "total_2": total_2,
            "total_matched": total_matched,
            "unmatched_1": unmatched_1,
            "unmatched_2": unmatched_2,
            "avg_iou": avg_iou,
            "avg_conf_diff": avg_conf_diff
        }

    print("Ground vs Regular:")
    print(summarize_all(ground_vs_regular))
    print("\nGround vs Split:")
    print(summarize_all(ground_vs_split))
    print("\nRegular vs Split:")
    print(summarize_all(regular_vs_split))

# Directory paths
regular_labels_dir = r"C:\Users\GenCyber\Documents\Yolov8\runs\detect\val42\labels"
split_labels_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New\output_labels"
ground_labels_dir = r"C:\Users\GenCyber\Documents\Yolov8\onion_balanced\onion_train_test\testing\labels\test"

# Compare all three sets of labels
all_results = compare_all(regular_labels_dir, split_labels_dir, ground_labels_dir)

# Summarize the overall comparison results across all images
summarize_overall_comparison(all_results)
