import pickle
import numpy as np

# Function to load data from a file
def load_data_from_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # Area of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Area of union
    union_area = box1_area + box2_area - inter_area

    # IoU calculation
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou

# Function to compare the results from different compression levels
def compare_results(data_level_0, data_other_level):
    differences = {
        "bounding_box_diff": [],
        "confidence_diff": [],
        "label_diff": 0,
        "iou_diff": []
    }

    for (prediction_level_0, file_level_0, _), (prediction_other, file_other, _) in zip(data_level_0, data_other_level):
        # Ensure we're comparing the same image
        if file_level_0 != file_other:
            print(f"File mismatch: {file_level_0} != {file_other}")
            continue

        # Compare bounding boxes, confidences, and labels
        for (box_0, score_0, label_0), (box_other, score_other, label_other) in zip(prediction_level_0, prediction_other):
            # Bounding box difference (Euclidean distance)
            box_diff = np.linalg.norm(np.array(box_0) - np.array(box_other))
            print('box0',box_0)
            print('bpx1',box_other)
            differences["bounding_box_diff"].append(box_diff)

            # Confidence score difference
            score_diff = abs(score_0 - score_other)
            differences["confidence_diff"].append(score_diff)

            # Check if labels are different
            if label_0 != label_other:
                differences["label_diff"] += 1

            # IoU difference
            iou_level_0 = calculate_iou(box_0, box_other)
            differences["iou_diff"].append(iou_level_0)

    return differences

# Function to print and summarize the differences
def print_differences(differences, level):
    print(f"\nSummary of Differences for Compression Level {level}:")
    if differences["bounding_box_diff"]:
        print(f"Average Bounding Box Difference: {np.mean(differences['bounding_box_diff']):.4f}")
    else:
        print("No bounding box differences detected.")
    
    if differences["confidence_diff"]:
        print(f"Average Confidence Score Difference: {np.mean(differences['confidence_diff']):.4f}")
    else:
        print("No confidence score differences detected.")
    
    if differences["iou_diff"]:
        print(f"Average IoU: {np.mean(differences['iou_diff']):.4f}")
    else:
        print("No IoU calculated.")

    print(f"Number of label mismatches: {differences['label_diff']}")

# Main function to compare results from levels 0 to 9
def main():
    # Load data for level 0 (reference)
    data_level_0 = load_data_from_file("lossy_data_clevel_0.pkl")

    # Compare with data from compression levels 1-9
    for level in range(1, 10):
        data_other_level = load_data_from_file(f"lossy_data_clevel_{level}.pkl")
        differences = compare_results(data_level_0, data_other_level)
        print_differences(differences, level)

if __name__ == "__main__":
    main()
