import pickle
import numpy as np
import cv2

# Function to load data from a file
def load_data_from_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
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

# Function to compare the results from lossless and lossy experiments
def compare_results(lossless_data, lossy_data):
    differences = {
        "bounding_box_diff": [],
        "confidence_diff": [],
        "label_diff": 0,
        "iou_diff": []
    }

    # Iterate through each image's predictions
    for (prediction_lossless, file_lossless, _), (prediction_lossy, file_lossy, _) in zip(lossless_data, lossy_data):
        # Ensure we're comparing the same image
        if file_lossless != file_lossy:
            print(f"File mismatch: {file_lossless} != {file_lossy}")
            continue

        # Compare bounding boxes, confidences, and labels
        for (box_lossless, score_lossless, label_lossless), (box_lossy, score_lossy, label_lossy) in zip(prediction_lossless, prediction_lossy):
            # Bounding box difference (Euclidean distance)
            box_diff = np.linalg.norm(np.array(box_lossless) - np.array(box_lossy))
            differences["bounding_box_diff"].append(box_diff)

            # Confidence score difference
            score_diff = abs(score_lossless - score_lossy)
            differences["confidence_diff"].append(score_diff)

            # Check if labels are different
            if label_lossless != label_lossy:
                differences["label_diff"] += 1

            # IoU difference
            iou_lossless = calculate_iou(box_lossless, box_lossy)
            differences["iou_diff"].append(iou_lossless)

    return differences

# Function to print and summarize the differences
def print_differences(differences):
    print("\nSummary of Differences:")
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

# Main function to load and compare the data
def main():
    # Load lossless and lossy data
    lossless_data = load_data_from_file("lossless_data.pkl")
    lossy_data = load_data_from_file("lossy_data_clevel_4.pkl")

    # Compare the results
    differences = compare_results(lossless_data, lossy_data)

    # Print the results
    print_differences(differences)

if __name__ == "__main__":
    main()
