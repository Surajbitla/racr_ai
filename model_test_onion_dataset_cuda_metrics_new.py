from pathlib import Path
import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Define the path to your dataset
dataset_path = 'onion/testing'

weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize global variables for metrics calculations
all_detections = []
all_annotations = []

# Custom dataset class to load images
class OnionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        # Get annotation path by replacing .jpg with .txt
        annotation_path = img_path.replace('.jpg', '.txt')
        
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()
        
        # Load ground truth boxes
        ground_truth = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    # YOLO format: class x_center y_center width height
                    # All values are normalized (0-1)
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert normalized coordinates to pixel coordinates
                    img_width, img_height = original_image.size
                    x = int((x_center - width/2) * img_width)
                    y = int((y_center - height/2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    
                    ground_truth.append([x, y, w, h])
        
        if self.transform:
            image = self.transform(image)
        return image, original_image, self.image_files[idx], ground_truth

# Custom dataset transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Load your dataset
dataset = OnionDataset(root=dataset_path, transform=transform)

# Custom collate_fn to avoid batching the PIL Image
def custom_collate_fn(batch):
    images, original_images, image_files, ground_truth = zip(*batch)
    return torch.stack(images, 0), original_images, image_files, ground_truth

# DataLoader with custom collate function
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Initialize the YOLO model
yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
m = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
m2 = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)


def postprocess(outputs, original_img_size, conf_threshold=0.25, iou_threshold=0.45):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Adjust based on the structure of outputs

    outputs = outputs.detach().cpu().numpy()
    outputs = np.transpose(np.squeeze(outputs))
    rows = outputs.shape[0]

    boxes = []
    scores = []
    class_ids = []

    img_w, img_h = original_img_size
    input_height, input_width = 640, 640

    x_factor = img_w / input_width
    y_factor = img_h / input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= conf_threshold:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    detections = []
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            print(f"Class: {class_names[class_id]}, Score: {score:.2f}, Box: {box}")
            detections.append((box, score, class_id))

    return detections


def draw_detections(image, detections, class_names, padding=2):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.
    Adds moderate padding to ensure text labels are clearly visible.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, class_id in detections:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        color = 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[class_id]}: {score:.2f}"
        
        # Determine the position for the label text
        text_size = draw.textbbox((0, 0), label, font)
        text_width = text_size[2]
        text_height = text_size[3]
        
        # Adjust the label position if it goes outside the image boundary
        label_x = x1
        label_y = y1 - text_height - padding if y1 - text_height - padding > 0 else y1 + h + padding
        
        # Ensure the text doesn't overflow beyond the right boundary
        if label_x + text_width > image.width:
            label_x = image.width - text_width
        
        # Draw the label background with adjusted padding and text
        draw.rectangle([label_x, label_y - text_height - padding, label_x + text_width, label_y], fill=color)
        draw.text((label_x, label_y - text_height - padding), label, fill=(255, 255, 255), font=font)

    return image

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def compute_metrics(all_detections, all_annotations, iou_threshold=0.5):
    """Compute precision, recall, and mAP metrics"""
    true_positives, false_positives, scores, num_annotations = [], [], [], 0

    for detections, annotations in zip(all_detections, all_annotations):
        detected_annotations = []
        
        for detection in detections:
            score = detection[1]
            scores.append(score)
            
            if len(annotations) == 0:
                false_positives.append(1)
                true_positives.append(0)
                continue
            
            ious = [calculate_iou(detection[0], ann) for ann in annotations]
            max_iou = max(ious)
            best_annotation = np.argmax(ious)
            
            if max_iou >= iou_threshold and best_annotation not in detected_annotations:
                true_positives.append(1)
                false_positives.append(0)
                detected_annotations.append(best_annotation)
            else:
                false_positives.append(1)
                true_positives.append(0)
        
        num_annotations += len(annotations)

    # Sort by score
    indices = np.argsort(-np.array(scores))
    false_positives = np.array(false_positives)[indices]
    true_positives = np.array(true_positives)[indices]

    # Cumulative sums
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # Recall and precision calculation
    recall = true_positives / (num_annotations if num_annotations > 0 else 1)
    precision = true_positives / (true_positives + false_positives)
    ap = np.trapz(precision, recall) if recall.size and precision.size else 0

    return {
        "precision": precision[-1] if precision.size else 0,
        "recall": recall[-1] if recall.size else 0,
        "mAP50": ap,
        "mAP50-95": ap
    }

m.eval()
m2.eval()

split_layer = 3

with torch.no_grad():
    for input_tensor, original_image, image_files, ground_truth in tqdm(data_loader, desc=f"Testing split at layer {split_layer}"):
        input_tensor = input_tensor.to(m.device)
        res = m(input_tensor, end=split_layer)
        logging.info("Switched.")
        if isinstance(res, NotDict):
            inner_dict = res.inner_dict
            for key in inner_dict:
                if isinstance(inner_dict[key], torch.Tensor):
                    inner_dict[key] = inner_dict[key].to(m2.device)
                    print(f"Intermediate tensors of {key} moved to the correct device.")
        else:
            print("res is not an instance of NotDict")
        out = m2(res, start=split_layer)

        detections = postprocess(out, original_image[0].size)
        print(f"Detections for {image_files[0]}:", detections)

        # Append detections for metrics calculation
        all_detections.append(detections)

        # Replace the empty ground_truth_boxes with the actual ground truth
        all_annotations.append(ground_truth[0])  # [0] because batch_size=1

        output_image = draw_detections(original_image[0], detections, class_names)
        
        output_path = os.path.join("output_images", f"output_with_detections_{image_files[0]}")
        print(f"Detections drawn and image saved as {output_path}.")

# Calculate and print final metrics
metrics = compute_metrics(all_detections, all_annotations)
print("\nMetrics Summary:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"mAP50: {metrics['mAP50']:.4f}")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
