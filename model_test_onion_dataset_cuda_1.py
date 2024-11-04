from pathlib import Path
import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, average_precision_score
from collections import defaultdict  # Import added here

from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Define the path to your dataset
dataset_path = 'onion/testing'
weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()
        if self.transform:
            image = self.transform(image)
        return image, original_image, self.image_files[idx]

# Custom dataset transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Load your dataset
dataset = OnionDataset(root=dataset_path, transform=transform)

# Custom collate_fn to avoid batching the PIL Image
def custom_collate_fn(batch):
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files

# DataLoader with custom collate function
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Initialize the YOLO model
yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
m = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
m2 = WrappedModel(config_path=yaml_file_path, weights_path=weight_path, participant_key='server')

def calculate_iou(box1, box2):
    """Compute the Intersection Over Union (IoU) between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y1_inter - y2_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

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

# Evaluation and metrics functions
def evaluate_predictions(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    tp, fp, scores, gt_count = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
    
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

    precision, recall, f1 = {}, {}, {}
    for cls in scores:
        # Threshold the predicted scores to convert them into binary predictions (e.g., threshold = 0.5)
        binary_preds = [1 if s >= 0.5 else 0 for s in scores[cls]]
        
        # Precision-recall curve
        precision[cls], recall[cls], _ = precision_recall_curve(tp[cls], scores[cls])
        
        # F1 Score Calculation
        f1[cls] = f1_score(tp[cls], binary_preds)
    
    mAP = np.mean([average_precision_score(tp[cls], scores[cls]) for cls in scores])
    
    return precision, recall, f1, mAP


def plot_pr_curve(precision, recall, mAP):
    plt.figure()
    for cls in precision:
        plt.plot(recall[cls], precision[cls], label=f'Class {cls}')
    plt.title(f'Precision-Recall Curve (mAP={mAP:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig("PR_curve.png")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap="Blues")
    plt.title(f'Confusion Matrix{" (Normalized)" if normalize else ""}')
    plt.savefig("confusion_matrix.png" if not normalize else "confusion_matrix_normalized.png")
    plt.show()

def plot_f1_curve(confidences, f1_scores, classes):
    plt.figure()
    for cls in confidences:
        plt.plot(confidences[cls], f1_scores[cls], label=f'Class {cls}')
    plt.title('F1-Confidence Curve')
    plt.xlabel('Confidence')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig("F1_curve.png")
    plt.show()

m.eval()
m2.eval()

split_layer = 5

all_pred_boxes, all_pred_classes, all_pred_scores = [], [], []
all_gt_boxes, all_gt_classes = [], []

with torch.no_grad():
    for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer}"):
        input_tensor = input_tensor.to(m.device)
        res = m(input_tensor, end=split_layer)
        logging.info("Switched.")
        if isinstance(res, NotDict):
            inner_dict = res.inner_dict
            for key in inner_dict:
                if isinstance(inner_dict[key], torch.Tensor):
                    inner_dict[key] = inner_dict[key].to(m2.device)
        else:
            print("res is not an instance of NotDict")
        out = m2(res, start=split_layer)

        detections = postprocess(out, original_image[0].size)
        print(f"Detections for {image_files[0]}:", detections)

        pred_boxes = [d[0] for d in detections]
        pred_scores = [d[1] for d in detections]
        pred_classes = [d[2] for d in detections]

        all_pred_boxes.extend(pred_boxes)
        all_pred_classes.extend(pred_classes)
        all_pred_scores.extend(pred_scores)

        output_image = draw_detections(original_image[0], detections, class_names)
        output_path = os.path.join("output_images", f"output_with_detections_{image_files[0]}")
        os.makedirs("output_images", exist_ok=True)
        output_image.save(output_path)
        print(f"Detections drawn and image saved as {output_path}.")

    precision, recall, f1, mAP = evaluate_predictions(all_pred_boxes, all_pred_scores, all_pred_classes, all_gt_boxes, all_gt_classes)
    plot_pr_curve(precision, recall, mAP)
    plot_confusion_matrix(all_gt_classes, all_pred_classes, classes=class_names)
    plot_f1_curve(precision, recall, class_names)
