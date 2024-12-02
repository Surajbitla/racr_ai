from pathlib import Path
import os
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
import torch.nn.functional as F

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

# Initialize counters
seen = 0
nt = torch.zeros(1)  # number of targets

# Define class names at the beginning of your script
names = {0: "with_weeds", 1: "without_weeds"}  # or use your class_names list
# class_names is already defined as ["with_weeds", "without_weeds"]
names = {i: name for i, name in enumerate(class_names)}

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

# Initialize confusion matrix before the main loop
nc = len(class_names)  # number of classes
confusion_matrix = ConfusionMatrix(nc=nc)
stats = []  # List to store statistics for each image

def postprocess(pred, conf_thres=0.25, iou_thres=0.45):
    """
    Post-process outputs (YOLOv8 format)
    """
    if isinstance(pred, tuple):
        outputs = pred[0]  # Take the first element of the tuple
    else:
        outputs = pred
        
    if outputs is None:
        return torch.zeros((0, 6))
    
    # Reshape predictions [batch, channels, height]
    outputs = outputs.squeeze(0).permute(1, 0).contiguous()
    
    # Split predictions
    box_xy = outputs[:, :2]  # x, y
    box_wh = outputs[:, 2:4]  # w, h
    objectness = outputs[:, 4]  # objectness score
    class_probs = outputs[:, 5]  # class probabilities
    
    # Apply sigmoid to scores
    objectness = torch.sigmoid(objectness)
    class_probs = torch.sigmoid(class_probs)
    
    # Convert boxes to xyxy format
    boxes = torch.zeros_like(outputs[:, :4])
    boxes[:, 0] = box_xy[:, 0] - box_wh[:, 0] / 2  # x1
    boxes[:, 1] = box_xy[:, 1] - box_wh[:, 1] / 2  # y1
    boxes[:, 2] = box_xy[:, 0] + box_wh[:, 0] / 2  # x2
    boxes[:, 3] = box_xy[:, 1] + box_wh[:, 1] / 2  # y2
    
    # Calculate confidence scores
    scores = objectness * class_probs
    
    # Filter by confidence
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = torch.zeros_like(scores)  # Assuming single class
    
    if len(boxes) > 0:
        # Apply NMS
        keep = torchvision.ops.nms(
            boxes,
            scores,
            iou_thres
        )
        
        # Format output: [x1, y1, x2, y2, score, class_id]
        output = torch.cat((
            boxes[keep],
            scores[keep].unsqueeze(1),
            class_ids[keep].unsqueeze(1)
        ), dim=1)
        
        # Keep only top-k predictions
        if len(output) > 100:  # Adjust this threshold as needed
            scores = output[:, 4]
            _, indices = torch.sort(scores, descending=True)
            output = output[indices[:100]]
        
        return output
    
    return torch.zeros((0, 6), device=outputs.device)

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
        label = f"{class_names[class_id]} {score:.2f}"
        
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

def compute_metrics(stats, iou_threshold=0.5):
    """Compute precision, recall, and mAP metrics"""
    true_positives = []
    false_positives = []
    scores = []
    total_gt = 0
    
    # Process each image's statistics
    for stat in stats:
        pred_boxes, pred_scores, pred_labels, targets = stat
        
        # Convert targets to numpy for easier handling
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        total_gt += len(targets)
        detected = set()  # Track which ground truth boxes have been detected
        
        # Sort predictions by confidence
        if len(pred_scores) > 0:
            sort_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[sort_idx]
            pred_scores = pred_scores[sort_idx]
        
        # Process each prediction
        for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
            scores.append(score.item())
            
            if len(targets) == 0:
                false_positives.append(1)
                true_positives.append(0)
                continue
            
            # Calculate IoU with all ground truth boxes
            ious = []
            for target in targets:
                iou = calculate_iou(
                    box.cpu().numpy(),
                    target[:4]  # first 4 elements are box coordinates
                )
                ious.append(iou)
            
            # Get best IoU and corresponding ground truth box index
            best_iou = max(ious)
            best_gt_idx = np.argmax(ious)
            
            if best_iou >= iou_threshold and best_gt_idx not in detected:
                true_positives.append(1)
                false_positives.append(0)
                detected.add(best_gt_idx)
            else:
                true_positives.append(0)
                false_positives.append(1)
    
    # Convert to numpy arrays
    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    scores = np.array(scores)
    
    # Sort by score
    indices = np.argsort(-scores)
    true_positives = true_positives[indices]
    false_positives = false_positives[indices]
    
    # Compute cumulative sums
    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)
    
    # Compute precision and recall
    recall = true_positives / total_gt if total_gt > 0 else np.zeros_like(true_positives)
    precision = true_positives / (true_positives + false_positives)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for r in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= r) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= r])
        ap += p / 11.0
    
    return precision, recall, ap, total_gt

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Rescale boxes (xyxy) from img1_shape to img0_shape
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """
    Clip boxes (xyxy) to image shape (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def plot_box(box, img, color=(128, 128, 128), label=None, line_thickness=3):
    """
    Plot one bounding box on image img
    """
    tl = line_thickness or round(0.002 * (img.size[0] + img.size[1]) / 2) + 1  # line/font thickness
    draw = ImageDraw.Draw(img)
    
    # Ensure coordinates are in correct order (x1 <= x2 and y1 <= y2)
    x1, y1 = min(float(box[0]), float(box[2])), min(float(box[1]), float(box[3]))
    x2, y2 = max(float(box[0]), float(box[2])), max(float(box[1]), float(box[3]))
    
    c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
    draw.rectangle([c1[0], c1[1], c2[0], c2[1]], outline=color, width=tl)
    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        font = ImageFont.load_default()
        # Get text bbox
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw label background
        draw.rectangle(
            [c1[0], c1[1] - text_height - 3, c1[0] + text_width, c1[1]], 
            fill=color
        )
        # Draw text
        draw.text((c1[0], c1[1] - text_height - 3), label, fill=(255, 255, 255), font=font)

m.eval()
m2.eval()

split_layer = 3

with torch.no_grad():
    for batch_i, (input_tensor, original_images, image_files, targets) in enumerate(tqdm(data_loader)):
        input_tensor = input_tensor.to(m.device)
        
        # First part of split model
        res = m(input_tensor, end=split_layer)
        if isinstance(res, NotDict):
            inner_dict = res.inner_dict
            for key in inner_dict:
                if isinstance(inner_dict[key], torch.Tensor):
                    inner_dict[key] = inner_dict[key].to(m2.device)
        
        # Second part of split model
        pred = m2(res, start=split_layer)
        
        # Run NMS with different confidence thresholds
        predn = postprocess(pred, conf_thres=0.25, iou_thres=0.45)
        
        # Get image info
        si = 0  # batch index (always 0 for batch_size=1)
        img_shape = original_images[si].size
        
        # Scale boxes to original image size
        scale = min(640 / img_shape[0], 640 / img_shape[1])
        if predn.shape[0]:
            predn[:, [0, 2]] = predn[:, [0, 2]] * img_shape[1] / scale
            predn[:, [1, 3]] = predn[:, [1, 3]] * img_shape[0] / scale
        
        # Process predictions
        if len(predn):
            # Get predictions
            predn_boxes = predn[:, :4]
            predn_scores = predn[:, 4]
            predn_labels = predn[:, 5]
            
            # Store statistics
            stats.append((
                predn_boxes,  # boxes
                predn_scores,  # scores
                predn_labels,  # labels
                targets[si]     # ground truth
            ))
            
            # Draw boxes on image
            for box, score, label in zip(predn_boxes, predn_scores, predn_labels):
                plot_box(box, original_images[si], label=f"{class_names[int(label)]} {score:.2f}")
        else:
            stats.append((
                torch.zeros((0, 4)),  # empty boxes
                torch.zeros(0),       # empty scores
                torch.zeros(0),       # empty labels
                targets[si]           # ground truth
            ))

print("\nFinal Statistics:")
print(f"Total images processed: {len(stats)}")
print(f"Total ground truth boxes: {nt.sum()}")
print(f"Number of statistics entries: {len(stats)}")

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    """
    Clip bounding xyxy bounding boxes to image shape (height, width)
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

# Compute metrics
if len(stats) > 0:
    precision, recall, ap, total_targets = compute_metrics(stats)
    
    print("\nFinal Statistics:")
    print(f"Total images processed: {len(stats)}")
    print(f"Total ground truth boxes: {total_targets}")
    print(f"Number of statistics entries: {len(stats)}")
    
    print("\nDetection Metrics:")
    print(f"Precision: {precision.mean():.4f}")
    print(f"Recall: {recall.mean():.4f}")
    print(f"AP@0.5: {ap:.4f}")
else:
    print("No detections to evaluate")

# Print confusion matrix
confusion_matrix.print()

# Print additional metrics details
print('\nDetailed Metrics:')
print(f'* Images processed: {len(stats)}')
print(f'* Total targets: {nt.sum()}')
print(f'* Precision: {precision.mean():.4f}')
print(f'* Recall: {recall.mean():.4f}')
print(f'* mAP@0.5: {ap:.4f}')
