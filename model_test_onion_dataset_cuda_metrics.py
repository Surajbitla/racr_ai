from pathlib import Path
import os
import logging
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics
import itertools
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

# IoU calculation
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def draw_detections(image, detections, class_names):
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
        text_size = draw.textbbox((0, 0), label, font)
        draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill=color)
        draw.text((x1, y1 - text_size[3]), label, fill=(255, 255, 255), font=font)

    return image

def postprocess(outputs, original_img_size, conf_threshold=0.25, iou_threshold=0.45):
    if isinstance(outputs, tuple):
        outputs = outputs[0]
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
            boxes.append([left, top, left + width, top + height])

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

# Initialize Ultralytics metrics
metrics = DetMetrics(save_dir="output_metrics", on_plot=True)

confidences = []
pred_classes = []
target_classes = []
ious = []
gt_boxes = []  # Ground truth boxes should be loaded here

m.eval()
m2.eval()

split_layer = 5

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
        out = m2(res, start=split_layer)

        detections = postprocess(out, original_image[0].size)
        print(f"Detections for {image_files[0]}:", detections)

        # Placeholder: ground truth boxes and class ids should be loaded here for each image
        gt_boxes = []  # Load ground truth bounding boxes here
        gt_class_ids = []  # Load ground truth class ids here

        # For each detection, compute IoU with ground truth and accumulate true positives
        for detection in detections:
            box, score, pred_class = detection

            # Placeholder: accumulate confidences and predicted classes
            confidences.append(score)
            pred_classes.append(pred_class)
            target_classes.append(0)  # Replace with actual ground truth class

            # Compute IoU for each detection with the ground truth
            for gt_box in gt_boxes:
                iou_value = calculate_iou(box, gt_box)
                ious.append(iou_value)

        # Save detection images
        output_image = draw_detections(original_image[0], detections, class_names)
        output_path = os.path.join("output_images", f"output_with_detections_{image_files[0]}")
        os.makedirs("output_images", exist_ok=True)
        output_image.save(output_path)
        print(f"Detections drawn and image saved as {output_path}.")

# Ensure tp has the correct shape (num_detections, num_iou_thresholds)
if len(ious) > 0:
    tp = np.zeros((len(ious), len(np.linspace(0.5, 0.95, 10))))
    for i, iou_value in enumerate(ious):
        for t, iou_thresh in enumerate(np.linspace(0.5, 0.95, 10)):
            tp[i, t] = 1 if iou_value > iou_thresh else 0
else:
    print("Warning: No IoU values were calculated, skipping metric processing.")
    tp = np.array([])

# Convert lists to numpy arrays for the metric processing
confidences = np.array(confidences)
pred_classes = np.array(pred_classes)
target_classes = np.array(target_classes)

# Check if arrays are not empty before processing metrics
if len(confidences) > 0 and len(pred_classes) > 0 and len(target_classes) > 0 and tp.size > 0:
    metrics.process(tp=tp, conf=confidences, pred_cls=pred_classes, target_cls=target_classes)
    metrics.plot_metrics()  # Plot mAP, precision-recall curves, etc.
    print("Metrics and graphs saved to 'output_metrics'.")
else:
    print("Warning: Empty arrays detected, skipping metric processing.")
