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
from src.tracr.experiment_design.models.model_hooked import WrappedModel, HookExitException

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Define the path to your dataset
dataset_path = 'onion/testing'

weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]

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

m.eval()
m2.eval()

split_layer = 5

with torch.no_grad():
    for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer}"):
        out, layer_outputs = m(input_tensor, end=split_layer)
        logging.info("Switched.")
        out, layer_outputs = m2(out, start=split_layer)

        detections = postprocess(out, original_image[0].size)
        print(f"Detections for {image_files[0]}:", detections)

        output_image = draw_detections(original_image[0], detections, class_names)
        
        # Save the image with detections
        output_path = os.path.join("output_images", f"output_with_detections_{image_files[0]}")
        os.makedirs("output_images", exist_ok=True)
        output_image.save(output_path)
        print(f"Detections drawn and image saved as {output_path}.")
