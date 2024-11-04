import torch
import os
import pickle
import time
import socket
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import blosc2
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from src.tracr.experiment_design.models.model_hooked import WrappedModel, HookExitException

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Define the path to your dataset
dataset_path = 'onion/testing'
weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

# Initialize the YOLO model
yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)

# Blosc2 Compression
def compress_data(data):
    serialized_data = pickle.dumps(data)
    compressed_data = blosc2.compress(serialized_data, clevel=4, filter=blosc2.Filter.SHUFFLE, codec=blosc2.Codec.ZSTD)
    size_bytes = len(compressed_data)
    return compressed_data, size_bytes

# Load the model
model.eval()

print(device)
server_address = ('10.0.0.219', 12345)  # Update with your server's address
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)

# IoU Calculation
def calculate_iou(pred_box, true_box):
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box
    x1_true, y1_true, x2_true, y2_true = true_box

    # Calculate the (x, y)-coordinates of the intersection
    x1 = max(x1_pred, x1_true)
    y1 = max(y1_pred, y1_true)
    x2 = min(x2_pred, x2_true)
    y2 = min(y2_pred, y2_true)

    # Compute the area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both boxes
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)

    # Compute the union area
    union_area = pred_area + true_area - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou

# Precision, Recall Calculation
def calculate_precision_recall(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    return precision, recall

# Method to get true labels from the corresponding text files
def get_true_label(image_file):
    # Assuming the labels are stored in the same directory as the images, but with .txt extension
    label_file = image_file.replace('.jpg', '.txt')  # Replace image extension with .txt
    label_path = os.path.join(dataset_path, label_file)  # Full path to the label file
    
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file for {image_file} not found at {label_path}.")
    
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            # Assuming each line in the label file follows the format: class_id x_center y_center width height
            class_id, x_center, y_center, width, height = map(float, line.split())
            labels.append(int(class_id))  # Append only the class ID (ignore the coordinates for now)
    
    return labels


def test_split_performance(client_socket, split_layer_index):
    correct = 0
    total = 0
    start_time = time.time()
    host_times = []
    travel_times = []
    server_times = []
    
    true_labels = []  # Store true class labels
    pred_labels = []  # Store predicted class labels
    pred_scores = []  # Store confidence scores for precision-recall curve

    with torch.no_grad():
        for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
            input_tensor = input_tensor.to(model.device)
            host_start_time = time.time()
            out = model(input_tensor, end=split_layer_index)
            data_to_send = (out, original_image[0].size)
            compressed_output, compressed_size = compress_data(data_to_send)
            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            travel_start_time = time.time()
            client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
            client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
            client_socket.sendall(compressed_output)

            data = client_socket.recv(4096)
            predictions, server_processing_time = pickle.loads(data)
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)
            server_times.append(server_processing_time)

            # Loop over each prediction and append the class ID and confidence score
            current_pred_labels = []
            current_pred_scores = []
            for prediction in predictions:
                current_pred_labels.append(prediction[2])  # Assuming prediction[2] is the class ID
                current_pred_scores.append(prediction[1])  # Assuming prediction[1] is the confidence score

            # Extend the true labels list with labels from the ground truth file
            current_true_labels = get_true_label(image_files[0])  # Define function to get labels for the current image

            # Check if true and predicted labels match at the image level
            if len(current_true_labels) != len(current_pred_labels):
                print(f"Label mismatch in image: {image_files[0]}")
                print(f"True labels count: {len(current_true_labels)}, Predicted labels count: {len(current_pred_labels)}")
            
            # Update evaluation lists
            true_labels.extend(current_true_labels)
            pred_labels.extend(current_pred_labels)
            pred_scores.extend(current_pred_scores)

    end_time = time.time()
    processing_time = end_time - start_time

    total_host_time = sum(host_times)
    total_travel_time = sum(travel_times) - sum(server_times)
    total_server_time = sum(server_times)
    total_processing_time = total_host_time + total_travel_time + total_server_time
    print(f"Total Host Time: {total_host_time:.2f} s, Total Travel Time: {total_travel_time:.2f} s, Total Server Time: {total_server_time:.2f} s")

    # Check for consistency in length of true and predicted labels
    if len(true_labels) != len(pred_labels):
        print(f"Warning: Mismatch in total label counts. True: {len(true_labels)}, Pred: {len(pred_labels)}")
        
        # Strategy 1: Truncate longer list to match shorter one
        min_length = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_length]
        pred_labels = pred_labels[:min_length]

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    # mAP Calculation (example using average_precision_score)
    average_precision = average_precision_score(true_labels, pred_scores)
    print(f"Average Precision (AP): {average_precision:.2f}")

    return total_host_time, total_travel_time, total_server_time, total_processing_time


# total_layers = 23  # len(list(model.backbone.body.children()))
# time_taken = []

# for split_layer_index in range(1, total_layers):
#     host_time, travel_time, server_time, processing_time = test_split_performance(client_socket, split_layer_index)
#     print(f"Split at layer {split_layer_index}, Processing Time: {processing_time:.2f} seconds")
#     time_taken.append((split_layer_index, host_time, travel_time, server_time, processing_time))

# best_split, host_time, travel_time, server_time, min_time = min(time_taken, key=lambda x: x[4])
# print(f"Best split at layer {best_split} with time {min_time:.2f} seconds")
# for i in time_taken:
#     split_layer_index, host_time, travel_time, server_time, processing_time = i
#     print(f"Layer {split_layer_index}: Host Time = {host_time}, Travel Time = {travel_time}, Server Time = {server_time}, Total Processing Time = {processing_time}")

split_layer_index = 3
host_time,travel_time,server_time,processing_time = test_split_performance(client_socket, split_layer_index)
client_socket.close()
