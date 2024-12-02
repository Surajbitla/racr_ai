import torchvision
import os
import zlib
import pickle
import time
import sys
import socket
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
import blosc2
import pandas as pd
import subprocess
import time
import os
from datetime import datetime
from src.tracr.experiment_design.models.model_hooked import WrappedModel, HookExitException

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Function to generate and rename the battery report
def generate_battery_report(prefix="before"):
    # Set the output path explicitly
    output_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New"
    output_file = os.path.join(output_dir, "battery-report.html")

    # Generate the battery report and save it to the specified path
    subprocess.run(f"powercfg /batteryreport /output {output_file}", shell=True)

    # Generate the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define destination file path with a unique name
    dest_file = os.path.join(output_dir, f"{timestamp}_battery_report_{prefix}.html")
    
    # Rename the battery report to avoid overwriting
    if os.path.exists(output_file):
        os.rename(output_file, dest_file)
        print(f"Battery report saved as {dest_file}")
    else:
        print(f"Battery report not found at {output_file}")

# Step 1: Generate the battery report before running the YOLO code
generate_battery_report(prefix="before")

# Step 2: Run the entire YOLO processing code
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

def compress_data_lossless(data):
    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data)
    size_bytes = sys.getsizeof(compressed_data)
    return compressed_data,size_bytes

# Custom lossy compression using quantization
def compress_data_lossy_batched(data, bits=8):
    levels = 2**bits - 1
    d = [1] * data[0].ndim
    d[0] = -1
    
    # Unpack the data tuple
    tensor_data, original_img_size = data
    mins = torch.amin(tensor_data, dim=(1,2,3)).view(d)
    maxs = torch.amax(tensor_data, dim=(1,2,3)).view(d)
    
    # Normalize and quantize
    data_normalized = (tensor_data - mins) / (maxs - mins)
    data_quantized = torch.round(data_normalized * levels) / levels
    
    # Pack the necessary information for decompression
    compressed_data = (data_quantized, mins, maxs, original_img_size)
    
    # Serialize the data
    serialized_data = pickle.dumps(compressed_data)
    size_bytes = sys.getsizeof(serialized_data)
    return serialized_data, size_bytes


# Load the model
model.eval()

print(device)
server_address = ('10.0.0.219', 12345)  # Update with your server's address
# server_address = ('10.33.76.187', 12345)
print(server_address)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)

# Function to save data to a file
def save_data_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

# Function to save detections in YOLO format to a .txt file
def save_detections_to_txt(detections, image_filename, original_img_size, output_dir="output_labels"):
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, image_filename.replace('.jpg', '.txt'))
    
    with open(txt_filename, 'w') as f:
        for detection in detections:
            box, score, class_id = detection
            # Convert bounding box coordinates from pixel space to normalized YOLO format
            x_center = (box[0] + box[2] / 2) / original_img_size[0]
            y_center = (box[1] + box[3] / 2) / original_img_size[1]
            width = box[2] / original_img_size[0]
            height = box[3] / original_img_size[1]
            # Save as: class_id x_center y_center width height confidence_score
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
    print(f"Saved detection results to {txt_filename}")

# Modify the test_split_performance function to save detections on host side
def test_split_performance(client_socket, split_layer_index, mode, bits=8):
    correct = 0
    total = 0
    start_time = time.time()
    # Lists to store times
    host_times = []
    compression_times = []
    travel_times = []
    server_times = []

    compression = []

    experiment_results = []  # To store results

    with torch.no_grad():
        for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
            # Measure host processing time
            input_tensor = input_tensor.to(model.device)
            host_start_time = time.time()
            # Processing with the model...
            out = model(input_tensor, end=split_layer_index)
            data_to_send = (out, original_image[0].size)

            # Compression: Lossless or lossy
            compression_start_time = time.time()
            if mode == "lossless":
                compressed_output, compressed_size = compress_data_lossless(data_to_send)
            elif mode == "lossy":
                compressed_output, compressed_size = compress_data_lossy_batched(data_to_send, bits)
            else:
                raise ValueError(f"Unknown mode received: {mode}")
            
            compression.append(compressed_size)
            compression_end_time = time.time()
            compression_times.append(compression_end_time - compression_start_time)

            # Convert mode to bytes (e.g., "lossless" -> b'lossless')
            mode_bytes = mode.encode('utf-8')
            mode_length = len(mode_bytes)

            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            # Send mode to server
            travel_start_time = time.time()
            client_socket.sendall(mode_length.to_bytes(4, 'big'))  # Send length of mode string first
            client_socket.sendall(mode_bytes)  # Send mode string itself

            # Send data to server
            client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
            client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
            client_socket.sendall(compressed_output)

            # Receive and unpack server response
            data = client_socket.recv(4096)
            prediction, server_processing_time = pickle.loads(data)
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)  # This includes server time
            server_times.append(server_processing_time)  # Log server time for each image
            experiment_results.append((prediction, image_files[0], compressed_size))
            print(prediction)
            # Save detections in YOLO format on the host side
            save_detections_to_txt(prediction, image_files[0], original_image[0].size)

    end_time = time.time()
    processing_time = end_time - start_time
    # accuracy = 100 * correct / total
    # print(f'Accuracy of the model on the test images: {accuracy} %')
    # print(f"Compressed Size in bytes: {compressed_size}")
    # Calculate average times for each part
    total_host_time = sum(host_times)
    total_travel_time = sum(travel_times) - sum(server_times)   # Correcting travel time
    total_server_time = sum(server_times)
    total_processing_time = total_host_time + total_travel_time + total_server_time
    print(f"Total Host Time: {total_host_time:.2f} s, Total Travel Time: {total_travel_time:.2f} s, Total Server Time: {total_server_time:.2f} s")

    total_compression_time = sum(compression_times)
    total_compression_size = sum(compression)
    print(f"Total Compression Time: {total_compression_time:.2f} s, Total Compression Size: {total_compression_size}")

    filename = f'{mode}_data_clevel_{bits}.pkl'
    save_data_to_file(experiment_results, filename)

    return (total_host_time, total_travel_time, total_server_time, total_processing_time)

# total_layers = 23 # len(list(model.backbone.body.children()))
# print(total_layers)
# time_taken = []

# for split_layer_index in range(1, total_layers):  # Assuming layer 0 is not a viable split point
#     host_time,travel_time,server_time,processing_time = test_split_performance(client_socket, split_layer_index)
#     print(f"Split at layer {split_layer_index}, Processing Time: {processing_time:.2f} seconds")
#     time_taken.append((split_layer_index, host_time, travel_time, server_time, processing_time))

# best_split, host_time, travel_time, server_time, min_time = min(time_taken, key=lambda x: x[4])
# print(f"Best split at layer {best_split} with time {min_time:.2f} seconds")
# for i in time_taken:
#     split_layer_index, host_time, travel_time, server_time, processing_time = i
#     print(f"Layer {split_layer_index}: Host Time = {host_time}, Travel Time = {travel_time}, Server Time = {server_time}, Total Processing Time = {processing_time}")

# df = pd.DataFrame(time_taken, columns=["Split Layer Index", "Host Time", "Travel Time", "Server Time", "Total Processing Time"])

# # Save the DataFrame to an Excel file
# df.to_excel("split_layer_times.xlsx", index=False)

split_layer_index = 3
mode = 'lossy'
bits = 6
print(f"Running in mode: {mode} with bits={bits}")
host_time,travel_time,server_time,processing_time = test_split_performance(client_socket, split_layer_index, mode, bits)
client_socket.close()


# Step 3: Generate the battery report after the YOLO code finishes
generate_battery_report(prefix="after")
print("Battery reports generated before and after YOLO processing.")

# Step 4: Call the mwh_extractor.py script using subprocess
try:
    # Run the mwh_extractor.py script after YOLO processing completes
    subprocess.run(['python', 'mwh_extractor.py'], check=True)
    print("mwh_extractor.py executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while running mwh_extractor.py: {e}")