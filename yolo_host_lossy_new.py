import torchvision
import os
import zlib
import pickle
import time
import sys
import socket
from pathlib import Path
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
from src.tracr.experiment_design.models.model_hooked import WrappedModel,NotDict

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Function to generate and rename the battery report
def generate_battery_report(prefix="before"):
    output_dir = r"C:\Users\GenCyber\Documents\RACR_AI_New"
    output_file = os.path.join(output_dir, "battery-report.html")
    subprocess.run(f"powercfg /batteryreport /output {output_file}", shell=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = os.path.join(output_dir, f"{timestamp}_battery_report_{prefix}.html")
    if os.path.exists(output_file):
        os.rename(output_file, dest_file)
        print(f"Battery report saved as {dest_file}")
    else:
        print(f"Battery report not found at {output_file}")

# Initialize model and dataset parameters
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

# Compression functions
def compress_data_lossless(data):
    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data)
    size_bytes = sys.getsizeof(compressed_data)
    return compressed_data, size_bytes

def lossy_compress(data, bits=8):
    """
    Compress data using lossy quantization.
    Args:
        data: tuple of (tensor_data/NotDict, img_size)
        bits: number of bits for quantization (1-8)
    Returns:
        tuple of (compressed_data, size_in_bytes)
    """
    if not isinstance(data, tuple) or len(data) != 2:
        raise ValueError("Data must be a tuple of (tensor_data/NotDict, img_size)")
        
    tensor_data, img_size = data
    
    # Handle NotDict object
    if isinstance(tensor_data, NotDict):
        inner_dict = tensor_data.inner_dict
        compressed_inner_dict = {}
        
        for key, value in inner_dict.items():
            if isinstance(value, torch.Tensor):
                # Apply quantization to each tensor in the dictionary
                d = [1] * value.ndim
                d[0] = -1
                
                mins = torch.amin(value, dim=(1,2,3)).view(d) if value.ndim >= 4 else torch.amin(value)
                maxs = torch.amax(value, dim=(1,2,3)).view(d) if value.ndim >= 4 else torch.amax(value)
                
                data_range = maxs - mins
                data_range[data_range == 0] = 1.0
                
                data_normalized = (value - mins) / data_range
                levels = 2**bits - 1
                data_quantized = torch.round(data_normalized * levels) / levels
                
                compressed_inner_dict[key] = {
                    'quantized': data_quantized.cpu().numpy(),
                    'mins': mins.cpu().numpy(),
                    'maxs': maxs.cpu().numpy(),
                }
            else:
                # Store non-tensor values as is
                compressed_inner_dict[key] = value
                
        compressed_data = {
            'type': 'NotDict',
            'data': compressed_inner_dict,
            'bits': bits
        }
    else:
        raise ValueError(f"Unsupported data type: {type(tensor_data)}")
    
    # Serialize and get size
    serialized = pickle.dumps((compressed_data, img_size))
    size_bytes = sys.getsizeof(serialized)
    
    print(f"Lossy compression stats:")
    print(f"- Compressed size: {size_bytes} bytes")
    print(f"- Quantization bits: {bits}")
    
    return serialized, size_bytes

def save_data_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def save_detections_to_txt(detections, image_filename, original_img_size, output_dir="output_labels"):
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, image_filename.replace('.jpg', '.txt'))
    
    with open(txt_filename, 'w') as f:
        for detection in detections:
            box, score, class_id = detection
            x_center = (box[0] + box[2] / 2) / original_img_size[0]
            y_center = (box[1] + box[3] / 2) / original_img_size[1]
            width = box[2] / original_img_size[0]
            height = box[3] / original_img_size[1]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")
    print(f"Saved detection results to {txt_filename}")

def test_split_performance(client_socket, split_layer_index, mode, compression_level=8):
    try:
        host_times = []
        compression_times = []
        travel_times = []
        server_times = []
        compression = []
        experiment_results = []

        yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
        model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        
        dataset = OnionDataset(root=dataset_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: list(zip(*x)))

        start_time = time.time()

        with torch.no_grad():
            for input_tensor, original_image, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
                input_tensor = torch.stack(input_tensor).to(model.device)
                host_start_time = time.time()
                
                # Get model output
                out = model(input_tensor, end=split_layer_index)
                data_to_send = (out, original_image[0].size)

                # Compress data
                compression_start_time = time.time()
                if mode == "lossless":
                    compressed_output, compressed_size = compress_data_lossless(data_to_send)
                elif mode == "lossy":
                    compressed_output, compressed_size = lossy_compress(data_to_send, bits=compression_level)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                if compressed_output is None:
                    raise RuntimeError("Compression failed")
                
                compression.append(compressed_size)
                compression_end_time = time.time()
                compression_times.append(compression_end_time - compression_start_time)

                # Send data to server
                mode_bytes = mode.encode('utf-8')
                mode_length = len(mode_bytes)

                host_end_time = time.time()
                host_times.append(host_end_time - host_start_time)

                # Network communication
                travel_start_time = time.time()
                client_socket.sendall(mode_length.to_bytes(4, 'big'))
                client_socket.sendall(mode_bytes)
                client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
                client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
                client_socket.sendall(compressed_output)

                # Receive response
                data = client_socket.recv(4096)
                prediction, server_processing_time = pickle.loads(data)
                travel_end_time = time.time()
                
                travel_times.append(travel_end_time - travel_start_time)
                server_times.append(server_processing_time)
                experiment_results.append((prediction, image_files[0], compressed_size))
                
                save_detections_to_txt(prediction, image_files[0], original_image[0].size)

        # Calculate and print statistics
        end_time = time.time()
        total_host_time = sum(host_times)
        total_travel_time = sum(travel_times) - sum(server_times)
        total_server_time = sum(server_times)
        total_processing_time = total_host_time + total_travel_time + total_server_time
        
        print(f"\nPerformance Statistics:")
        print(f"Total Host Time: {total_host_time:.2f} s")
        print(f"Total Travel Time: {total_travel_time:.2f} s")
        print(f"Total Server Time: {total_server_time:.2f} s")
        print(f"Total Compression Time: {sum(compression_times):.2f} s")
        print(f"Total Compression Size: {sum(compression)} bytes")
        print(f"Average Compression Size: {sum(compression)/len(compression):.2f} bytes per image")

        filename = f'{mode}_data_compression_{compression_level}.pkl'
        save_data_to_file(experiment_results, filename)

        return (total_host_time, total_travel_time, total_server_time, total_processing_time)
        
    except Exception as e:
        print(f"Error in test_split_performance: {str(e)}")
        raise

if __name__ == "__main__":
    generate_battery_report(prefix="before")

    server_address = ('10.0.0.219', 12345)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    split_layer_index = 3
    mode = 'lossy'
    compression_level = 4  # 1-8, where 1 is highest compression
    print(f"Compression mode: {mode}, level: {compression_level}")
    
    host_time, travel_time, server_time, processing_time = test_split_performance(
        client_socket, split_layer_index, mode, compression_level
    )
    
    client_socket.close()
    generate_battery_report(prefix="after")
    
    try:
        subprocess.run(['python', 'mwh_extractor.py'], check=True)
        print("mwh_extractor.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running mwh_extractor.py: {e}")