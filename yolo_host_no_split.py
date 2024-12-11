import os
import pickle
import time
import sys
import socket
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import blosc2

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("yolo_logger")

# Dataset configuration
dataset_path = 'onion/testing'

# Network configuration
server_address = ('10.0.0.61', 12345)  # Update with your server's IP

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
        original_size = image.size
        if self.transform:
            image = self.transform(image)
        return image, original_size, self.image_files[idx]

def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images in batch"""
    images = torch.stack([item[0] for item in batch], 0)
    original_sizes = [item[1] for item in batch]
    image_files = [item[2] for item in batch]
    return images, original_sizes, image_files

def compress_blosc2(data, clevel=4):
    """Compress using blosc2"""
    try:
        serialized_data = pickle.dumps(data)
        
        # Calculate padding needed for 8-byte alignment
        remainder = len(serialized_data) % 8
        if remainder != 0:
            padding = b'\0' * (8 - remainder)
            serialized_data = serialized_data + padding
            
        compressed_data = blosc2.compress(serialized_data, 
                                        clevel=clevel,
                                        typesize=8)
        if compressed_data is None:
            raise RuntimeError("Compression failed")
        return compressed_data
    except Exception as e:
        logger.error(f"Compression error: {e}")
        raise

def run_experiment(use_compression=False):
    """Run the experiment with or without compression"""
    # Initialize dataset
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    dataset = OnionDataset(root=dataset_path, transform=transform)
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    
    results = {
        'times': {
            'host': [],
            'travel': [],
            'server': [],
            'compression': [] if use_compression else None
        },
        'sizes': {
            'original': [],
            'compressed': [] if use_compression else None
        },
        'predictions': []
    }
    
    try:
        for batch_idx, (images, original_sizes, image_files) in enumerate(data_loader):
            print(f"\rProcessing image {batch_idx+1}/{len(data_loader)}", end='')
            
            # Host processing time (minimal in this case)
            host_start = time.time()
            image = images[0]  # Get first image tensor
            
            # Convert tensor to numpy array in correct format (HWC)
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0)  # CHW -> HWC
                image = (image * 255).numpy().astype(np.uint8)  # Scale to 0-255 and convert to uint8
            
            data_to_send = (image, original_sizes[0])
            host_time = time.time() - host_start
            results['times']['host'].append(host_time)
            
            # Compression if enabled
            if use_compression:
                comp_start = time.time()
                compressed_data = compress_blosc2(data_to_send)
                comp_time = time.time() - comp_start
                results['times']['compression'].append(comp_time)
                
                results['sizes']['original'].append(sys.getsizeof(pickle.dumps(data_to_send)))
                results['sizes']['compressed'].append(sys.getsizeof(compressed_data))
                
                # Send compression flag and data
                client_socket.sendall(b'1')
                data_length = len(compressed_data)
                client_socket.sendall(data_length.to_bytes(4, 'big'))
                client_socket.sendall(compressed_data)
            else:
                # Send uncompressed data
                client_socket.sendall(b'0')
                serialized_data = pickle.dumps(data_to_send)
                data_length = len(serialized_data)
                client_socket.sendall(data_length.to_bytes(4, 'big'))
                client_socket.sendall(serialized_data)
                
                results['sizes']['original'].append(sys.getsizeof(serialized_data))
            
            # Network and server time
            travel_start = time.time()
            
            # Receive results
            response_length = int.from_bytes(client_socket.recv(4), 'big')
            response = b''
            while len(response) < response_length:
                chunk = client_socket.recv(min(4096, response_length - len(response)))
                if not chunk:
                    raise ConnectionError("Connection broken")
                response += chunk
            
            travel_time = time.time() - travel_start
            results['times']['travel'].append(travel_time)
            
            # Process response
            detections, server_time = pickle.loads(response)
            results['times']['server'].append(server_time)
            results['predictions'].append((detections, image_files[0]))
            
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        raise
    finally:
        client_socket.close()
    
    return results

def save_experiment_results(results, output_folder):
    """Save experiment results to files"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate totals
    total_host_time = sum(results['times']['host'])
    total_travel_time = sum(results['times']['travel'])
    total_server_time = sum(results['times']['server'])
    total_compression_time = sum(results['times']['compression']) if results['times']['compression'] is not None else 0
    
    # Calculate total experiment time
    total_experiment_time = total_host_time + total_travel_time + total_server_time
    if results['times']['compression'] is not None:
        total_experiment_time += total_compression_time
    
    # Save detailed metrics
    metrics = {
        'host_times': results['times']['host'],
        'travel_times': results['times']['travel'],
        'server_times': results['times']['server'],
        'original_sizes': results['sizes']['original'],
        'compressed_sizes': results['sizes'].get('compressed', []),
        'compression_times': results['times'].get('compression', []),
        'total_times': {
            'host': total_host_time,
            'travel': total_travel_time,
            'server': total_server_time,
            'compression': total_compression_time if results['times']['compression'] is not None else None,
            'total_experiment': total_experiment_time
        }
    }
    
    metrics_file = os.path.join(output_folder, "metrics.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save predictions
    predictions_file = os.path.join(output_folder, "predictions.pkl")
    with open(predictions_file, 'wb') as f:
        pickle.dump(results['predictions'], f)
    
    # Save summary
    summary_file = os.path.join(output_folder, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Experiment Summary\n")
        f.write("=================\n\n")
        
        # Total experiment time
        f.write("Total Experiment Time\n")
        f.write("-----------------\n")
        f.write(f"Total time: {total_experiment_time:.2f} seconds\n")
        f.write(f"Number of images: {len(results['times']['host'])}\n")
        f.write(f"Average time per image: {total_experiment_time/len(results['times']['host']):.4f} seconds\n\n")
        
        # Detailed timing statistics
        f.write("Detailed Timing Statistics\n")
        f.write("------------------------\n")
        f.write("Per Image (seconds):\n")
        f.write(f"Host processing: {np.mean(results['times']['host']):.4f} ± {np.std(results['times']['host']):.4f}\n")
        
        # Only include compression stats if compression was used
        if results['times']['compression'] is not None:
            compression_times = results['times']['compression']
            f.write(f"Compression: {np.mean(compression_times):.4f} ± {np.std(compression_times):.4f}\n")
            
        f.write(f"Network travel: {np.mean(results['times']['travel']):.4f} ± {np.std(results['times']['travel']):.4f}\n")
        f.write(f"Server processing: {np.mean(results['times']['server']):.4f} ± {np.std(results['times']['server']):.4f}\n\n")
        
        # Total times
        f.write("Total Times (seconds):\n")
        f.write(f"Total host time: {total_host_time:.2f}\n")
        if results['times']['compression'] is not None:
            f.write(f"Total compression time: {total_compression_time:.2f}\n")
        f.write(f"Total network travel time: {total_travel_time:.2f}\n")
        f.write(f"Total server processing time: {total_server_time:.2f}\n\n")
        
        # Size statistics if compression was used
        f.write("Size Statistics\n")
        f.write("--------------\n")
        if results['sizes']['compressed'] is not None:
            orig_size_avg = np.mean(results['sizes']['original'])
            comp_size_avg = np.mean(results['sizes']['compressed'])
            total_orig_size = sum(results['sizes']['original'])
            total_comp_size = sum(results['sizes']['compressed'])
            
            f.write("Per Image (bytes):\n")
            f.write(f"Original size: {orig_size_avg:.2f}\n")
            f.write(f"Compressed size: {comp_size_avg:.2f}\n")
            f.write(f"Compression ratio: {comp_size_avg/orig_size_avg:.4f}\n\n")
            
            f.write("Total (bytes):\n")
            f.write(f"Total original size: {total_orig_size:,}\n")
            f.write(f"Total compressed size: {total_comp_size:,}\n")
            f.write(f"Overall compression ratio: {total_comp_size/total_orig_size:.4f}\n")
        else:
            # Just show original size stats
            orig_size_avg = np.mean(results['sizes']['original'])
            total_orig_size = sum(results['sizes']['original'])
            f.write("Per Image (bytes):\n")
            f.write(f"Original size: {orig_size_avg:.2f}\n\n")
            f.write("Total (bytes):\n")
            f.write(f"Total original size: {total_orig_size:,}\n")

def main():
    # Create output folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = "no_split_experiments"
    os.makedirs(output_base, exist_ok=True)
    
    # Run without compression
    print("\nRunning experiment without compression...")
    results_no_comp = run_experiment(use_compression=False)
    save_experiment_results(results_no_comp, 
                          os.path.join(output_base, f"{timestamp}_no_compression"))
    
    # Run with compression
    print("\nRunning experiment with compression...")
    results_comp = run_experiment(use_compression=True)
    save_experiment_results(results_comp, 
                          os.path.join(output_base, f"{timestamp}_with_compression"))
    
    print("\nExperiments completed!")

if __name__ == "__main__":
    main() 