import torchvision
import os
import pickle
import time
import sys
import socket
from pathlib import Path
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import blosc2
import io
import zlib
import cv2
from datetime import datetime
import subprocess
import pandas as pd

from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tracr_logger")

# Dataset and model configuration
dataset_path = 'onion/testing'
weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network configuration
server_address = ('10.0.0.219', 12345)  # Update with your server's IP

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

def custom_collate_fn(batch):
    """Custom collate function to handle PIL Images in batch"""
    images = torch.stack([item[0] for item in batch], 0)
    original_images = [item[1] for item in batch]  # Keep PIL images as list
    image_files = [item[2] for item in batch]
    return images, original_images, image_files

def compress_blosc2_zfp(data, clevel=4):
    """Compress using blosc2 with ZFP codec for lossy compression"""
    serialized_data = pickle.dumps(data)
    try:
        # Calculate appropriate typesize based on data length
        data_length = len(serialized_data)
        typesize = 8  # Use 8 bytes (64-bit) alignment
        
        # Pad data if necessary to match typesize
        padding_size = (typesize - (data_length % typesize)) % typesize
        padded_data = serialized_data + b'\0' * padding_size
        
        compressed_data = blosc2.compress(padded_data, 
                                        clevel=clevel,
                                        filter=blosc2.Filter.NOFILTER,
                                        codec=blosc2.Codec.ZFP_ACC,
                                        typesize=typesize)
        if compressed_data is None:
            raise RuntimeError("Compression failed")
        
        size_bytes = sys.getsizeof(compressed_data)
        return compressed_data, size_bytes
    except Exception as e:
        logger.error(f"Compression error: {e}")
        logger.error(f"Data type: {type(data[0])}")
        logger.error(f"Data shape: {data[0].shape if hasattr(data[0], 'shape') else 'No shape'}")
        raise

def compress_jpeg(data, quality=50):
    """Compress tensors using JPEG compression"""
    if isinstance(data[0], torch.Tensor):
        tensor = data[0].cpu().numpy()
        tensor = ((tensor - tensor.min()) * (255/(tensor.max() - tensor.min()))).astype(np.uint8)
        img = Image.fromarray(tensor[0])
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        compressed = buffer.getvalue()
        
        final_data = pickle.dumps((compressed, data[1]))
        size_bytes = sys.getsizeof(final_data)
        return final_data, size_bytes
    return None, 0

def compress_svd(data, k=10):
    """Compress using SVD by keeping top k singular values"""
    if isinstance(data[0], torch.Tensor):
        tensor = data[0].cpu().numpy()
        compressed_arrays = []
        
        for i in range(tensor.shape[0]):
            U, S, Vt = np.linalg.svd(tensor[i], full_matrices=False)
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]
            compressed_arrays.append((U_k, S_k, Vt_k))
            
        compressed_data = pickle.dumps((compressed_arrays, data[1]))
        size_bytes = sys.getsizeof(compressed_data)
        return compressed_data, size_bytes
    return None, 0

def compress_quantization(data, bits=8):
    """Compress using uniform quantization"""
    if isinstance(data[0], torch.Tensor):
        tensor = data[0].cpu().numpy()
        data_min = tensor.min()
        data_max = tensor.max()
        scaled = (tensor - data_min) / (data_max - data_min)
        levels = 2**bits
        quantized = np.round(scaled * (levels-1)) / (levels-1)
        compressed_data = pickle.dumps((quantized, data_min, data_max, data[1]))
        size_bytes = sys.getsizeof(compressed_data)
        return compressed_data, size_bytes
    return None, 0

def save_detections_to_txt(prediction, image_file, original_img_size):
    """Save detection results in YOLO format"""
    txt_filename = os.path.splitext(image_file)[0] + '.txt'
    with open(txt_filename, 'w') as f:
        for box, score, class_id in prediction:
            x_center = (box[0] + box[2]/2) / original_img_size[0]
            y_center = (box[1] + box[3]/2) / original_img_size[1]
            width = box[2] / original_img_size[0]
            height = box[3] / original_img_size[1]
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n")

def save_data_to_file(data, filename):
    """Save experiment results to file"""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def create_experiment_folder():
    """Create folder for experiment results"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"compression_experiments_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def get_tensor_size(data):
    """Get size of tensor or NotDict in bytes"""
    if isinstance(data, NotDict):
        # For NotDict, we need to get size of all tensors in inner_dict
        total_size = 0
        for tensor in data.inner_dict.values():
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.element_size() * tensor.nelement()
        return total_size
    elif isinstance(data, torch.Tensor):
        return data.element_size() * data.nelement()
    else:
        raise TypeError(f"Unsupported data type for size calculation: {type(data)}")

def compress_none(data):
    """Baseline - no compression"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    serialized = pickle.dumps((tensor, metadata))
    return serialized, tensor_size, sys.getsizeof(serialized)

def compress_lossless_zlib(data, level=9):
    """Regular lossless compression using zlib"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    if isinstance(tensor, NotDict):
        # Convert NotDict to regular dict for serialization
        tensor_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in tensor.inner_dict.items()}
        serialized = pickle.dumps(tensor_dict)
    else:
        serialized = pickle.dumps(tensor.cpu())
    compressed = zlib.compress(serialized, level)
    final_data = pickle.dumps((compressed, metadata))
    return final_data, tensor_size, sys.getsizeof(compressed)

def pad_data(data, typesize=8):
    """Pad data to be multiple of typesize"""
    padding_size = (typesize - (len(data) % typesize)) % typesize
    return data + b'\0' * padding_size

def compress_lossless_blosc2(data, clevel=9):
    """Lossless compression using blosc2"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    if isinstance(tensor, NotDict):
        tensor_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in tensor.inner_dict.items()}
        serialized = pickle.dumps(tensor_dict)
    else:
        serialized = pickle.dumps(tensor.cpu())
    
    # Pad the serialized data
    padded_data = pad_data(serialized, typesize=8)
    
    try:
        compressed = blosc2.compress(padded_data, 
                                   clevel=clevel,
                                   typesize=8)
        if compressed is None:
            raise RuntimeError("Compression failed")
        final_data = pickle.dumps((compressed, metadata))
        return final_data, tensor_size, sys.getsizeof(compressed)
    except Exception as e:
        logger.error(f"Blosc2 compression error: {e}")
        logger.error(f"Data size before padding: {len(serialized)}")
        logger.error(f"Data size after padding: {len(padded_data)}")
        raise

def compress_lossy_blosc2_zfp_rate(data, rate=4):
    """Lossy compression using blosc2-ZFP rate mode"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    
    debug_tensor_info(tensor, "Input to ZFP rate compression")
    
    try:
        if isinstance(tensor, NotDict):
            compressed_dict = {}
            for k, v in tensor.inner_dict.items():
                if isinstance(v, torch.Tensor):
                    # Convert to numpy and ensure proper alignment
                    v_np = v.detach().cpu().numpy().astype(np.float32)
                    
                    # Calculate padding needed for 8-byte alignment
                    bytes_per_element = v_np.itemsize
                    elements_per_alignment = 8 // bytes_per_element
                    total_elements = v_np.size
                    padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
                    
                    if padding_elements > 0:
                        # Create padding shape
                        pad_shape = list(v_np.shape)
                        pad_shape[-1] += padding_elements
                        # Create padded array
                        padded = np.zeros(pad_shape, dtype=v_np.dtype)
                        # Copy original data
                        slicing = tuple(slice(0, s) for s in v_np.shape)
                        padded[slicing] = v_np
                        v_np = padded
                    
                    compressed_dict[k] = {
                        'data': v_np,
                        'original_shape': v.shape,
                        'padded_shape': v_np.shape,
                        'padding_elements': padding_elements
                    }
                else:
                    compressed_dict[k] = v
            
            data_to_compress = {
                'type': 'NotDict',
                'data': compressed_dict
            }
        else:
            tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
            
            # Same padding logic for single tensor
            bytes_per_element = tensor_np.itemsize
            elements_per_alignment = 8 // bytes_per_element
            total_elements = tensor_np.size
            padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
            
            if padding_elements > 0:
                pad_shape = list(tensor_np.shape)
                pad_shape[-1] += padding_elements
                padded = np.zeros(pad_shape, dtype=tensor_np.dtype)
                slicing = tuple(slice(0, s) for s in tensor_np.shape)
                padded[slicing] = tensor_np
                tensor_np = padded
            
            data_to_compress = {
                'type': 'Tensor',
                'data': tensor_np,
                'original_shape': tensor.shape,
                'padded_shape': tensor_np.shape,
                'padding_elements': padding_elements
            }
        
        # Verify alignment before compression
        serialized = pickle.dumps(data_to_compress, protocol=4)
        if len(serialized) % 8 != 0:
            # Add padding to serialized data if needed
            padding = b'\0' * (8 - (len(serialized) % 8))
            serialized += padding
        
        # Compress with blosc2
        compressed = blosc2.compress(serialized,
                                   codec=blosc2.Codec.ZFP_RATE,
                                   clevel=rate,
                                   typesize=8)
        
        if compressed is None:
            raise RuntimeError("Compression failed")
        
        # Package with metadata
        final_package = {
            'compressed_data': compressed,
            'metadata': metadata,
            'compression_info': {
                'mode': 'zfp_rate',
                'rate': rate
            }
        }
        
        final_data = pickle.dumps(final_package, protocol=4)
        compressed_size = len(compressed)
        
        print(f"Compression stats:")
        print(f"Original size: {tensor_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Ratio: {compressed_size/tensor_size:.3f}")
        
        return final_data, tensor_size, compressed_size
        
    except Exception as e:
        logger.error(f"ZFP rate compression error: {str(e)}")
        logger.error("Tensor info during error:")
        debug_tensor_info(tensor, "Error state")
        raise

def compress_lossy_blosc2_zfp_accuracy(data, accuracy=0.1):
    """Lossy compression using blosc2-ZFP accuracy mode"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    
    debug_tensor_info(tensor, "Input to ZFP accuracy compression")
    
    try:
        if isinstance(tensor, NotDict):
            compressed_dict = {}
            for k, v in tensor.inner_dict.items():
                if isinstance(v, torch.Tensor):
                    # Convert to numpy and ensure proper alignment
                    v_np = v.detach().cpu().numpy().astype(np.float32)
                    v_np = np.ascontiguousarray(v_np)
                    
                    # Calculate padding needed for 8-byte alignment
                    bytes_per_element = v_np.itemsize
                    elements_per_alignment = 8 // bytes_per_element
                    total_elements = v_np.size
                    padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
                    
                    if padding_elements > 0:
                        pad_shape = list(v_np.shape)
                        pad_shape[-1] += padding_elements
                        padded = np.zeros(pad_shape, dtype=v_np.dtype)
                        slicing = tuple(slice(0, s) for s in v_np.shape)
                        padded[slicing] = v_np
                        v_np = padded
                    
                    compressed_dict[k] = {
                        'data': v_np,
                        'original_shape': v.shape,
                        'padded_shape': v_np.shape,
                        'padding_elements': padding_elements
                    }
                else:
                    compressed_dict[k] = v
            
            data_to_compress = {
                'type': 'NotDict',
                'data': compressed_dict
            }
        else:
            tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
            tensor_np = np.ascontiguousarray(tensor_np)
            
            bytes_per_element = tensor_np.itemsize
            elements_per_alignment = 8 // bytes_per_element
            total_elements = tensor_np.size
            padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
            
            if padding_elements > 0:
                pad_shape = list(tensor_np.shape)
                pad_shape[-1] += padding_elements
                padded = np.zeros(pad_shape, dtype=tensor_np.dtype)
                slicing = tuple(slice(0, s) for s in tensor_np.shape)
                padded[slicing] = tensor_np
                tensor_np = padded
            
            data_to_compress = {
                'type': 'Tensor',
                'data': tensor_np,
                'original_shape': tensor.shape,
                'padded_shape': tensor_np.shape,
                'padding_elements': padding_elements
            }
        
        # Verify alignment before compression
        serialized = pickle.dumps(data_to_compress, protocol=4)
        if len(serialized) % 8 != 0:
            # Add padding to serialized data if needed
            padding = b'\0' * (8 - (len(serialized) % 8))
            serialized += padding
        
        # Compress with blosc2
        compressed = blosc2.compress(serialized,
                                   codec=blosc2.Codec.ZFP_ACC,
                                   clevel=int(accuracy * 100),
                                   typesize=8)
        
        if compressed is None:
            raise RuntimeError("Compression failed")
        
        # Package with metadata
        final_package = {
            'compressed_data': compressed,
            'metadata': metadata,
            'compression_info': {
                'mode': 'zfp_accuracy',
                'accuracy': accuracy
            }
        }
        
        final_data = pickle.dumps(final_package, protocol=4)
        compressed_size = len(compressed)
        
        print(f"Compression stats:")
        print(f"Original size: {tensor_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Ratio: {compressed_size/tensor_size:.3f}")
        
        return final_data, tensor_size, compressed_size
        
    except Exception as e:
        logger.error(f"ZFP accuracy compression error: {str(e)}")
        logger.error("Tensor info during error:")
        debug_tensor_info(tensor, "Error state")
        raise

def compress_lossy_blosc2_zfp_precision(data, precision=12):
    """Lossy compression using blosc2-ZFP precision mode"""
    tensor, metadata = data
    tensor_size = get_tensor_size(tensor)
    
    try:
        if isinstance(tensor, NotDict):
            compressed_dict = {}
            for k, v in tensor.inner_dict.items():
                if isinstance(v, torch.Tensor):
                    v_np = v.detach().cpu().numpy().astype(np.float32)
                    v_np = np.ascontiguousarray(v_np)
                    
                    # Calculate padding
                    bytes_per_element = v_np.itemsize
                    elements_per_alignment = 8 // bytes_per_element
                    total_elements = v_np.size
                    padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
                    
                    if padding_elements > 0:
                        pad_shape = list(v_np.shape)
                        pad_shape[-1] += padding_elements
                        padded = np.zeros(pad_shape, dtype=v_np.dtype)
                        slicing = tuple(slice(0, s) for s in v_np.shape)
                        padded[slicing] = v_np
                        v_np = padded
                    
                    compressed_dict[k] = {
                        'data': v_np,
                        'original_shape': v.shape,
                        'padded_shape': v_np.shape,
                        'padding_elements': padding_elements
                    }
                else:
                    compressed_dict[k] = v
            
            data_to_compress = {
                'type': 'NotDict',
                'data': compressed_dict
            }
        else:
            tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
            tensor_np = np.ascontiguousarray(tensor_np)
            
            # Calculate padding
            bytes_per_element = tensor_np.itemsize
            elements_per_alignment = 8 // bytes_per_element
            total_elements = tensor_np.size
            padding_elements = (elements_per_alignment - (total_elements % elements_per_alignment)) % elements_per_alignment
            
            if padding_elements > 0:
                pad_shape = list(tensor_np.shape)
                pad_shape[-1] += padding_elements
                padded = np.zeros(pad_shape, dtype=tensor_np.dtype)
                slicing = tuple(slice(0, s) for s in tensor_np.shape)
                padded[slicing] = tensor_np
                tensor_np = padded
            
            data_to_compress = {
                'type': 'Tensor',
                'data': tensor_np,
                'original_shape': tensor.shape,
                'padded_shape': tensor_np.shape,
                'padding_elements': padding_elements
            }
        
        # Serialize and ensure alignment
        serialized = pickle.dumps(data_to_compress, protocol=4)
        if len(serialized) % 8 != 0:
            padding = b'\0' * (8 - (len(serialized) % 8))
            serialized += padding
        
        # Map precision to valid clevel range (0-9)
        clevel = min(9, max(0, precision % 10))
        
        # Compress with blosc2
        compressed = blosc2.compress(serialized,
                                   codec=blosc2.Codec.ZFP_PREC,
                                   clevel=clevel,
                                   typesize=8)
        
        if compressed is None:
            raise RuntimeError("Compression failed")
        
        # Package with metadata
        final_package = {
            'compressed_data': compressed,
            'metadata': metadata,
            'compression_info': {
                'mode': 'zfp_precision',
                'precision': precision
            }
        }
        
        final_data = pickle.dumps(final_package, protocol=4)
        compressed_size = len(compressed)
        
        print(f"Compression stats:")
        print(f"Original size: {tensor_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Ratio: {compressed_size/tensor_size:.3f}")
        
        return final_data, tensor_size, compressed_size
        
    except Exception as e:
        logger.error(f"ZFP precision compression error: {str(e)}")
        logger.error("Tensor info during error:")
        debug_tensor_info(tensor, "Error state")
        raise

def compress_custom_quantization(data, bits=8):
    """Custom quantization compression from yolo_host_lossy_new.py"""
    tensor, metadata = data
    
    if not isinstance(tensor, NotDict):
        raise ValueError("Custom quantization expects NotDict input")
        
    inner_dict = tensor.inner_dict
    compressed_inner_dict = {}
    
    for key, value in inner_dict.items():
        if isinstance(value, torch.Tensor):
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
            compressed_inner_dict[key] = value
            
    compressed_data = {
        'type': 'NotDict',
        'data': compressed_inner_dict,
        'bits': bits
    }
    
    serialized = pickle.dumps((compressed_data, metadata))
    return serialized, get_tensor_size(tensor), sys.getsizeof(serialized)

def run_compression_experiment(client_socket, split_layer_index, compression_config):
    """Run experiment for a specific compression configuration"""
    mode = compression_config['mode']
    params = compression_config.get('params', {})
    compress_func = compression_config['function']
    
    print(f"\nRunning {mode} compression with params: {params}")
    
    results = {
        'mode': mode,
        'params': params,
        'times': {
            'host': [],
            'compression': [],
            'travel': [],
            'server': []
        },
        'sizes': {
            'original': [],
            'compressed': []
        },
        'predictions': []
    }

    total_images = len(data_loader)
    print(f"Processing {total_images} images...")

    with torch.no_grad():
        for batch_idx, (input_tensor, original_image, image_files) in enumerate(data_loader):
            try:
                print(f"\rProcessing image {batch_idx+1}/{total_images}", end='')
                input_tensor = input_tensor.to(device)
                
                # Host processing
                host_start = time.time()
                out = model(input_tensor, end=split_layer_index)
                data_to_send = (out, original_image[0].size)
                
                # Compression
                compression_start = time.time()
                compressed_output, orig_size, comp_size = compress_func(data_to_send, **params)
                compression_end = time.time()
                
                results['sizes']['original'].append(orig_size)
                results['sizes']['compressed'].append(comp_size)
                results['times']['compression'].append(compression_end - compression_start)
                
                host_end = time.time()
                results['times']['host'].append(host_end - host_start)

                # Network communication
                travel_start = time.time()
                mode_bytes = mode.encode('utf-8')
                client_socket.sendall(len(mode_bytes).to_bytes(4, 'big'))
                client_socket.sendall(mode_bytes)
                client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
                client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
                client_socket.sendall(compressed_output)

                # Receive results
                data = client_socket.recv(4096)
                prediction, server_time = pickle.loads(data)
                travel_end = time.time()
                
                results['times']['travel'].append(travel_end - travel_start)
                results['times']['server'].append(server_time)
                results['predictions'].append((prediction, image_files[0]))
                
            except Exception as e:
                logger.error(f"Error processing image {image_files[0]}: {str(e)}")
                continue

    print(f"\nCompleted {mode} compression experiment")
    return results

def save_experiment_results(results, output_folder):
    """Save experiment results to files"""
    mode = results['mode']
    params = '_'.join(f"{k}_{v}" for k, v in results['params'].items())
    base_name = f"{mode}_{params}"
    
    # Save predictions
    predictions_file = f"{output_folder}/{base_name}_predictions.pkl"
    with open(predictions_file, 'wb') as f:
        pickle.dump(results['predictions'], f)
    print(f"Saved predictions to {predictions_file}")
    
    # Save metrics
    metrics = {
        'times': results['times'],
        'sizes': results['sizes']
    }
    metrics_file = f"{output_folder}/{base_name}_metrics.pkl"
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved metrics to {metrics_file}")
    
    # Save immediate summary
    summary_file = f"{output_folder}/{base_name}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Results for {base_name}:\n")
        f.write(f"Original size (avg): {np.mean(results['sizes']['original']):.2f} bytes\n")
        f.write(f"Compressed size (avg): {np.mean(results['sizes']['compressed']):.2f} bytes\n")
        f.write(f"Compression ratio: {np.mean(results['sizes']['compressed']) / np.mean(results['sizes']['original']):.2f}\n")
        f.write(f"\nTiming Information:\n")
        f.write(f"Host processing: {sum(results['times']['host']):.2f}s\n")
        f.write(f"Compression: {sum(results['times']['compression']):.2f}s\n")
        f.write(f"Network travel: {sum(results['times']['travel']):.2f}s\n")
        f.write(f"Server processing: {sum(results['times']['server']):.2f}s\n")
        f.write(f"Total time: {sum(results['times']['host']) + sum(results['times']['compression']) + sum(results['times']['travel']) + sum(results['times']['server']):.2f}s\n")
    
    print(f"Saved summary to {summary_file}")

def save_intermediate_results(results, output_folder, experiment_name):
    """Save results immediately after each experiment"""
    # Create experiment-specific folder
    exp_folder = os.path.join(output_folder, experiment_name)
    os.makedirs(exp_folder, exist_ok=True)
    
    # Save detailed metrics
    metrics = {
        'original_sizes': results['sizes']['original'],
        'compressed_sizes': results['sizes']['compressed'],
        'compression_times': results['times']['compression'],
        'host_times': results['times']['host'],
        'travel_times': results['times']['travel'],
        'server_times': results['times']['server']
    }
    
    metrics_file = os.path.join(exp_folder, "metrics.pkl")
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save predictions
    predictions_file = os.path.join(exp_folder, "predictions.pkl")
    with open(predictions_file, 'wb') as f:
        pickle.dump(results['predictions'], f)
    
    # Save summary
    summary_file = os.path.join(exp_folder, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Size statistics
        orig_size_avg = np.mean(results['sizes']['original'])
        comp_size_avg = np.mean(results['sizes']['compressed'])
        compression_ratio = comp_size_avg / orig_size_avg
        
        f.write("Size Statistics:\n")
        f.write(f"Original size (avg): {orig_size_avg:.2f} bytes\n")
        f.write(f"Compressed size (avg): {comp_size_avg:.2f} bytes\n")
        f.write(f"Compression ratio: {compression_ratio:.4f}\n\n")
        
        # Timing statistics
        f.write("Timing Statistics:\n")
        f.write(f"Host processing: {sum(results['times']['host']):.4f}s\n")
        f.write(f"Compression: {sum(results['times']['compression']):.4f}s\n")
        f.write(f"Network travel: {sum(results['times']['travel']):.4f}s\n")
        f.write(f"Server processing: {sum(results['times']['server']):.4f}s\n")
        total_time = (sum(results['times']['host']) + 
                     sum(results['times']['compression']) + 
                     sum(results['times']['travel']) + 
                     sum(results['times']['server']))
        f.write(f"Total time: {total_time:.4f}s\n")
        
        # Add compression ratio distribution
        ratios = np.array(results['sizes']['compressed']) / np.array(results['sizes']['original'])
        f.write("\nCompression Ratio Distribution:\n")
        f.write(f"Min ratio: {np.min(ratios):.4f}\n")
        f.write(f"Max ratio: {np.max(ratios):.4f}\n")
        f.write(f"Mean ratio: {np.mean(ratios):.4f}\n")
        f.write(f"Std ratio: {np.std(ratios):.4f}\n")
        
        # Add timing distribution
        f.write("\nTiming Distribution (seconds):\n")
        f.write("Compression time - Mean: {:.4f}, Std: {:.4f}\n".format(
            np.mean(results['times']['compression']),
            np.std(results['times']['compression'])
        ))
        f.write("Network time - Mean: {:.4f}, Std: {:.4f}\n".format(
            np.mean(results['times']['travel']),
            np.std(results['times']['travel'])
        ))
    
    print(f"\nResults saved to {exp_folder}")
    return metrics

def save_comparative_summary(all_metrics, output_folder):
    """Save a summary comparing all compression methods"""
    summary_file = os.path.join(output_folder, "comparative_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Comparative Summary of All Compression Methods\n")
        f.write("==========================================\n\n")
        
        # Table header
        f.write(f"{'Method':<30} {'Orig Size (B)':<15} {'Comp Size (B)':<15} {'Ratio':<10} {'Comp Time (s)':<15} {'Total Time (s)':<15}\n")
        f.write("-" * 100 + "\n")
        
        for method, metrics in all_metrics.items():
            orig_size = np.mean(metrics['original_sizes'])
            comp_size = np.mean(metrics['compressed_sizes'])
            ratio = comp_size / orig_size
            comp_time = np.mean(metrics['compression_times'])
            total_time = (np.sum(metrics['host_times']) + 
                         np.sum(metrics['compression_times']) + 
                         np.sum(metrics['travel_times']) + 
                         np.sum(metrics['server_times']))
            
            f.write(f"{method:<30} {orig_size:>14.0f} {comp_size:>14.0f} {ratio:>9.3f} {comp_time:>14.3f} {total_time:>14.3f}\n")
        
        f.write("\nNotes:\n")
        f.write("- Sizes are in bytes\n")
        f.write("- Ratio is compressed size / original size (smaller is better)\n")
        f.write("- Times are in seconds\n")
        f.write("- Total time includes host processing, compression, network travel, and server processing\n")
    
    print(f"\nComparative summary saved to {summary_file}")

    # Also save as CSV for easier analysis
    csv_file = os.path.join(output_folder, "compression_results.csv")
    with open(csv_file, 'w') as f:
        f.write("Method,OriginalSize,CompressedSize,CompressionRatio,CompressionTime,TotalTime\n")
        for method, metrics in all_metrics.items():
            orig_size = np.mean(metrics['original_sizes'])
            comp_size = np.mean(metrics['compressed_sizes'])
            ratio = comp_size / orig_size
            comp_time = np.mean(metrics['compression_times'])
            total_time = (np.sum(metrics['host_times']) + 
                         np.sum(metrics['compression_times']) + 
                         np.sum(metrics['travel_times']) + 
                         np.sum(metrics['server_times']))
            f.write(f"{method},{orig_size},{comp_size},{ratio},{comp_time},{total_time}\n")
    
    print(f"Results also saved as CSV to {csv_file}")

def compress_sz3(data, tolerance=1e-4):
    """Compress using SZ3 lossy compressor"""
    try:
        import sz3
        tensor, metadata = data
        tensor_size = get_tensor_size(tensor)
        
        if isinstance(tensor, NotDict):
            tensor_dict = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                          for k, v in tensor.inner_dict.items()}
            compressed = sz3.compress(tensor_dict, tolerance=tolerance)
        else:
            compressed = sz3.compress(tensor.cpu().numpy(), tolerance=tolerance)
            
        final_data = pickle.dumps((compressed, metadata))
        return final_data, tensor_size, sys.getsizeof(compressed)
    except ImportError:
        logger.error("SZ3 not installed. Install with: pip install sz3")
        raise

def compress_tensorfloat(data, bits=16):
    """Compress using TensorFloat (specifically for ML tensors)"""
    try:
        import tensor_compression as tc
        tensor, metadata = data
        tensor_size = get_tensor_size(tensor)
        
        if isinstance(tensor, NotDict):
            compressed_dict = {}
            for k, v in tensor.inner_dict.items():
                if isinstance(v, torch.Tensor):
                    compressed_dict[k] = tc.compress_tensor(v.cpu(), bits=bits)
                else:
                    compressed_dict[k] = v
            compressed = compressed_dict
        else:
            compressed = tc.compress_tensor(tensor.cpu(), bits=bits)
            
        final_data = pickle.dumps((compressed, metadata))
        return final_data, tensor_size, sys.getsizeof(final_data)
    except ImportError:
        logger.error("TensorFloat not installed. Install with: pip install tensor-compression")
        raise

def compress_mgard(data, tolerance=0.01):
    """Compress using MGARD for scientific data"""
    try:
        import mgard
        tensor, metadata = data
        tensor_size = get_tensor_size(tensor)
        
        if isinstance(tensor, NotDict):
            tensor_dict = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                          for k, v in tensor.inner_dict.items()}
            compressed = mgard.compress(tensor_dict, tolerance=tolerance)
        else:
            compressed = mgard.compress(tensor.cpu().numpy(), tolerance=tolerance)
            
        final_data = pickle.dumps((compressed, metadata))
        return final_data, tensor_size, sys.getsizeof(compressed)
    except ImportError:
        logger.error("MGARD not installed. Install with: pip install mgard")
        raise

def verify_compression(original_data, compressed_data, decompressed_data):
    """Verify compression is actually modifying and restoring data"""
    if isinstance(original_data, NotDict):
        for k, v in original_data.inner_dict.items():
            if isinstance(v, torch.Tensor):
                orig_tensor = v.cpu().numpy()
                decomp_tensor = decompressed_data.inner_dict[k].cpu().numpy()
                diff = np.abs(orig_tensor - decomp_tensor).max()
                print(f"Key {k}: Max difference = {diff}")
                if diff == 0:
                    print(f"Warning: No difference in {k} - compression might not be working")
    else:
        orig_tensor = original_data.cpu().numpy()
        decomp_tensor = decompressed_data.cpu().numpy()
        diff = np.abs(orig_tensor - decomp_tensor).max()
        print(f"Max difference = {diff}")
        if diff == 0:
            print("Warning: No difference - compression might not be working")

def monitor_compression(original_size, compressed_size, method_name):
    """Monitor compression effectiveness"""
    ratio = compressed_size / original_size
    print(f"\nCompression Method: {method_name}")
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {ratio:.3f}")
    if ratio >= 1:
        print("Warning: Data size increased after compression!")
    return ratio

def debug_tensor_info(tensor, name=""):
    """Print debug information about a tensor or NotDict"""
    print(f"\nDebug info for {name}:")
    if isinstance(tensor, NotDict):
        print("Type: NotDict")
        for k, v in tensor.inner_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"Key {k}:")
                print(f"  Shape: {v.shape}")
                print(f"  Dtype: {v.dtype}")
                print(f"  Device: {v.device}")
                print(f"  Size in bytes: {v.element_size() * v.nelement()}")
                print(f"  Min/Max: {v.min().item():.6f}/{v.max().item():.6f}")
    else:
        print("Type: Tensor")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Size in bytes: {tensor.element_size() * tensor.nelement()}")
        print(f"Min/Max: {tensor.min().item():.6f}/{tensor.max().item():.6f}")

def send_data_with_retry(client_socket, data, max_retries=3):
    """Send data with retry logic"""
    for attempt in range(max_retries):
        try:
            data_length = len(data)
            client_socket.sendall(data_length.to_bytes(4, 'big'))
            client_socket.sendall(data)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Send attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)
    return False

def receive_data_with_retry(client_socket, max_retries=3):
    """Receive data with retry logic"""
    for attempt in range(max_retries):
        try:
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                raise ConnectionError("Connection closed by server")
            data_length = int.from_bytes(length_bytes, 'big')
            data = receive_full_message(client_socket, data_length)
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Receive attempt {attempt + 1} failed: {str(e)}")
            time.sleep(1)
    raise ConnectionError("Failed to receive data after retries")

def main():
    # Initialize model and dataset
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    
    print(f"\nInitializing dataset from {dataset_path}")
    dataset = OnionDataset(root=dataset_path, transform=transform)
    global data_loader
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    print(f"Dataset loaded with {len(dataset)} images")
    
    print("\nInitializing model...")
    yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
    global model
    model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
    model.eval()
    model = model.to(device)
    print(f"Model initialized on {device}")
    # Split layer configuration
    split_layer_index = 3  # Adjust based on your needs

    # Compression configurations
    compression_configs = [
        # {
        #     'mode': 'no_compression',
        #     'function': compress_none
        # },
        # {
        #     'mode': 'lossless_zlib',
        #     'function': compress_lossless_zlib,
        #     'params': {'level': 9}
        # },
        {
            'mode': 'lossless_blosc2',
            'function': compress_lossless_blosc2,
            'params': {'clevel': 4}
        },
        # {
        #     'mode': 'lossy_zfp_rate',
        #     'function': compress_lossy_blosc2_zfp_rate,
        #     'params': {'rate': 4}
        # },
        # {
        #     'mode': 'lossy_zfp_rate',
        #     'function': compress_lossy_blosc2_zfp_rate,
        #     'params': {'rate': 8}
        # },
        # {
        #     'mode': 'lossy_zfp_accuracy',
        #     'function': compress_lossy_blosc2_zfp_accuracy,
        #     'params': {'accuracy': 0.01}
        # },
        # {
        #     'mode': 'lossy_zfp_accuracy',
        #     'function': compress_lossy_blosc2_zfp_accuracy,
        #     'params': {'accuracy': 0.001}
        # },
        # {
        #     'mode': 'lossy_zfp_precision',
        #     'function': compress_lossy_blosc2_zfp_precision,
        #     'params': {'precision': 4}
        # },
        # {
        #     'mode': 'lossy_zfp_precision',
        #     'function': compress_lossy_blosc2_zfp_precision,
        #     'params': {'precision': 8}
        # },
        # {
        #     'mode': 'custom_quantization',
        #     'function': compress_custom_quantization,
        #     'params': {'bits': 2}
        # },
        # {
        #     'mode': 'custom_quantization',
        #     'function': compress_custom_quantization,
        #     'params': {'bits': 4}
        # },
        # {
        #     'mode': 'custom_quantization',
        #     'function': compress_custom_quantization,
        #     'params': {'bits': 6}
        # },
        # {
        #     'mode': 'custom_quantization',
        #     'function': compress_custom_quantization,
        #     'params': {'bits': 8}
        # },
        # {
        #     'mode': 'custom_quantization',
        #     'function': compress_custom_quantization,
        #     'params': {'bits': 10}
        # }
    ]
    # Connect to server
    server_address = ('10.0.0.61', 12345)
    print(f"\nConnecting to server at {server_address}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    print("Connected to server")
    
    try:
        # Create output folder
        output_folder = create_experiment_folder()
        print(f"\nSaving results to: {output_folder}")
        
        print("\nStarting compression experiments...")
        all_metrics = {}
        for idx, config in enumerate(compression_configs):
            mode = config['mode']
            params = config.get('params', {})
            experiment_name = f"{mode}"
            if params:
                experiment_name += "_" + "_".join(f"{k}_{v}" for k, v in params.items())
            
            print(f"\nExperiment {idx+1}/{len(compression_configs)}: {experiment_name}")
            
            try:
                results = run_compression_experiment(client_socket, split_layer_index, config)
                metrics = save_intermediate_results(results, output_folder, experiment_name)
                all_metrics[experiment_name] = metrics
                print(f"Completed experiment: {experiment_name}")
            except Exception as e:
                logger.error(f"Error in experiment {experiment_name}: {e}")
                continue
        
        # Save comparative summary
        save_comparative_summary(all_metrics, output_folder)
        
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise
    finally:
        client_socket.close()
        print("Client socket closed")

if __name__ == "__main__":
    main()