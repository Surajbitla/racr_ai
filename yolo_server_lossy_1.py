import torch
import os
import pickle
import time
import socket
from pathlib import Path
import logging
import numpy as np
import blosc2
from PIL import Image
import io
import cv2
import torchvision.transforms as transforms
import zlib

from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tracr_logger")

# Model configuration
weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Server configuration
host = '0.0.0.0'
port = 12345

# Post-processing configuration
conf_threshold = 0.25
iou_threshold = 0.45

def decompress_blosc2_zfp(compressed_data):
    """Decompress data compressed with blosc2 ZFP codec"""
    try:
        # First, unpack the final package
        final_package = pickle.loads(compressed_data)
        
        if not isinstance(final_package, dict):
            raise ValueError(f"Expected dictionary in final package, got {type(final_package)}")
            
        compressed = final_package['compressed_data']
        metadata = final_package['metadata']
        compression_info = final_package.get('compression_info', {})
        
        # Decompress the blosc2 data
        decompressed_bytes = blosc2.decompress(compressed)
        
        # Remove any null padding before unpickling
        decompressed_bytes = decompressed_bytes.rstrip(b'\x00')
        
        try:
            data_dict = pickle.loads(decompressed_bytes)
        except Exception as e:
            logger.error(f"Failed to unpickle decompressed data: {str(e)}")
            logger.error(f"Decompressed data size: {len(decompressed_bytes)}")
            raise
        
        if data_dict['type'] == 'NotDict':
            # Handle NotDict case
            reconstructed_dict = {}
            for k, v in data_dict['data'].items():
                if isinstance(v, dict) and 'data' in v:
                    # Get the original data and shape
                    data = v['data']
                    original_shape = v['original_shape']
                    
                    # Remove padding if present
                    if len(original_shape) == 4:
                        tensor = torch.from_numpy(
                            data[:original_shape[0],
                                :original_shape[1],
                                :original_shape[2],
                                :original_shape[3]]).float()
                    else:
                        # Handle other tensor shapes
                        tensor = torch.from_numpy(data).float()
                        tensor = tensor.reshape(original_shape)
                    
                    reconstructed_dict[k] = tensor
                else:
                    reconstructed_dict[k] = v
            out = NotDict(reconstructed_dict)
        else:
            # Handle single tensor case
            data = data_dict['data']
            original_shape = data_dict['original_shape']
            
            # Remove padding if present
            if len(original_shape) == 4:
                out = torch.from_numpy(
                    data[:original_shape[0],
                        :original_shape[1],
                        :original_shape[2],
                        :original_shape[3]]).float()
            else:
                # Handle other tensor shapes
                out = torch.from_numpy(data).float()
                out = out.reshape(original_shape)
        
        return (out, metadata)
        
    except Exception as e:
        logger.error(f"Decompression error: {str(e)}")
        logger.error(f"Received data type: {type(compressed_data)}")
        if isinstance(final_package, dict):
            logger.error(f"Final package keys: {final_package.keys()}")
        else:
            logger.error(f"Final package type: {type(final_package)}")
        raise

def decompress_jpeg(compressed_data):
    """Decompress JPEG compressed data"""
    data = pickle.loads(compressed_data)
    compressed_img, metadata = data
    
    img = Image.open(io.BytesIO(compressed_img))
    tensor = torch.from_numpy(np.array(img)).float()
    tensor = tensor.unsqueeze(0)
    
    return (tensor, metadata)

def decompress_svd(compressed_data):
    """Decompress SVD compressed data"""
    compressed_arrays, metadata = pickle.loads(compressed_data)
    reconstructed = []
    
    for U_k, S_k, Vt_k in compressed_arrays:
        reconstructed_matrix = U_k @ np.diag(S_k) @ Vt_k
        reconstructed.append(reconstructed_matrix)
    
    tensor = torch.from_numpy(np.array(reconstructed))
    return (tensor, metadata)

def decompress_quantization(compressed_data):
    """Decompress quantization compressed data"""
    quantized, data_min, data_max, metadata = pickle.loads(compressed_data)
    reconstructed = quantized * (data_max - data_min) + data_min
    tensor = torch.from_numpy(reconstructed)
    return (tensor, metadata)

def decompress_none(compressed_data):
    """Decompress uncompressed data"""
    return pickle.loads(compressed_data)

def decompress_lossless_zlib(compressed_data):
    """Decompress zlib compressed data"""
    data = pickle.loads(compressed_data)
    compressed_tensor, metadata = data
    tensor_dict = pickle.loads(zlib.decompress(compressed_tensor))
    if isinstance(tensor_dict, dict):
        # Reconstruct NotDict
        for k, v in tensor_dict.items():
            if isinstance(v, np.ndarray):
                tensor_dict[k] = torch.from_numpy(v)
        out = NotDict(tensor_dict)
    else:
        out = torch.from_numpy(tensor_dict) if isinstance(tensor_dict, np.ndarray) else tensor_dict
    return (out, metadata)

def decompress_lossless_blosc2(compressed_data):
    """Decompress blosc2 lossless compressed data"""
    data = pickle.loads(compressed_data)
    compressed_tensor, metadata = data
    tensor_dict = pickle.loads(blosc2.decompress(compressed_tensor))
    if isinstance(tensor_dict, dict):
        # Reconstruct NotDict
        for k, v in tensor_dict.items():
            if isinstance(v, np.ndarray):
                tensor_dict[k] = torch.from_numpy(v)
        out = NotDict(tensor_dict)
    else:
        out = torch.from_numpy(tensor_dict) if isinstance(tensor_dict, np.ndarray) else tensor_dict
    return (out, metadata)

def decompress_custom_quantization(compressed_data):
    """Decompress custom quantization data"""
    try:
        compressed_dict, metadata = pickle.loads(compressed_data)
        
        if not isinstance(compressed_dict, dict) or 'type' not in compressed_dict:
            raise ValueError("Invalid compressed data format")
            
        if compressed_dict['type'] != 'NotDict':
            raise ValueError("Expected NotDict type in compressed data")
            
        reconstructed_dict = {}
        for key, value in compressed_dict['data'].items():
            if isinstance(value, dict):
                quantized = value['quantized']
                mins = value['mins']
                maxs = value['maxs']
                
                reconstructed = torch.from_numpy(quantized).float()
                reconstructed = reconstructed * (maxs - mins) + mins
                reconstructed_dict[key] = reconstructed
            else:
                reconstructed_dict[key] = value
                
        return (NotDict(reconstructed_dict), metadata)
    except Exception as e:
        logger.error(f"Custom quantization decompression error: {str(e)}")
        raise

def decompress_sz3(compressed_data):
    """Decompress SZ3 compressed data"""
    try:
        import sz3
        data = pickle.loads(compressed_data)
        compressed_tensor, metadata = data
        if isinstance(compressed_tensor, dict):
            tensor_dict = {k: torch.from_numpy(sz3.decompress(v)) if isinstance(v, bytes) else v 
                          for k, v in compressed_tensor.items()}
            out = NotDict(tensor_dict)
        else:
            out = torch.from_numpy(sz3.decompress(compressed_tensor))
        return (out, metadata)
    except ImportError:
        logger.error("SZ3 not installed. Install with: pip install sz3")
        raise

def decompress_tensorfloat(compressed_data):
    """Decompress TensorFloat compressed data"""
    try:
        import tensor_compression as tc
        data = pickle.loads(compressed_data)
        compressed_tensor, metadata = data
        if isinstance(compressed_tensor, dict):
            tensor_dict = {k: tc.decompress_tensor(v) if isinstance(v, bytes) else v 
                          for k, v in compressed_tensor.items()}
            out = NotDict(tensor_dict)
        else:
            out = tc.decompress_tensor(compressed_tensor)
        return (out, metadata)
    except ImportError:
        logger.error("TensorFloat not installed. Install with: pip install tensor-compression")
        raise

def decompress_mgard(compressed_data):
    """Decompress MGARD compressed data"""
    try:
        import mgard
        data = pickle.loads(compressed_data)
        compressed_tensor, metadata = data
        if isinstance(compressed_tensor, dict):
            tensor_dict = {k: torch.from_numpy(mgard.decompress(v)) if isinstance(v, bytes) else v 
                          for k, v in compressed_tensor.items()}
            out = NotDict(tensor_dict)
        else:
            out = torch.from_numpy(mgard.decompress(compressed_tensor))
        return (out, metadata)
    except ImportError:
        logger.error("MGARD not installed. Install with: pip install mgard")
        raise

def receive_full_message(conn, expected_length):
    """Receive complete message from socket"""
    data_chunks = []
    bytes_recd = 0
    while bytes_recd < expected_length:
        chunk = conn.recv(min(expected_length - bytes_recd, 4096))
        if chunk == b'':
            raise RuntimeError("Socket connection broken")
        data_chunks.append(chunk)
        bytes_recd += len(chunk)
    return b''.join(data_chunks)

def postprocess(outputs, original_img_size, conf_threshold=0.25, iou_threshold=0.45):
    """Post-process model outputs to get detections"""
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
            x, y, w, h = outputs[i][0:4]
            left = int((x - w/2) * x_factor)
            top = int((y - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    detections = []
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detections.append((boxes[i], scores[i], class_ids[i]))

    return detections

def main():
    # Initialize model
    yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
    model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path, participant_key='server')
    model.eval()
    model = model.to(device)
    print(f"Model initialized on {device}")

    # Decompression configurations
    decompression_functions = {
        'no_compression': decompress_none,
        'lossless_zlib': decompress_lossless_zlib,
        'lossless_blosc2': decompress_lossless_blosc2,
        'lossy_zfp_rate': decompress_blosc2_zfp,
        'lossy_zfp_accuracy': decompress_blosc2_zfp,
        'lossy_zfp_precision': decompress_blosc2_zfp,
        'custom_quantization': decompress_custom_quantization,
        # Optional decompression methods
        # 'sz3': decompress_sz3,
        # 'tensorfloat': decompress_tensorfloat,
        # 'mgard': decompress_mgard
    }
    # Setup server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on port {port}...")

    while True:
        conn = None
        try:
            print("Waiting for connection...")
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            
            # Set socket timeouts
            conn.settimeout(60)  # 60 second timeout
            
            while True:
                try:
                    # Receive compression mode
                    mode_length_bytes = conn.recv(4)
                    if not mode_length_bytes:
                        print("Client disconnected")
                        break
                    
                    mode_length = int.from_bytes(mode_length_bytes, 'big')
                    compression_mode = conn.recv(mode_length).decode('utf-8')
                    
                    if compression_mode not in decompression_functions:
                        raise ValueError(f"Unknown compression mode: {compression_mode}")
                    
                    logger.info(f"Processing request with mode: {compression_mode}")
                    
                    # Receive data length and data
                    split_layer_index = int.from_bytes(conn.recv(4), 'big')
                    data_length = int.from_bytes(conn.recv(4), 'big')
                    
                    logger.debug(f"Receiving {data_length} bytes of data")
                    compressed_data = receive_full_message(conn, data_length)
                    
                    try:
                        # Process the data
                        decompression_start = time.time()
                        received_data = decompression_functions[compression_mode](compressed_data)
                        decompression_time = time.time() - decompression_start
                        
                        out, original_img_size = received_data
                        
                        # Process data
                        server_start_time = time.time()
                        with torch.no_grad():
                            if isinstance(out, NotDict):
                                inner_dict = out.inner_dict
                                for key, value in inner_dict.items():
                                    if isinstance(value, torch.Tensor):
                                        inner_dict[key] = value.to(device)
                            
                            res, _ = model(out, start=split_layer_index)
                            detections = postprocess(res, original_img_size)
                        
                        server_processing_time = time.time() - server_start_time
                        total_server_time = server_processing_time + decompression_time
                        
                        # Send results back
                        response_data = pickle.dumps((detections, total_server_time))
                        response_length = len(response_data)
                        conn.sendall(response_length.to_bytes(4, 'big'))
                        conn.sendall(response_data)
                        
                        logger.debug(f"Processed request: {len(detections)} detections, "
                                   f"server time: {total_server_time:.3f}s")
                    
                    except Exception as e:
                        logger.error(f"Error processing request: {str(e)}")
                        # Send error response
                        error_response = pickle.dumps(([], 0.0))
                        response_length = len(error_response)
                        conn.sendall(response_length.to_bytes(4, 'big'))
                        conn.sendall(error_response)
                
                except socket.timeout:
                    logger.error("Socket timeout")
                    break
                except ConnectionError as e:
                    logger.error(f"Connection error: {str(e)}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    break
            
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        
        finally:
            if conn:
                conn.close()
                print("Connection closed")
    
    server_socket.close()
    print("Server shutdown complete")

if __name__ == "__main__":
    main() 