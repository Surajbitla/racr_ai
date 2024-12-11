import torch
import os
import pickle
import time
import socket
import logging
from ultralytics import YOLO
import blosc2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("yolo_logger")

# Model configuration
weight_path = 'runs/detect/train16/weights/best.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Server configuration
host = '0.0.0.0'
port = 12345

def decompress_blosc2(compressed_data):
    """Decompress blosc2 compressed data"""
    try:
        decompressed = blosc2.decompress(compressed_data)
        # Remove any padding before unpickling
        decompressed = decompressed.rstrip(b'\0')
        return pickle.loads(decompressed)
    except Exception as e:
        logger.error(f"Decompression error: {e}")
        raise

def main():
    # Load YOLO model
    model = YOLO(weight_path)
    model.to(device)  # Ensure model is on correct device
    print(f"Model loaded on {device}")

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
            
            conn.settimeout(60)  # 60 second timeout
            
            while True:
                try:
                    # Check compression flag
                    compression_flag = conn.recv(1)
                    if not compression_flag:
                        print("Client disconnected")
                        break
                    
                    use_compression = compression_flag == b'1'
                    
                    # Receive data length and data
                    data_length = int.from_bytes(conn.recv(4), 'big')
                    received_data = b''
                    while len(received_data) < data_length:
                        chunk = conn.recv(min(4096, data_length - len(received_data)))
                        if not chunk:
                            raise ConnectionError("Connection broken")
                        received_data += chunk
                    
                    # Process the data
                    server_start = time.time()
                    
                    try:
                        # Decompress if needed
                        if use_compression:
                            image_array, original_size = decompress_blosc2(received_data)
                        else:
                            image_array, original_size = pickle.loads(received_data)
                        
                        # Verify image array
                        if image_array is None or image_array.size == 0:
                            raise ValueError("Received empty image array")
                            
                        if not isinstance(image_array, np.ndarray):
                            raise TypeError(f"Expected numpy array, got {type(image_array)}")
                            
                        # Ensure image is in correct format (HWC, uint8)
                        if image_array.dtype != np.uint8:
                            logger.warning("Converting image to uint8")
                            image_array = (image_array * 255).astype(np.uint8)
                        
                        # Run inference
                        results = model(image_array, device=device)
                        
                        # Extract detections in simple format
                        detections = []
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                # Convert to [x, y, w, h] format
                                w = x2 - x1
                                h = y2 - y1
                                detections.append(([x1, y1, w, h], conf, cls))
                        
                    except Exception as e:
                        logger.error(f"Processing error: {str(e)}")
                        detections = []  # Return empty detections on error
                    
                    server_time = time.time() - server_start
                    
                    # Send results back
                    response_data = pickle.dumps((detections, server_time))
                    response_length = len(response_data)
                    conn.sendall(response_length.to_bytes(4, 'big'))
                    conn.sendall(response_data)
                    
                    logger.debug(f"Processed request: {len(detections)} detections, "
                               f"server time: {server_time:.3f}s")
                
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