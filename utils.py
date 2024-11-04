import os
import subprocess
import pickle
import zlib
import blosc2
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import socket
import json
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.processing_times = []
        self.network_times = []
        self.compression_times = []
        
    def add_metric(self, category, time_taken):
        if category == 'processing':
            self.processing_times.append(time_taken)
        elif category == 'network':
            self.network_times.append(time_taken)
        elif category == 'compression':
            self.compression_times.append(time_taken)
            
    def get_statistics(self):
        return {
            'processing': {
                'avg': np.mean(self.processing_times),
                'max': max(self.processing_times),
                'min': min(self.processing_times)
            },
            'network': {
                'avg': np.mean(self.network_times),
                'max': max(self.network_times),
                'min': min(self.network_times)
            },
            'compression': {
                'avg': np.mean(self.compression_times),
                'max': max(self.compression_times),
                'min': min(self.compression_times)
            }
        }

def generate_battery_report(prefix="before", output_dir=None):
    """Generate and save battery report with timestamp"""
    if not output_dir:
        return
        
    output_file = os.path.join(output_dir, "battery-report.html")
    subprocess.run(f"powercfg /batteryreport /output {output_file}", shell=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_file = os.path.join(output_dir, f"{timestamp}_battery_report_{prefix}.html")
    
    if os.path.exists(output_file):
        os.rename(output_file, dest_file)
        print(f"Battery report saved as {dest_file}")
    else:
        print(f"Battery report not found at {output_file}")

def compress_data_lossless(data):
    """Compress data using zlib"""
    serialized_data = pickle.dumps(data)
    compressed_data = zlib.compress(serialized_data)
    size_bytes = sys.getsizeof(compressed_data)
    return compressed_data, size_bytes

def compress_data_lossy(data, clevel):
    """Compress data using blosc2"""
    serialized_data = pickle.dumps(data)
    compressed_data = blosc2.compress(serialized_data, 
                                    clevel=clevel, 
                                    filter=blosc2.Filter.SHUFFLE, 
                                    codec=blosc2.Codec.ZSTD)
    size_bytes = sys.getsizeof(compressed_data)
    return compressed_data, size_bytes

def save_data_to_file(data, filename):
    """Save data to pickle file"""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def save_detections_to_txt(detections, image_filename, original_img_size, output_dir):
    """Save YOLO format detections to text file"""
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, image_filename.replace('.jpg', '.txt'))
    
    try:
        with open(txt_filename, 'w') as f:
            if isinstance(detections, (list, tuple)):
                for detection in detections:
                    if len(detection) >= 6:  # [x1, y1, x2, y2, conf, cls]
                        x1, y1, x2, y2, conf, cls = detection[:6]
                        # Convert to YOLO format
                        x_center = (x1 + x2) / (2 * original_img_size[0])
                        y_center = (y1 + y2) / (2 * original_img_size[1])
                        width = (x2 - x1) / original_img_size[0]
                        height = (y2 - y1) / original_img_size[1]
                        f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
    except Exception as e:
        logger.warning(f"Could not save detections for {image_filename}: {e}")

def save_split_experiment_results(results):
    """Save split experiment results to Excel"""
    df = pd.DataFrame(results, 
                     columns=["Split Layer Index", "Host Time", 
                             "Travel Time", "Server Time", 
                             "Total Processing Time"])
    df.to_excel("split_layer_times.xlsx", index=False) 

def send_heartbeat(socket):
    """Send heartbeat message"""
    try:
        socket.sendall(b'heartbeat')
        return True
    except:
        return False

def check_connection(socket):
    """Check if connection is still alive"""
    try:
        # Check if socket is still connected
        socket.settimeout(5)
        socket.sendall(b'ping')
        response = socket.recv(1024)
        socket.settimeout(None)
        return response == b'pong'
    except:
        return False

def check_server_status(server_address, port):
    """Check if server is running and get its status"""
    try:
        status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        status_socket.settimeout(5)
        status_socket.connect((server_address, port + 1))
        status_data = status_socket.recv(4096)
        status_socket.close()
        return json.loads(status_data.decode())
    except:
        return None