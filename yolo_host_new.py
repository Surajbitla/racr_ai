import os
import torch
import socket
import logging
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import config
import utils
from dataset import OnionDataset
from src.tracr.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yolo_host.log')
    ]
)
logger = logging.getLogger("tracr_logger")

class YOLOHost:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.setup_dataset()
        self.setup_network()

    def setup_model(self):
        yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
        self.model = WrappedModel(config_path=yaml_file_path, 
                                weights_path=config.WEIGHT_PATH)
        self.model.eval()

    def setup_dataset(self):
        dataset = OnionDataset(root=config.DATASET_PATH)
        self.data_loader = DataLoader(dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    collate_fn=dataset.collate_fn)

    def setup_network(self):
        """Setup network connection with retry mechanism"""
        max_retries = 5
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((config.SERVER_ADDRESS, config.SERVER_PORT))
                logger.info(f"Connected to server at {config.SERVER_ADDRESS}:{config.SERVER_PORT}")
                return
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError("Failed to connect to server after maximum retries")

    def process_single_split(self, split_layer_index, mode, clevel=4):
        host_times, compression_times = [], []
        travel_times, server_times = [], []
        compression_sizes = []
        experiment_results = []

        if config.GENERATE_BATTERY_REPORT:
            utils.generate_battery_report("before", config.BATTERY_REPORT_DIR)

        with torch.no_grad():
            for input_tensor, original_image, image_files in tqdm(self.data_loader):
                # Host processing
                host_start = time.time()
                input_tensor = input_tensor.to(self.model.device)
                out = self.model(input_tensor, end=split_layer_index)
                data_to_send = (out, original_image[0].size)

                # Compression
                compress_start = time.time()
                if mode == "lossless":
                    compressed_output, size = utils.compress_data_lossless(data_to_send)
                else:
                    compressed_output, size = utils.compress_data_lossy(data_to_send, clevel)
                
                compression_sizes.append(size)
                compression_times.append(time.time() - compress_start)
                host_times.append(time.time() - host_start)

                # Network communication
                travel_start = time.time()
                self._send_to_server(mode, split_layer_index, compressed_output)
                prediction, server_time = self._receive_from_server()
                
                travel_times.append(time.time() - travel_start)
                server_times.append(server_time)
                
                # Save results
                experiment_results.append((prediction, image_files[0], size))
                if prediction is not None:  # Only save if we have predictions
                    utils.save_detections_to_txt(prediction, 
                                              image_files[0], 
                                              original_image[0].size,
                                              config.OUTPUT_LABELS_DIR)

        if config.GENERATE_BATTERY_REPORT:
            utils.generate_battery_report("after", config.BATTERY_REPORT_DIR)

        return self._calculate_metrics(host_times, travel_times, 
                                    server_times, compression_times, 
                                    compression_sizes, experiment_results)

    def _send_to_server(self, mode, split_layer_index, compressed_output):
        """Simplified send with minimal overhead"""
        try:
            # Send mode
            mode_bytes = mode.encode('utf-8')
            self.client_socket.sendall(len(mode_bytes).to_bytes(4, 'big'))
            self.client_socket.sendall(mode_bytes)
            
            # Send split layer index
            self.client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
            
            # Send data length and data
            self.client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
            self.client_socket.sendall(compressed_output)
            
        except socket.error as e:
            logger.error(f"Send error: {e}")
            self.setup_network()
            raise

    def _receive_from_server(self):
        """Receive data from server with chunked transfer"""
        try:
            # First receive the size of the incoming data
            size_bytes = self.client_socket.recv(4)
            total_size = int.from_bytes(size_bytes, 'big')
            
            # Receive data in chunks
            data = b''
            while len(data) < total_size:
                chunk_size = min(config.BUFFER_SIZE, total_size - len(data))
                chunk = self.client_socket.recv(chunk_size)
                if not chunk:
                    raise ConnectionError("Connection lost while receiving data")
                data += chunk
                
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            self.setup_network()  # Try to reconnect
            raise

    def _calculate_metrics(self, host_times, travel_times, server_times, 
                         compression_times, compression_sizes, experiment_results):
        metrics = {
            'total_host_time': sum(host_times),
            'total_travel_time': sum(travel_times) - sum(server_times),
            'total_server_time': sum(server_times),
            'total_compression_time': sum(compression_times),
            'total_compression_size': sum(compression_sizes),
            'experiment_results': experiment_results
        }
        return metrics

    def run(self):
        max_retries = config.MAX_RETRIES
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if config.RUN_SPLIT_EXPERIMENTS:
                    results = []
                    for split_idx in range(1, config.TOTAL_LAYERS):
                        metrics = self.process_single_split(split_idx, 
                                                         config.COMPRESSION_MODE,
                                                         config.COMPRESSION_LEVEL)
                        results.append((split_idx, *self._extract_times(metrics)))
                    utils.save_split_experiment_results(results)
                else:
                    metrics = self.process_single_split(config.SPLIT_LAYER_INDEX,
                                                     config.COMPRESSION_MODE,
                                                     config.COMPRESSION_LEVEL)
                    self._log_metrics(metrics)
                break  # Success, exit the retry loop
                
            except ConnectionError as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Connection error: {e}. Retrying ({retry_count}/{max_retries})...")
                    time.sleep(config.RETRY_DELAY)
                    self.setup_network()  # Try to reconnect
                else:
                    logger.error("Max retries reached. Giving up.")
                    raise
            except Exception as e:
                logger.error(f"Unrecoverable error: {e}")
                raise

    def cleanup(self):
        self.client_socket.close()

    def _extract_times(self, metrics):
        """Extract timing information from metrics"""
        return (
            metrics['total_host_time'],
            metrics['total_travel_time'],
            metrics['total_server_time'],
            metrics['total_host_time'] + metrics['total_travel_time'] + metrics['total_server_time']
        )

    def _log_metrics(self, metrics):
        """Log performance metrics"""
        logger.info("Performance Metrics:")
        logger.info(f"Total Host Time: {metrics['total_host_time']:.2f}s")
        logger.info(f"Total Travel Time: {metrics['total_travel_time']:.2f}s")
        logger.info(f"Total Server Time: {metrics['total_server_time']:.2f}s")
        logger.info(f"Total Compression Time: {metrics['total_compression_time']:.2f}s")
        logger.info(f"Total Compression Size: {metrics['total_compression_size']} bytes")

if __name__ == "__main__":
    host = YOLOHost()
    try:
        host.run()
    finally:
        host.cleanup() 