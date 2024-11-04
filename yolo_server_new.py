import torch
import socket
import pickle
import logging
from pathlib import Path
import time
import os
import zlib
import blosc2
import config
import utils
from src.tracr.experiment_design.models.model_hooked import WrappedModel
import signal
import json
import threading
import select

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("yolo_server")

class YOLOServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.setup_network()
        self.running = True
        self.start_time = time.time()
        self.request_count = 0
        self.performance_monitor = utils.PerformanceMonitor()
        self.setup_signal_handlers()
        self.setup_status_endpoint()

    def setup_model(self):
        """Initialize the YOLO model"""
        try:
            yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
            self.model = WrappedModel(config_path=yaml_file_path, 
                                    weights_path=config.WEIGHT_PATH)
            self.model.eval()
            logger.info(f"Model initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def setup_network(self):
        """Setup network socket and start listening"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Add socket options to allow port reuse
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((config.SERVER_ADDRESS, config.SERVER_PORT))
            self.server_socket.listen(5)  # Allow queue of up to 5 connections
            logger.info(f"Server listening on {config.SERVER_ADDRESS}:{config.SERVER_PORT}")
        except Exception as e:
            logger.error(f"Failed to setup network: {e}")
            raise

    def receive_data(self, client_socket):
        """Simplified receive with minimal overhead"""
        try:
            # Receive mode
            mode_length = int.from_bytes(client_socket.recv(4), 'big')
            mode = client_socket.recv(mode_length).decode('utf-8')
            
            # Receive split layer index
            split_layer_index = int.from_bytes(client_socket.recv(4), 'big')
            
            # Receive data
            data_length = int.from_bytes(client_socket.recv(4), 'big')
            compressed_data = client_socket.recv(data_length)
            
            return mode, split_layer_index, compressed_data
        except Exception as e:
            logger.error(f"Receive error: {e}")
            raise

    def process_data(self, mode, split_layer_index, compressed_data):
        """Process received data like in original code"""
        try:
            start_time = time.time()
            
            # Decompress data
            if mode == "lossless":
                serialized_data = zlib.decompress(compressed_data)
            else:  # lossy
                serialized_data = blosc2.decompress(compressed_data)
            
            data = pickle.loads(serialized_data)
            intermediate_output, original_size = data

            # Process with remaining layers
            with torch.no_grad():
                predictions = self.model(intermediate_output, start=split_layer_index)

            processing_time = time.time() - start_time
            return predictions, processing_time
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def send_response(self, client_socket, predictions, processing_time):
        """Send processed results back to client in chunks"""
        try:
            response = pickle.dumps((predictions, processing_time))
            # Send size first
            client_socket.sendall(len(response).to_bytes(4, 'big'))
            
            # Send data in chunks
            for i in range(0, len(response), config.BUFFER_SIZE):
                chunk = response[i:i + config.BUFFER_SIZE]
                client_socket.sendall(chunk)
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            raise

    def handle_client(self, client_socket, addr):
        """Handle individual client connection with heartbeat"""
        logger.info(f"New connection from {addr}")
        client_socket.settimeout(config.SOCKET_TIMEOUT)
        
        try:
            while True:
                # Check for incoming data
                ready = select.select([client_socket], [], [], 1.0)
                if ready[0]:
                    # Check first byte to determine if it's a heartbeat
                    first_byte = client_socket.recv(1, socket.MSG_PEEK)
                    if first_byte == b'p':  # ping
                        # Handle heartbeat
                        client_socket.recv(4)  # clear the 'ping'
                        client_socket.sendall(b'pong')
                        continue
                    
                    # Regular data processing
                    mode, split_layer_index, compressed_data = self.receive_data(client_socket)
                    self.request_count += 1
                    
                    process_start = time.time()
                    predictions, processing_time = self.process_data(mode, split_layer_index, compressed_data)
                    self.performance_monitor.add_metric('processing', time.time() - process_start)
                    
                    self.send_response(client_socket, predictions, processing_time)
                    
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
            logger.info(f"Connection closed with {addr}")

    def run(self):
        """Main server loop"""
        logger.info("Server started and ready for connections")
        
        while self.running:
            try:
                # Set a timeout on accept() to allow checking self.running periodically
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, addr = self.server_socket.accept()
                    self.handle_client(client_socket, addr)
                except socket.timeout:
                    continue  # No connection received, check if we should keep running
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal")
                self.running = False
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                # Don't exit the loop, continue listening for new connections
                continue

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.server_socket.close()
            logger.info("Server shutdown complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Starting graceful shutdown...")
        self.running = False

    def setup_status_endpoint(self):
        """Setup a simple status endpoint"""
        status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        status_socket.bind((config.SERVER_ADDRESS, config.SERVER_PORT + 1))
        status_socket.listen(1)
        
        def status_handler():
            while self.running:
                try:
                    client, addr = status_socket.accept()
                    status = {
                        'running': self.running,
                        'device': str(self.device),
                        'uptime': time.time() - self.start_time,
                        'processed_requests': self.request_count,
                        'performance': self.performance_monitor.get_statistics()
                    }
                    client.sendall(json.dumps(status).encode())
                    client.close()
                except:
                    pass

        threading.Thread(target=status_handler, daemon=True).start()

def main():
    """Main function to run the server"""
    server = None
    try:
        server = YOLOServer()
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        if server:
            server.cleanup()

if __name__ == "__main__":
    main() 