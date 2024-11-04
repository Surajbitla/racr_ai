"""
Configuration settings for YOLO processing system
"""

# Network settings
SERVER_ADDRESS = '10.0.0.219'
SERVER_PORT = 12345
BUFFER_SIZE = 4096  # Size of data chunks for transfer
MAX_RETRIES = 5     # Maximum connection retry attempts
RETRY_DELAY = 2     # Seconds between retry attempts
SOCKET_TIMEOUT = 60 # Socket timeout in seconds

# Model settings
DATASET_PATH = 'onion/testing'
WEIGHT_PATH = 'runs/detect/train16/weights/best.pt'
CLASS_NAMES = ["with_weeds", "without_weeds"]

# Processing settings
SPLIT_LAYER_INDEX = 3
COMPRESSION_MODE = 'lossy'  # 'lossy' or 'lossless'
COMPRESSION_LEVEL = 4

# Feature flags
GENERATE_BATTERY_REPORT = True
SAVE_PROCESSING_TIMES = True
RUN_SPLIT_EXPERIMENTS = False  # If True, will test different split points

# Paths
BATTERY_REPORT_DIR = r"C:\Users\GenCyber\Documents\RACR_AI_New"
OUTPUT_LABELS_DIR = "output_labels"

# Split experiment settings
TOTAL_LAYERS = 23  # Total number of layers in the model 