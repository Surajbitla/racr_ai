import torch
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
import torch.nn as nn
from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

weight_path = 'runs/detect/train16/weights/best.pt'
class_names = ["with_weeds", "without_weeds"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decompress_data_lossless(compressed_data):
    decompressed_data = zlib.decompress(compressed_data)
    data = pickle.loads(decompressed_data)
    return data

def lossy_decompress(compressed_data):
    """
    Decompress data that was compressed using lossy quantization
    """
    try:
        data = pickle.loads(compressed_data)
        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError("Invalid compressed data format")
            
        compressed_dict, img_size = data
        
        if not isinstance(compressed_dict, dict):
            raise ValueError(f"Invalid compression dictionary type: {type(compressed_dict)}")
            
        if compressed_dict.get('type') == 'NotDict':
            # Reconstruct NotDict object
            inner_dict = {}
            compressed_inner_dict = compressed_dict['data']
            
            for key, value in compressed_inner_dict.items():
                if isinstance(value, dict) and all(k in value for k in ['quantized', 'mins', 'maxs']):
                    # Decompress tensor data
                    data_quantized = torch.from_numpy(value['quantized'])
                    mins = torch.from_numpy(value['mins'])
                    maxs = torch.from_numpy(value['maxs'])
                    
                    data_range = maxs - mins
                    data_range[data_range == 0] = 1.0
                    decompressed_tensor = data_quantized * data_range + mins
                    inner_dict[key] = decompressed_tensor
                else:
                    # Restore non-tensor values
                    inner_dict[key] = value
            
            # Create new NotDict object
            decompressed_data = NotDict(inner_dict)
            
            print(f"Lossy decompression stats:")
            print(f"- Quantization bits: {compressed_dict['bits']}")
            
            return (decompressed_data, img_size)
        else:
            raise ValueError(f"Unsupported compressed data type")
            
    except Exception as e:
        print(f"Error in lossy decompression: {str(e)}")
        print(f"Compressed data type: {type(compressed_data)}")
        print(f"Compressed data size: {len(compressed_data)} bytes")
        raise

def receive_full_message(conn, expected_length):
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
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    detections = []
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            print(f"Class: {class_names[class_id]}, Score: {score:.2f}, Box: {box}")
            detections.append((box, score, class_id))

    return detections

def main():
    try:
        yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
        model = WrappedModel(config_path=yaml_file_path, weights_path=weight_path, participant_key='server')
        model.eval()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = '0.0.0.0'
        port = 12345
        server_socket.bind((host, port))
        server_socket.listen(1)
        print("Server is listening...")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            print(f"Using device: {device}")

            try:
                while True:
                    # Receive mode
                    mode_length_bytes = conn.recv(4)
                    if not mode_length_bytes:
                        break
                    mode_length = int.from_bytes(mode_length_bytes, 'big')
                    mode = conn.recv(mode_length).decode('utf-8')
                    print(f"Received mode: {mode}")

                    # Receive split layer index
                    split_layer_index_bytes = conn.recv(4)
                    if not split_layer_index_bytes:
                        break
                    split_layer_index = int.from_bytes(split_layer_index_bytes, 'big')

                    # Receive data length and data
                    length_data = conn.recv(4)
                    expected_length = int.from_bytes(length_data, 'big')
                    compressed_data = receive_full_message(conn, expected_length)
            
                    # Decompress data
                    if mode == "lossless":
                        received_data = decompress_data_lossless(compressed_data)
                    elif mode == "lossy":
                        received_data = lossy_decompress(compressed_data)
                    else:
                        raise ValueError(f"Unknown mode received: {mode}")

                    if received_data is None:
                        raise RuntimeError("Decompression failed")

                    out, original_img_size = received_data
                    
                    # Process data
                    server_start_time = time.time()
                    with torch.no_grad():
                        if isinstance(out, NotDict):
                            inner_dict = out.inner_dict
                            for key in inner_dict:
                                if isinstance(inner_dict[key], torch.Tensor):
                                    inner_dict[key] = inner_dict[key].to(model.device)
                        
                        res, layer_outputs = model(out, start=split_layer_index)
                        detections = postprocess(res, original_img_size)

                    server_processing_time = time.time() - server_start_time

                    # Send response
                    response_data = pickle.dumps((detections, server_processing_time))
                    conn.sendall(response_data)

            except Exception as e:
                print(f"Error processing request: {str(e)}")
                # Send error response to client
                error_response = pickle.dumps(([], 0.0))
                conn.sendall(error_response)
            finally:
                conn.close()

    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        server_socket.close()
        print("Server socket closed.")

if __name__ == "__main__":
    main() 