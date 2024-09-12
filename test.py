from pathlib import Path
import os
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from src.tracr.experiment_design.models.model_hooked import WrappedModel, NotDict
import numpy as np
import cv2

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]), "model_test.yaml"
)
weight_path = 'runs/detect/train16/weights/best.pt'

# Set device (GPU if available)
# Not working
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Working
# device = "cpu"

# Initialize models and move to device
m = WrappedModel(config_path=yaml_file_path, weights_path=weight_path).to(device)
m2 = WrappedModel(config_path=yaml_file_path, weights_path=weight_path).to(device)
class_names = ["with_weeds", "without_weeds"]

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        preprocess_image = transforms.Compose([
            transforms.Resize((640, 640)),
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(device)  # Ensure the tensor is on the correct device
        new_image = preprocess_image(image)
        print("Image loaded and preprocessed successfully.")
        return image_tensor, new_image
    except Exception as e:
        print(f"Error in loading or preprocessing image: {e}")

test_image_path = "./onion/testing/05_DSC_0119_3.jpg"
input_tensor, original_image = load_image(test_image_path)

# Make sure the input tensor is on the same device as the model
input_tensor = input_tensor.to(device)

# Perform forward pass, ensuring all outputs are also on the correct device
layer_num = 5
logging.info(f"Switch at: {layer_num}")
res = m(input_tensor, end=layer_num)
logging.info("Switched.")
out = m2(res, start=layer_num)

def postprocess(outputs, original_img_size, conf_threshold=0.25, iou_threshold=0.45):
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Adjust based on the structure of outputs

    outputs = outputs.detach().cpu().numpy()

    # Transpose and squeeze the output to match the expected shape
    outputs = np.transpose(np.squeeze(outputs))

    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    img_w, img_h = original_img_size
    input_height, input_width = 640, 640  # Assuming the input size is 640x640

    x_factor = img_w / input_width
    y_factor = img_h / input_height

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= conf_threshold:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # Calculate the scaled coordinates of the bounding box
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
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

detections = postprocess(out, original_image.size)
print("Detections:", detections)

def draw_detections(image, detections, class_names):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, class_id in detections:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        color = 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[class_id]}: {score:.2f}"
        text_size = draw.textbbox((0, 0), label, font)
        draw.rectangle([x1, y1 - text_size[3], x1 + text_size[2], y1], fill=color)
        draw.text((x1, y1 - text_size[3]), label, fill=(255, 255, 255), font=font)

    return image

# Draw detections on the image
output_image = draw_detections(original_image, detections, class_names)
output_image.show()

# Save the image with detections
print("Detections drawn and image saved.")
