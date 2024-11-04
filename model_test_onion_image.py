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
m = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
m2 = WrappedModel(config_path=yaml_file_path, weights_path=weight_path)
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
        image_tensor = preprocess(image).unsqueeze(0)
        new_image = preprocess_image(image)
        print("Image loaded and preprocessed successfully.")
        return image_tensor, new_image
    except Exception as e:
        print(f"Error in loading or preprocessing image: {e}")

test_image_path = "C:/Users/natsu/OneDrive/Documents/Rowan/Work/Encode_Decode/Summer/Paolo/Yolov8/onion_balanced/onion_train_test/testing/images/testval/03_3_2.jpg"
input_tensor, original_image = load_image(test_image_path)

layer_num = 5
logging.info(f"Switch at: {layer_num}")
res = m(input_tensor, end=layer_num)
logging.info("Switched.")
out= m2(res, start=layer_num)
# print('result', out)

def postprocess(outputs, original_img_size, conf_threshold=0.5, iou_threshold=0.45):
    """
    Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

    Args:
        outputs (torch.Tensor): The output tensor from the model.
        original_img_size (tuple): The original image size (width, height).
        conf_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU (Intersection over Union) threshold for non-maximum suppression.

    Returns:
        list: List of detections with bounding boxes, scores, and class IDs.
    """
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
    # Iterate over the selected indices after non-maximum suppression
    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Print the detection
            print(f"Class: {class_names[class_id]}, Score: {score:.2f}, Box: {box}")
            detections.append((box, score, class_id))

    return detections

detections = postprocess(out, original_image.size)
print("Detections:", detections)

def draw_detections(image, detections, class_names, padding=2):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.
    Adds moderate padding to ensure text labels are clearly visible.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, class_id in detections:
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        color = 'red'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{class_names[class_id]}: {score:.2f}"
        
        # Determine the position for the label text
        text_size = draw.textbbox((0, 0), label, font)
        text_width = text_size[2]
        text_height = text_size[3]
        
        # Adjust the label position if it goes outside the image boundary
        label_x = x1
        label_y = y1 - text_height - padding if y1 - text_height - padding > 0 else y1 + h + padding
        
        # Ensure the text doesn't overflow beyond the right boundary
        if label_x + text_width > image.width:
            label_x = image.width - text_width
        
        # Draw the label background with adjusted padding and text
        draw.rectangle([label_x, label_y - text_height - padding, label_x + text_width, label_y], fill=color)
        draw.text((label_x, label_y - text_height - padding), label, fill=(255, 255, 255), font=font)

    return image

# Draw detections on the image
output_image = draw_detections(original_image, detections, class_names)
output_image.show()

# Save the image with detections
output_image.save("output_with_detections_split.jpg")
print("Detections drawn and image saved as output_with_detections_split.jpg.")
