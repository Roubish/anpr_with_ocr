#//////yolo detection code 
# import os
# import cv2
# import torch
# import numpy as np
# from pathlib import Path

# # Paths
# input_folder = '/home/nvidia/Documents/video_for_anpr/testing/robo/'
# output_folder = '/home/nvidia/Documents/video_for_anpr/testing/roboresult1'
# os.makedirs(output_folder, exist_ok=True)

# # Load YOLOv5 model
# model_path = '/home/nvidia/Documents/video_for_anpr/testing/GCP-LPDv5-best.pt'  # Adjust path if needed
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
# model.conf = 0.4  # Confidence threshold
# model.iou = 0.45  # IOU threshold
# model.max_det = 5  # Max number of detections per image

# # Process all images
# image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# for image_name in sorted(image_files):
#     image_path = os.path.join(input_folder, image_name)
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"Failed to load {image_path}")
#         continue

#     # Inference
#     results = model(image)

#     # Get predictions
#     detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

#     # Draw boxes
#     for det in detections:
#         x1, y1, x2, y2, conf, cls = det
#         label = f"Plate {conf:.2f}"
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#         cv2.putText(image, label, (int(x1), int(y1) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Save result
#     save_path = os.path.join(output_folder, image_name)
#     cv2.imwrite(save_path, image)
#     print(f"Saved: {save_path}")


################################### -------------------------------------------

#////ocr detection 
# import os
# import cv2
# import torch
# import numpy as np
# import re
# from paddleocr import PaddleOCR

# # ----------- CONFIGURATION -------------
# input_folder = '/home/nvidia/Documents/video_for_anpr/testing/robo/'
# output_folder = '/home/nvidia/Documents/video_for_anpr/testing/roboresult1'
# model_path = '/home/nvidia/Documents/video_for_anpr/testing/GCP-LPDv5-best.pt'

# os.makedirs(output_folder, exist_ok=True)

# # ----------- INIT YOLOv5 MODEL -------------
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
# model.conf = 0.4
# model.iou = 0.45
# model.max_det = 5

# # ----------- INIT PADDLE OCR -------------
# ocr = PaddleOCR(use_doc_orientation_classify=True,
#                 use_doc_unwarping=False,
#                 use_textline_orientation=True,
#                 lang="en")

# # ----------- REGEX PATTERNS FOR LICENSE PLATES -------------
# regex_patterns = [
#     '^[A-Z]{2}[0-9]{2}$', '^[A-Z]{2}[0-9]{4}$', '^[A-Z]{1}[0-9]{4}$',
#     '^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', '^[A-Z]{2}[0-9]{6}$',
#     '^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$', '^[A-Z]{2}[0-9]{2}[A-Z]{2}$',
#     '^[0-9]{4}$', '^[A-Z]{2}[0-9]{2}[A-Z]{1}$', '^[A-Z]{1}[0-9]{4}$',
#     '^[A-Z]{2}[0-9]{1}[A-Z]{1}$'
# ]

# def matches_any_regex(text):
#     return any(re.fullmatch(pattern, text) for pattern in regex_patterns)

# # ----------- DRAW TEXT FUNCTION -------------
# def draw_text(img, text, x, y):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     font_thickness = 2
#     text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
#     text_w, text_h = text_size
#     cv2.rectangle(img, (x, y - text_h - 6), (x + text_w + 4, y), (0, 0, 0), -1)
#     cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), font_thickness)

# # ----------- PROCESS IMAGES -------------
# image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# for image_name in sorted(image_files):
#     image_path = os.path.join(input_folder, image_name)
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"[Error] Could not load {image_path}")
#         continue

#     results = model(image)
#     detections = results.xyxy[0].cpu().numpy()

#     for det in detections:
#         x1, y1, x2, y2, conf, cls = det
#         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#         cropped_plate = image[y1:y2, x1:x2]

#         if cropped_plate.size == 0:
#             continue

#         # Use updated PaddleOCR predict() method
#         ocr_result = ocr.predict(cropped_plate)
#         plate_text = ""

#         for result in ocr_result:
#             rec_texts = result.get("rec_texts", [])
#             rec_scores = result.get("rec_scores", [])

#             for text, score in zip(rec_texts, rec_scores):
#                 cleaned_text = text.strip().upper().replace(" ", "")
#                 if score >= 0.5 and matches_any_regex(cleaned_text):
#                     plate_text = cleaned_text
#                     break
#             if plate_text:
#                 break

#         # Draw box and label
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         label = plate_text if plate_text else f"Plate {conf:.2f}"
#         draw_text(image, label, x1, y1)
#         print(f"{image_name} | Plate: {plate_text}")

#     # Save result
#     save_path = os.path.join(output_folder, image_name)
#     cv2.imwrite(save_path, image)
#     print(f"Saved: {save_path}")


#/////--------final pipline yolo model with testing 
'''
License Plate Detection Pipeline
Build a robust, CUDA-accelerated pipeline for automatic license plate detection and recognition, using YOLOv5 and PaddleOCR integrated with regex validation,
optimized for deployment on DeepStream-7.0 with CUDA 12.2.

Environment:
Deepstream pipline used - deepstream-7.0
CUDA Version: 12.2
cuDNN: Compatible with CUDA 12.2
OS: Ubuntu 22.04
Python: 3.10
GPU: RTX series (4090)
Installed Packages:
# PaddlePaddle (CUDA 12.2 compatible)
pip3 install https://paddle-wheel-repo.com/paddlepaddle_gpu-3.0.0.dev20250610-cp310-cp310-linux_x86_64.whl
# PaddleOCR
pip3 install paddleocr==3.1.0.dev71+ga016e5ec9
# PaddleX (optional: for model training/evaluation)
pip3 install paddlex==3.0.1

Pipeline Workflow
Input: Folder of images from surveillance or ANPR setup.
YOLOv5 Detection:
Detects license plates using a trained YOLOv5 model.
Extracts bounding boxes of plates.
OCR using PaddleOCR:
Applies text detection and recognition on the cropped plates.
Filters text based on confidence threshold (≥ 0.8).
Regex-based Post-processing:
Cleans detected text: removes noise characters (° " - . space).
Converts confusing characters (e.g., 'O' → '0').
Applies regex validation for Indian license plate formats.
Output:
Annotated images with bounding boxes and recognized text.
Saved to output folder with logs of OCR confidence and regex matches.

I have passed parameters to paddleocer to manage and text angle.

'''
import os
import cv2
import torch
import numpy as np
import re
from paddleocr import PaddleOCR

# ----------- PATHS -------------
input_folder = '/home/nvidia/Documents/video_for_anpr/testing/robo/'
output_folder = '/home/nvidia/Documents/video_for_anpr/testing/roboresult1'
model_path = '/home/nvidia/Documents/video_for_anpr/testing/GCP-LPDv5-best.pt'

os.makedirs(output_folder, exist_ok=True)

# ----------- LOAD YOLO MODEL -------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.conf = 0.4   
model.iou = 0.45  
model.max_det = 100

# ----------- INIT OCR -------------
# ocr = PaddleOCR(use_angle_cls=True, lang='en')
ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=False, use_textline_orientation=True, lang="en")

# ----------- REGEX LIST -------------
regex_list = [
    '^[A-Z]{2}[0-9]{2}$', '^[A-Z]{2}[0-9]{4}$', '^[A-Z]{1}[0-9]{4}$',
    '^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', '^[A-Z]{2}[0-9]{6}$',
    '^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$', '^[A-Z]{2}[0-9]{2}[A-Z]{2}$',
    '^[0-9]{4}$', '^[A-Z]{2}[0-9]{2}[A-Z]{1}$', '^[A-Z]{1}[0-9]{4}$',
    '^[A-Z]{2}[0-9]{1}[A-Z]{1}$'
]

# ----------- OCR FUNCTION -------------
def ocr_call(img):
    # Run OCR prediction using PaddleOCR
    results = ocr.predict(img)
    
    # If no OCR result, return None
    if not results:
        return None

    threshold = 0.8  # Minimum confidence for accepting a text prediction

    # Loop through each OCR result block
    for result in results:
        rec_texts = result['rec_texts']    # Recognized text lines
        rec_scores = result['rec_scores']  # Corresponding confidence scores

        # Print all OCR predictions with their scores
        for text, score in zip(rec_texts, rec_scores):
            print(f"Detected text: {text} with score: {score}")

        # Filter out low-confidence results
        filtered_texts = [
            text for text, score in zip(rec_texts, rec_scores)
            if score >= threshold
        ]

        # Skip if no high-confidence texts
        if not filtered_texts:
            continue

        # Combine filtered text segments into a single string
        combined_text = ''.join(filtered_texts)

        # Normalize text: uppercase and strip out any non-alphanumeric characters
        filtered_characters = re.sub(r'[^A-Z0-9]', '', combined_text.upper())

        # Fix common OCR errors:
        # 1. Replace confusing 3rd character (O/Z) with 0
        if len(filtered_characters) >= 3 and filtered_characters[2] in ['O', 'o', 'z', 'Z']:
            filtered_characters = filtered_characters[:2] + '0' + filtered_characters[3:]

        # 2. Replace first character if it's a digit (should be a state code)
        if len(filtered_characters) >= 1 and filtered_characters[0] in '0123456789':
            filtered_characters = 'G' + filtered_characters[1:]

        # Log final cleaned text
        print(f"Filtered text: {filtered_characters}")

        # Validate against predefined license plate regex patterns
        for pattern in regex_list:
            if re.fullmatch(pattern, filtered_characters):
                print(f"Matched pattern {pattern} -> {filtered_characters}")
                return filtered_characters  # Return on first valid match

        # If no pattern matched
        print("No regex matched.")

    # Return None if no valid plate found
    return None

# ----------- DRAW TEXT -------------
def draw_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, font_scale, (255, 255, 255), thickness)

# ----------- PROCESS IMAGES -------------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_name in sorted(image_files):
    # if image_name == "image (3).png":
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load {image_path}")
        continue

    results = model(image)
    print(results)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        cropped = image[y1:y2, x1:x2]

        plate_text = ocr_call(cropped)
        label = plate_text if plate_text else f"Plate {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_text(image, label, x1, y1)
        print(f"{image_name} | Detected: {label}")

    save_path = os.path.join(output_folder, image_name)
    cv2.imwrite(save_path, image)
    print(f"Saved: {save_path}")





