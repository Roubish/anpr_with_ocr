## Pipeline Workflow
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
