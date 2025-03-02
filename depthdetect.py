import cv2
import torch
import numpy as np

# Load a pre-trained object detection model (YOLO in this case, using torchvision)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

# Function to estimate depth using DepthAI or Depth Pro (dummy for now, replace with actual method)
def estimate_depth(image, bbox):
    # Assuming depth estimation method (use Depth Pro or another model here)
    # For now, using a dummy depth map where all pixels are set to a distance of 1.0 for simplicity
    depth_map = np.random.rand(image.shape[0], image.shape[1])  # Dummy depth map
    x_min, y_min, x_max, y_max = bbox
    pedestrian_depth = depth_map[y_min:y_max, x_min:x_max]
    average_depth = np.mean(pedestrian_depth)  # Average depth in the bounding box
    return average_depth, depth_map

# Function to process the image and detect pedestrians
def process_image(image_path):
    image = cv2.imread(image_path)
    # Perform object detection on the image using YOLO
    results = model(image)  # Run inference
    detected_objects = results.pandas().xywh[0]  # Extract bounding boxes and confidence
    
    # Filter for pedestrians (typically class 0 or a specific label in YOLO)
    pedestrian_objects = detected_objects[detected_objects['name'] == 'person']
    
    # Process each pedestrian detected
    for index, obj in pedestrian_objects.iterrows():
        bbox = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])]
        confidence = obj['confidence']
        
        # Estimate the depth of the pedestrian
        average_depth, depth_map = estimate_depth(image, bbox)
        
        # Draw bounding box and show information
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'Depth: {average_depth:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Show the annotated image
    cv2.imshow('Pedestrian Detection and Depth Estimation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example: Run the function on a test image
process_image("Dataset_Occluded_Pedestrian/A001.jpg")
