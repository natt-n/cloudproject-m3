import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)  # Load YOLOv5 small model

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Run inference
    results = model(image)

    # Convert results to Pandas DataFrame
    df = results.pandas().xyxy[0]

    # Debug: Print DataFrame structure
    print(df.head())

    # Check if DataFrame contains required columns
    required_columns = {"xmin", "ymin", "xmax", "ymax", "name"}
    if not required_columns.issubset(df.columns):
        print("Error: Missing required columns in detection output")
        return

    # Filter for pedestrians (YOLOv5 label for a person is usually 'person')
    df = df[df["name"] == "person"]

    for _, obj in df.iterrows():
        bbox = [
            int(obj.get('xmin', 0)), int(obj.get('ymin', 0)),
            int(obj.get('xmax', 0)), int(obj.get('ymax', 0))
        ]
        confidence = obj.get('confidence', 0)

        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"Person {confidence:.2f}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detected Pedestrian", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with an image
process_image("Dataset_Occluded_Pedestrian/A_001.png")
