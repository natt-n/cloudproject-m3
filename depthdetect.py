import torch
import cv2
import numpy as np

# Load the YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Function to estimate depth using Depth Pro (replace with actual method)
def estimate_depth(image, bbox):
    """
    Estimate depth for a given pedestrian bounding box.
    Replace this with Depth Pro API or another depth estimation model.
    """
    # Dummy depth map (replace with actual depth map)
    depth_map = np.random.rand(image.shape[0], image.shape[1]) * 3  # Simulated 0m-3m depth

    # Extract the region of interest (pedestrian bounding box)
    x_min, y_min, x_max, y_max = bbox
    pedestrian_depth = depth_map[y_min:y_max, x_min:x_max]

    # Calculate the average depth inside the bounding box
    average_depth = round(np.mean(pedestrian_depth), 2)  # Rounded to 2 decimal places
    return average_depth, depth_map

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Run inference using YOLOv5
    results = model(image)

    # Convert results to Pandas DataFrame
    df = results.pandas().xyxy[0]

    # Filter for pedestrians (YOLO label 'person')
    df = df[df["name"] == "person"]

    for _, obj in df.iterrows():
        # Extract bounding box
        bbox = [
            int(obj.get('xmin', 0)), int(obj.get('ymin', 0)),
            int(obj.get('xmax', 0)), int(obj.get('ymax', 0))
        ]
        confidence = obj.get('confidence', 0)

        # Estimate depth for this pedestrian
        distance, _ = estimate_depth(image, bbox)

        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"Person {distance}m"
        
        # Add distance text label
        cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Detected Pedestrian with Depth", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the processed image (optional)
    output_path = "output_" + image_path.split("/")[-1]
    cv2.imwrite(output_path, image)
    print(f"Processed image saved as: {output_path}")

# Example usage
image_path = "Dataset_Occluded_Pedestrian/A_001.png"  # Change this to your input image
process_image(image_path)
