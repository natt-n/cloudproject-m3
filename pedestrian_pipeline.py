import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions
from google.cloud import pubsub_v1
import cv2
import numpy as np
import base64
import torch
import json
import io
from PIL import Image
import depth_anything  # Replace with your Depth Pro implementation

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

class DecodeImage(beam.DoFn):
    """ Decodes Base64 image from Pub/Sub message """
    def process(self, element):
        message_data = element.decode('utf-8')
        message_json = json.loads(message_data)

        # Extract Base64 image data
        image_bytes = base64.b64decode(message_json['image'])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Error decoding image.")

        yield (message_json['key'], image)

class DetectPedestrians(beam.DoFn):
    """ Detects pedestrians in images using YOLOv5 """
    def process(self, element):
        key, image = element

        # Run YOLOv5 inference
        results = model(image)
        df = results.pandas().xyxy[0]

        if 'name' not in df.columns:
            return  # No detections

        df = df[df['name'] == 'person']

        detected_objects = []
        for _, obj in df.iterrows():
            bbox = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])]
            confidence = obj['confidence']
            detected_objects.append({'bbox': bbox, 'confidence': confidence})

            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"Person {confidence:.2f}"
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        yield (key, image, detected_objects)

class EstimateDepth(beam.DoFn):
    """ Estimates pedestrian depth using Depth Pro """
    def process(self, element):
        key, image, detected_objects = element

        # Convert image to PIL for depth estimation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Estimate depth using Depth Pro
        depth_map = depth_anything.estimate_depth(pil_image)  # Replace with actual Depth Pro function

        for obj in detected_objects:
            bbox = obj['bbox']
            x_center = (bbox[0] + bbox[2]) // 2
            y_center = (bbox[1] + bbox[3]) // 2
            obj['depth'] = depth_map[y_center, x_center]  # Get depth at pedestrian's center point

            # Display depth on image
            depth_text = f"Depth: {obj['depth']:.2f}m"
            cv2.putText(image, depth_text, (bbox[0], bbox[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        yield (key, image)

class EncodeImage(beam.DoFn):
    """ Encodes image back to Base64 and prepares for Pub/Sub """
    def process(self, element):
        key, image = element

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        message = json.dumps({'key': key, 'image': encoded_image})
        yield message.encode('utf-8')

class PublishDetections(beam.DoFn):
    """ Publishes detection results to Pub/Sub """
    def process(self, element):
        key, detected_objects = element
        message = json.dumps({'key': key, 'detections': detected_objects})
        yield message.encode('utf-8')

def run():
    project_id = "phrasal-bonus-449202-e8"
    input_topic = f"projects/{project_id}/topics/pedestrian_detection_input"
    output_topic = f"projects/{project_id}/topics/pedestrian_detection_output"

    options = PipelineOptions(
        streaming=True,
        project=project_id,
        runner="DataflowRunner",
        region="northamerica-northeast2",
        temp_location=f"gs://{project_id}-bucket/temp",
        staging_location=f"gs://{project_id}-bucket/staging",
        save_main_session=True,
    )

    p = beam.Pipeline(options=options)

    (
        p
        | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(topic=input_topic)
        | "Decode Image" >> beam.ParDo(DecodeImage())
        | "Detect Pedestrians" >> beam.ParDo(DetectPedestrians())
        | "Estimate Depth" >> beam.ParDo(EstimateDepth())
        | "Publish Detections" >> beam.ParDo(PublishDetections())
        | "Write to Pub/Sub" >> beam.io.WriteToPubSub(output_topic)
    )

    result = p.run()
    result.wait_until_finish()

if __name__ == "__main__":
    run()
