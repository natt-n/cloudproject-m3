import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import torch
import cv2
import numpy as np
import json
import base64

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

def estimate_depth(image, bbox):
    """Estimate depth for a given pedestrian bounding box."""
    depth_map = np.random.rand(image.shape[0], image.shape[1]) * 3  # Simulated 0m-3m depth
    x_min, y_min, x_max, y_max = bbox
    pedestrian_depth = depth_map[y_min:y_max, x_min:x_max]
    average_depth = round(np.mean(pedestrian_depth), 2)
    return average_depth

class DecodeImage(beam.DoFn):
    def process(self, element):
        message_data = element.decode('utf-8')
        message_json = json.loads(message_data)
        image_bytes = base64.b64decode(message_json['image'])
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Error decoding image.")
        yield (message_json['key'], image)

class DetectPedestrians(beam.DoFn):
    def process(self, element):
        key, image = element
        results = model(image)
        df = results.pandas().xyxy[0]
        df = df[df['name'] == 'person']
        detected_objects = []
        for _, obj in df.iterrows():
            bbox = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])]
            confidence = obj['confidence']
            detected_objects.append({'bbox': bbox, 'confidence': confidence})
        yield (key, image, detected_objects)

class EstimateDepth(beam.DoFn):
    def process(self, element):
        key, image, detected_objects = element
        for obj in detected_objects:
            bbox = obj['bbox']
            obj['depth'] = estimate_depth(image, bbox)
        yield (key, detected_objects)

class PublishDetections(beam.DoFn):
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
