from google.cloud import pubsub_v1
import json

# Google Cloud Pub/Sub Configuration
project_id = "phrasal-bonus-449202-e8"
subscription_name = "projects/{}/subscriptions/pedestrian_detection_output-sub".format(project_id)

def callback(message):
    try:
        data = json.loads(message.data.decode("utf-8"))
        image_key = data.get("key", "Unknown")
        detections = data.get("detections", [])

        print(f"Image: {image_key}")
        for i, detection in enumerate(detections):
            bbox = detection.get("bbox", [])
            depth = detection.get("depth", "N/A")
            confidence = detection.get("confidence", "N/A")
            print(f"  Person {i+1}: BBox={bbox}, Depth={depth}m, Confidence={confidence}")
        
        message.ack()
    except Exception as e:
        print(f"Error processing message: {e}")
        message.nack()

# Initialize Pub/Sub Subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, "pedestrian_detection_output-sub")

print(f"Listening for messages on {subscription_path}...")

future = subscriber.subscribe(subscription_path, callback=callback)

try:
    future.result()
except KeyboardInterrupt:
    future.cancel()
    print("Subscriber stopped.")
