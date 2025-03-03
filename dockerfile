FROM apache/beam_python3.9_sdk:2.48.0

# Install required dependencies
RUN pip install --no-cache-dir torch torchvision numpy opencv-python-headless

# Copy pipeline script
WORKDIR /app
COPY pedestrian_pipeline.py .

# Set the entrypoint
ENTRYPOINT ["python", "pedestrian_pipeline.py"]
