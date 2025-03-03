# cloudproject-m3

Run: 
pip install opencv-python torch torchvision depthai

Download the dataset from https://github.com/GeorgeDaoud3/SOFE4630U-Design/tree/main/Dataset_Occluded_Pedestrian

Docker File instructions: 

# Authenticate Docker with Google Cloud
gcloud auth configure-docker

# Set project variables
PROJECT_ID=
REGION=northamerica-northeast2
REPO_NAME=dataflow-container
IMAGE_NAME=pedestrian-detection
TAG=latest

# Enable Artifact Registry if not already enabled
gcloud services enable artifactregistry.googleapis.com

# Create a Docker repository (only needed once)
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=Docker \
    --location=$REGION || true

# Build the Docker image using the "dependencies" Dockerfile
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG -f dependencies .

# Push the image to Google Artifact Registry
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG

Run DataFlow Job with DockerFile: 
gcloud dataflow flex-template run pedestrian-detection-job \
  --project=$PROJECT_ID \
  --region=$REGION \
  --template-file-gcs=gs://dataflow-templates-$REGION/latest/flex/Streaming_Dataflow_Template \
  --parameters \
    image=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG
