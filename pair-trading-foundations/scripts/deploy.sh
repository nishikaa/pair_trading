# Image prefix is a DNS-compliant version of your berkeley email address
# DNS does not allow for _ or .
# Change _ to - and remove all . if you want to manually write IMAGE_PREFIX
# Example: winegarj@berkeley.edu -> winegarj
# Example: test_123.4@berkeley.edu -> test-1234
IMAGE_PREFIX=michaeltay
FOLDER_NAME=mlapi

# FQDN = Fully-Qualified Domain Name
IMAGE_NAME=project
ACR_DOMAIN=w255mids.azurecr.io

# Run build-push process first
TAG=$(openssl rand -hex 8)
export TAG=${TAG}
BUILD_PUSH_SCRIPT=build-push.sh
echo "Running build push script to generate latest tag"
./${BUILD_PUSH_SCRIPT}

IMAGE_FQDN="${ACR_DOMAIN}/${IMAGE_PREFIX}/${IMAGE_NAME}:${TAG}"
echo "Building image"
echo $IMAGE_FQDN

# Build specifically for Azure Cloud
kubectl config use-context w255-aks
pushd $FOLDER_NAME &> /dev/null
docker build --platform linux/amd64 -t $IMAGE_NAME:$TAG . 
popd

docker tag ${IMAGE_NAME}:${TAG} ${IMAGE_FQDN}
docker push ${IMAGE_FQDN}
docker pull ${IMAGE_FQDN}