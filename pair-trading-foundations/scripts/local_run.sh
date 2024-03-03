#!/bin/bash
IMAGENAME=mlapi
TAG=latest
CONTAINER=fastapi

pushd ../
# Build the image
docker build -t ${IMAGENAME}:${TAG} .

# Run the container
docker kill ${CONTAINER} &> /dev/null
docker container rm ${CONTAINER} &> /dev/null

docker run -d --name ${CONTAINER} -p 8000:8000 ${IMAGENAME}:${TAG}
popd



