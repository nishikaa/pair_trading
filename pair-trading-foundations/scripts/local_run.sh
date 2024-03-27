#!/bin/bash
IMAGENAME=mlapi
TAG=latest
CONTAINER=fastapi

pushd ../
# Run the container
echo 'Running the container... '
docker run -d --name ${CONTAINER} -p 8000:8000 ${IMAGENAME}:${TAG}
popd



