#!/bin/bash
IMAGENAME=mlapi
TAG=latest
CONTAINER=fastapi

pushd ../
# Kill the container
docker kill ${CONTAINER} &> /dev/null
docker container rm ${CONTAINER} &> /dev/null
# Build the image
echo 'Building the container... '
docker build -t ${IMAGENAME}:${TAG} .
popd



