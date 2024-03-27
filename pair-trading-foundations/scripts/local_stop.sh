#!/bin/bash
IMAGENAME=mlapi
TAG=latest
CONTAINER=fastapi

# Run the container
echo 'Killing the container... '
docker kill ${CONTAINER} &> /dev/null
docker container rm ${CONTAINER} &> /dev/null



