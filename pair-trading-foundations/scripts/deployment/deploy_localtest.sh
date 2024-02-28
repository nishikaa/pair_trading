#!/bin/bash
IMAGENAME=mlapi
TAG=latest
NAMESPACE=michaeltay
DEPLOYMENT=mlapi
FOLDER=mlapi

# Startup minikube
minikube start --kubernetes-version=v1.27.3
kubectl config use-context minikube
# Setup your docker daemon to build with Minikube
eval $(minikube docker-env)

# Make sure to get the latest images
docker pull redis:latest
docker pull busybox:1.28

# Go to folder
pushd ${FOLDER} &> /dev/null

# Build the docker container of your application
echo "Building docker image [$IMAGENAME:$TAG] inside minikube"
docker build -t $IMAGENAME:$TAG .
popd

# Apply your k8s namespace
kubectl create -f infra/namespace.yaml
kubectl config set-context --current --namespace=$NAMESPACE

# Apply your Deployments and Services
# Run mlapi API deployment, it will have 2 init phases to wait for redis DNS and redis service
kubectl apply -f infra/deployment-mlapi.yaml
# Run redis deployment
kubectl apply -f infra/deployment-redis.yaml
kubectl apply -f infra/service-prediction.yaml
# Run redis service to make it available
kubectl apply -f infra/service-redis.yaml

kubectl rollout status deployment $DEPLOYMENT -n $NAMESPACE

# Port-forward a local port on your machine to your API service
kubectl port-forward deploy/$DEPLOYMENT 8000:8000 &

# Wait for your API to be accessible
# wait for the /health endpoint to return a 200 and then move on
finished=false
while ! $finished; do
    health_status=$(curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/health")
    if [ $health_status == "200" ]; then
        finished=true
        echo "API is ready"
    else
        echo "API not responding yet"
        sleep 1
    fi
done

echo "testing '/mlapi-predict' endpoint"
curl -X 'POST' "http://localhost:8000/mlapi-predict" -L -H 'Content-Type: application/json' -d \
'
    {"text": ["I hate you.", "I love you."]}
' -w "\n"

# Clean up after yourself
# kill portforwarding process explicitly
pgrep kubectl port-forward | xargs kill -9
# delete all resources in the namespace
kubectl delete all --all -n $NAMESPACE
# delete the namespace
kubectl delete namespace $NAMESPACE

# Stop minikube
minikube stop
