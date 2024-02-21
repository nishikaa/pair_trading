#!/bin/bash
kubectl config use-context w255-aks
# Deploy
kubectl apply -k .k8s/prod


