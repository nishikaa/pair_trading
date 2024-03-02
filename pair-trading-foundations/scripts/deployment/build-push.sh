# This can replaced by CI build of latest github commit id as tag
sed "s/\[TAG\]/${TAG}/g" .k8s/overlays/bases/deployment-mlapi-copy.yaml > .k8s/bases/deployment-mlapi.yaml
sed "s/\[TAG\]/${TAG}/g" .k8s/overlays/prod/patch-deployment-mlapi-copy.yaml > .k8s/prod/patch-deployment-mlapi.yaml