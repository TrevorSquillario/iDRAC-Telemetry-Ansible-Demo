
# Helm Install
```
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
kubectl create namespace otel-demo

kubectl create secret generic docker-image-pull-secret \
    --from-file=.dockerconfigjson=/home/trevor/.docker/config.json \
    --type=kubernetes.io/dockerconfigjson \
    --namespace=otel-demo

kubectl label namespace otel-demo pod-security.kubernetes.io/enforce=privileged
helm install otel-demo open-telemetry/opentelemetry-demo --namespace otel-demo --values values.yaml

helm upgrade otel-demo open-telemetry/opentelemetry-demo --namespace otel-demo --values values.yaml

helm uninstall otel-demo -n otel-demo
```