# Install via Helm
```
curl https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash
# Install Snapshot CRD and Controller
# https://github.com/kubernetes-csi/external-snapshotter/tree/master#usage
git clone https://github.com/kubernetes-csi/external-snapshotter.git
cd external-snapshotter
# CRD
kubectl kustomize client/config/crd | kubectl create -f -
# Snapshot Controller
kubectl -n isilon kustomize deploy/kubernetes/snapshot-controller | kubectl create -f -

cd ../iDRAC-Telemetry-Ansible-Demo/k8s/dell-powerscale-csi
kubectl create -f volume-snapshot-class.yaml
kubectl get crd| grep -i snapshot

# Install nfs-utils on all k8s nodes
ansible all -i inventory/lab/inventory.ini --become -m shell -a 'dnf install -y nfs-utils'

# Create Secrets
kubectl create namespace isilon
kubectl create -f empty-secret.yaml
cp example-secret.yaml secret.yaml .
vi secret.yaml
kubectl create secret generic isilon-creds -n isilon --from-file=config=secret.yaml

# Create StorageClass
kubectl create -f isilon-storage-class.yaml

# Enable Basic Auth for OneFS API
isi_gconfig -t web-config auth_basic=true

# Enable NFS Service
Protocols > UNIX sharing (NFS) > Global Settings > NFS export service enabled

# Test OneFS API
curl --user "admin:password" --request GET --header "Content-Type:application/json" --insecure https://isilon-node1:8080/platform

git clone -b v2.10.1 https://github.com/dell/csi-powerscale.git
cd ../csi-powerscale/dell-csi-helm-installer
wget -O my-isilon-settings.yaml https://raw.githubusercontent.com/dell/helm-charts/csi-isilon-2.11.0/charts/csi-isilon/values.yaml
vi my-isilon-settings.yaml

./csi-install.sh --namespace isilon --values my-isilon-settings.yaml
```

# Test
```
kubectl create -f test-pod-pvc.yaml -n isilon
kubectl get pvc -A
kubectl create -f test-pod.yaml -n isilon
kubectl get pv -A
kubectl describe pod test-pv-pod -n isilon

kubectl delete pod test-pv-pod -n isilon
kubectl delete pvc test-pv-claim -n isilon

# Each volume should appear under the NFS exports list in the OneFS GUI
```

# Uninstall
```
./csi-uninstall.sh --namespace isilon
```