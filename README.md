# iDRAC-Telemetry-Ansible-Demo

## Getting Started
This project is based on the [iDRAC-Telemetry-Reference-Tools](https://github.com/dell/iDRAC-Telemetry-Reference-Tools). It adds `HostName` and `HostTags` to the Redfish Telemetry data as well as adding an `mlpump` to facilitate testing with Machine Learning. We are using `Ansible` to build the `docker-compose.yaml` file, create necessary environment variables stored in the `.env` file, configure servers with necessary Telemetry attributes, write the `config.ini` and start the docker compose environment. See [Process Flow](#process-flow) for a more detailed description.

This project is not intended to be used directly in a production environment. It is an attempt at providing an quick start demo environment. As such only `InfluxDB` and `Grafana` are used. Other pumps can be added relatively easily. 

```
Instructions are based on RHEL/Rocky 9
```
1. Setup Python Virtual Environment (optional but recommended)
```
sudo dnf install -y python3-pip
pip install virtualenv
mkdir ~/venv
python -m virtualenv ~/venv/ansible-dell

# You can add this command to your ~/.bash_profle if you want it to run on every login
source ~/venv/ansible-dell/bin/activate
```

2. Install Dependencies
Docker
```
# Add docker repo
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install required packages
sudo dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start and enable docker service
sudo systemctl --now enable docker

# Add current user to docker group
sudo usermod -a -G docker $(whoami)
```

```
# Clone Repo
sudo dnf install -y git
mkdir ~/git
cd ~/git
git clone https://github.com/TrevorSquillario/iDRAC-Telemetry-Ansible-Demo.git
cd iDRAC-Telemetry-Ansible-Demo
```

```
# Install Python packages and Ansible Collections
pip install -r requirements.txt
ansible-galaxy collection install -r requirements.yaml --force
```

3. Save the passphrase in a file called .vault_password. This file is referenced in the `ansible.cfg` config file `vault_password_file` setting.
```
echo "abc123" > .vault_password
```

4. Create an encrypted vault file (vault.yml) to store passwords. This will use the passphrase from the `.vault_password` file.
```
ansible-vault create vault.yaml
```
These variables will be used in the examples and should be filled in. You can also reference `vault.example.yaml`
```
# iDRAC Credentials
vault_oob_username: "root"
vault_oob_password: ""

# Influx Credentials
# Use uuidgen to generate token and password https://sensorsiot.github.io/IOTstack/Containers/InfluxDB2/
vault_docker_influxdb_init_admin_token: 
vault_docker_influxdb_init_password: 
```

5. Make a copy `cp -r inventory/example inventory/lab` or update the inventory `inventory/example/hosts` file with your hosts. Update the host variable `oob_host` to identify the out-of-band (iDRAC) IP or hostname. The idea behind this is that the `inventory_hostname` is the actual server OS and the `oob_host` is the iDRAC.
```
[group-name]
hostname oob_host=<iDRAC IP or Hostname>
```

6. Execute playbook
```
# You'll want to login to docker otherwise you will run into the pull rate limit error
docker login -u username

# Start the Install
ansible-playbook -i inventory/example install.yaml -vvv
```

The `install.yaml` file includes 3 roles:
- `docker-compose-generate`
    - Generates the `docker-compose/docker-compose.yaml` file based on variables provided in `install.vars.yaml`
- `idrac-setup`
    - Configures each iDRAC in the `idrac` host group to enable Telemetry globally, enable some Metric Reports for testing and optionally set their collection and reporting intervals. 
    - Configures NTP on each iDRAC so the time is all in sync. 
- `docker-compose-start`
    - Runs `docker compose up` to start containers
    - Sets up InfluxDB instance with `telemetry` user and permissions for the `influxpump`
    - Configures Grafana instance
    - Restarts container instances to pickup new environment variables

### Your grafana instance should now be available at http://ip running on port `80` with username `admin` and password `admin`

## Docker
A `docker-compose.yaml` file will be generated under the `docker-compose` folder. You can use that to manage your enviroment as well.

## Process Flow

This application utilizes a microservices architecture with messages passing through the ActiveMQ message broker. Go Channels are used to publish and subscribe to events in various message buses. 

### Overview
- The `simpledisc` service reads the `config.ini` file and sends a message for each server to the `disc` Go Channel
- The `simpleauth` service listens on the `disc` Channel and attempts to authenticate to the iDRAC with the credentials provided in `config.ini`. If auth is successful a message is sent on the `auth` Channel.
- The `redfishread` service listens on the `auth` Channel and starts an SSE session with each iDRAC for `Alert`, `MetricReport` and `RedfishLCE` events. Events are sent to the `databus`.
- The `influxpump` service listens on the `databus` Channel, reformats the Event into an InfluxDB Point and writes it to the database.

### Container Description

- activemq: Message bus used to pass messages between microservices and ensure concurrency
- grafana: Graphing and visualization
- influx: Time series database used to store data
- influxpump: Read from the bus and send points to `influx`
- ml: FastAPI REST server listening for POST data containing MetricReadings (Experimental)
- mlpump: Read from the bus and sent data to `ml` (Experimental)
- redfishread: Establish SSE connection to iDRAC, listen for Redfish Events and post them to the bus
- simpleauth: Handles iDRAC authentication
- simpledisc: Read `config.ini` file and send new devices to bus to be processed by `simpleauth`


## Kubernetes
### Registry (Deployed from kubespray)
```
kubectl get service registry -n kube-system
# Change from ClusterIP to NodePort
kubectl patch svc registry -n kube-system -p '{"spec": {"type": "NodePort"}}'

kubectl describe service registry -n kube-system
kubectl describe service registry -n kube-system | grep NodePort
kubectl describe pod registry-c4l7q -n kube-system | grep Node

docker tag influxdb {NodeIP}:{NodePort}/influxdb:latest
sudo vi /etc/docker/daemon.json
{
  "insecure-registries":["{NodeIP}:{NodePort}"]
}

docker push {NodeIP}:{NodePort}/influxdb:latest
```

### Helm
```
helm dependency update ./redfishread
helm install test redfishread --dry-run
```

## CI/CD
### GitHub Actions

View GHCR Images
https://github.com/TrevorSquillario?tab=packages


## Troubleshooting
### Reset
```
# Reset ALL
docker system prune -a --volumes

# Reset All Volumes
docker volume rm $(docker volume ls -q)
```

```
docker cp ./cmd/redfishread/redfishread.go redfishread:/build/cmd/redfishread/redfishread.go
```

### Docker Error: You have reached your pull rate limit
Login to docker to fix
```
docker login -u username
```

### Fix for  curl: (6) Could not resolve host: influx
Apparently the search suffix breaks internal name resolution for curl
```
docker exec -it setup /bin/sh
vi /etc/resolve.conf
# search x.x.x.x
```