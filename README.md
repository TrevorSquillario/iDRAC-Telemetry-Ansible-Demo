# iDRAC-Telemetry-Ansible-Demo

## Getting Started
1. Setup Python Virtual Environment (optional but recommended)
```
pip install virtualenv
mkdir ~/venv
python -m virtualenv ~/venv/ansible-dell
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
sudo dnf install python3-pip
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
    - Generates the `docker-compose/.env` file which is used by docker compose to configure the containers
- `idrac-setup`
    - Configures each iDRAC in the `idrac` host group to enable Telemetry globally and enable some Metric Reports for testing. 
    - Configures NTP on each iDRAC so the time is all in sync. 
- `docker-compose-start`
    - Runs `docker compose up` to start containers
    - Sets up InfluxDB instance with user and permissions for the `influxpump`
    - Configures Grafana instance
    - Restarts container instances to pickup new environment variables

### Your grafana instance should now be available at http://ip running on port `80` with username `admin` and password `admin`

# Docker
A `docker-compose.yaml` file will be generated under the `docker-compose` folder. You can use that to manage your enviroment as well.

# Kubernetes
```
cd docker-compose
kompose convert -f docker-compose.yaml -o k8s
```

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