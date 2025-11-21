# Scripts Directory

Efficiently manage and maintain the developer example application using the following scripts. These utilities help automate cleanup, manage resources, and streamline your workflow.

## Prerequisites

- Docker and Docker Compose must be installed.
- The `uv` package manager must be installed.
- Scripts assume you are running in the project root directory unless otherwise specified.
- Note: Volume cleanup scripts automatically manage service lifecycle - manual shutdown is not required.

---

## Scripts

### `deploy-nmp.sh`

Deployment script for NVIDIA NeMo microservices (NMP) setup. This is a comprehensive deployment script with specialized configuration for enterprise environments.

**Key Features:**

- **Helm Chart Management**: Uses repository-based Helm chart fetching with automatic latest version detection. The script fetches the NeMo microservices Helm chart from the NGC repository (`nemo-microservices/nemo-microservices-helm-chart`) and uses the latest version by default. You can override the version using the `--helm-chart-version` flag.

- **Minikube Configuration**: Automatically configures Minikube with 500 GB disk space to handle large container images. This prevents "no space left on device" errors during container image pulls. The disk size setting persists in the systemd service configuration for automatic application across restarts.

- **Enterprise Configuration**: Contains advanced deployment logic and enterprise-specific configurations.

- **Requirements**: Requires specific NMP credentials and environment setup (NGC_API_KEY, NVIDIA_API_KEY).

**Usage:**

```bash
./scripts/deploy-nmp.sh [--helm-chart-version VERSION]
```

**Environment Variables:**

- `NGC_API_KEY`: Required for NGC login and container downloads
- `NVIDIA_API_KEY`: Required for remote NIM access
- `HELM_CHART_REPO`: Helm chart repository (default: `nemo-microservices/nemo-microservices-helm-chart`)
- `HELM_CHART_VERSION`: Helm chart version (default: latest, empty string)

> **Note:** This script is designed for Minikube-based deployments. For production Kubernetes deployments, refer to the [Helm Installation Guide](11-helm-installation.md).

### `generate_openapi.py`

Python script to generate the OpenAPI specification for the API.

- Imports the FastAPI app and writes the OpenAPI schema to `openapi.json` (or a user-specified path).
- Validates the output path for safety.
- Can be run as `python scripts/generate_openapi.py [output_path.json]`.

### `run.sh`

- Stops any running containers, then starts the main application stack using Docker Compose.
- Builds images as needed.
- Runs MongoDB in detached mode without attaching logs, to reduce log noise.

### `run-dev.sh`

- Stops any running containers, then starts the application stack with both the main and development Docker Compose files.
- Builds images as needed.
- Runs MongoDB, Elasticsearch, and Kibana in detached mode (no logs attached).
- Ensures development UIs are available.

### `stop.sh`

- Stops all running containers for both the main and development Docker Compose files.

### Volume Cleanup Scripts

- `clear_es_volume.sh`, `clear_redis_volume.sh`, `clear_mongodb_volume.sh`, `clear_mlflow_volume.sh`—Each script:
  - Stops the relevant service container (Elasticsearch, Redis, MongoDB, or MLflow).
  - Removes the associated Docker volume to clear all stored data.
  - Restarts the service container to ensure the service is running with a fresh, empty volume.
  - Prints status messages for each step.
- `clear_all_volumes.sh`—A convenience script to clear all persistent data volumes used by the application. It sequentially calls the four volume cleanup scripts above (Elasticsearch, Redis, MongoDB, and MLflow) and restarts all services.

### `check_requirements.sh`

A script to ensure your `requirements.txt` is in sync with your `pyproject.toml`:

- Uses `uv` to generate a temporary list of installed packages.
- Compares this list to `requirements.txt`.
- If out of sync, prints a diff and instructions to update.
- Exits with an error if not up to date, otherwise confirms success.

### `quick-test.sh`

A minimal script to verify that the API is running and responsive:

- Sends a POST request to `http://localhost:8001/jobs` with a test payload.
- Useful for smoke-testing the local API after startup.
