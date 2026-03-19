# CI Runner Architecture: Why Minikube Works

## Background

The `ai-model-distillation-for-financial-data` notebook requires a full Kubernetes
cluster (via Minikube) with GPU passthrough to deploy NeMo Microservices. Intuitively,
running Minikube inside a Kubernetes pod (the ARC runner) should fail — you would be
attempting to run "Kubernetes inside Kubernetes inside a container." However, empirical
testing shows that **Minikube starts and operates correctly** in this environment.

This document explains **why** it works, the architectural implications, and caveats
to be aware of.

---

## Environment Facts (from feasibility test)

| Observation | Value | Significance |
|---|---|---|
| Container detection | `NO (bare metal/VM-like)` | No `/.dockerenv`, no container cgroup markers |
| PID 1 | `run.sh` | ARC runner entrypoint, not `systemd` or `init` |
| Docker version | `28.5.1` | Full-featured Docker daemon (not a minimal DinD) |
| Docker socket | `/var/run/docker.sock` present | Mounted from the host node |
| `nvidia-ctk` | Available on `PATH` | Host-level NVIDIA Container Toolkit |
| `nvidia-smi` (direct) | Not on `PATH` | GPU drivers are on the host, not exposed as binaries |
| `nvidia-smi` (via docker) | Works with `--gpus all` | Docker daemon has GPU runtime configured |
| `minikube start --driver=docker --gpus=all` | Succeeds | Minikube container created on the host |
| Kernel | Host kernel (not a guest/nested kernel) | Runner shares the host kernel |

---

## Root Cause: Host Docker Socket Mounting

The ARC (Actions Runner Controller) runner pod is configured to **mount the host
node's Docker socket** (`/var/run/docker.sock`) into the pod. This is the single
architectural decision that makes everything work.

### How It Works

```
┌──────────────────────── K8s Host Node ─────────────────────────────┐
│                                                                     │
│  Docker Daemon (v28.5.1)                                           │
│  ├── Configured with nvidia-ctk (GPU runtime)                      │
│  ├── Has access to all host GPUs                                   │
│  └── Listens on /var/run/docker.sock                               │
│         ▲                                                           │
│         │  volume mount                                             │
│         │                                                           │
│  ┌──────┴──────── ARC Runner Pod ──────────────────┐               │
│  │                                                  │               │
│  │  /var/run/docker.sock ──► Host Docker Daemon     │               │
│  │                                                  │               │
│  │  When runner executes:                           │               │
│  │    docker run ...        ──► runs on HOST        │               │
│  │    docker compose up ... ──► runs on HOST        │               │
│  │    minikube start        ──► creates container   │               │
│  │                               on HOST            │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                     │
│  ┌── minikube container (lives on host, not inside pod) ──┐        │
│  │                                                         │        │
│  │  Kubernetes cluster (single-node)                       │        │
│  │  ├── kube-apiserver                                     │        │
│  │  ├── kubelet                                            │        │
│  │  ├── GPU passthrough via host nvidia runtime            │        │
│  │  └── NeMo Microservices (Helm-installed)                │        │
│  │                                                         │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
│  ┌── Docker Compose containers (also on host) ────────────┐        │
│  │  redis, mongodb, elasticsearch, mlflow, api, celery     │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution Flow

1. **GitHub Actions triggers a job** on `arc-runner-set-oke-org-poc-2-gpu`.
2. ARC creates a pod on the Kubernetes cluster. The pod spec includes a volume mount
   for `/var/run/docker.sock` from the host node.
3. The runner's `docker` CLI connects to the **host's Docker daemon** through the
   mounted socket — not a Docker daemon running inside the pod.
4. When `minikube start --driver=docker --gpus=all` runs:
   - Minikube asks Docker to create a container.
   - Docker creates that container **on the host**, as a sibling to the runner pod
     (not nested inside it).
   - The `--gpus=all` flag works because the host's Docker daemon has `nvidia-ctk`
     configured as a runtime.
5. The minikube container now has a fully functional Kubernetes cluster with GPU access.
6. `kubectl` commands from the runner communicate with this minikube cluster through
   the standard kubeconfig that `minikube start` writes to `~/.kube/config`.
7. Docker Compose services (Redis, MongoDB, etc.) also run as host-level containers.

### Why It Appears as "Bare Metal"

The environment detection reports "bare metal/VM-like" because:

- **No `/.dockerenv`**: This file is created by Docker inside containers, but the ARC
  runner pod is managed by Kubernetes (containerd), not Docker directly.
- **No container cgroup markers**: On cgroup v2 systems (modern kernels), the
  traditional `kubepods` or `docker` markers in `/proc/1/cgroup` may not be present.
- **Host kernel is shared**: The runner sees the host's kernel (`uname -r`) directly.
- **Docker commands operate at host level**: Since all Docker operations go through the
  host socket, the runner effectively has host-level Docker access.

---

## Key Implications for CI

### What Works

| Capability | Status | Notes |
|---|---|---|
| `minikube start --driver=docker --gpus=all` | Works | Container created on host |
| `helm install` NeMo Microservices | Works | Minikube cluster has GPU access |
| `docker compose up` (Data Flywheel) | Works | Containers created on host |
| `nvidia-smi` via Docker | Works | Host Docker has nvidia runtime |
| `/etc/hosts` append (`tee -a`) | Works | Can add DNS entries |
| `kubectl` operations | Works | Standard kubeconfig from minikube |

### What Does NOT Work

| Capability | Status | Notes |
|---|---|---|
| `sed -i` on `/etc/hosts` | Fails | Mounted filesystem does not support atomic rename. Use `tee -a` (append) instead. `deploy-nmp.sh` uses `tee -a` for first-time writes, so this is only an issue if entries need to be updated. |
| `systemctl` / systemd | Not PID 1 | PID 1 is `run.sh`, not systemd. `deploy-nmp.sh` may call systemd commands but typically handles this gracefully. |
| `nvidia-smi` (direct binary) | Not on PATH | Use `docker run --gpus all ... nvidia-smi` instead. The GPU is accessible through Docker, just not as a host binary. |

### Caveats and Risks

1. **Shared Docker Daemon**: All runner pods on the same host node share a single
   Docker daemon. If two CI jobs run simultaneously on the same node, they may
   experience:
   - Port conflicts (e.g., two minikube instances trying to bind the same ports)
   - Container name collisions
   - Shared docker network namespace

2. **Host-Level Side Effects**: Containers created by the CI job are **not** cleaned
   up when the runner pod is deleted. The cleanup step in the workflow must explicitly
   `minikube delete` and `docker rm` all resources. Orphaned containers will persist
   on the host.

3. **Security**: The runner has root-equivalent access to the host's Docker daemon.
   Any code running in CI can create privileged containers, mount host filesystems,
   and access GPUs. This is acceptable for a trusted CI environment but should not
   be used for untrusted workloads (e.g., pull requests from external contributors).

4. **Resource Contention**: Minikube and Docker Compose services consume host
   resources (CPU, memory, GPU memory) without Kubernetes resource limits. A single
   CI job running the full notebook may consume significant resources on the host node.

---

## Recommendations

1. **Ensure only one notebook CI job runs at a time** on a given host node to avoid
   resource and port conflicts. Use GitHub Actions concurrency groups:
   ```yaml
   concurrency:
     group: notebook-ci-${{ github.ref }}
     cancel-in-progress: true
   ```

2. **Always clean up** in a `post` or `if: always()` step:
   ```yaml
   - name: Cleanup
     if: always()
     run: |
       minikube delete 2>/dev/null || true
       docker compose -f deploy/docker-compose.yaml down -v 2>/dev/null || true
       docker rm -f $(docker ps -aq) 2>/dev/null || true
       docker network prune -f 2>/dev/null || true
   ```

3. **Use `tee -a`** (not `sed -i`) for `/etc/hosts` modifications. Since the runner
   is ephemeral, there is no need to clean up `/etc/hosts` entries after the job.

4. **Install tools at runtime**: `minikube`, `kubectl`, `helm`, and `uv` are not
   pre-installed on the runner. The CI workflow must install them as part of the job.
   Consider caching these binaries if job startup time is a concern.

---

## Verification

The `test-minikube-feasibility.yaml` workflow validates all of the above. Run it
manually via `workflow_dispatch` or push a change to the workflow file to trigger it:

```bash
gh workflow run "Test Minikube Feasibility"
```

The final summary step produces a pass/fail checklist for every prerequisite the
notebook requires.
