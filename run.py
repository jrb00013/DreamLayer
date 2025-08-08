
# This file allows the user to build, run, and develop the DreamLayer code reliably with a single command
# python run.py

import os
import docker
import sys
import docker_utils as docker_utils
NETWORK_NAME = "dreamlayer"
current_dir = os.path.dirname(os.path.abspath(__file__))
webapp_dir = os.path.join(current_dir, "client")

print("Making sure the Docker network is up")
docker_utils.ensure_network(NETWORK_NAME)

comfyui_dockerfile = '''
# Use specific PyTorch version compatible with ComfyUI
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CI=true

# Set the working directory
WORKDIR /app

# Update package lists and install basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip first
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Copy ComfyUI files
COPY ComfyUI/ ./ComfyUI/

# Install ComfyUI requirements
RUN pip install -r ComfyUI/requirements.txt

# Expose the port used by ComfyUI
EXPOSE 8188

# Set the default command to start ComfyUI
CMD ["python3", "ComfyUI/main.py"]
'''
with open("ComfyUIDockerfile", 'w+') as f:
    f.write(comfyui_dockerfile)

frontend_dockerfile = '''
# Use Pytorch as the base image (shared with ComfyUI)
FROM pytorch/pytorch

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CI=true

# Set the working directory
WORKDIR /app

# Update package lists and install basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    lsof \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip first
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Pre-install Node.js (shared with frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY dream_layer_backend/ ./dream_layer_backend/
COPY start_dream_layer.sh .

# Install backend requirements
RUN pip install -r dream_layer_backend/requirements.txt

# Expose ports used by Dream Layer backend
EXPOSE 5001 5002 5003 5004

# Set the default command to start the backend
CMD ["./start_dream_layer.sh", "--backend"]
'''
with open("DreamLayerBackendDockerfile", 'w+') as f:
    f.write(frontend_dockerfile)

frontend_dockerfile = '''
# Use Pytorch as the base image (shared with other services)
FROM pytorch/pytorch

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CI=true

# Set the working directory
WORKDIR /app

# Update package lists and install basic utilities
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Node.js (shared with backend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copy frontend files
COPY dream_layer_frontend/ ./dream_layer_frontend/

# Install frontend dependencies
WORKDIR /app/dream_layer_frontend
RUN npm install && npm install -g vite

# Expose port used by Dream Layer frontend
EXPOSE 8080

# Set the default command to start the frontend
WORKDIR /app/dream_layer_frontend
CMD ["sh", "-c", "npm install && npm run dev"]

'''
with open("DreamLayerFrontendDockerfile", 'w+') as f:
    f.write(frontend_dockerfile)


# Build Docker images
if "build" in sys.argv:
    docker_utils.build_image("ComfyUIDockerfile", "dreamlayer-comfyui")
    docker_utils.build_image("DreamLayerBackendDockerfile", "dreamlayer-backend") 
    docker_utils.build_image("DreamLayerFrontendDockerfile", "dreamlayer-frontend")


# ComfyUI container
comfyui = dict(
    image="dreamlayer-comfyui",
    name="dreamlayer-comfyui",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    detach=True,
    volumes={
        current_dir: {"bind": "/app", "mode": "rw"}
    },
    ports={"8188/tcp": 8188},
    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
    command=["python3", "ComfyUI/main.py", "--listen", "0.0.0.0", "--enable-cors-header", "http://localhost:8080"]
)

# Backend container
backend = dict(
    image="dreamlayer-backend",
    name="dreamlayer-backend",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    detach=True,
    volumes={
        current_dir: {"bind": "/app", "mode": "rw"},
        os.path.join(current_dir, "ComfyUI", "output"): {"bind": "/app/Dream_Layer_Resources/output/", "mode": "rw"},
    },
    ports={
        "5001/tcp": 5001,
        "5002/tcp": 5002,
        "5003/tcp": 5003,
        "5004/tcp": 5004
    },
    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
    environment={
        "COMFY_API_URL": "http://dreamlayer-comfyui:8188"
    },
)

# Frontend container
frontend = dict(
    image="dreamlayer-frontend",
    name="dreamlayer-frontend",
    network=NETWORK_NAME,
    restart_policy={"Name": "always"},
    detach=True,
    volumes={
        os.path.join(current_dir, "dream_layer_frontend"): {"bind": "/app", "mode": "rw"}
    },
    ports={"8080/tcp": ("0.0.0.0", 8080)},
    working_dir="/app/",
    environment={
        "VITE_BACKEND_API_BASE_URL": "http://localhost:5002"
    },
    command=["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8080"]
)

# Run containers
docker_utils.run_container(comfyui)
docker_utils.run_container(backend)
docker_utils.run_container(frontend)
