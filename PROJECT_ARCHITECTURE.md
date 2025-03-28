# Project Architecture: Distributed Inference System

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architectural Components](#architectural-components)
4. [Model Sharding Procedure](#model-sharding-procedure)
5. [Data Flow](#data-flow)
6. [Setup and Deployment](#setup-and-deployment)
7. [API Endpoints](#api-endpoints)
8. [Troubleshooting](#troubleshooting)

## Overview

This project implements a distributed inference system for large language models using Django. The system allows splitting large AI models into smaller shards that can be distributed across multiple worker nodes, enabling distributed inference even when a single machine cannot hold the entire model in memory.

The system consists of a central Django application that manages worker nodes, model shards, and inference requests, along with a network of worker nodes that perform the actual inference computations.

### System Architecture Diagram

mermaid
graph TD
    A[Django Master Application] --> B[Dashboard App]
    A --> C[Shard Management App]
    C --> D[Worker Node Management]
    C --> E[Model Sharding]
    C --> F[Inference Coordination]
    
    B --> G[Web Interface]
    G --> H[Node Status View]
    G --> I[Inference Request Form]
    
    F --> J[Worker Node 1]
    F --> K[Worker Node 2]
    F --> L[Worker Node N]
    
    J --> M[Model Shard 1]
    K --> N[Model Shard 2]
    L --> O[Model Shard N]
    
    P[Client] --> G
    P --> Q[REST API]
    Q --> C


## Project Structure

The Django project follows a modular structure with two main apps:

- **dashboard**: Provides the web interface for monitoring system status and submitting inference requests
- **shard_mgmt**: Handles all aspects of worker node management, model sharding, and distributed inference


├── dashboard/               # Dashboard application
│   ├── templates/           # HTML templates for the web interface
│   ├── views.py             # View functions for dashboard and inference UI
│   ├── urls.py              # URL routing for dashboard views
│   └── management/          # Django management commands
│       └── commands/        
│           └── shard_model.py  # Model sharding command (duplicate for compatibility)
│
├── shard_mgmt/              # Shard management application
│   ├── models.py            # Data models for nodes, shards, and inference requests
│   ├── views.py             # View functions for node and inference APIs
│   ├── forms.py             # Forms for node creation and inference requests
│   ├── urls.py              # URL routing for shard management APIs
│   ├── admin.py             # Django admin interface configuration
│   └── management/          # Django management commands
│       └── commands/
│           └── shard_model.py  # Model sharding command
│
├── master/                  # Django project configuration
│   ├── settings.py          # Django settings
│   ├── urls.py              # Main URL configuration
│   ├── wsgi.py              # WSGI configuration
│   └── asgi.py              # ASGI configuration
│
├── manage.py                # Django command-line utility
├── Dockerfile               # Docker configuration
└── requirements.txt         # Python dependencies


## Architectural Components

### Models

The system uses three main data models to manage distributed inference:

#### WorkerNode (shard_mgmt/models.py)

Represents a computational node that can host model shards and perform inference:

python
class WorkerNode(models.Model):
    hostname = models.CharField(max_length=255)
    ip_address = models.CharField(max_length=255)
    port = models.IntegerField(default=5000)
    is_active = models.BooleanField(default=False)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


Each worker node provides an HTTP API for loading models, running inference, and checking health status.

#### ModelShard (shard_mgmt/models.py)

Represents a portion of a machine learning model assigned to a specific worker node:

python
class ModelShard(models.Model):
    node = models.ForeignKey(WorkerNode, on_delete=models.CASCADE, related_name='shards')
    model_name = models.CharField(max_length=255)
    shard_id = models.IntegerField()
    is_loaded = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


Multiple shards of the same model can be distributed across different worker nodes.

#### InferenceRequest (shard_mgmt/models.py)

Represents a request to run inference on a specific model:

python
class InferenceRequest(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    model_name = models.CharField(max_length=255)
    prompt = models.TextField()
    result = models.TextField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)


Inference requests track their execution status and store both input prompts and generated outputs.

### Views

The system's functionality is exposed through Django views that handle HTTP requests:

#### Dashboard Views (dashboard/views.py)

- `dashboard`: Displays system status, including node health and recent inference requests
- `inference`: Provides a form for submitting new inference requests

#### Shard Management Views (shard_mgmt/views.py)

- `node_management`: Interface for adding and managing worker nodes
- `node_status`: API for checking worker node health status
- `add_node`, `remove_node`: APIs for worker node management
- `submit_inference`: API for submitting inference requests
- `inference_status`, `recent_inferences`: APIs for monitoring inference progress
- `process_inference_request`: Background task handler for inference execution

## Model Sharding Procedure

The model sharding process is a key feature that enables distributed inference. It is implemented as a Django management command in `shard_mgmt/management/commands/shard_model.py`.

### Sharding Process Overview

1. **Load the original model**: The complete model is loaded using Hugging Face's Transformers library.
2. **Analyze model architecture**: The command determines the model's architecture (GPT-2, OPT, etc.) and identifies layers that can be distributed.
3. **Calculate layer distribution**: Based on the requested number of shards, the command calculates how to distribute layers across shards.
4. **Create and save shards**: For each shard, a new model instance is created containing only the relevant layers, along with metadata.
5. **Register in database**: Optionally, the shards can be registered in the database and associated with a worker node.

### Executing the Sharding Command

The sharding command can be run via the Django management interface:

bash
python manage.py shard_model --model_name="gpt2" --num_shards=4 --output_dir="model_shards" --device="cpu" --register_db --node_id=1


Parameters:
- `model_name`: Hugging Face model identifier
- `num_shards`: Number of shards to create
- `output_dir`: Directory where shards will be saved
- `device`: Computing device (cpu or cuda)
- `register_db`: Flag to register shards in the database
- `node_id`: Worker node ID if registering shards

### Shard Structure

Each shard includes:

1. **Model weights**: A subset of the original model's layers
2. **Configuration file**: Model architecture settings
3. **Metadata file**: Information about the shard's position in the overall model

The metadata file (`metadata.json`) includes:
- Model name
- Shard ID
- Total number of shards
- Start and end layer indices
- Total layer count
- Model architecture type

## Data Flow

### Inference Request Flow

The flow of an inference request in the system:

mermaid
sequenceDiagram
    participant User
    participant Django
    participant WorkerNode
    
    User->>Django: Submit inference request
    Django->>Django: Create InferenceRequest record
    Django->>Django: Start background thread
    Django-->>User: Return request ID
    
    par Background Processing
        Django->>Django: Look up available shards
        alt Shards available
            Django->>WorkerNode: Send inference request with shard IDs
            WorkerNode->>WorkerNode: Load specified shards
            WorkerNode->>WorkerNode: Run inference
            WorkerNode-->>Django: Return results
        else No shards available
            Django->>WorkerNode: Request to load full model
            WorkerNode->>WorkerNode: Load model
            Django->>WorkerNode: Send inference request
            WorkerNode->>WorkerNode: Run inference
            WorkerNode-->>Django: Return results
        end
        Django->>Django: Update InferenceRequest status and result
    end
    
    User->>Django: Poll for inference status
    Django-->>User: Return current status and results if complete


### Worker Node Management Flow

The flow for adding and managing worker nodes:

mermaid
sequenceDiagram
    participant Admin
    participant Django
    participant WorkerNode
    
    Admin->>Django: Submit node details (IP, port)
    Django->>WorkerNode: Health check request
    alt Node reachable
        WorkerNode-->>Django: Health data
        Django->>Django: Create WorkerNode record
        Django-->>Admin: Confirm addition
    else Node unreachable
        Django-->>Admin: Report connection error
    end
    
    loop Health Monitoring
        Django->>WorkerNode: Periodic health check
        alt Node healthy
            WorkerNode-->>Django: Health data
            Django->>Django: Update last_heartbeat
        else Node unreachable
            Django->>Django: Mark node as inactive
        end
    end


## Setup and Deployment

### Prerequisites

- Docker and Docker Compose
- At least one worker node running the worker API service
- Sufficient disk space for model storage

### Environment Variables

- `SECRET_KEY`: Django secret key
- `DEBUG`: Enable debug mode (0 or 1)
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`: Redis configuration
- `MODEL_CACHE_DIR`: Directory for storing model shards

### Deployment Steps

1. **Clone the repository**

bash
git clone <repository-url>
cd <project-directory>


2. **Build and start the Docker container**

bash
docker build -t distributed-inference .
docker run -d -p 8000:8000 -v /path/to/model/storage:/app/model_cache distributed-inference


3. **Apply database migrations**

bash
docker exec -it <container_id> python manage.py migrate


4. **Create superuser for admin access**

bash
docker exec -it <container_id> python manage.py createsuperuser


5. **Accessing the application**

- Dashboard: http://localhost:8000/
- Admin interface: http://localhost:8000/admin/
- Node management: http://localhost:8000/shard/nodes/

### Worker Node Setup

Each worker node should run a compatible API service that implements:

- `/health` endpoint for status checks
- `/load_model` endpoint for loading models or shards
- `/unload_model` endpoint for freeing resources
- `/inference` endpoint for processing inference requests

## API Endpoints

### Dashboard Endpoints

- `GET /` - Dashboard home showing system status
- `GET /inference/` - Inference submission form

### Shard Management Endpoints

- `GET /shard/nodes/` - Node management interface
- `GET /shard/api/nodes/status/` - Get status of all worker nodes
- `POST /shard/api/nodes/add/` - Add a new worker node
- `POST /shard/api/nodes/remove/<int:node_id>/` - Remove a worker node
- `POST /shard/api/inference/submit/` - Submit an inference request
- `GET /shard/api/inference/status/<int:request_id>/` - Check inference status
- `GET /shard/api/inference/recent/` - Get recent inference requests

## Troubleshooting

### Common Issues

1. **Worker nodes not connecting**
   - Verify network connectivity
   - Check that the worker API is running
   - Ensure firewall rules allow traffic on the specified port

2. **Model sharding failures**
   - Check available memory on the machine running the sharding command
   - Verify the model is supported by the sharding process
   - Check disk space for shard storage

3. **Inference request failures**
   - Check that worker nodes are active
   - Verify that model shards are properly loaded
   - Check logs for detailed error messages
