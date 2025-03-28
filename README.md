# Local Distributed Parallel LLM Inferencing

A distributed inferencing platform that simplifies LLM sharding and parallel inferencing within a local system. This system leverages parallel computing libraries to break down and process large language model (LLM) inference tasks in a distributed manner.

## System Architecture

The architecture follows a STAR topology with a single Master Node (Hub) coordinating multiple Worker Nodes, all running on consumer-grade laptops connected via a local network.

- **Master Node (Hub)**: Acts as the centralized controller and the single point of inference for the user.
- **Worker Nodes**: Operate on the local network and connect to the Master Node.

## Project Structure

The project is organized with a clear separation of responsibilities between components:

### Master Node (`master/`)

The Django-based Master Node serves as the central coordinator for the entire system:

- **`master/settings.py`**: Django configuration file containing database settings, Redis connection parameters, and security configurations.
- **`master/urls.py`**: URL routing definitions for the web dashboard and REST API endpoints.
- **`master/models/`**:
  - **`node.py`**: Defines the `WorkerNode` model for tracking connected worker instances, their capabilities, and health status.
  - **`request.py`**: Contains the `InferenceRequest` model that tracks user prompts, assigned workers, and inference results.
  - **`model.py`**: Implements the `LLMModel` and `ModelShard` models for tracking available models and their distribution across workers.

- **`master/views/`**:
  - **`dashboard.py`**: Renders the main web interface for users to submit prompts and monitor inference.
  - **`api.py`**: REST API endpoints for worker registration, status updates, and result collection.
  - **`nodes.py`**: Node management interface for adding, testing, and removing worker nodes.

- **`master/services/`**:
  - **`orchestrator.py`**: Core logic for distributing inference tasks across available worker nodes.
  - **`sharding.py`**: Handles model partitioning for large models that need to be split across multiple workers.
  - **`result_aggregator.py`**: Collects and combines partial results from workers for final output.

- **`master/static/`**: Contains CSS, JavaScript, and image files for the web dashboard.
- **`master/templates/`**: HTML templates for the user interface.

### Worker Node (`worker/`)

The Flask-based Worker Node handles the actual model loading and inference execution:

- **`worker/app.py`**: Main Flask application that defines API endpoints for the master to communicate with.
- **`worker/models/`**:
  - **`model_manager.py`**: Handles downloading, caching, and loading of LLM models from Hugging Face.
  - **`inference.py`**: Contains the inference pipeline that processes prompts using loaded models.
  - **`sharded_model.py`**: Implements specialized handling for partial model shards in distributed inference.

- **`worker/utils/`**:
  - **`gpu_utils.py`**: Utilities for GPU memory management and CUDA optimization.
  - **`cache.py`**: Model caching functions to avoid redundant downloads.
  - **`auth.py`**: Authentication mechanisms for secure communication with the master.

- **`worker/config.py`**: Configuration parameters and environment variable processing.
- **`worker/requirements.txt`**: Python dependencies specific to worker nodes.

### Docker Configuration (`docker-compose.yml`)

The Docker Compose configuration orchestrates multi-container deployment:

- **Services**:
  - **`master`**: Configures the Django master service with network settings and volume mounts.
  - **`worker1`, `worker2`, etc.**: Worker node services that can be scaled horizontally.
  - **`redis`**: Message broker for asynchronous communication between nodes.

- **Networks**: Defines internal networks for secure container communication.
- **Volumes**: 
  - **`model-cache`**: Persistent storage for downloaded models.
  - **`db-data`**: Database persistence for the master node.

- **Environment Variables**: Configuration for GPU support, authentication, and service discovery.

### Data Flow and Request Lifecycle

A typical inference request follows this path through the system:

1. User submits a prompt through the master node's web interface
2. Master's `orchestrator.py` analyzes the request and selected model
3. For standard models:
   - Master selects the most available worker node
   - Request is forwarded to the worker via REST API
   - Worker performs inference and returns results
   
4. For sharded models:
   - Master's `sharding.py` divides the model into parts
   - Parts are assigned to different workers based on capability and load
   - Workers load their respective shards
   - Inference is performed in coordinated fashion, with intermediate tensors passed between workers
   - Final worker in the chain returns complete results
   
5. Results are stored in the database and displayed to the user

This architecture allows for flexible scaling and efficient use of distributed computing resources for LLM inference.

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- Local network connectivity between all nodes
- 8GB+ RAM on each node (16GB+ recommended for larger models)

## Installation

### Docker Setup (Recommended)

The easiest way to get started is using Docker and docker-compose.

1. **Clone the repository**:
   bash
   git clone https://github.com/yourusername/local-distributed-llm.git
   cd local-distributed-llm
   

2. **Configure Environment**:
   
   Modify the `docker-compose.yml` file to set your desired configuration:
   
   - To enable GPU support, change `USE_GPU=0` to `USE_GPU=1` in the build args and environment variables.
   - Update the `SECRET_KEY` for production use.

3. **Start the services**:
   bash
   docker-compose up -d
   

4. **Access the Dashboard**:
   
   Open your browser and navigate to `http://localhost:8000`

### Manual Setup

If you prefer to run without Docker, follow these steps:

1. **Clone the repository**:
   bash
   git clone https://github.com/yourusername/local-distributed-llm.git
   cd local-distributed-llm
   

2. **Set up the Master Node**:
   bash
   cd master
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Initialize the database
   python manage.py migrate
   
   # Create a superuser (for admin access)
   python manage.py createsuperuser
   
   # Start the server
   python manage.py runserver 0.0.0.0:8000
   

3. **Set up Worker Nodes** (repeat on each worker machine):
   bash
   cd worker
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Start the worker service
   export MODEL_CACHE_DIR=/path/to/model/cache  # Optional: Set custom model cache directory
   export USE_GPU=0  # Set to 1 to enable GPU support
   
   # Start the Flask server
   gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 300 app:app
   

## Configuration

### Master Node Configuration

The Master Node can be configured through environment variables:

- `DEBUG`: Set to 1 for development, 0 for production
- `SECRET_KEY`: Django secret key (change for production)
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `MODEL_CACHE_DIR`: Directory to cache downloaded models

### Worker Node Configuration

Worker Nodes can be configured through environment variables:

- `MODEL_CACHE_DIR`: Directory to cache downloaded models
- `USE_GPU`: Set to 1 to enable GPU support, 0 for CPU only
- `AUTH_ENABLED`: Set to 1 to enable authentication
- `AUTH_KEY`: Authentication key when AUTH_ENABLED is 1

## Usage

### Adding Worker Nodes

1. Ensure your Worker Node services are running
2. In the Master Node dashboard, navigate to "Node Management"
3. Click "Add Node" and enter the hostname/IP and port of the Worker Node
4. The system will automatically check if the node is reachable
5. Once added, the node will appear in the nodes list

### Running Inference

1. Navigate to the "Inference" tab in the dashboard
2. Select a model from the dropdown (or enter a Hugging Face model name)
3. Enter your prompt in the text area
4. Click "Submit" to start the inference process
5. The system will automatically distribute the workload across available nodes
6. Results will appear in the "Recent Requests" section once completed

### Model Sharding

For large models that don't fit on a single machine:

1. In the "Model Management" tab, select a model to shard
2. Choose the number of shards and assign them to different nodes
3. The system will automatically handle the sharding and reassembly during inference

## Troubleshooting

### Node Connection Issues

If you're having trouble connecting to Worker Nodes:

- Ensure all machines are on the same network
- Check firewall settings and ensure ports are open
- Verify that the Worker Node service is running
- Use the "Test Connection" button in Node Management to diagnose issues

### GPU Issues

If you're experiencing problems with GPU acceleration:

- Verify that CUDA is properly installed
- Check that the appropriate GPU drivers are installed
- Ensure the `USE_GPU` environment variable is set to 1
- Check GPU memory usage with `nvidia-smi` command

### Model Loading Failures

If models fail to load:

- Check disk space on the worker nodes
- Ensure internet connectivity for downloading models
- Review the logs for specific error messages
- Try with a smaller model to test the system

## Monitoring and Logs

### Master Node Logs

bash
# If using Docker
docker-compose logs -f master

# If running manually
tail -f master/logs/django.log


### Worker Node Logs

bash
# If using Docker
docker-compose logs -f worker1

# If running manually
tail -f worker/logs/worker.log


## Performance Tuning

For optimal performance:

- Use GPU acceleration when available
- Adjust the number of worker processes based on your hardware
- Consider using smaller models or more worker nodes for faster inference
- Ensure sufficient RAM and disk space for model caching

## License

This project is licensed under the MIT License - see the LICENSE file for details.
