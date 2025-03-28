# Local Distributed Parallel LLM Inferencing

A distributed inferencing platform that simplifies LLM sharding and parallel inferencing within a local system. This system leverages parallel computing libraries to break down and process large language model (LLM) inference tasks in a distributed manner.

## System Architecture

The architecture follows a STAR topology with a single Master Node (Hub) coordinating multiple Worker Nodes, all running on consumer-grade laptops connected via a local network.

- **Master Node (Hub)**: Acts as the centralized controller and the single point of inference for the user.
- **Worker Nodes**: Operate on the local network and connect to the Master Node.

## Project Structure

The project is organized into the following main components:

- **master/**: Contains the Django-based Master Node implementation that:
  - Manages the web dashboard for user interaction
  - Coordinates inference tasks across worker nodes
  - Handles model management and sharding configuration
  - Processes result aggregation from workers

- **worker/**: Contains the Flask-based Worker Node implementation that:
  - Loads and runs LLM inference tasks
  - Manages local model caching
  - Handles tensor parallelism for distributed inference
  - Reports status and results back to the master node

- **docker-compose.yml**: Defines the multi-container Docker application with:
  - Service definitions for master and worker nodes
  - Network configuration for node communication
  - Volume mappings for persistent model storage
  - Environment variable settings for customization

This modular architecture allows for easy scaling by adding more worker nodes as needed, while maintaining a single point of control through the master node.

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
